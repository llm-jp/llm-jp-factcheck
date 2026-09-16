"""Multi-turn chat with independent fact-checks for each model response."""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
from pathlib import Path
from uuid import uuid4

import streamlit as st

from chat import stream_chat_response
from pipeline import PipelineConfig, run_factcheck

logger = logging.getLogger(__name__)
DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"

LABELS = {
    "Supported": "supported",
    "Partially supported": "partially-supported",
    "Partially refuted": "partially-refuted",
    "Refuted": "refuted",
    "Not enough information": "nei",
}


def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("num_evidences must be greater than zero")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with a model and fact-check responses using independent Chatbot and Factchecker connections.",
        epilog=(
            "Configure OpenAI-compatible or Azure OpenAI endpoints with CHATBOT_* and FACTCHECKER_* "
            "environment variables."
        ),
    )
    parser.add_argument(
        "--engine",
        default=None,
        help="Factchecker model or Azure deployment (overrides FACTCHECKER_MODEL; default: gpt-4-0613).",
    )
    parser.add_argument(
        "--chat-engine",
        "--chat_engine",
        default=None,
        help="Chatbot model or Azure deployment (overrides CHATBOT_MODEL; defaults to the Factchecker model).",
    )
    parser.add_argument(
        "--tokenizer_name",
        "--tokenizer-name",
        type=str,
        default=None,
        help="Retrieval tokenizer (overrides TOKENIZER_NAME; default: llm-jp/llm-jp-3-13b).",
    )
    parser.add_argument(
        "--es_host",
        "--es-host",
        type=str,
        default=None,
        help="Elasticsearch host (overrides ES_HOST; default: http://10.2.73.12:9200).",
    )
    parser.add_argument(
        "--es_dump_index",
        "--es-dump-index",
        type=str,
        default=None,
        help="Search index (overrides ES_DUMP_INDEX; default: llm-jp-corpus-v3).",
    )
    parser.add_argument("--num_evidences", "--num-evidences", type=positive_integer, default=1)
    parser.add_argument("--embedding", default="intfloat/multilingual-e5-base")
    parser.add_argument("--decomposition-prompt", "--decomposition_prompt", default=None)
    parser.add_argument("--verification-prompt", "--verification_prompt", default=None)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic chat and fact-check results without API calls, Elasticsearch, or model downloads.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Whether to log debug messages.")
    args = parser.parse_args(argv)
    if not args.mock:
        from dotenv import load_dotenv

        load_dotenv(DOTENV_PATH, override=False)
        if args.engine is None:
            args.engine = os.getenv("FACTCHECKER_MODEL", "").strip() or None
        if args.chat_engine is None:
            args.chat_engine = os.getenv("CHATBOT_MODEL", "").strip() or None
    if args.engine is None:
        args.engine = "gpt-4-0613"
    for name, environment_variable, default in (
        ("tokenizer_name", "TOKENIZER_NAME", "llm-jp/llm-jp-3-13b"),
        ("es_host", "ES_HOST", "http://10.2.73.12:9200"),
        ("es_dump_index", "ES_DUMP_INDEX", "llm-jp-corpus-v3"),
    ):
        if getattr(args, name) is None:
            value = os.getenv(environment_variable, "").strip() if not args.mock else ""
            setattr(args, name, value or default)
    return args


def _text(value: object) -> str:
    return html.escape(str(value), quote=True)


def _badge(label: str) -> str:
    class_name = LABELS.get(label, "nei")
    return f'<span class="verdict {class_name}">{_text(label)}</span>'


def _render_result(result: dict, index: int) -> None:
    st.markdown(
        f'<div class="claim-heading"><span class="eyebrow">Claim {index}</span>'
        f'<p class="claim-text">{_text(result["claim"])}</p></div>',
        unsafe_allow_html=True,
    )
    if result.get("no_evidence"):
        with st.container(border=True):
            st.markdown(_badge("Not enough information"), unsafe_allow_html=True)
            st.caption("No evidence was retrieved for this claim. No verification call was made.")
        return

    for evidence_index, evidence in enumerate(result["evidences"], 1):
        verification = evidence["verification"]
        with st.container(border=True):
            st.markdown(f'<div class="eyebrow evidence-number">Evidence {evidence_index}</div>', unsafe_allow_html=True)
            st.markdown(_badge(verification["label"]), unsafe_allow_html=True)
            st.markdown(
                f'<div class="section-label">Rationale</div>'
                f'<p class="rationale">{_text(verification["rationale"])}</p>',
                unsafe_allow_html=True,
            )
            with st.expander("Evidence passage", expanded=False):
                st.markdown(
                    f'<div class="evidence-passage">{_text(evidence["passage"])}</div>',
                    unsafe_allow_html=True,
                )
            with st.expander("Source details"):
                st.text(f"Source: {evidence.get('dataset') or 'Not available'}")
                training_step = evidence.get("training_step")
                st.text(f"Training step: {training_step if training_step is not None else 'Not available'}")


def _render_results(run: dict) -> None:
    claims = run.get("claims", [])
    results = run.get("results", [])
    if not claims and not results:
        if run.get("state") == "complete":
            st.info("No claims were found in this response.")
        return

    with st.container(border=True):
        st.markdown(
            '<div class="results-header"><h3 class="results-heading">Fact-check results</h3>'
            f'<span class="results-count">{len(results)} / {len(claims)} claims checked</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("Each verdict applies to one claim–evidence pair.")
        if run.get("mock"):
            st.caption("Synthetic results for interface testing.")
        # Keep sections at stable positions while results arrive. Streamlit reuses
        # nested blocks during a run, so moving expanders can retain unrelated content.
        with st.container():
            for index, result in enumerate(results, 1):
                _render_result(result, index)
        pending_message = st.empty()
        if claims and not results and run.get("state") == "running":
            pending_message.caption("Retrieving evidence. Results will appear as claims are checked.")


def _progress(event) -> float:
    if event.stage == "complete":
        return 1.0
    if event.stage == "decomposition":
        return 0.03
    if event.stage == "preparation":
        return 0.12
    if event.total_claims:
        fractions = {"retrieval": 0.08, "ranking": 0.25, "verification": 0.72, "claim_complete": 1.0}
        completed = max(event.current_claim - 1, 0) + fractions.get(event.stage, 0)
        return min(0.98, 0.15 + 0.83 * completed / event.total_claims)
    return 0.15


def _run_factcheck(response: dict, preceding_messages: list[dict], args: argparse.Namespace, results_area) -> None:
    # Snapshot only the conversation preceding this response, including its user prompt.
    context = json.dumps(
        [{"role": message["role"], "content": message["content"]} for message in preceding_messages],
        ensure_ascii=False,
        indent=2,
    )
    run = {
        "document": response["content"],
        "context": context,
        "claims": [],
        "results": [],
        "state": "running",
        "error": None,
    }
    factcheck = run_factcheck
    if args.mock:
        from mock_backend import run_mock_factcheck

        factcheck = run_mock_factcheck
        run["mock"] = True
    st.session_state["factchecks"][response["id"]] = run
    with st.status("Extracting claims from this response…", expanded=True) as status:
        progress = st.progress(0.0)
        message = st.empty()
        message.caption(
            "Simulating decomposition, evidence retrieval, and verification with synthetic data."
            if args.mock
            else "Decomposition → evidence retrieval → verification. Initial model loading may take some time."
        )
        try:
            config = PipelineConfig(
                engine=args.engine,
                tokenizer_name=args.tokenizer_name,
                es_host=args.es_host,
                es_dump_index=args.es_dump_index,
                num_evidences=args.num_evidences,
                embedding=args.embedding,
                decomposition_prompt=args.decomposition_prompt,
                verification_prompt=args.verification_prompt,
            )
            complete = False
            previous_progress = 0.0
            for event in factcheck(run["document"], context, config):
                status.update(label=event.message)
                message.caption(event.message)
                previous_progress = max(previous_progress, _progress(event))
                progress.progress(previous_progress)
                if event.claims is not None:
                    run["claims"] = event.claims
                if event.stage == "claim_complete" and event.result is not None:
                    run["results"].append(event.result)
                if event.claims is not None or event.stage == "claim_complete":
                    with results_area.container():
                        _render_results(run)
                if event.stage == "complete":
                    complete = True
            if not complete:
                raise RuntimeError("Fact-checking ended before completion. Please retry.")
        except Exception as exc:
            logger.exception("Fact-checking failed")
            run["state"] = "error"
            run["error"] = str(exc)
        else:
            run["state"] = "complete"
        finally:
            progress.empty()


def _render_message(message: dict, index: int, args: argparse.Namespace) -> None:
    with st.chat_message(message["role"]):
        _render_author(message["role"])
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_response_controls(message, index, args)


def _render_author(role: str) -> None:
    name = "You" if role == "user" else "Assistant"
    st.markdown(f'<div class="message-author">{name}</div>', unsafe_allow_html=True)


def _render_response_controls(message: dict, index: int, args: argparse.Namespace) -> None:
    response_id = message["id"]
    st.button(
        "Fact-check response",
        key=f"factcheck_{response_id}",
        type="primary",
        on_click=_request_factcheck,
        args=(response_id,),
    )
    if st.session_state.get("recheck_confirmation") == response_id:
        with st.container(border=True):
            st.markdown('<div class="recheck-heading">Run fact-check again?</div>', unsafe_allow_html=True)
            st.caption("This will replace the existing results for this response.")
            confirm, cancel = st.columns(2)
            with confirm:
                st.button(
                    "Run again",
                    key=f"confirm_factcheck_{response_id}",
                    type="primary",
                    on_click=_confirm_factcheck,
                    args=(response_id,),
                )
            with cancel:
                st.button(
                    "Cancel",
                    key=f"cancel_factcheck_{response_id}",
                    on_click=_cancel_factcheck,
                    args=(response_id,),
                )
    activity_area = st.empty()
    results_area = st.empty()
    if st.session_state.get("pending_factcheck") == response_id:
        st.session_state.pop("pending_factcheck")
        with activity_area.container():
            _run_factcheck(message, st.session_state["messages"][:index], args, results_area)
    if response_id in st.session_state["factchecks"]:
        run = st.session_state["factchecks"][response_id]
        if run["state"] == "complete":
            activity_area.caption("Fact-check complete")
        elif run["state"] == "error":
            activity_area.error(f"Fact-checking failed: {run['error']}. Completed results have been retained.")
        else:
            activity_area.warning("This fact-check was interrupted. Completed results are shown below. You can retry.")
        with results_area.container():
            _render_results(run)


def _request_factcheck(response_id: str) -> None:
    """Run an initial check directly; require confirmation for an existing run."""
    st.session_state.pop("pending_factcheck", None)
    if response_id in st.session_state["factchecks"]:
        st.session_state["recheck_confirmation"] = response_id
    else:
        st.session_state.pop("recheck_confirmation", None)
        st.session_state["pending_factcheck"] = response_id


def _confirm_factcheck(response_id: str) -> None:
    if st.session_state.get("recheck_confirmation") == response_id:
        st.session_state.pop("recheck_confirmation")
        st.session_state["pending_factcheck"] = response_id


def _cancel_factcheck(response_id: str) -> None:
    if st.session_state.get("recheck_confirmation") == response_id:
        st.session_state.pop("recheck_confirmation")


def _generate_response(args: argparse.Namespace) -> None:
    history = [{"role": message["role"], "content": message["content"]} for message in st.session_state["messages"]]
    generate = stream_chat_response
    if args.mock:
        from mock_backend import stream_mock_chat_response

        generate = stream_mock_chat_response
    st.session_state["chat_error"] = None
    response_area = st.empty()
    with response_area.container():
        with st.chat_message("assistant"):
            _render_author("assistant")
            output = st.empty()
            fragments = []
            try:
                with st.spinner("Generating response…"):
                    stream = generate(history, model=args.chat_engine or args.engine)
                    try:
                        for fragment in stream:
                            fragments.append(fragment)
                            output.markdown("".join(fragments) + " ▌")
                    finally:
                        close = getattr(stream, "close", None)
                        if close is not None:
                            close()
                content = "".join(fragments)
                if not content.strip():
                    raise ValueError("The model returned an empty response.")
            except Exception as exc:
                logger.exception("Response generation failed")
                st.session_state["chat_error"] = str(exc)
            else:
                message = {"id": uuid4().hex, "role": "assistant", "content": content}
                st.session_state["messages"].append(message)
                output.markdown(content)
                _render_response_controls(message, len(st.session_state["messages"]) - 1, args)
    if st.session_state["chat_error"]:
        response_area.empty()


def _submit_message() -> None:
    """Consume one input event before rendering; reruns never resend it."""
    prompt = st.session_state["chat_input"]
    st.session_state["input_error"] = None
    if not prompt or not prompt.strip():
        st.session_state["input_error"] = "Enter a message before sending."
        return
    st.session_state["messages"].append({"id": uuid4().hex, "role": "user", "content": prompt})
    _request_response()


def _request_response() -> None:
    st.session_state["chat_error"] = None
    st.session_state["pending_response"] = True


def _new_chat() -> None:
    st.session_state["messages"] = []
    st.session_state["factchecks"] = {}
    st.session_state["chat_error"] = None
    st.session_state["input_error"] = None
    st.session_state.pop("pending_response", None)
    st.session_state.pop("chat_input", None)
    st.session_state.pop("pending_factcheck", None)
    st.session_state.pop("recheck_confirmation", None)


def main(args: argparse.Namespace) -> None:
    st.set_page_config(page_title="LLM-jp Fact-Check", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(
        f"<style>{Path(__file__).with_name('styles.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True
    )
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("factchecks", {})
    st.session_state.setdefault("chat_error", None)

    heading, actions = st.columns([5, 1])
    with heading:
        st.markdown(
            '<header class="app-header"><div class="app-brand"><span class="brand-symbol" aria-hidden="true">'
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
            'stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M20 11V6a3 3 0 0 0-3-3H7a3 3 0 0 0-3 3v9a3 3 0 0 0 3 3h1v3l4-3h2"/>'
            '<path d="m15 15 2 2 4-5"/></svg></span>'
            "<div><h1>LLM-jp Fact-Check</h1></div></div>"
            '<p class="app-description">Chat with a language model and check its responses against evidence.</p></header>',
            unsafe_allow_html=True,
        )
    with actions:
        st.button("New chat", key="new_chat", use_container_width=True, on_click=_new_chat)

    if args.mock:
        st.info("Mock mode: chat responses and fact-check results use synthetic data. No external services are called.")

    if not st.session_state["messages"]:
        st.markdown(
            '<section class="welcome-panel"><span class="welcome-icon" aria-hidden="true">'
            '<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1.7" '
            'stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M30 29H15l-7 6v-9a5 5 0 0 1-3-4V11a5 5 0 0 1 5-5h20a5 5 0 0 1 5 5v13a5 5 0 0 1-5 5Z"/>'
            '<path d="M18 34v1a4 4 0 0 0 4 4h12l7 5V20a4 4 0 0 0-4-4M13 14h14M13 21h9"/>'
            "</svg></span><h2>Start a conversation</h2>"
            '<p class="welcome-description">Ask a question or explore a topic. '
            "You can check any response to examine the claims it makes.</p>"
            '<div class="workflow-grid">'
            '<div class="workflow-step"><span class="step-number">01</span>'
            "<div><h3>Discuss a topic</h3><p>Send a message below and continue with follow-up questions.</p></div></div>"
            '<div class="workflow-step"><span class="step-number">02</span>'
            "<div><h3>Check a response</h3><p>Select <strong>Fact-check response</strong> below a reply "
            "to inspect its claims and evidence.</p></div></div></div></section>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<h2 class="conversation-heading">Conversation</h2>', unsafe_allow_html=True)
    for index, message in enumerate(st.session_state["messages"]):
        _render_message(message, index, args)

    if st.session_state.pop("pending_response", False):
        _generate_response(args)
    elif (
        st.session_state["messages"]
        and st.session_state["messages"][-1]["role"] == "user"
        and not st.session_state["chat_error"]
    ):
        st.session_state["chat_error"] = "Response generation was interrupted. Select Retry response to try again."

    if st.session_state["chat_error"]:
        st.error(f"Response generation failed: {st.session_state['chat_error']}")
        st.button("Retry response", key="retry_response", on_click=_request_response)
    if st.session_state.get("input_error"):
        st.warning(st.session_state["input_error"])
    st.chat_input("Message the model", key="chat_input", on_submit=_submit_message)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main(args)
