"""Multi-turn chat with independent fact-checks for each model response."""

from __future__ import annotations

import argparse
import base64
import html
import json
import logging
import os
from concurrent.futures import wait
from pathlib import Path
from uuid import uuid4

import streamlit as st

from chat import stream_chat_response
from jobs import FactcheckJob
from pipeline import PipelineConfig, run_factcheck

logger = logging.getLogger(__name__)
DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"
FACTCHECK_ERROR_MESSAGE = "Unable to complete the fact-check. Please try again."
CHAT_ERROR_MESSAGE = "Unable to generate a response. Please select Retry response."
CHAT_INTERRUPTED_MESSAGE = "Response generation was interrupted. Select Retry response to try again."

LABELS = {
    "Fully supported": "supported",
    "Inferentially supported": "inferentially-supported",
    "Partially supported": "partially-supported",
    "Fully refuted": "refuted",
    "Inferentially refuted": "inferentially-refuted",
    "Not enough information": "nei",
}


def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with an LLM and check its responses against evidence retrieved from its training data.",
        epilog=(
            "Configure OpenAI-compatible or Azure OpenAI endpoints with CHATBOT_* and FACTCHECKER_* "
            "environment variables."
        ),
    )
    parser.add_argument(
        "--engine",
        default=None,
        help="Factchecker model or Azure deployment (overrides FACTCHECKER_MODEL; default: gpt-5.4-2026-03-05).",
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
    parser.add_argument(
        "--max-concurrency",
        "--max_concurrency",
        type=positive_integer,
        default=8,
        help="Maximum simultaneous Checkworthy, retrieval, or Verification requests (default: 8).",
    )
    parser.add_argument("--decomposition-prompt", "--decomposition_prompt", default=None)
    parser.add_argument("--checkworthiness-prompt", "--checkworthiness_prompt", default=None)
    parser.add_argument("--verification-prompt", "--verification_prompt", default=None)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic chat and fact-check results without API calls, Elasticsearch, or tokenizer downloads.",
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
        args.engine = "gpt-5.4-2026-03-05"
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


def _progress_message(message: str) -> str:
    return (
        '<div class="progress-message" role="status">'
        '<span class="progress-spinner" aria-hidden="true"></span>'
        f"<span>{_text(message)}</span></div>"
    )


def _render_result(result: dict, index: int, *, stopped: bool = False, paused: bool = False) -> None:
    processing = (
        not stopped
        and not paused
        and result.get("status") in {"checkworthiness", "retrieval", "verification"}
        and result.get("is_checkworthy") is not False
        and not result.get("no_evidence")
    )
    activity = (
        '<span class="claim-activity" role="status">'
        '<span class="claim-spinner" aria-hidden="true"></span>Processing</span>'
        if processing
        else ""
    )
    st.markdown(
        f'<div class="claim-heading"><div class="claim-title-row">'
        f'<span class="eyebrow">Claim {index}</span>{activity}</div>'
        f'<p class="claim-text">{_text(result["claim"])}</p></div>',
        unsafe_allow_html=True,
    )
    if result.get("is_checkworthy") is False:
        with st.container(border=True):
            st.caption("Not check-worthy. Retrieval and verification were skipped.")
        return
    if result.get("no_evidence"):
        with st.container(border=True):
            st.markdown(_badge("Not enough information"), unsafe_allow_html=True)
            st.caption("No evidence was retrieved for this claim. No verification call was made.")
        return

    if paused and result.get("status") != "complete":
        st.caption("Paused")
    elif result.get("status") in {"checkworthiness", "retrieval"}:
        pending = {
            "checkworthiness": "Assessing check-worthiness…",
            "retrieval": "Check-worthy. Waiting for evidence search to complete…",
        }
        st.caption("Processing stopped before this claim was verified." if stopped else pending[result["status"]])

    for evidence_index, evidence in enumerate(result["evidences"], 1):
        verification = evidence.get("verification")
        with st.container(border=True):
            st.markdown(f'<div class="eyebrow evidence-number">Evidence {evidence_index}</div>', unsafe_allow_html=True)
            with st.container():
                if verification is None:
                    st.caption(
                        "Verification did not complete."
                        if stopped
                        else "Verification paused."
                        if paused
                        else "Waiting for verification…"
                    )
                else:
                    st.markdown(_badge(verification["label"]), unsafe_allow_html=True)
                    if verification.get("rationale"):
                        st.markdown(
                            f'<div class="section-label">Rationale</div>'
                            f'<p class="rationale">{_text(verification["rationale"])}</p>',
                            unsafe_allow_html=True,
                        )
            st.markdown(
                '<div class="section-label">Source</div>'
                f'<p class="evidence-source">{_text(evidence.get("dataset") or "Not available")}</p>',
                unsafe_allow_html=True,
            )
            with st.expander("Evidence passage", expanded=False):
                st.markdown(
                    f'<div class="evidence-passage">{_text(evidence["passage"])}</div>',
                    unsafe_allow_html=True,
                )


def _summarize_results(run: dict) -> dict:
    """Count completed evidence verdicts from the latest claim snapshots."""
    counts = dict.fromkeys(LABELS, 0)
    skipped = without_evidence = 0
    for result in run.get("claim_states", run.get("results", [])):
        if result.get("is_checkworthy") is False:
            skipped += 1
            continue
        if result.get("no_evidence"):
            without_evidence += 1
            continue
        for evidence in result.get("evidences", []):
            label = (evidence.get("verification") or {}).get("label")
            if label in counts:
                counts[label] += 1
    return {
        "labels": counts,
        "verdicts": sum(counts.values()),
        "skipped_claims": skipped,
        "claims_without_evidence": without_evidence,
    }


def _render_summary(run: dict) -> None:
    summary = _summarize_results(run)
    cards = "".join(
        f'<div class="verdict-summary-card {style}" data-verdict="{_text(label)}">'
        f"<dt>{'<br>'.join(_text(label).rsplit(' ', 1))}</dt><dd>{summary['labels'][label]}</dd></div>"
        for label, style in LABELS.items()
    )
    st.markdown(
        '<section class="verdict-summary" aria-label="Verdict summary">'
        '<div class="summary-totals">'
        f"<span>Evidence verdicts <strong>{summary['verdicts']}</strong></span>"
        f"<span>Skipped claims <strong>{summary['skipped_claims']}</strong></span>"
        f"<span>Claims without evidence <strong>{summary['claims_without_evidence']}</strong></span>"
        f'</div><dl class="verdict-summary-grid">{cards}</dl></section>',
        unsafe_allow_html=True,
    )


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
            f'<span class="results-count">{len(results)} / {len(claims)} claims processed</span></div>',
            unsafe_allow_html=True,
        )
        _render_summary(run)
        if run.get("mock"):
            st.caption("Synthetic results for interface testing.")
        # Keep every claim in its original position while individual results arrive.
        with st.container():
            for index, result in enumerate(run.get("claim_states", results), 1):
                with st.container():
                    _render_result(
                        result,
                        index,
                        stopped=run.get("state") in {"error", "interrupted"},
                        paused=run.get("state") == "paused",
                    )


def _progress(event) -> float:
    if event.progress is not None:
        return event.progress
    if event.stage == "complete":
        return 1.0
    if event.stage == "decomposition":
        return 0.03
    if event.stage == "checkworthiness":
        return 0.05 + 0.15 * event.current_claim / max(event.total_claims, 1)
    if event.stage == "preparation":
        return 0.22
    if event.total_claims:
        if event.stage == "retrieval":
            return 0.25 + 0.25 * event.current_claim / event.total_claims
        completed = event.current_claim if event.stage == "claim_complete" else max(event.current_claim - 1, 0)
        return min(0.98, 0.5 + 0.48 * completed / event.total_claims)
    return 0.5


def _start_factcheck(response: dict, preceding_messages: list[dict], args: argparse.Namespace) -> None:
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
        "claim_states": [],
        "state": "running",
        "error": None,
        "progress": 0.0,
        "message": "Extracting claims from this response…",
        "completed_results": {},
    }
    factcheck = run_factcheck
    if args.mock:
        from mock_backend import run_mock_factcheck

        factcheck = run_mock_factcheck
        run["mock"] = True
    st.session_state["factchecks"][response["id"]] = run
    try:
        config = PipelineConfig(
            engine=args.engine,
            tokenizer_name=args.tokenizer_name,
            es_host=args.es_host,
            es_dump_index=args.es_dump_index,
            num_evidences=args.num_evidences,
            max_concurrency=args.max_concurrency,
            decomposition_prompt=args.decomposition_prompt,
            checkworthiness_prompt=args.checkworthiness_prompt,
            verification_prompt=args.verification_prompt,
        )
        st.session_state["factcheck_jobs"][response["id"]] = FactcheckJob(factcheck(run["document"], context, config))
    except Exception:
        logger.exception("Fact-checking could not start")
        run["state"] = "error"
        run["error"] = FACTCHECK_ERROR_MESSAGE


def _run_factcheck(response_id: str) -> None:
    run = st.session_state["factchecks"][response_id]
    job = st.session_state["factcheck_jobs"][response_id]
    try:
        for _ in range(100):
            future = job.advance()
            # Drain quick intermediate events without waiting for the next UI tick.
            if not future.done():
                wait([future], timeout=0.005)
            if not future.done():
                break
            try:
                event = job.consume()
            except StopIteration:
                raise RuntimeError("Fact-checking ended before completion. Please retry.") from None
            run["message"] = event.message
            run["progress"] = max(run["progress"], _progress(event))
            if event.claims is not None:
                run["claims"] = event.claims
                run["claim_states"] = [
                    {"claim": claim, "status": "checkworthiness", "evidences": []} for claim in event.claims
                ]
            if event.claim_update is not None:
                run["claim_states"][event.current_claim - 1] = event.claim_update
            if event.stage == "claim_complete" and event.result is not None:
                run["claim_states"][event.current_claim - 1] = event.result
                run["completed_results"][event.current_claim - 1] = event.result
                run["results"] = [run["completed_results"][index] for index in sorted(run["completed_results"])]
            if event.stage == "complete":
                run["state"] = "complete"
                break
    except Exception:
        logger.exception("Fact-checking failed")
        run["state"] = "error"
        run["error"] = FACTCHECK_ERROR_MESSAGE
    finally:
        if run["state"] in {"complete", "error"} or (
            job.pending is not None and job.pending.done() and job.pending.exception() is not None
        ):
            st.session_state["factcheck_jobs"].pop(response_id, None)
            job.close()


def _render_message(message: dict, index: int, args: argparse.Namespace) -> None:
    with st.chat_message(message["role"]):
        model = message.get("model") or ("Mock model" if args.mock else args.chat_engine or args.engine)
        _render_author(message["role"], model)
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_response_controls(message, index, args)


def _render_author(role: str, model: str) -> None:
    name = "You" if role == "user" else model
    st.markdown(f'<div class="message-author">{_text(name)}</div>', unsafe_allow_html=True)


def _render_response_controls(message: dict, index: int, args: argparse.Namespace) -> None:
    run = st.session_state["factchecks"].get(message["id"])
    polling = (run and run["state"] == "running") or st.session_state.get("pending_factcheck") == message["id"]
    st.fragment(run_every=0.25 if polling else None)(_response_controls)(message, index, args)


def _response_controls(message: dict, index: int, args: argparse.Namespace) -> None:
    response_id = message["id"]
    if st.session_state.get("pending_factcheck") == response_id:
        st.session_state.pop("pending_factcheck")
        _start_factcheck(message, st.session_state["messages"][:index], args)
    run = st.session_state["factchecks"].get(response_id)
    if run and run["state"] in {"running", "paused"} and response_id not in st.session_state["factcheck_jobs"]:
        run["state"] = "interrupted"
    # Apply queued events before emitting UI elements. Replacing an st.empty()
    # results tree for every event briefly collapses it and moves the viewport.
    if run and run["state"] == "running":
        _run_factcheck(response_id)
        if run["state"] != "running":
            st.rerun()
    with st.container(key=f"factcheck_action_{response_id}"):
        _factcheck_button(response_id)
    if run:
        if run["state"] != "complete":
            with st.container(key=f"factcheck_activity_{response_id}"):
                if run["state"] == "running":
                    st.progress(run["progress"])
                    st.markdown(_progress_message(run["message"]), unsafe_allow_html=True)
                elif run["state"] == "paused":
                    st.progress(run["progress"])
                    st.caption(
                        "Fact-check paused. Requests already sent may finish; no new requests will start. "
                        "Resume to continue."
                    )
                elif run["state"] == "error":
                    message = FACTCHECK_ERROR_MESSAGE
                    if run.get("claim_states") or run.get("results"):
                        message += " Available results are shown below."
                    st.error(message)
                elif run["state"] == "interrupted":
                    st.warning("This fact-check was interrupted. Available results are shown below. You can retry.")
        with st.container(key=f"factcheck_results_{response_id}"):
            _render_results(run)


def _factcheck_button(response_id: str) -> None:
    run = st.session_state["factchecks"].get(response_id)
    state = run["state"] if run else None
    if state == "running":
        label, key, callback = "Pause fact-check", "pause_factcheck", _pause_factcheck
    elif state == "paused":
        label, key, callback = "Resume fact-check", "resume_factcheck", _resume_factcheck
    else:
        label, key, callback = "Fact-check response", "factcheck", _request_factcheck
    if st.button(label, key=f"{key}_{response_id}", type="primary"):
        callback(response_id)
        st.rerun()


def _pause_factcheck(response_id: str) -> None:
    if job := st.session_state["factcheck_jobs"].get(response_id):
        job.control.pause()
        st.session_state["factchecks"][response_id]["state"] = "paused"


def _resume_factcheck(response_id: str) -> None:
    if job := st.session_state["factcheck_jobs"].get(response_id):
        job.control.resume()
        st.session_state["factchecks"][response_id]["state"] = "running"


def _request_factcheck(response_id: str) -> None:
    """Run an initial check directly; require confirmation for an existing run."""
    if response_id in st.session_state["factcheck_jobs"]:
        return
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


def _dismiss_factcheck_confirmation() -> None:
    st.session_state.pop("recheck_confirmation", None)


@st.dialog("Run fact-check again?", on_dismiss=_dismiss_factcheck_confirmation)
def _show_factcheck_confirmation(response_id: str) -> None:
    st.write("This will replace the existing results for this response.")
    confirm, cancel = st.columns(2)
    with confirm:
        if st.button("Run again", key=f"confirm_factcheck_{response_id}", type="primary", use_container_width=True):
            _confirm_factcheck(response_id)
            st.rerun()
    with cancel:
        if st.button("Cancel", key=f"cancel_factcheck_{response_id}", use_container_width=True):
            _cancel_factcheck(response_id)
            st.rerun()


def _generate_response(args: argparse.Namespace) -> None:
    history = [{"role": message["role"], "content": message["content"]} for message in st.session_state["messages"]]
    model = args.chat_engine or args.engine
    display_model = "Mock model" if args.mock else model
    generate = stream_chat_response
    if args.mock:
        from mock_backend import stream_mock_chat_response

        generate = stream_mock_chat_response
    st.session_state["chat_error"] = None
    response_area = st.empty()
    with response_area.container():
        with st.chat_message("assistant"):
            _render_author("assistant", display_model)
            output = st.empty()
            fragments = []
            try:
                with st.spinner("Generating response…"):
                    stream = generate(history, model=model)
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
            except Exception:
                logger.exception("Response generation failed")
                st.session_state["chat_error"] = CHAT_ERROR_MESSAGE
            else:
                message = {"id": uuid4().hex, "role": "assistant", "content": content, "model": display_model}
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
    for job in st.session_state.get("factcheck_jobs", {}).values():
        job.close()
    st.session_state["factcheck_jobs"] = {}
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
    st.session_state.setdefault("factcheck_jobs", {})
    st.session_state.setdefault("chat_error", None)

    logo = base64.b64encode(Path(__file__).with_name("assets").joinpath("llm-jp.png").read_bytes()).decode("ascii")
    heading, actions = st.columns([5, 1])
    with heading:
        st.markdown(
            '<header class="app-header"><div class="app-brand">'
            f'<img class="brand-symbol" src="data:image/png;base64,{logo}" alt="LLM-jp logo">'
            "<div><h1>LLM-jp Fact-Check</h1></div></div>"
            '<p class="app-description">Chat with an LLM and check its responses against evidence '
            "from its training data.</p></header>",
            unsafe_allow_html=True,
        )
    with actions:
        st.button("New chat", key="new_chat", use_container_width=True, on_click=_new_chat)

    if args.mock:
        st.info("Mock mode: chat responses and fact-check results use synthetic data. No external services are called.")

    if not st.session_state["messages"]:
        # A nested chat input is inline; the conversation composer stays pinned below.
        with st.container():
            st.markdown(
                '<section class="welcome-panel"><span class="welcome-icon" aria-hidden="true">'
                '<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1.7" '
                'stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M30 29H15l-7 6v-9a5 5 0 0 1-3-4V11a5 5 0 0 1 5-5h20a5 5 0 0 1 5 5v13a5 5 0 0 1-5 5Z"/>'
                '<path d="M18 34v1a4 4 0 0 0 4 4h12l7 5V20a4 4 0 0 0-4-4M13 14h14M13 21h9"/>'
                "</svg></span><h2>Start a conversation</h2></section>",
                unsafe_allow_html=True,
            )
            st.chat_input("Message the model", key="chat_input", on_submit=_submit_message)
            if st.session_state.get("input_error"):
                st.warning(st.session_state["input_error"])
            st.markdown(
                '<div class="workflow-grid">'
                '<div class="workflow-step"><span class="step-number">01</span>'
                "<div><h3>Discuss a topic</h3><p>Send a message and continue with follow-up questions.</p></div></div>"
                '<div class="workflow-step"><span class="step-number">02</span>'
                "<div><h3>Check a response</h3><p>Select <strong>Fact-check response</strong> to compare claims "
                "with training data and inspect the evidence.</p></div></div></div>",
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
        st.session_state["chat_error"] = CHAT_INTERRUPTED_MESSAGE

    if st.session_state["chat_error"]:
        # Older sessions may still contain exception text; display only known messages.
        st.error(
            CHAT_INTERRUPTED_MESSAGE
            if st.session_state["chat_error"] == CHAT_INTERRUPTED_MESSAGE
            else CHAT_ERROR_MESSAGE
        )
        st.button("Retry response", key="retry_response", on_click=_request_response)
    if st.session_state["messages"]:
        if st.session_state.get("input_error"):
            st.warning(st.session_state["input_error"])
        st.chat_input("Message the model", key="chat_input", on_submit=_submit_message)

    if response_id := st.session_state.get("recheck_confirmation"):
        _show_factcheck_confirmation(response_id)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main(args)
