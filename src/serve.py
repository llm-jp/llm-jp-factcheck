import argparse
import dataclasses
import logging
import hmac
from typing import Optional

import streamlit as st
import openai
from checkworthy import identify_checkworthiness
from decompose import decompose_document_into_claims
from retrieval import create_elasticsearch_client, search_documents
from transformers import AutoTokenizer
from verify import verify_claim

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Claim:
    claim_id: int
    claim: str
    is_checkworthy: Optional[bool] = None
    evidences: Optional[list] = None
    is_veridied: Optional[bool] = None
    rationale: Optional[str] = None

    def render(self) -> None:
        st.write(f"### Claim {self.claim_id}")

        st.code(self.claim, wrap_lines=True, language=None)

        if self.is_checkworthy is not None:
            if self.is_checkworthy:
                st.write("**Checkworthy**: :green[TRUE]")
            else:
                st.write("**Checkworthy**: :red[FALSE]")

        if self.evidences is not None:
            st.write("**Evidences**")
            with st.expander("Expand to view the evidences"):
                for i, evidence in enumerate(self.evidences, 1):
                    st.write(f"**Evidence {i}**")
                    st.caption(
                        f"Dataset: {evidence['dataset']} | Training Step: {evidence['training_step']}"
                    )
                    st.code(evidence["passage"], wrap_lines=True, language=None)

        if self.is_veridied is not None:
            assert self.rationale is not None
            if self.is_veridied:
                st.write(
                    f"**Verification result**: :green[Supported] (Rationale: {self.rationale})"
                )
            else:
                st.write(
                    f"**Verification result**: :red[Not Supported] (Rationale: {self.rationale})"
                )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chatbot-endpoint", type=str, default="http://localhost:24681/v1"
    )
    parser.add_argument(
        "--chatbot-model-name",
        type=str,
        default="/data/shared/llm-jp-3-172b-instruct3/",
    )
    parser.add_argument("--engine", type=str, default="gpt-4-0613")
    parser.add_argument("--tokenizer_name", type=str, default="llm-jp/llm-jp-3-13b")
    parser.add_argument("--es_host", type=str, default="http://10.2.73.12:9200")
    parser.add_argument("--es_dump_index", type=str, default="llm-jp-corpus-v3")
    parser.add_argument("--num_evidences", type=int, default=1)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Whether to log debug messages."
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run the pipeline.

    Args:
        args (argparse.Namespace): The command-line arguments.
    """
    st.set_page_config(
        page_title="LLM-jp Factcheck",
        page_icon="🔍️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    def check_password():
        """Returns `True` if the user had the correct password."""

        def password_entered():
            """Checks whether a password entered by the user is correct."""
            if hmac.compare_digest(
                st.session_state["password"], st.secrets["password"]
            ):
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # Don't store the password.
            else:
                st.session_state["password_correct"] = False

        # Return True if the password is validated.
        if st.session_state.get("password_correct", False):
            return True

        # Show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        if "password_correct" in st.session_state:
            st.error("😕 Password incorrect")
        return False

    if check_password() is False:
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    @st.cache_resource
    def _get_tokenizer(tokenizer_name_or_path: str):
        return AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    @st.cache_resource
    def _create_elasticsearch_client(host: str):
        return create_elasticsearch_client(host)

    @st.cache_resource
    def _create_chatbot_client(endpoint: str, api_key: str = "EMPTY"):
        return openai.OpenAI(base_url=endpoint, api_key=api_key)

    @st.dialog("Fact-check", width="large")
    def factcheck():
        response = st.session_state.messages[-1]["content"]
        context = ""
        for m in st.session_state.messages[:-1]:
            context += f"{m['role']}: {m['content']}\n"
        claims = []

        analysis_placeholder = st.empty()

        def render():
            with analysis_placeholder.container():
                st.write("### Response")
                st.code(response, wrap_lines=True, language=None)
                for claim in claims:
                    claim.render()

        render()

        with st.spinner("Decomposing the response into atomic claims..."):
            for i, claim in enumerate(
                decompose_document_into_claims(
                    document=response,
                    context=context,
                    model=args.engine,
                ),
                1,
            ):
                claims.append(Claim(claim_id=i, claim=claim))
            render()

        with st.spinner("Identifying check-worthy claims..."):
            checkworthy_labels = identify_checkworthiness(claims, args.engine)
            for claim, is_checkworthy in zip(claims, checkworthy_labels):
                claim.is_checkworthy = is_checkworthy
            render()

        tokenizer = _get_tokenizer(args.tokenizer_name)
        es = _create_elasticsearch_client(args.es_host)

        with st.spinner("Retrieving the evidences..."):
            for claim in claims:
                if claim.is_checkworthy is False:
                    continue

                claim_token_ids = tokenizer.encode(
                    claim.claim, add_special_tokens=False
                )
                evidences = []
                hits = search_documents(
                    es,
                    args.es_dump_index,
                    body={
                        "query": {
                            "match": {"token_ids": " ".join(map(str, claim_token_ids))}
                        },
                    },
                    size=args.num_evidences,
                    max_concurrent_shard_requests=64,
                )
                for hit in hits:
                    passage_token_ids = list(
                        map(int, hit["_source"]["token_ids"].split())
                    )
                    passage = tokenizer.decode(passage_token_ids).strip()
                    dataset = hit["_source"]["dataset_name"]
                    training_step = hit["_source"]["iteration"]
                    evidences.append(
                        {
                            "passage": passage,
                            "dataset": dataset,
                            "training_step": training_step,
                        }
                    )
                claim.evidences = evidences
                render()

        with st.spinner("Verifying the check-worthy claims..."):
            for claim in claims:
                if claim.is_checkworthy is False:
                    continue

                result = verify_claim(
                    claim.claim, [e["passage"] for e in claim.evidences], args.engine
                )
                claim.is_veridied = result["label"]
                claim.rationale = result["rationale"]
                render()

    with st.sidebar:
        st.title("LLM-jp Fact-Check")
        if st.button(":pencil: New conversation"):
            st.session_state.messages = []

    chatbot_client = _create_chatbot_client(args.chatbot_endpoint)

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if i == len(st.session_state.messages) - 1:
                st.button(":mag: Fact-check", on_click=factcheck)

    if prompt := st.chat_input("Input your message"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            stream = chatbot_client.chat.completions.create(
                model=args.chatbot_model_name,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
