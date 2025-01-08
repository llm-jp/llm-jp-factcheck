import argparse
import dataclasses
import logging
import hmac
from typing import Optional

import streamlit as st
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
        st.write(f"#### Claim {self.claim_id}")

        st.code(self.claim)

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
                    st.code(evidence["passage"], wrap_lines=True)

        if self.is_veridied is not None:
            assert self.rationale is not None
            if self.is_veridied:
                st.write("**Verification result**: :green[Supported]")
            else:
                st.write("**Verification result**: :red[Not Supported]")
            st.write(f"**Rationale**: {self.rationale}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
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

    st.title("LLM-jp Fact-Check")

    with st.form("form", clear_on_submit=False):
        st.write(
            "Enter the document (and the context if applicable) you want to fact-check."
        )
        context = st.text_area("Context")
        document = st.text_area("Model output")
        submitted = st.form_submit_button("Submit")

    if submitted and document.strip() != "":
        st.subheader("Fact-Check Results")
        analysis_placeholder = st.empty()
        with st.spinner("Decomposing the document into atomic claims..."):
            claims = []
            for i, claim in enumerate(
                decompose_document_into_claims(
                    document=document,
                    context=context,
                    model=args.engine,
                ),
                1,
            ):
                claims.append(Claim(claim_id=i, claim=claim))

        with analysis_placeholder.container():
            for claim in claims:
                claim.render()

        with st.spinner("Identifying check-worthy claims..."):
            checkworthy_labels = identify_checkworthiness(claims, args.engine)
            for claim, is_checkworthy in zip(claims, checkworthy_labels):
                claim.is_checkworthy = is_checkworthy

        with analysis_placeholder.container():
            for claim in claims:
                claim.render()

        @st.cache_resource
        def _get_tokenizer(tokenizer_name_or_path: str):
            return AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        @st.cache_resource
        def _create_elasticsearch_client(host: str):
            return create_elasticsearch_client(host)

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

                with analysis_placeholder.container():
                    for claim in claims:
                        claim.render()

        with st.spinner("Verifying the check-worthy claims..."):
            for claim in claims:
                if claim.is_checkworthy is False:
                    continue

                result = verify_claim(
                    claim.claim, [e["passage"] for e in claim.evidences], args.engine
                )
                claim.is_veridied = result["label"]
                claim.rationale = result["rationale"]

                with analysis_placeholder.container():
                    for claim in claims:
                        claim.render()

        #     st.subheader("Result of verifiation")
        #     for claim in checkworthy_claims:
        #         st.markdown(f"**Claim**: {claim.strip()}")

        #         with st.spinner("Retrieving the evidences..."):
        #             claim_token_ids = tokenizer.encode(claim, add_special_tokens=False)
        #             evidences = []
        #             hits = search_documents(
        #                 es,
        #                 args.es_dump_index,
        #                 body={
        #                     "query": {
        #                         "match": {
        #                             "token_ids": " ".join(map(str, claim_token_ids))
        #                         }
        #                     },
        #                 },
        #                 size=args.num_evidences,
        #                 max_concurrent_shard_requests=64,
        #             )
        #             for hit in hits:
        #                 passage_token_ids = list(
        #                     map(int, hit["_source"]["token_ids"].split())
        #                 )
        #                 passage = tokenizer.decode(passage_token_ids).strip()
        #                 dataset = hit["_source"]["dataset_name"]
        #                 training_step = hit["_source"]["iteration"]
        #                 raw_corpus_path = hit["_source"]["raw_corpus_path"]
        #                 # TODO: Fix hard-coded prefix
        #                 raw_corpus_path = raw_corpus_path.replace(
        #                     "home/shared/corpus/llm-jp-corpus/", ""
        #                 )
        #                 raw_corpus_path = raw_corpus_path.replace(
        #                     "training_resharded_tokenize_ver3.0/", ""
        #                 )
        #                 raw_corpus_path = raw_corpus_path.lstrip("/")
        #                 raw_corpus_idx = hit["_source"]["raw_corpus_idx"]
        #                 evidences.append(
        #                     {
        #                         "passage": passage,
        #                         "dataset": dataset,
        #                         "training_step": training_step,
        #                         "raw_corpus_path": raw_corpus_path,
        #                         "raw_corpus_idx": raw_corpus_idx,
        #                     }
        #                 )

        #         # TODO: Trace meta data

        #         for i, evidence in enumerate(evidences, 1):
        #             with st.expander(f"Evidence {i}"):
        #                 st.markdown(f"Dataset: {evidence['dataset']}")
        #                 st.markdown(f"Training Step: {evidence['training_step']:,}")
        #                 st.markdown(evidence["passage"])
        #                 st.markdown("---")

        #         with st.spinner("Verifying the check-worthy claims..."):
        #             result = verify_claim(
        #                 claim, [e["passage"] for e in evidences], args.engine
        #             )

        #         st.markdown(
        #             f"**Result**: {'Supported' if result['label'] else 'Not Supported'}"
        #         )
        #         st.markdown(f"**Rationale**: {result['rationale']}")
        #         st.markdown("--")


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
