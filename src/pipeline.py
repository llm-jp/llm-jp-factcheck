import argparse
import logging

import streamlit as st
from checkworthy import identify_checkworthiness
from decompose import decompose_document_into_claims
from retrieval import chunk_document, create_elasticsearch_client, create_relevance_scorer, search_documents
from transformers import AutoTokenizer
from verify import verify_claim

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--tokenizer_name", type=str, default="llm-jp/llm-jp-13b-v1.0")
    parser.add_argument("--es_host", type=str, default="http://localhost:9200")
    parser.add_argument("--es_index", type=str, default="memorization-analysis-dev")
    parser.add_argument("--num_evidences", type=int, default=3)
    parser.add_argument("--embedding", type=str, default="intfloat/multilingual-e5-base")
    parser.add_argument("-v", "--verbose", action="store_true", help="Whether to log debug messages.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run the pipeline.

    Args:
        args (argparse.Namespace): The command-line arguments.
    """
    st.title("LLM-jp Fact-Check")

    with st.form("form", clear_on_submit=False):
        st.write("Enter the document (and the context if applicable) you want to fact-check.")
        context = st.text_area("Context")
        document = st.text_area("Document")
        submitted = st.form_submit_button("Submit")

    if submitted:
        with st.spinner("Decomposing the document into claims..."):
            claims = decompose_document_into_claims(
                document=document,
                context=context,
                model=args.engine,
            )

        st.subheader("Result of claim extraction")
        for i, claim in enumerate(claims, 1):
            st.markdown(f"- Claim {i}: {claim}")

        with st.spinner("Identifying check-worthy claims..."):
            checkworthy_labels = identify_checkworthiness(claims, args.engine)
            checkworthy_claims = [claim for claim, label in zip(claims, checkworthy_labels) if label]

        st.subheader("Result of check-worthy prediction")
        for i, label in enumerate(checkworthy_labels, 1):
            st.markdown(f"- Claim {i}: {label}")

        @st.cache_resource
        def _get_tokenizer(tokenizer_name_or_path: str):
            return AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        @st.cache_resource
        def _create_elasticsearch_client(host: str):
            return create_elasticsearch_client(host)

        @st.cache_resource
        def _create_relevance_scorer(embedding: str):
            return create_relevance_scorer(embedding)

        with st.spinner("Retrieving evidence documents..."):
            tokenizer = _get_tokenizer(args.tokenizer_name)
            es = _create_elasticsearch_client(args.es_host)
            scorer = _create_relevance_scorer(args.embedding)

            evidences = []
            for claim in checkworthy_claims:
                claim_token_ids = tokenizer(claim, add_special_tokens=False)["input_ids"]
                hits = search_documents(
                    es,
                    args.es_index,
                    body={"query": {"match": {"token_ids": " ".join(map(str, claim_token_ids))}}},
                    size=3,
                )
                evidences_of_claim = []
                for hit in hits:
                    document = tokenizer.decode(list(map(int, hit["_source"]["token_ids"].split()))).strip()
                    dataset = hit["_source"]["dataset_name"]
                    training_step = hit["_source"]["iteration"]
                    for passage in chunk_document(document):
                        score = scorer(claim, passage)
                        evidences_of_claim.append((passage, dataset, training_step, score))

                evidences_of_claim.sort(key=lambda x: x[3], reverse=True)
                evidences.append(evidences_of_claim[: args.num_evidences])

        with st.spinner("Verifying the check-worthy claims..."):
            results = []
            for claim, evidences_of_claim in zip(checkworthy_claims, evidences):
                results.append(verify_claim(claim, evidences_of_claim, args.engine))

        st.subheader("Overall result")
        for claim, result_of_claim, evidences_of_claim in zip(checkworthy_claims, results, evidences):
            st.markdown(f"**Claim**: {claim.strip()}")
            st.markdown(f"**Result**: {result_of_claim['label']}")
            st.markdown(f"**Rationale**: {result_of_claim['rationale']}")
            with st.expander("Evidence"):
                for i, (passage, dataset, training_step, _) in enumerate(evidences_of_claim, 1):
                    st.markdown(f"Evidence {i}. Dataset: {dataset}. Training Step: {training_step}.")
                    st.markdown(passage)
                    st.markdown("--")


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
