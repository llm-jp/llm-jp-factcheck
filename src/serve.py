import argparse
import json
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
    parser.add_argument("--engine", type=str, default="gpt-4-0613")
    parser.add_argument("--tokenizer_name", type=str, default="llm-jp/llm-jp-13b-v1.0")
    parser.add_argument("--es_host", type=str, default="http://localhost:9200")
    parser.add_argument("--es_dump_index", type=str, default="llm-jp-search-v1.0")
    parser.add_argument("--es_meta_index", type=str, default="llm-jp-search-for-meta-v1.0")
    parser.add_argument("--num_evidences", type=int, default=1)
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
        text = st.text_area("Document")
        submitted = st.form_submit_button("Submit")

    if submitted:
        st.subheader("Input context")
        st.markdown(context)

        st.subheader("Input document")
        st.markdown(text)

        with st.spinner("Decomposing the document into claims..."):
            claims = decompose_document_into_claims(
                document=text,
                context=context,
                model=args.engine,
            )

        st.subheader("Result of claim detection")
        for i, claim in enumerate(claims, 1):
            st.markdown(f"- Claim {i}: {claim}")

        with st.spinner("Identifying check-worthy claims..."):
            checkworthy_labels = identify_checkworthiness(claims, args.engine)
            checkworthy_claims = [claim for claim, label in zip(claims, checkworthy_labels) if label]

        st.subheader("Result of check-worthy prediction")
        for i, label in enumerate(checkworthy_labels, 1):
            st.markdown(f"- Claim {i}: {'Checkworthy' if label else 'Not Checkworthy'}")

        @st.cache_resource
        def _get_tokenizer(tokenizer_name_or_path: str):
            return AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        @st.cache_resource
        def _create_elasticsearch_client(host: str):
            return create_elasticsearch_client(host)

        @st.cache_resource
        def _create_relevance_scorer(embedding: str):
            return create_relevance_scorer(embedding)

        tokenizer = _get_tokenizer(args.tokenizer_name)
        es = _create_elasticsearch_client(args.es_host)
        scorer = _create_relevance_scorer(args.embedding)

        st.subheader("Result of verifiation")
        for claim in checkworthy_claims:
            st.markdown(f"**Claim**: {claim.strip()}")

            with st.spinner("Retrieving the evidences..."):
                claim_token_ids = tokenizer.encode(claim, add_special_tokens=False)
                evidence_candidates = []
                hits = search_documents(
                    es,
                    args.es_dump_index,
                    body={
                        "query": {"match": {"token_ids": " ".join(map(str, claim_token_ids))}},
                    },
                    size=3,
                    max_concurrent_shard_requests=64,
                )

            with st.spinner("Extracting the most related snippets..."):
                for hit in hits:
                    text = tokenizer.decode(list(map(int, hit["_source"]["token_ids"].split()))).strip()
                    dataset = hit["_source"]["dataset_name"]
                    training_step = hit["_source"]["iteration"]
                    for passage in chunk_document(text):
                        score = scorer(claim, passage)
                        evidence_candidates.append(
                            {
                                "passage": passage,
                                "dataset": dataset,
                                "training_step": training_step,
                                "score": score,
                            }
                        )
                evidences = sorted(evidence_candidates, key=lambda x: x["score"], reverse=True)[: args.num_evidences]

            with st.spinner("Retrieving the meta information of the evidences..."):
                for evidence in evidences:
                    evidence_token_ids = tokenizer.encode(evidence["passage"], add_special_tokens=False)
                    hits = search_documents(
                        es,
                        args.es_meta_index,
                        body={
                            "query": {"match": {"token_ids": " ".join(map(str, evidence_token_ids))}},
                        },
                        size=1,
                        max_concurrent_shard_requests=64,
                    )
                    if hits:
                        evidence["meta"] = json.loads(hits[0]["_source"]["meta"])
                    else:
                        evidence["meta"] = {}

            for i, evidence in enumerate(evidences, 1):
                with st.expander(f"Evidence {i}"):
                    st.markdown(f"Dataset: {evidence['dataset']}")
                    st.markdown(f"Training Step: {evidence['training_step']}")
                    st.markdown("Meta information:")
                    st.markdown("\n".join(f"- {k}: {v}" for k, v in evidence["meta"].items()))
                    st.markdown(evidence["passage"])
                    st.markdown("---")

            with st.spinner("Verifying the check-worthy claims..."):
                result = verify_claim(claim, [e["passage"] for e in evidences], args.engine)

            st.markdown(f"**Result**: {'Supported' if result['label'] else 'Not Supported'}")
            st.markdown(f"**Rationale**: {result['rationale']}")
            st.markdown("--")


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
