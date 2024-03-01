import argparse
import logging

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
    parser.add_argument("--document", type=str, required=True)
    parser.add_argument("--context", type=str, default=None)
    parser.add_argument("--engine", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--tokenizer_name", type=str, default="llm-jp/llm-jp-13b-v1.0")
    parser.add_argument("--es_host", type=str, default="http://localhost:9200")
    parser.add_argument("--es_index", type=str, default="memorization-analysis-dev")
    parser.add_argument("--num_evidences", type=int, default=5)
    parser.add_argument("--embedding", type=str, default="intfloat/multilingual-e5-base")
    parser.add_argument("-v", "--verbose", action="store_true", help="Whether to log debug messages.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run the pipeline.

    Args:
        args (argparse.Namespace): The command-line arguments.
    """
    logger.info("Running the fact-checking pipeline.")
    logger.info(f"Document: {args.document}")
    logger.info(f"Context: {args.context}")

    logger.info("Decompose the document into claims.")
    claims = decompose_document_into_claims(
        document=args.document,
        context=args.context,
        model=args.engine,
    )
    logger.info(f"Extracted {len(claims)} claims.")

    logger.info("Identify check-worthy claims.")
    checkworthy_labels = identify_checkworthiness(claims, args.engine)
    checkworthy_claims = [claim for claim, label in zip(claims, checkworthy_labels) if label]
    logger.info(f"Identified {len(checkworthy_claims)} check-worthy claims.")

    logger.info("Retrieve evidence documents.")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    es = create_elasticsearch_client(args.es_host)
    scorer = create_relevance_scorer(args.embedding)
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

        evidences_of_claim.sort(key=lambda x: x[1], reverse=True)
        evidences.append(evidences_of_claim[: args.num_evidences])

    logger.info("Verify the check-worthy claims.")
    for claim, evidences_of_claim in zip(checkworthy_claims, evidences):
        result = verify_claim(claim, evidences_of_claim, args.engine)
        logger.info(f"Claim: {claim.strip()}")
        logger.info(f"Result: {result['label']}")
        logger.info(f"Rationale: {result['rationale']}")
        logger.info("Evidence:")
        for i, (passage, dataset, training_step, _) in enumerate(evidences_of_claim, 1):
            logger.info(f"{i}. Dataset: {dataset} Training Step: {training_step}\n{passage}")


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
