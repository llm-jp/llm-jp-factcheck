from typing import Callable

import torch
from elasticsearch import Elasticsearch
from transformers import AutoModel, AutoTokenizer


def create_elasticsearch_client(host: str) -> Elasticsearch:
    """Create an Elasticsearch client.

    Args:
        host (str): The Elasticsearch host.

    Returns:
        Elasticsearch: The Elasticsearch client.
    """
    return Elasticsearch(host)


def search_documents(es: Elasticsearch, index: str, body: dict, **kwargs) -> list[dict]:
    """Search for documents in an index.

    Args:
        es (Elasticsearch): The Elasticsearch client.
        index (str): The name of the Elasticsearch index.
        body (dict): The body of the request.
        **kwargs: Additional keyword arguments.

    Returns:
        list[dict]: The list of documents that match the query.
    """
    res = es.options(request_timeout=2_400).search(
        index=index,
        body=body,
        **kwargs,
    )
    return res["hits"]["hits"]


def create_relevance_scorer(model_name: str) -> Callable[[str, str], float]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name)
    model.eval()

    def calculate_relevance_score(query: str, document: str) -> float:
        """Calculate the relevance score between a query and a document.

        Args:
            query (str): The query.
            document (str): The document.

        Returns:
            float: The relevance score.
        """
        query = f"passage: {query}"
        document = f"passage: {document}"

        with torch.no_grad():
            query_last_hidden_states = model(**tokenizer(query, return_tensors="pt")).last_hidden_state[0]
        query_embedding = query_last_hidden_states.mean(dim=0)

        with torch.no_grad():
            document_last_hidden_states = model(**tokenizer(document, return_tensors="pt")).last_hidden_state[0]
        document_embedding = document_last_hidden_states.mean(dim=0)

        return torch.cosine_similarity(query_embedding, document_embedding, dim=0).item()

    return calculate_relevance_score


if __name__ == "__main__":
    scorer = create_relevance_scorer("intfloat/multilingual-e5-small")
    print(scorer("The capital of France is Paris.", "Paris is the capital of France."))
    print(scorer("The capital of France is Paris.", "Paris is the capital of the US."))
    print(scorer("The capital of France is Paris.", "フランスの首都はパリです。"))
