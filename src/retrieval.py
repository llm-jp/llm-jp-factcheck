from typing import Callable

import torch
from elasticsearch import Elasticsearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

    def calculate_relevance_score(query: str, passage: str) -> float:
        """Calculate the relevance score between a query and a document.

        Args:
            query (str): The query.
            passage (str): The passage.

        Returns:
            float: The relevance score.
        """
        query = f"passage: {query}"
        passage = f"passage: {passage}"

        with torch.no_grad():
            query_embedding = model(**tokenizer(query, return_tensors="pt")).last_hidden_state[0].mean(dim=0)
            passage_embedding = model(**tokenizer(passage, return_tensors="pt")).last_hidden_state[0].mean(dim=0)

        return torch.cosine_similarity(query_embedding, passage_embedding, dim=0).item()

    return calculate_relevance_score


def chunk_document(document: str, chunk_size: int = 500, chunk_overlap: int = 200) -> list[str]:
    """Chunk a document into smaller pieces (called passages).

    Args:
        document (str): The document.
        chunk_size (int, optional): The size of each chunk. Defaults to 500.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 200.

    Returns:
        list[str]: The list of passages.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(document)


if __name__ == "__main__":
    scorer = create_relevance_scorer("intfloat/multilingual-e5-small")
    print(scorer("The capital of France is Paris.", "Paris is the capital of France."))
    print(scorer("The capital of France is Paris.", "Paris is the capital of the US."))
    print(scorer("The capital of France is Paris.", "フランスの首都はパリです。"))

    document = "The capital of France is Paris. The Eiffel Tower is in Paris."
    print(chunk_document(document, chunk_size=50, chunk_overlap=20))
