from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch


def create_elasticsearch_client(host: str) -> Elasticsearch:
    """Create an Elasticsearch client.

    Args:
        host (str): The Elasticsearch host.

    Returns:
        Elasticsearch: The Elasticsearch client.
    """
    from elasticsearch import Elasticsearch

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


def chunk_document(document: str, chunk_size: int = 500, chunk_overlap: int = 200) -> list[str]:
    """Chunk a document into smaller pieces (called passages).

    Args:
        document (str): The document.
        chunk_size (int, optional): The size of each chunk. Defaults to 500.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 200.

    Returns:
        list[str]: The list of passages.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(document)


def create_relevance_scorer(model_name: str) -> Callable[[str, str], float]:
    import torch
    from transformers import AutoModel, AutoTokenizer

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
            query_embedding = (
                model(**tokenizer(query, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length))
                .last_hidden_state[0]
                .mean(dim=0)
            )
            passage_embedding = (
                model(**tokenizer(passage, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length))
                .last_hidden_state[0]
                .mean(dim=0)
            )

        return torch.cosine_similarity(query_embedding, passage_embedding, dim=0).item()

    return calculate_relevance_score
