from __future__ import annotations

from typing import TYPE_CHECKING

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
