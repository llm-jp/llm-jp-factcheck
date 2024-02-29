from elasticsearch import Elasticsearch


def search_documents(host: str, index: str, body: dict, **kwargs) -> list[dict]:
    """Search for documents in an index.

    Args:
        host (str): The Elasticsearch host.
        index (str): The name of the Elasticsearch index.
        body (dict): The body of the request.
        **kwargs: Additional keyword arguments.

    Returns:
        list[dict]: The list of documents that match the query.
    """
    es = Elasticsearch(host)
    res = es.options(request_timeout=2_400).search(
        index=index,
        body=body,
        **kwargs,
    )
    return res["hits"]["hits"]
