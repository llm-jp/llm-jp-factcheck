import json
from logging import getLogger
from textwrap import dedent

from utils import client

logger = getLogger(__name__)

SYSTEM_PROMPT = dedent(
    """\
    You are provided with a document. Your task is to decompose the text into atomic claims so that each claim represents one context-independent fact.
    For example, the document "Mary is a five-year old girl, she likes playing piano and she doesn't like cookies." is decomposed into the following claims:
    - Mary is a five-year old girl.
    - Mary likes playing piano.
    - Mary doesn't like cookies.
    """
)

USER_PROMPT = dedent(
    """\
    Decompose the following document into atomic claims:
    ---
    {document}
    """
)

TOOL = {
    "type": "function",
    "function": {
        "name": "createClaimList",
        "description": "Create a list of atomic claims, each representing one context-independent fact.",
        "parameters": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "description": "A list of claims.",
                    "items": {"type": "string"},
                },
            },
            "required": ["claims"],
        },
    },
}

TOOL_CHOICE = {"type": "function", "function": {"name": "createClaimList"}}


def decompose_document_into_claims(document: str, model: str) -> list[str]:
    """Decompose a document into a list of statements.

    Args:
        document (str): A document.
        model (str): A model.

    Returns:
        list[str]: A list of statements.
    """
    if document.strip() == "":
        return []

    ret = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(document=document)},
        ],
        tools=[TOOL],
        tool_choice=TOOL_CHOICE,
    )
    for tool_call in ret.choices[0].message.tool_calls:
        if tool_call.function.name == "createClaimList":
            try:
                claims = json.loads(tool_call.function.arguments).get("claims", [])
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse JSON: {tool_call.function.arguments}")
            if not isinstance(claims, list) or not all(isinstance(claim, str) for claim in claims):
                raise ValueError(f"Invalid claims: {tool_call.function.arguments}")
            claims = [claim.strip() for claim in claims if claim.strip()]
            return claims
    raise ValueError("Failed to extract claims")


if __name__ == "__main__":
    document = (
        "The first thing to do is to understand the problem. "
        "The second thing to do is to decompose the problem into smaller problems. "
        "The third thing to do is to solve the smaller problems."
    )
    claims = decompose_document_into_claims(document, "gpt-4-1106-preview")
    print(claims)
