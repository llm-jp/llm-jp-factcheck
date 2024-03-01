import json
from logging import getLogger
from textwrap import dedent
from typing import Optional

from utils import client

logger = getLogger(__name__)

SYSTEM_PROMPT = dedent(
    """\
    You are provided with a document (or an utterance) with context.
    Your task is to decompose the document into atomic claims.
    Each claim represents one fact.
    Every claim should be context-independent, i.e., it should be understandable alone without the context.
    For example, pronouns should be replaced with the actual names.

    Example:
        Input:
            Context: What do you know about Mary?
            Document: She likes playing piano and doesn't like cookies.
        Output:
            Claims:
                - Mary likes playing piano.
                - Mary doesn't like cookies.

    Example:
        Input:
            Context: アメリカの初代大統領は誰ですか？
            Document: ジョージ・ワシントンです。
        Output:
            Claims:
                - アメリカの初代大統領はジョージ・ワシントンです。
    """
)

USER_PROMPT = dedent(
    """\
    Decompose the following document into atomic claims:
    ---
    [Context]
    {context}
    ---
    [Document]
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


def decompose_document_into_claims(document: str, model: str, context: Optional[str] = None) -> list[str]:
    """Decompose a document into a list of claims.

    Args:
        document (str): A document.
        model (str): A model.
        context (str, optional): A context. Defaults to None.

    Returns:
        list[str]: A list of claims.
    """
    if document.strip() == "":
        return []

    ret = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(document=document, context=context)},
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
    context = "When did the First World War end?"
    document = "It ended on 11 November 1918. It lasted for four years."
    claims = decompose_document_into_claims(document, context=context, model="gpt-4-1106-preview")
    print(claims)
