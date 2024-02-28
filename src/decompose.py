import json
from textwrap import dedent
from utils import run_chat_completion

SYSTEM_PROMPT = dedent(
    """\
    You are provided with a document. Your task is to decompose the text into atomic claims.
    For example, the document "Mary is a five-year old girl, she likes playing piano and she doesn't like cookies." is decomposed into the following claims:
    1. Mary is a five-year old girl.
    2. Mary likes playing piano.
    3. Mary doesn't like cookies.
    """
)

USER_PROMPT = dedent(
    """\
    Decompose the following document into atomic claims:
    ---
    {document}
    """
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_claim_list",
            "description": "Decompose text into atomic claims, each representing one context-independent fact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "description": "A list of claims, each representing one context-independent fact.",
                        "items": {
                            "type": "string",
                        },
                    },
                },
                "required": ["claims"],
            },
        },
    }
]

TOOL_CHOICE = {"type": "function", "function": {"name": "create_claim_list"}}


def decompose_document_into_claims(document: str, model: str) -> list[str]:
    """Decompose a document into a list of statements.

    Args:
        document (str): A document.
        model (str): A model.

    Returns:
        list[str]: A list of statements.
    """
    ret = run_chat_completion(
        model,
        SYSTEM_PROMPT,
        USER_PROMPT.format(document=document),
        tools=TOOLS,
        tool_choice=TOOL_CHOICE,
    )
    for tool_call in ret.choices[0].message.tool_calls:
        if tool_call.function.name == "create_claim_list":
            claims = json.loads(tool_call.function.arguments).get("claims", [])
            assert isinstance(claims, list) and all(
                isinstance(claim, str) for claim in claims
            )
            return claims
    return []
