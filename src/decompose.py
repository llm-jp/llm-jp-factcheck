from __future__ import annotations

from pathlib import Path

from clients import get_client
from prompts import PROMPT_DIR, render_prompt
from utils import parse_json_response

DEFAULT_PROMPT_PATH = PROMPT_DIR / "decomposition_8shot.json"

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "decomposition",
        "strict": True,
        "description": "Create a list of self-contained claims from generated text.",
        "schema": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "description": "Claims in the same language and order as the source text.",
                    "items": {"type": "string"},
                },
            },
            "required": ["claims"],
            "additionalProperties": False,
        },
    },
}


def decompose_document_into_claims(
    document: str,
    model: str,
    context: str | None = None,
    *,
    prompt_path: str | Path | None = None,
) -> list[str]:
    """Decompose generated text using an editable prompting template."""
    if not isinstance(document, str):
        raise TypeError("document must be a string.")
    if context is not None and not isinstance(context, str):
        raise TypeError("context must be a string or None.")
    if not document.strip():
        return []
    messages = render_prompt(
        DEFAULT_PROMPT_PATH if prompt_path is None else prompt_path,
        {"document": document, "context": context or ""},
        {"document"},
    )
    response = get_client("factchecker").chat.completions.create(
        model=model,
        messages=messages,
        response_format=RESPONSE_FORMAT,
    )
    payload = parse_json_response(response)
    if set(payload) != {"claims"}:
        raise ValueError("Decomposition response must contain exactly 'claims'.")
    claims = payload["claims"]
    if not isinstance(claims, list) or any(not isinstance(claim, str) or not claim.strip() for claim in claims):
        raise ValueError("Decomposition response 'claims' must be a list of nonempty strings.")
    return [claim.strip() for claim in claims]
