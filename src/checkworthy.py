"""Identify claims worth checking using an editable prompting template."""

from __future__ import annotations

from pathlib import Path

from clients import get_client
from prompts import PROMPT_DIR, render_prompt
from utils import parse_json_response

DEFAULT_PROMPT_PATH = PROMPT_DIR / "checkworthiness.json"

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "checkworthiness",
        "strict": True,
        "description": "Decide whether one claim should be fact-checked.",
        "schema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "boolean",
                    "description": "True if this claim is check-worthy, false otherwise.",
                },
            },
            "required": ["label"],
            "additionalProperties": False,
        },
    },
}


def identify_checkworthiness(claim: str, model: str, *, prompt_path: str | Path | None = None) -> bool:
    """Judge one claim independently through the Factchecker connection."""
    if not isinstance(claim, str):
        raise TypeError("claim must be a string.")
    if not claim.strip():
        raise ValueError("claim must be a nonempty string.")
    messages = render_prompt(
        DEFAULT_PROMPT_PATH if prompt_path is None else prompt_path,
        {"claim": claim},
        {"claim"},
    )
    response = get_client("factchecker").chat.completions.create(
        model=model,
        messages=messages,
        response_format=RESPONSE_FORMAT,
    )
    payload = parse_json_response(response)
    if set(payload) != {"label"}:
        raise ValueError("Check-worthiness response must contain exactly 'label'.")
    label = payload["label"]
    if not isinstance(label, bool):
        raise ValueError("Check-worthiness response 'label' must be a boolean.")
    return label
