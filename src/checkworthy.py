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
        "description": "Decide whether each claim should be fact-checked, preserving input order.",
        "schema": {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "description": "One boolean per claim: true if check-worthy, false otherwise.",
                    "items": {"type": "boolean"},
                },
            },
            "required": ["labels"],
            "additionalProperties": False,
        },
    },
}


def identify_checkworthiness(claims: list[str], model: str, *, prompt_path: str | Path | None = None) -> list[bool]:
    """Return one check-worthiness label per claim through the Factchecker connection."""
    if not isinstance(claims, list):
        raise TypeError("claims must be a list of nonempty strings.")
    if any(not isinstance(claim, str) or not claim.strip() for claim in claims):
        raise ValueError("claims must contain only nonempty strings.")
    if not claims:
        return []
    messages = render_prompt(
        DEFAULT_PROMPT_PATH if prompt_path is None else prompt_path,
        {"claims": "\n".join(f"- {claim}" for claim in claims)},
        {"claims"},
    )
    response = get_client("factchecker").chat.completions.create(
        model=model,
        messages=messages,
        response_format=RESPONSE_FORMAT,
    )
    payload = parse_json_response(response)
    if set(payload) != {"labels"}:
        raise ValueError("Check-worthiness response must contain exactly 'labels'.")
    labels = payload["labels"]
    if not isinstance(labels, list) or any(not isinstance(label, bool) for label in labels):
        raise ValueError("Check-worthiness response 'labels' must be a list of booleans.")
    if len(labels) != len(claims):
        raise ValueError(f"Expected {len(claims)} check-worthiness labels, but got {len(labels)}.")
    return labels
