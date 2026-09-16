from __future__ import annotations

from pathlib import Path

from clients import get_client
from prompts import PROMPT_DIR, render_prompt
from utils import parse_tool_response

DEFAULT_PROMPT_PATH = PROMPT_DIR / "verification.json"
VERIFICATION_LABELS = (
    "Supported",
    "Partially supported",
    "Partially refuted",
    "Refuted",
    "Not enough information",
)

TOOL = {
    "type": "function",
    "function": {
        "name": "setVerificationResult",
        "description": "Verify one claim against one evidence passage using the five specified labels.",
        "parameters": {
            "type": "object",
            "properties": {
                "rationale": {"type": "string", "description": "Explain the label using the supplied evidence."},
                "label": {"type": "string", "enum": list(VERIFICATION_LABELS)},
            },
            "required": ["rationale", "label"],
            "additionalProperties": False,
        },
    },
}

TOOL_CHOICE = {"type": "function", "function": {"name": "setVerificationResult"}}


def verify_claim(
    claim: str,
    evidence: str,
    model: str,
    *,
    prompt_path: str | Path | None = None,
) -> dict[str, str]:
    """Predict a label and rationale for a single claim–evidence pair."""
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must be a nonempty string.")
    if not isinstance(evidence, str):
        raise TypeError("evidence must be one string, not a list of passages.")
    if not evidence.strip():
        return {"label": "Not enough information", "rationale": "No evidence was provided."}
    messages = render_prompt(
        DEFAULT_PROMPT_PATH if prompt_path is None else prompt_path,
        {"claim": claim, "evidence": evidence},
        {"claim", "evidence"},
    )
    response = get_client("factchecker").chat.completions.create(
        model=model,
        messages=messages,
        tools=[TOOL],
        tool_choice=TOOL_CHOICE,
    )
    payload = parse_tool_response(response, "setVerificationResult")
    if set(payload) != {"label", "rationale"}:
        raise ValueError("Verification response must contain exactly 'label' and 'rationale'.")
    label = payload["label"]
    rationale = payload["rationale"]
    if not isinstance(label, str) or label not in VERIFICATION_LABELS:
        raise ValueError(f"Verification response 'label' must be one of {VERIFICATION_LABELS}.")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Verification response 'rationale' must be a nonempty string.")
    return {"label": label, "rationale": rationale.strip()}
