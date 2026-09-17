from __future__ import annotations

from pathlib import Path

from clients import get_client
from prompts import PROMPT_DIR, render_prompt
from utils import parse_json_response

DEFAULT_PROMPT_PATH = PROMPT_DIR / "verification.yaml"
VERIFICATION_LABELS = (
    "Fully supported",
    "Inferentially supported",
    "Partially supported",
    "Fully refuted",
    "Inferentially refuted",
    "Not enough information",
)
MODEL_LABELS = dict(zip(("完全支持", "推定支持", "部分支持", "完全否定", "推定否定", "不明"), VERIFICATION_LABELS))

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "verification",
        "strict": True,
        "description": "Assign one of the six verdict labels to one claim-evidence pair.",
        "schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": list(MODEL_LABELS)},
                "rationale": {
                    "type": "string",
                    "description": "A brief justification in one English sentence of at most 40 words.",
                },
            },
            "required": ["label", "rationale"],
            "additionalProperties": False,
        },
    },
}


def verify_claim(
    claim: str,
    evidence: str,
    model: str,
    *,
    prompt_path: str | Path | None = None,
) -> dict[str, str]:
    """Predict a six-class verdict with a brief evidence-based justification."""
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
        response_format=RESPONSE_FORMAT,
    )
    payload = parse_json_response(response)
    if set(payload) != {"label", "rationale"}:
        raise ValueError("Verification response must contain exactly 'label' and 'rationale'.")
    label = payload["label"]
    if not isinstance(label, str) or label not in MODEL_LABELS:
        raise ValueError(f"Verification response 'label' must be one of {tuple(MODEL_LABELS)}.")
    rationale = payload["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Verification response 'rationale' must be a nonempty string.")
    return {"label": MODEL_LABELS[label], "rationale": rationale.strip()}
