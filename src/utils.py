import json
from typing import Any


def parse_json_response(response: Any) -> dict:
    """Read a completed Structured Outputs response, rejecting refusals and partial output."""
    choices = getattr(response, "choices", None)
    if not choices or len(choices) != 1:
        raise ValueError("Expected exactly one model response choice.")
    message = getattr(choices[0], "message", None)
    if getattr(message, "refusal", None):
        raise ValueError("The model refused to produce the requested structured output.")
    finish_reason = getattr(choices[0], "finish_reason", None)
    if finish_reason != "stop":
        raise ValueError(f"Structured output did not complete normally (finish_reason={finish_reason!r}).")
    if getattr(message, "tool_calls", None):
        raise ValueError("Expected JSON response content, but the model returned tool calls.")
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Expected nonempty JSON response content.")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("The model returned invalid JSON response content.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object in the model response.")
    return payload
