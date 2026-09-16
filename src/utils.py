import json
from typing import Any


def parse_tool_response(response: Any, expected_name: str) -> dict:
    """Require one function call with a JSON object, or raise a useful error."""
    choices = getattr(response, "choices", None)
    if not choices or len(choices) != 1:
        raise ValueError("Expected exactly one model response choice.")
    message = getattr(choices[0], "message", None)
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls or len(tool_calls) != 1:
        raise ValueError(f"Expected exactly one {expected_name} tool call.")
    function = getattr(tool_calls[0], "function", None)
    if getattr(function, "name", None) != expected_name:
        raise ValueError(f"Expected a {expected_name} tool call.")
    arguments = getattr(function, "arguments", None)
    if not isinstance(arguments, str):
        raise ValueError(f"Invalid JSON arguments for {expected_name}.")
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON arguments for {expected_name}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object from {expected_name}.")
    return payload
