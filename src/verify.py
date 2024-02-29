import json
from textwrap import dedent
from typing import Any

from utils import client

SYSTEM_PROMPT = dedent(
    """\
    You are provided with a claim and a list of evidences.
    Your task is to verify whether the claim is supported by the evidences.
    """
)

USER_PROMPT = dedent(
    """\
    Verify the following claim using the provided evidences:
    ---
    Claim: {claim}
    Passages:
    {passages}
    """
)

TOOL = {
    "type": "function",
    "function": {
        "name": "setVerificationResult",
        "description": "Verify a claim using a list of passages.",
        "parameters": {
            "type": "object",
            "properties": {
                "rationale": {
                    "type": "string",
                    "description": "A rationale for the verification result.",
                },
                "label": {
                    "type": ["boolean", "null"],
                    "description": "A label for the verification result. "
                    "True if the claim is supported by the evidences, and False if it is not. "
                    "Otherwise, null.",
                },
            },
            "required": ["rationale", "label"],
        },
    },
}

TOOL_CHOICE = {"type": "function", "function": {"name": "setVerificationResult"}}


def verify_claim(claim: str, passages: list[str], model: str) -> dict[str, Any]:
    """Verify a claim.

    Args:
        claim (str): A claim.
        passages (list[str]): A list of passages.
        model (str): A model.
    """
    formatted_passages = "\n".join(f"- {i} {passage}" for i, passage in enumerate(passages, start=1))
    ret = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(claim=claim, passages=formatted_passages)},
        ],
        tools=[TOOL],
        tool_choice=TOOL_CHOICE,
    )
    for tool_call in ret.choices[0].message.tool_calls:
        if tool_call.function.name == "setVerificationResult":
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse JSON: {tool_call.function.arguments}")
            if "rationale" not in arguments or "label" not in arguments:
                raise ValueError(f"Failed to extract rationale and label: {tool_call.function.arguments}")
            rationale = arguments.get("rationale")
            if not isinstance(rationale, str):
                raise ValueError(f"Invalid rationale: {tool_call.function.arguments}")
            label = arguments.get("label")
            if not isinstance(label, bool) and label is not None:
                raise ValueError(f"Invalid label: {tool_call.function.arguments}")
            return arguments
        raise ValueError("Failed to extract claims")


if __name__ == "__main__":
    claim = "The earth is flat."
    passages = [
        "The earth is round.",
        "Some people believe that the earth is flat, but they are wrong.",
    ]
    print(verify_claim(claim, passages, "gpt-4-1106-preview"))
