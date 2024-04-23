import json
from textwrap import dedent
from typing import Any

from utils import client

SYSTEM_PROMPT = dedent(
    """\
    You are provided with a claim and a list of evidences.
    Your task is to verify whether the claim is supported by the evidences.

    Example:
        Input:
            Claim: The earth is flat.
            Evidences:
                1. The earth is round.
                2. Some people believe that the earth is flat, but they are wrong.
        Output:
            Label (boolean): false
            Rationale (string): The claim is not supported by any of the evidences.
    """
)

USER_PROMPT = dedent(
    """\
    Verify the following claim using the provided evidences:
    ---
    [Claim]
    {claim}
    ---
    [Evidences]
    {evidences}
    """
)

TOOL = {
    "type": "function",
    "function": {
        "name": "setVerificationResult",
        "description": "Verify a claim according to the provided evidences.",
        "parameters": {
            "type": "object",
            "properties": {
                "rationale": {
                    "type": "string",
                    "description": "A rationale for the verification result.",
                },
                "label": {
                    "type": ["boolean", "null"],
                    "description": "A label for the verification result. True if the claim is supported by the evidences, and False if it is not. Otherwise, null.",
                },
            },
            "required": ["rationale", "label"],
        },
    },
}

TOOL_CHOICE = {"type": "function", "function": {"name": "setVerificationResult"}}


def verify_claim(claim: str, evidences: list[str], model: str) -> dict[str, Any]:
    """Verify a claim.

    Args:
        claim (str): A claim.
        evidences (list[str]): A list of evidences.
        model (str): A model.
    """
    formatted_evidences = "\n".join(f"- {i} {passage}" for i, passage in enumerate(evidences, start=1))
    ret = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(claim=claim, evidences=formatted_evidences)},
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
                raise ValueError(f"Failed to extract rationale and/or label: {tool_call.function.arguments}")
            rationale = arguments.get("rationale")
            if not isinstance(rationale, str):
                raise ValueError(f"Invalid rationale: {tool_call.function.arguments}")
            label = arguments.get("label")
            if not isinstance(label, bool) and label is not None:
                raise ValueError(f"Invalid label: {tool_call.function.arguments}")
            return arguments
        raise ValueError("Failed to extract claims")


if __name__ == "__main__":
    claim = "The First World War ended in 1920."
    evidences = [
        "The First World War ended in 1918.",
        "The First World War lasted from 1914 to 1918.",
    ]
    print(verify_claim(claim, evidences, "gpt-4-1106-preview"))
