import json
from logging import getLogger
from textwrap import dedent

from utils import client

logger = getLogger(__name__)

SYSTEM_PROMPT = dedent(
    """\
    You are provided with texts. Your task is to identify whether each of them is worth fact-checking.
    For example, the following texts are check-worthy:
    - Friends is a great TV series
    - The Stanford Prison Experiment was conducted in the basement of Encina Hall.
    while the following texts are not check-worthy:
    - I think Apple is a good company.
    - Are you sure Preslav is a professor in MBZUAI?
    - As a language model, I can't provide these info.
    """
)

USER_PROMPT = dedent(
    """\
    Identify whether the following texts are check-worthy in the context of fact-checking:
    ---
    {texts}
    """
)

TOOL = {
    "type": "function",
    "function": {
        "name": "setCheckworthyLabels",
        "description": "Set check-worthy labels by identifying whether the texts are worth fact-checking.",
        "parameters": {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "description": "A list of labels, which is True if the text is check-worthy and False otherwise.",
                    "items": {
                        "type": "boolean",
                    },
                },
            },
            "required": ["labels"],
        },
    },
}

TOOL_CHOICE = {"type": "function", "function": {"name": "setCheckworthyLabels"}}


def identify_checkworthiness(claims: list[str], model: str) -> list[bool]:
    """Decompose a document into a list of statements.

    Args:
        claims (str): A list of claims.
        model (str): A model.

    Returns:
        list[str]: A list of statements.
    """
    if not claims:
        return []

    texts = "\n".join(f"- {claim}" for claim in claims)
    ret = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(texts=texts)},
        ],
        tools=[TOOL],
        tool_choice=TOOL_CHOICE,
    )
    for tool_call in ret.choices[0].message.tool_calls:
        if tool_call.function.name == "setCheckworthyLabels":
            try:
                labels = json.loads(tool_call.function.arguments).get("labels", [])
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse JSON: {tool_call.function.arguments}")
            if not isinstance(labels, list) or not all(isinstance(label, bool) for label in labels):
                logger.error(f"Invalid labels: {tool_call.function.arguments}")
            if len(labels) != len(claims):
                raise ValueError(f"Expected {len(claims)} labels, but got {len(labels)}.")
            return labels
    raise ValueError("Failed to identify check-worthiness.")


if __name__ == "__main__":
    claims = [
        "The capital of France is Paris.",
        "The first prime number is 1.",
        "I think Google is a good company.",
        "Do you think it will be sunny tomorrow?",
    ]
    labels = identify_checkworthiness(claims, "gpt-4-1106-preview")
    print(labels)
