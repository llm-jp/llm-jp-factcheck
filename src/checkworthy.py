import json
from logging import getLogger
from textwrap import dedent
from utils import run_chat_completion

logger = getLogger(__name__)

SYSTEM_PROMPT = dedent(
    """\
    You are provided with texts. Your task is to identify whether the texts are checkworthy in the context of fact-checking.
    For example, the following texts are checkworthy:
    - Friends is a great TV series
    - The Stanford Prison Experiment was conducted in the basement of Encina Hall.
    while the following texts are not checkworthy:
    - I think Apple is a good company.
    - Are you sure Preslav is a professor in MBZUAI?
    - As a language model, I can't provide these info.
    """
)

USER_PROMPT = dedent(
    """\
    Identify whether the following texts are checkworthy in the context of fact-checking:
    ---
    {texts}
    """
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_checkworthy_labels",
            "description": "Identify whether the texts are checkworthy in the context of fact-checking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "labels": {
                        "type": "array",
                        "description": "A list of labels, which is True if the text is checkworthy and False otherwise.",
                        "items": {
                            "type": "boolean",
                        },
                    },
                },
                "required": ["labels"],
            },
        },
    }
]

TOOL_CHOICE = {"type": "function", "function": {"name": "set_checkworthy_labels"}}


def identify_checkworthiness(claims: list[str], model: str) -> list[bool]:
    """Decompose a document into a list of statements.

    Args:
        claims (str): A list of claims.
        model (str): A model.

    Returns:
        list[str]: A list of statements.
    """
    texts = "\n".join(f"{i}. {claim}" for i, claim in enumerate(claims, 1))
    ret = run_chat_completion(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT.format(texts=texts),
        tools=TOOLS,
        tool_choice=TOOL_CHOICE,
    )
    if ret is None:
        logger.error("Failed to run claim extraction.")
    for tool_call in ret.choices[0].message.tool_calls:
        if tool_call.function.name == "set_checkworthy_labels":
            try:
                labels = json.loads(tool_call.function.arguments).get("labels", [])
                assert isinstance(labels, list) and all(
                    isinstance(label, bool) for label in labels
                )
                return labels
            except AssertionError:
                logger.error(f"Invalid labels: {tool_call.function.arguments}")
            except Exception as e:
                logger.error(f"An error occurred: {e}")
    return []


if __name__ == "__main__":
    claims = [
        "The capital of France is Paris.",
        "The first prime number is 1.",
        "I think Google is a good company.",
        "Do you think it will be sunny tomorrow?",
    ]
    labels = identify_checkworthiness(claims, "gpt-4-1106-preview")
    print(labels)
