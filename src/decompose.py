from textwrap import dedent
from utils import run_chat_completion

SYSTEM_PROMPT = "You are good at decomposing and decontextualizing text."

# TODO: Evaluating the return value as Python code is a security risk. Use function calling.
USER_PROMPT = dedent(
    """\
    Your task is to decompose the text into atomic claims.
    Let's define a function named decompose(input:str).
    The returned value should be a list of strings, where each string should be a context-independent claim, representing one fact.
    For example, if a user call decompose("Mary is a five-year old girl, she likes playing piano and she doesn't like cookies."),
    you should return a python list without any other words:
    ["Mary is a five-year old girl.", "Mary likes playing piano.", "Mary doesn't like cookies."]
    Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!

    decompose("{document}")
    """
)


def decompose_document_into_claims(document: str, model: str) -> list[str]:
    """Decompose a document into a list of statements.

    Args:
        document (str): A document.
        model (str): A model.

    Returns:
        list[str]: A list of statements.
    """
    ret = run_chat_completion(
        model,
        SYSTEM_PROMPT,
        USER_PROMPT.format(document=document),
    )
    claims = eval(ret)
    assert isinstance(claims, list) and all(isinstance(claim, str) for claim in claims)
    return claims
