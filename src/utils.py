from time import sleep
from logging import getLogger
from os import getenv
from typing import Optional

from dotenv import load_dotenv
from openai import AzureOpenAI

logger = getLogger(__name__)

load_dotenv()

client = AzureOpenAI(
    api_key=getenv("AZURE_OPENAI_API_KEY"),
    api_version=getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=getenv("AZURE_OPENAI_ENDPOINT"),
)


def _run_chat_completion(
    model: str,
    system_prompt: str,
    user_prompt: str,
    **kwargs,
) -> str:
    """Run a chat completion.

    Args:
        model (str): A model.
        system_prompt (str): A system prompt.
        user_prompt (str): A user prompt.
        kwargs: Keyword arguments.

    Returns:
        str: A chat completion.
    """
    ret = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **kwargs,
    )
    return ret.choices[0].message.content


def run_chat_completion(
    model: str,
    system_prompt: str,
    user_prompt: str,
    num_retries: int = 3,
    sleep_time: int = 5,
    **kwargs,
) -> Optional[str]:
    """Run a chat completion.

    Args:
        model (str): A model.
        system_prompt (str): A system prompt.
        user_prompt (str): A user prompt.
        num_retries (int, optional): Number of retries. Defaults to 3.
        sleep_time (int, optional): Sleep time. Defaults to 5.
        kwargs: Keyword arguments.

    Returns:
        str: A chat completion.
    """
    for _ in range(num_retries):
        try:
            return _run_chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                **kwargs,
            )
        except Exception as e:
            logger.error(e)
            sleep(sleep_time)
    return None
