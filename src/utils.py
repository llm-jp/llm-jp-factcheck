from logging import getLogger
from os import getenv

from dotenv import load_dotenv
from openai import AzureOpenAI

logger = getLogger(__name__)

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=getenv("AZURE_OPENAI_API_KEY"),
    api_version=getenv("AZURE_OPENAI_API_VERSION"),
)
