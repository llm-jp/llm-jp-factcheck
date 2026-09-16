"""Independent OpenAI-compatible connections for chat and fact-checking."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from openai import OpenAI

ClientRole = Literal["chatbot", "factchecker"]
CONNECTION_FIELDS = ("API_TYPE", "ENDPOINT", "API_KEY", "API_VERSION")


@dataclass(frozen=True)
class APIConfig:
    api_type: str
    endpoint: str
    api_key: str = field(repr=False)
    api_version: str | None = None

    @classmethod
    def from_env(cls, role: ClientRole) -> APIConfig:
        """Read one role, using shared Azure settings only for unconfigured roles."""
        if role not in ("chatbot", "factchecker"):
            raise ValueError("Client role must be 'chatbot' or 'factchecker'.")
        prefix = role.upper()
        configured = any(f"{prefix}_{name}" in os.environ for name in CONNECTION_FIELDS)
        if not configured:
            prefix = "AZURE_OPENAI"

        def required(name: str) -> str:
            variable = f"{prefix}_{name}"
            value = os.getenv(variable, "").strip()
            if not value:
                raise ValueError(f"Missing {variable} for {role}. Configure {role.upper()}_* in .env.")
            return value

        api_type = required("API_TYPE").lower() if configured else "azure"
        if api_type not in ("openai", "azure"):
            raise ValueError(f"{prefix}_API_TYPE must be 'openai' or 'azure'.")
        endpoint = required("ENDPOINT").rstrip("/")
        try:
            url = urlsplit(endpoint)
            valid_url = url.scheme in ("http", "https") and bool(url.hostname) and url.port != 0
        except ValueError:
            valid_url = False
        if not valid_url or url.username or url.password or url.query or url.fragment:
            raise ValueError(f"{prefix}_ENDPOINT must be an HTTP(S) base URL without credentials, query, or fragment.")
        return cls(
            api_type=api_type,
            endpoint=endpoint,
            api_key=required("API_KEY"),
            api_version=required("API_VERSION") if api_type == "azure" else None,
        )


@lru_cache(maxsize=2)
def get_client(role: ClientRole) -> OpenAI:
    """Create and reuse each role's client on first use; imports need no credentials."""
    from dotenv import load_dotenv
    from openai import AzureOpenAI, OpenAI

    load_dotenv()
    config = APIConfig.from_env(role)
    if config.api_type == "azure":
        client = AzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.endpoint,
        )
        # The pinned SDK reads a global AD token even with an explicit API key.
        # These connections use API keys, so a token for another resource must not override them.
        client._azure_ad_token = None
    else:
        client = OpenAI(api_key=config.api_key, base_url=config.endpoint)
    # Do not inherit global organization or project routing on independent endpoints.
    client.organization = None
    client.project = None
    return client
