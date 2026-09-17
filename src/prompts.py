"""Load editable UTF-8 YAML prompts without caching their contents."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import yaml

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
_PLACEHOLDER = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)


def render_prompt(
    path: str | Path,
    values: Mapping[str, str],
    required_placeholders: set[str],
) -> list[dict[str, str]]:
    """Read a prompt, validate its placeholders, and return chat messages.

    YAML contains exactly ``system`` and ``user`` strings. Legacy JSON prompt
    files are also accepted. Placeholders use
    double braces (for example ``{{document}}``); ordinary JSON example braces
    are left alone. Substitution is a single pass, including for input that
    itself contains a placeholder-looking string.
    """
    path = Path(path)
    try:
        template = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load prompt {path}: {exc}") from exc
    if not isinstance(template, dict) or set(template) != {"system", "user"}:
        raise ValueError(f"Prompt {path} must contain exactly 'system' and 'user'.")
    if any(not isinstance(text, str) or not text.strip() for text in template.values()):
        raise ValueError(f"Prompt {path}: 'system' and 'user' must be nonempty strings.")

    placeholders = {name for text in template.values() for name in _PLACEHOLDER.findall(text)}
    unknown = placeholders - values.keys()
    if unknown:
        raise ValueError(f"Prompt {path} has unknown placeholders: {', '.join(sorted(unknown))}.")
    missing = required_placeholders - placeholders
    if missing:
        raise ValueError(f"Prompt {path} is missing placeholders: {', '.join(sorted(missing))}.")
    return [
        {"role": role, "content": _PLACEHOLDER.sub(lambda match: values[match.group(1)], template[role])}
        for role in ("system", "user")
    ]
