"""Stream a normal assistant reply while keeping research metadata local."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing

from clients import get_client


def stream_chat_response(messages: list[dict], model: str) -> Iterator[str]:
    """Yield reply fragments, succeeding only for a complete, nonempty reply.

    Callers must exhaust the iterator before storing the assistant message.
    Close this generator if rendering is interrupted to release the stream.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("Chat history must be a nonempty list of messages.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A model name or Azure deployment is required.")
    payload = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            raise ValueError("Chat messages must have a user or assistant role.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Chat messages must contain nonempty text.")
        payload.append({"role": message["role"], "content": content})
    if payload[-1]["role"] != "user":
        raise ValueError("The last chat message must be from the user.")

    has_text = False
    completed = False
    with closing(get_client("chatbot").chat.completions.create(model=model, messages=payload, stream=True)) as stream:
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if choices is None:
                raise ValueError("The response stream contains an invalid chunk.")
            if not choices:
                # Usage and Azure prompt annotations can have no choices.
                continue
            if len(choices) != 1:
                raise ValueError("The response stream contains unexpected completion choices.")
            choice = choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason is not None and finish_reason != "stop":
                raise ValueError(f"The assistant response did not complete (finish_reason={finish_reason}).")
            content = getattr(getattr(choice, "delta", None), "content", None)
            if content is not None and not isinstance(content, str):
                raise ValueError("The response stream contains invalid text.")
            if content:
                if completed:
                    raise ValueError("The response stream contains text after completion.")
                has_text = has_text or bool(content.strip())
                yield content
            # Azure may send annotation choices after the completion's stop.
            completed = completed or finish_reason == "stop"
        if not has_text:
            raise ValueError("The model returned an empty response. Please try again.")
        if not completed:
            raise ValueError("The response stream ended before the assistant finished. Please try again.")
