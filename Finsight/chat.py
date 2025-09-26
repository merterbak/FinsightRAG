from __future__ import annotations
import os
from typing import Dict, List, Optional
from mistralai import Mistral


def read_mistral_key(config: Optional[Dict] = None):
    if config:
        config_value = config.get("mistral_api_key")
        if isinstance(config_value, str) and config_value:
            return config_value

    env_value = os.getenv("MISTRAL_API_KEY")
    if env_value:
        return env_value

    raise RuntimeError(
        "Missing MISTRAL_API_KEY. Add it to your .env file or set it via the UI configuration."
    )


def stream_chat_with_mistral(messages: List[Dict[str, str]], config: Optional[Dict], model: str = "mistral-medium-2508", temperature: float = 0.2):
    client = Mistral(api_key=read_mistral_key(config))
    stream = client.chat.stream(
        model=model,
        messages=[{"role": item.get("role", "user"), "content": item.get("content", "")} for item in messages],
        temperature=float(temperature),
    )

    for chunk in stream:
        if not chunk or not getattr(chunk, "data", None):
            continue
        choices = getattr(chunk.data, "choices", None) or []
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None) if delta else None
        if content:
            yield content


