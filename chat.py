import requests
from typing import List, Dict, Generator
from config import get_api_key, validate_api_key


def chat_with_mistral(messages: List[Dict[str, str]], config: Dict, model: str = "mistral-small-latest", temperature: float = 0.2) -> str:
    api_key = get_api_key(config, "mistral")
    validate_api_key(api_key, "MISTRAL")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature)
    }
    
    resp = requests.post("https://api.mistral.ai/v1/chat/completions", json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def stream_chat_with_mistral(messages: List[Dict[str, str]], config: Dict, model: str = "mistral-small-latest", temperature: float = 0.2) -> Generator[str, None, None]:
    from mistralai import Mistral
    
    api_key = get_api_key(config, "mistral")
    validate_api_key(api_key, "MISTRAL")
    
    client = Mistral(api_key=api_key)
    
    mistral_messages = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]
    
    with client.chat_stream(model=model, messages=mistral_messages, temperature=float(temperature)) as stream:
        for chunk in stream:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    yield delta.content
