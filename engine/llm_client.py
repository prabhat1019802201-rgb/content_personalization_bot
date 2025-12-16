# engine/llm_client.py
"""
Unified LLM client.
Currently supports Ollama (local).
"""

import json
import requests
from typing import Optional


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b"


def call_llm(prompt: str, model: Optional[str] = None) -> str:
    """
    Call Ollama LLM and return raw text output.

    Args:
        prompt: Full prompt string
        model: Ollama model name (optional)

    Returns:
        str: raw model output text
    """

    payload = {
        "model": model or DEFAULT_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        # Ollama returns output under "response"
        return data.get("response", "").strip()

    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {e}")
