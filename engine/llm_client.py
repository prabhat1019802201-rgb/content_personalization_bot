import json
from typing import List, Dict, Tuple

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"


def call_ollama_raw(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 120,
) -> str:
    """
    Call Ollama's /api/generate endpoint and return the raw text response.

    NOTE: This assumes Ollama is installed and running locally.
    If it's not running, this will raise a ConnectionError.
    """
    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": user_prompt,
        # We want plain text output; we will parse JSON lines ourselves.
        "stream": False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            "Could not connect to Ollama at "
            f"{OLLAMA_URL}. Is Ollama installed and running?"
        ) from e

    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama returned status {resp.status_code}: {resp.text}"
        )

    # Ollama wraps the result as JSON: {"response": "...", ...}
    data = resp.json()
    return data.get("response", "")


def parse_variants_from_response(raw_text: str) -> List[Dict]:
    """
    The model is instructed to output 3 JSON objects, one per line.
    This helper tries to parse them into Python dicts.
    """
    variants = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Some models might wrap text with extra content; try to find JSON object.
        try:
            obj = json.loads(line)
            variants.append(obj)
        except json.JSONDecodeError:
            # try a best-effort extraction
            start = line.find("{")
            end = line.rfind("}")
            if start != -1 and end != -1 and start < end:
                try:
                    obj = json.loads(line[start : end + 1])
                    variants.append(obj)
                except json.JSONDecodeError:
                    continue
    return variants


def generate_variants(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
) -> List[Dict]:
    """
    High-level helper:
    - Call Ollama
    - Parse A/B/C variants from the response
    """
    raw = call_ollama_raw(system_prompt, user_prompt, model=model)
    variants = parse_variants_from_response(raw)

    # Fallback: if nothing parsed, treat the whole response as single variant
    if not variants:
        variants = [
            {
                "variant_tag": "A",
                "subject": "",
                "body": raw,
                "disclaimers": [],
            }
        ]
    return variants


if __name__ == "__main__":
    # Simple smoke test (will fail if Ollama not running)
    from engine.prompting import build_prompts_for_customer

    sp, up, cust, prod = build_prompts_for_customer("C00001")
    print("Calling Ollama for customer:", cust["customer_id"], cust["name"])

    try:
        vars_ = generate_variants(sp, up)
        print("Got variants:")
        for v in vars_:
            print(v.get("variant_tag"), ":", v.get("body")[:200], "...\n")
    except Exception as e:
        print("Error calling Ollama:", e)
