# test/test_llm_fixed.py
import requests, json

payload = {
    "model": "qwen2.5vl:7b",
    "system": "You are a bank marketing assistant.",
    "prompt": "Write a short 6-word EV loan banner headline.",
    "options": {"temperature": 0.2}
    # NOTE: we omit "format" because Ollama rejected "text" in your environment.
}

r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
print("STATUS:", r.status_code)
txt = r.text
print("\nRAW RESPONSE:\n", txt)

# Try to parse JSON safely (Ollama sometimes wraps output in a field)
try:
    data = r.json()
    # Common fields: 'response' or 'output'
    text = data.get("response") or data.get("output") or json.dumps(data)
    print("\nEXTRACTED TEXT:\n", text)
except Exception as e:
    print("\nCould not parse JSON:", e)
    print("Full text:")
    print(txt)
