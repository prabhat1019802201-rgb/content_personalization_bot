# test/test_vlm_review_fixed.py
import base64, requests, json, sys
IMG_PATH = "test_banner.png"

with open(IMG_PATH, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

# Ask the model to return JSON; also accept non-JSON and try to parse the last line
prompt = (
    f"Image (base64): {img_b64}\n"
    "Headline: Drive Green.\n"
    "Subtitle: Union Green Vehicle Loans.\n"
    "CTA: Know More.\n"
    "Return JSON only with keys: score (0-1), blocked (bool), comments (list of short strings)."
)

payload = {
    "model": "qwen2.5vl:7b",
    "system": "You are a creative reviewer. Return a short JSON with keys: score, blocked, comments.",
    "prompt": prompt,
    "format": "json",          # ask for JSON (works better for structured replies)
    "options": {"temperature": 0.0}
}

print("Sending VLM request (may take a little while)...")
r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
print("STATUS:", r.status_code)
text = r.text
print("\nRAW RESPONSE:\n", text[:1000])  # print first 1000 chars

# Try parsing smartly
try:
    # sometimes r.json() returns an outer dict; sometimes 'response' contains JSON string
    outer = r.json()
    candidate = outer.get("response") or outer.get("output") or outer
    if isinstance(candidate, str):
        # try parse candidate as JSON
        parsed = json.loads(candidate)
    else:
        parsed = candidate
    print("\nPARSED JSON:\n", json.dumps(parsed, indent=2))
except Exception as e:
    print("\nCould not parse structured JSON:", e)
    # Fallback: try to find JSON in last line
    last_line = text.strip().splitlines()[-1]
    try:
        parsed2 = json.loads(last_line)
        print("\nParsed JSON from last line:\n", json.dumps(parsed2, indent=2))
    except Exception:
        print("\nNo JSON found. Full model text returned. Use it for debugging.")
        print(text)
