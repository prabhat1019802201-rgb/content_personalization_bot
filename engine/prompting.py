# engine/prompting.py
"""
Prompt construction + LLM invocation.
Generates best-of-3 variants per channel.
"""

from typing import Dict, List
from engine.llm_client import call_llm
import sys

def _dbg(msg):
    print(f"[PROMPTING] {msg}", file=sys.stderr)
# -----------------------------------------------------------------------------
# Prompt builders
# -----------------------------------------------------------------------------

def _base_context(customer: dict, product: dict) -> str:
    return f"""
You are a senior bank marketing copywriter.

Customer profile:
- Name: {customer.get('name')}
- Age: {customer.get('age')}
- City: {customer.get('city')}
- Lifecycle stage: {customer.get('lifecycle_stage')}
- Risk profile: {customer.get('risk_profile')}
- Avg monthly balance: {customer.get('avg_monthly_balance')}

Product:
- Name: {product.get('name')}
- Category: {product.get('category')}
- Description: {product.get('description', '')}

Tone:
- Professional
- Trustworthy
- Simple banking language
""".strip()


def _channel_prompt(channel: str) -> str:
    if channel == "banner":
        return """
Generate 3 DIFFERENT banner headlines.
Each headline must be max 8 words.
No emojis.
Return as:
A: ...
B: ...
C: ...
""".strip()

    if channel == "whatsapp":
        return """
Generate 3 DIFFERENT WhatsApp marketing messages.
Each message:
- Friendly
- 2–3 short lines
- Includes CTA
Return as:
A: ...
B: ...
C: ...
""".strip()

    if channel == "email":
        return """
Generate 3 DIFFERENT marketing emails.
Each must include:
- Subject line
- Body (short paragraph)
Return as:
A:
Subject: ...
Body: ...

B:
Subject: ...
Body: ...

C:
Subject: ...
Body: ...
""".strip()

    return ""


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_prompts_per_channel(
    customer: dict,
    product: dict,
    use_llm: bool = True,
    n: int = 3,
    model: str | None = None,
):
    """
    Returns:
    {
      "banner": [variant_dict, ...],
      "whatsapp": [...],
      "email": [...]
    }
    """

    _dbg("build_prompts_per_channel called")
    _dbg(f"use_llm={use_llm}, model={model}")

    channels = ["banner", "whatsapp", "email"]
    results = {}

    customer_name = customer.get("name", "Customer")
    product_name = product.get("name", "our product")

    for channel in channels:
        variants = []

        for i in range(n):
            variant_tag = chr(ord("A") + i)

            prompt = f"""
You are a banking marketing expert.

Create a {channel.upper()} marketing message.

Customer:
- Name: {customer_name}
- Segment: {customer.get("segment")}
- City: {customer.get("city")}

Product:
- Name: {product_name}
- Category: {product.get("category")}

Rules:
- Tone: professional, friendly
- Bank: Union Bank of India
- Avoid emojis
- Short and clear

Return only the message text.
"""

            _dbg(f"Calling LLM for {channel} variant {variant_tag}")

            if use_llm:
                try:
                    from engine.llm_client import call_llm
                    text = call_llm(prompt, model=model)
                except Exception as e:
                    _dbg(f"LLM ERROR: {e}")
                    text = ""
            else:
                text = f"{product_name} designed for {customer_name}."

            variants.append({
                "variant_tag": variant_tag,
                "subject": f"{product_name} from Union Bank" if channel == "email" else None,
                "body": text.strip(),
                "disclaimer": "T&C apply"
            })

        results[channel] = variants

    _dbg("build_prompts_per_channel completed")
    return results

# -----------------------------------------------------------------------------
# Output parser
# -----------------------------------------------------------------------------

def _parse_variants(channel: str, text: str) -> List[dict]:
    """
    Parses A/B/C style output safely.
    """
    variants = []
    blocks = [b.strip() for b in text.split("\n") if b.strip()]

    current = None
    buf = []

    def flush(tag, lines):
        body = "\n".join(lines).strip()
        subject = None

        if channel == "email":
            for ln in lines:
                if ln.lower().startswith("subject"):
                    subject = ln.split(":", 1)[-1].strip()

        variants.append({
            "variant_tag": tag,
            "subject": subject,
            "body": body
        })

    for line in blocks:
        if line.startswith(("A:", "B:", "C:")):
            if current:
                flush(current, buf)
            current = line[0]
            buf = [line[2:].strip()]
        else:
            buf.append(line)

    if current:
        flush(current, buf)

    return variants
