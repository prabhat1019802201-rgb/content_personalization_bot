from typing import Dict, List
from engine.llm_client import call_llm
import sys

def _dbg(msg):
    print(f"[PROMPTING] {msg}", file=sys.stderr)

def _base_context(customer: dict, product: dict) -> str:
    return f"""
You are a senior marketing copywriter at Union Bank of India.

Customer Profile:
- Name: {customer.get('name')}
- Age: {customer.get('age')}
- City: {customer.get('city')}
- Lifecycle Stage: {customer.get('lifecycle_stage')}
- Risk Profile: {customer.get('risk_profile')}
- Avg Monthly Balance: ₹{customer.get('avg_monthly_balance')}

Product Context:
- Product Name: {product.get('name')}
- Category: {product.get('category')}
- Description: {product.get('description', 'N/A')}

Objective:
Create compliant, customer-centric banking communication
that is clear, professional, and persuasive.
""".strip()

# -----------------------------------------------------------------------------
# Curated channel instructions
# -----------------------------------------------------------------------------

def _channel_instructions(channel: str) -> str:
    if channel == "email":
        return """
Write a professional Indian bank marketing EMAIL.

Follow this structure:
1. Strong, benefit-driven subject line
2. Warm opening line referencing customer context
3. Clear value proposition of the product
4. 2–4 short bullet benefits
5. Clear CTA (visit branch / call / apply)
6. Polite, compliant closing

Tone:
- Trustworthy
- Professional
- Customer-centric
- No guarantees or exaggerated claims

Return format:
Subject: <subject line>

<body>
"""

    if channel == "whatsapp":
        return """
Write a WhatsApp marketing message.

Rules:
- 2–3 short lines
- Friendly but professional
- Clear CTA
- No emojis
- No guarantees

Example style:
Hello Mr. Kumar,
You may benefit from a Fixed Deposit offering better returns on surplus balance.
Call 1800-258-4588 to know more.
"""

    if channel == "banner":
        return """
Write ONE banner headline.

Rules:
- 5–8 words only
- Outcome-focused
- Aspirational
- No punctuation
- No emojis

Examples:
- Your Dream Home Is Within Reach
- Small Savings Big Future
- Secure Tomorrow Starting Today
"""

    return ""


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_prompts_per_channel(
    customer: dict,
    product: dict,
    use_llm: bool = True,
    n: int = 1,  # single output per channel
    model: str | None = None,
    enabled_channels: List[str] | None = None,
) -> Dict[str, List[dict]]:

    _dbg("build_prompts_per_channel called")
    _dbg(f"use_llm={use_llm}, model={model}")

    channels = enabled_channels or ["banner", "whatsapp", "email"]
    results = {}

    base_context = _base_context(customer, product)

    for channel in channels:
        instruction = _channel_instructions(channel)

        prompt = f"""
{base_context}

TASK:
{instruction}

Important rules:
- Bank: Union Bank of India
- RBI compliant language
- No guarantees or promises
- Clear and customer-friendly

Return only the final content.
"""

        _dbg(f"Calling LLM for channel={channel}")

        if use_llm:
            try:
                text = call_llm(prompt, model=model)
            except Exception as e:
                _dbg(f"LLM ERROR: {e}")
                text = ""
        else:
            text = f"{product.get('name')} designed for {customer.get('name')}."

        variant = {
            "variant_tag": "A",
            "body": text.strip(),
            "disclaimer": "T&C apply",
            "subject": None
        }

        # Extract subject if email
        if channel == "email":
            lines = text.splitlines()
            for ln in lines:
                if ln.lower().startswith("subject"):
                    variant["subject"] = ln.split(":", 1)[-1].strip()
                    variant["body"] = "\n".join(
                        l for l in lines if not l.lower().startswith("subject")
                    ).strip()
                    break

        results[channel] = [variant]

    _dbg("build_prompts_per_channel completed")
    return results
