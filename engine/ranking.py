from typing import Dict, List, Tuple
import re


# Simple CTA regex
CTA_RE = re.compile(
    r"\b(apply|open|start|begin|view|check|recharge|invest|learn|discover|login|log in|explore)\b",
    re.IGNORECASE,
)


def score_variant(variant: Dict, customer: Dict, channel: str = "sms") -> float:
    """
    Compute a simple score for a variant based on:
    - CTA presence
    - Personalization (name, city)
    - Length (ideal range depends on channel)
    - Compliance (bonus for compliant, penalty for non-compliant)
    """
    body = (variant.get("body") or "").strip()
    subject = (variant.get("subject") or "").strip()

    score = 0.0

    # 1) CTA presence in body
    if CTA_RE.search(body):
        score += 2.0

    # 2) Personalization: first name / city present
    name = (customer.get("name") or "").split()[0].lower()
    city = (customer.get("city") or "").lower()

    body_lower = body.lower()
    if name and name in body_lower:
        score += 1.0
    if city and city in body_lower:
        score += 0.5

    # 3) Length heuristic
    length = len(body)
    if channel == "sms":
        # prefer around 120–200 chars
        ideal_min, ideal_max = 120, 200
    else:  # email / app
        # allow longer messages
        ideal_min, ideal_max = 200, 600

    if ideal_min <= length <= ideal_max:
        score += 1.5
    else:
        # small penalty for being too short or too long
        score -= 0.5

    # 4) Compliance
    if variant.get("compliant", True):
        score += 1.0
    else:
        score -= 3.0

    return score


def rank_variants(variants: List[Dict], customer: Dict, channel: str) -> List[Tuple[float, Dict]]:
    """
    Compute scores and return a list sorted by score descending.
    Each item: (score, variant_dict)
    """
    scored = []
    for v in variants:
        s = score_variant(v, customer, channel=channel)
        scored.append((s, v))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored
