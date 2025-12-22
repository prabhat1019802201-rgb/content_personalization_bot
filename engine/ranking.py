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

# engine/ranking.py

PRODUCT_RULES = {
    "P001": {  # Savings
        "min_balance": 0,
        "lifecycle": ["active", "onboarding"],
        "weight": 1.0,
    },
    "P002": {  # Fixed Deposit
        "min_balance": 50000,
        "lifecycle": ["active"],
        "weight": 2.0,
    },
    "P003": {  # Credit Card
        "requires": "credit_card_holder",
        "weight": 2.5,
    },
    "P004": {  # Personal Loan
        "max_risk": ["low", "medium"],
        "weight": 2.0,
    },
    "P005": {  # Home Loan
        "min_balance": 100000,
        "lifecycle": ["active"],
        "weight": 3.0,
    },
    "P006": {  # Insurance
        "lifecycle": ["active", "dormant"],
        "weight": 1.5,
    },
    "P007": {  # Travel Insurance
        "min_balance": 50000,
        "weight": 2.0,
    },
}

def rank_customers_for_product(product_id, customers_df):
    rules = PRODUCT_RULES.get(product_id, {})
    ranked = []

    for _, c in customers_df.iterrows():
        score = 0

        # Base lifecycle score
        if c.get("lifecycle_stage") == "active":
            score += 3
        elif c.get("lifecycle_stage") == "onboarding":
            score += 1

        # Product-specific rules
        bal = float(c.get("avg_monthly_balance", 0) or 0)

        if "min_balance" in rules and bal >= rules["min_balance"]:
            score += rules["weight"]

        if "lifecycle" in rules and c.get("lifecycle_stage") in rules["lifecycle"]:
            score += rules["weight"]

        if "requires" in rules and c.get(rules["requires"]) is True:
            score += rules["weight"]

        if "max_risk" in rules and c.get("risk_profile") in rules["max_risk"]:
            score += rules["weight"]

        ranked.append((score, c["customer_id"]))

    ranked.sort(key=lambda x: x[0])         # ascending by score
    return [cid for _, cid in ranked]

