from typing import Dict, List, Tuple
import re

BEHAVIOR_PATTERN = re.compile(r"\b\d+\s+app login", re.IGNORECASE)

# Very simple banned phrases for prototype
BANNED_PHRASES = [
    "guaranteed",
    "assured return",
    "assured returns",
    "no eligibility checks",
    "pre-approved for everyone",
    # Behavioral exposure we want to avoid
    "we noticed you've been",
    "we noticed your recent",
    "we've noticed your",
    "we saw your recent",
    "recent app login activity",
    "with 3 app logins",
    "with 2 declined transactions",
    "declined transaction",
    "declined transactions",
]

RATE_TOKENS = [
    "%", "percent", "interest rate", "rate of return", "roi",
]

def check_banned_phrases(text: str) -> List[str]:
    lower = text.lower()
    found = [b for b in BANNED_PHRASES if b in lower]

    if BEHAVIOR_PATTERN.search(lower):
        found.append("explicit_app_login_count")

    return found

def needs_rate_disclaimer(text: str) -> bool:
    """
    Check if the text mentions rates/percentages and therefore
    needs a disclaimer.
    """
    lower = text.lower()
    return any(tok in lower for tok in RATE_TOKENS)


def apply_compliance_rules(variant: Dict, default_disclaimers: List[str]) -> Tuple[Dict, bool]:
    """
    Apply simple compliance rules to a single variant.

    Returns:
        (updated_variant, is_rejected)

    - If banned phrases found -> mark as rejected.
    - If rates mentioned -> append disclaimers if not already present.
    """
    v = dict(variant)  # copy
    body = v.get("body", "") or ""
    disclaimers = v.get("disclaimers") or []

    # 1) Check for banned phrases
    banned = check_banned_phrases(body)
    if banned:
        v["compliant"] = False
        v["rejection_reason"] = f"banned_phrases: {', '.join(banned)}"
        return v, True  # rejected

    # 2) Add disclaimers when rates are mentioned
    if needs_rate_disclaimer(body):
        # ensure all default disclaimers are present
        existing = set(d.strip() for d in disclaimers)
        for d in default_disclaimers:
            if d.strip() not in existing:
                disclaimers.append(d)
        v["disclaimers"] = disclaimers

    v["compliant"] = True
    return v, False

