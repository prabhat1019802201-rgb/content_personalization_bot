# engine/policy.py
"""
Policy & compliance helpers.

Expose:
    apply_policy_filter(variant: dict) -> (compliant: bool, reason: Optional[str], cleaned_variant: dict)
"""

import re
from typing import Dict, Tuple, Optional, List

# phrases that should cause outright rejection (non-compliant)
BANNED_PHRASES = {
    "guaranteed",
    "assured return",
    "assured returns",
    "no eligibility",
    "pre-approved for all",
    "pre approved",
    "pre-approved",
    "assured",
}

# phrases that reveal/point to customer behavior counts / sensitive nudges — we prefer to remove sentences with these
BEHAVIOR_PHRASES = [
    r"\bwe noticed\b",
    r"\bwe've noticed\b",
    r"\bwe have noticed\b",
    r"\byou've been\b",
    r"\byou have been\b",
    r"\brecent .* transactions\b",
    r"\b\d+ (?:transactions|logins|declines)\b",
    r"\bwe noticed your recent\b",
    r"\bwe noticed you\b",
]

# small helper to split into sentences (keeps punctuation)
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

# small phrase to require if rates or % present
RATE_DISCLAIMER = "Rates subject to change. T&C apply."

def _contains_banned(text: str) -> Optional[str]:
    low = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase in low:
            return phrase
    return None

def _strip_behavior_sentences(text: str) -> str:
    """
    Remove sentences that contain any banned behavior-mention phrases.
    Keeps other sentences.
    """
    if not text or not isinstance(text, str):
        return text or ""

    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    keep: List[str] = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        lowered = s_clean.lower()
        remove = False
        for patt in BEHAVIOR_PHRASES:
            if re.search(patt, lowered):
                remove = True
                break
        if not remove:
            keep.append(s_clean)
    # If nothing remains, return original but with behavior phrases neutralized (fallback)
    if not keep:
        # neutralize by replacing suspicious phrases with neutral tokens
        neutral = text
        for patt in BEHAVIOR_PHRASES:
            neutral = re.sub(patt, "", neutral, flags=re.IGNORECASE)
        return neutral.strip()
    return " ".join(keep)

def _needs_rate_disclaimer(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if "%" in t or "interest rate" in t or "rate of interest" in t:
        return True
    return False

def apply_policy_filter(variant: Dict) -> Tuple[bool, Optional[str], Dict]:
    """
    Apply simple compliance rules to a single variant.

    Args:
      variant: dict with keys like 'subject', 'body', 'disclaimers' (list)

    Returns:
      (compliant: bool, rejection_reason: Optional[str], cleaned_variant: dict)
    """
    v = dict(variant)  # shallow copy so we don't mutate caller's object
    subject = str(v.get("subject") or "")
    body = str(v.get("body") or "")
    disclaimers = list(v.get("disclaimers") or [])

    # 1) Ban check - reject if banned phrase present anywhere
    found = _contains_banned(subject + " " + body)
    if found:
        reason = f"banned_phrase:{found}"
        return False, reason, v

    # 2) Remove sentences that call out customer's recent behaviour counts/phrases
    cleaned_body = _strip_behavior_sentences(body)

    # If cleaning removed all content, fallback to a safe short version
    if not cleaned_body.strip():
        # try subject fallback or a generic nudge
        cleaned_body = (
            subject
            or "Explore this relevant offer tailored for you. T&C apply."
        )

    # 3) Rate handling - append disclaimer if % or rate mention present
    if _needs_rate_disclaimer(subject + " " + cleaned_body):
        if RATE_DISCLAIMER not in disclaimers:
            disclaimers.append(RATE_DISCLAIMER)

    # 4) Ensure no direct "you've been active" style fragments remain - final neutralization
    for patt in BEHAVIOR_PHRASES:
        cleaned_body = re.sub(patt, "", cleaned_body, flags=re.IGNORECASE)

    cleaned_body = re.sub(r'\s{2,}', ' ', cleaned_body).strip()

    # 5) Compose cleaned variant to return
    cleaned_variant = dict(v)
    cleaned_variant["subject"] = subject.strip()
    cleaned_variant["body"] = cleaned_body
    cleaned_variant["disclaimers"] = disclaimers

    # After cleaning, check again for banned phrases (edge cases)
    found2 = _contains_banned(cleaned_variant["subject"] + " " + cleaned_variant["body"])
    if found2:
        reason = f"banned_phrase_after_cleaning:{found2}"
        return False, reason, cleaned_variant

    # Passed all checks
    return True, None, cleaned_variant
