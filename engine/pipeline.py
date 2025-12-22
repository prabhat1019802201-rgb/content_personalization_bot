# engine/pipeline.py
"""
Stable GenAI personalization pipeline
- Analytics (Customer 360) when use_llm=False
- Marketing generation when use_llm=True
- Product override from user prompt
- Channel-aware generation (banner / whatsapp / email)
"""

import traceback
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from engine.data_loader import load_all
from engine.prompting import build_prompts_per_channel
from engine.ranking import rank_variants
from engine.policy import apply_policy_filter
from engine.creative import generate_creatives_for_variant
import engine.settings as settings


# =============================================================================
# Helpers
# =============================================================================

def compute_engagement(events_df, customer_id, days=30):
    cutoff = datetime.now() - timedelta(days=days)
    sub = events_df[events_df["customer_id"] == customer_id].copy()

    if not sub.empty:
        sub["event_ts"] = pd.to_datetime(sub["event_ts"], errors="coerce")
        sub = sub[sub["event_ts"] >= cutoff]

    score = 0.0
    for _, ev in sub.iterrows():
        et = str(ev.get("event_type", "")).lower()
        if et in ("fund_transfer", "card_payment"):
            score += 3
        elif et in ("bill_pay", "app_login"):
            score += 1
        elif et == "declined_txn":
            score += 0.5
    return round(score, 2)


def compute_value_score(cust_row):
    bal = float(cust_row.get("avg_monthly_balance") or 0)
    score = np.log1p(bal)
    if cust_row.get("credit_card_holder"):
        score *= 1.2
    return round(float(score), 2)


def infer_segment(cust_row, engagement):
    lifecycle = str(cust_row.get("lifecycle_stage") or "").lower()
    risk = str(cust_row.get("risk_profile") or "").lower()
    bal = float(cust_row.get("avg_monthly_balance") or 0)

    if lifecycle == "onboarding":
        return "ONBOARDING"
    if lifecycle == "dormant":
        return "WINBACK"
    if lifecycle == "active":
        if bal >= 100000 and risk in ("low", ""):
            return "HIGH_VALUE"
        return "STANDARD"
    return "UNCLASSIFIED"


def get_recent_events(events_df, customer_id, n=5):
    sub = events_df[events_df["customer_id"] == customer_id].copy()
    if sub.empty:
        return []
    sub["event_ts"] = pd.to_datetime(sub["event_ts"], errors="coerce")
    return (
        sub.sort_values("event_ts", ascending=False)
        .head(n)
        .to_dict(orient="records")
    )


# =============================================================================
# Product Recommendation
# =============================================================================

def recommend_product(products_df, cust_row):
    scored = []

    bal = float(cust_row.get("avg_monthly_balance") or 0)
    lifecycle = cust_row.get("lifecycle_stage", "")
    risk = cust_row.get("risk_profile", "")

    for _, r in products_df.iterrows():
        score = 0
        rules = r.get("eligibility_rules") or {}

        if isinstance(rules, str):
            try:
                import json
                rules = json.loads(rules)
            except Exception:
                rules = {}

        min_bal = float(rules.get("min_balance", 0) or 0)
        allowed_risk = rules.get("risk", [])

        if bal < min_bal:
            continue

        score += min(bal / 50000, 5)

        if lifecycle == "active":
            score += 2
        elif lifecycle == "dormant":
            score += 1

        if allowed_risk:
            score += 2 if risk in allowed_risk else -1

        category = (r.get("category") or "").lower()
        if category == "insurance" and bal > 50000:
            score += 2
        if category == "savings":
            score += 1

        scored.append((score, r.to_dict()))

    if not scored:
        return products_df.iloc[0].to_dict()

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def extract_product_from_prompt(user_text: str, products_df):
    if not user_text:
        return None

    text = user_text.lower()

    for _, r in products_df.iterrows():
        name = (r.get("name") or "").lower()
        category = (r.get("category") or "").lower()

        # split name into keywords
        name_tokens = name.split()

        # match if ANY keyword appears
        if any(tok in text for tok in name_tokens):
            return r.to_dict()

        # match category (insurance, loan, fd, savings)
        if category and category in text:
            return r.to_dict()

    return None

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def generate_for_customer(
    customer_id: str,
    user_request: Optional[str] = None,
    use_llm: bool = False,
    text_model_choice: Optional[str] = None,
    custom_model: Optional[str] = None,  # kept for UI compatibility
    image_model_choice: Optional[str] = None,
    creative_kit: Optional[dict] = None,
    creative_width: int = 1200,
    creative_height: int = 400,
    creative_steps: int = 20,
) -> Dict[str, Any]:

    try:
        print(f"🔥 generate_for_customer | use_llm={use_llm}")

        customers_df, events_df, products_df = load_all()

        customer_id = str(customer_id).strip()
        customers_df["customer_id"] = customers_df["customer_id"].astype(str).str.strip()

        if customer_id not in set(customers_df["customer_id"]):
            raise ValueError(f"Customer {customer_id} not found")

        cust_row = customers_df.loc[
            customers_df["customer_id"] == customer_id
        ].iloc[0].to_dict()

        engagement = compute_engagement(events_df, customer_id)
        value_score = compute_value_score(cust_row)
        segment = infer_segment(cust_row, engagement)
        cust_row["segment"] = segment

        metrics = {
            "engagement_score": engagement,
            "value_score": value_score,
        }

        recent_events = get_recent_events(events_df, customer_id)

        # -------------------------------------------------
        # Product selection
        # -------------------------------------------------
        recommended_products = recommend_products(products_df, cust_row, top_k=3)

        product = recommended_products[0] if recommended_products else recommend_product(
          products_df, cust_row
     )

        if user_request:
            print("🧠 USER PROMPT:", user_request)
            override = extract_product_from_prompt(user_request, products_df)
            if override:
                print(f"🔁 Product overridden by prompt → {override.get('name')}")
                product = override

        # -------------------------------------------------
        # ANALYTICS ONLY
        # -------------------------------------------------
        if not use_llm:
            return {
                "customer": cust_row,
                "product": product,
                "recommended_products": recommended_products,
                "recent_events": recent_events,
                "metrics": metrics,
            }

        # -------------------------------------------------
        # MARKETING GENERATION
        # -------------------------------------------------
        creative_kit = creative_kit or {}
        enabled_channels = [
            ch for ch, enabled in creative_kit.items() if enabled
        ]

        per_channel_raw = build_prompts_per_channel(
            cust_row,
            product,
            use_llm=True,
            n=3,
            model=text_model_choice,
        )

        variants_scored = {}
        selected = {}

        for channel, variants in per_channel_raw.items():
            if channel not in enabled_channels:
                continue

            processed = []
            for v in variants:
                compliant, _, cleaned = apply_policy_filter(v)
                cleaned = cleaned if cleaned and cleaned.get("body") else v
                processed.append({"variant": cleaned, "compliant": True})

            try:
                ranked = rank_variants(processed, cust_row, channel)
            except TypeError:
                ranked = rank_variants(processed)

            chosen = None
            for item in ranked:
                v = item.get("variant") if isinstance(item, dict) else None
                if v and v.get("body"):
                    chosen = v
                    break

            if not chosen and variants:
                chosen = variants[0]

            selected[channel] = chosen
            variants_scored[channel] = ranked

        # -------------------------------------------------
        # IMAGE GENERATION (BANNER ONLY)
        # -------------------------------------------------
        creative_result = None
        if (
            creative_kit.get("banner")
            and image_model_choice
            and image_model_choice != "No image"
            and selected.get("banner")
        ):
            mapped = settings.map_image_choice(image_model_choice)
            creative_result = generate_creatives_for_variant(
                campaign_id=f"cust_{customer_id}",
                variant_tag="A",
                product_key=product.get("category", "generic"),
                variant_brief=selected["banner"]["body"][:200],
                headline=selected["banner"].get("subject") or product.get("name"),
                subtitle=selected["banner"]["body"][:120],
                cta_text="Know More",
                n_images=1,
                customer=cust_row,            
                product=product,   
                output_sizes=[(creative_width, creative_height)],
                image_model_choice=mapped.get("backend"),
                steps=creative_steps,
                device=mapped.get("device"),
            )

        return {
                   "customer": cust_row,
                   "product": product,
                   "recommended_products": recommended_products,
                   "recent_events": recent_events,
                   "metrics": metrics,
                   "variants_scored": variants_scored,
                   "selected": selected,
                   "creative_result": creative_result,
        }

    except Exception:
        traceback.print_exc()
        raise


# =============================================================================
# Intent Detection
# =============================================================================

def detect_intent(user_text: str) -> str:
    t = user_text.lower()

    if any(k in t for k in [
        "generate", "marketing", "campaign", "promotion", "message"
    ]):
        return "MARKETING_GEN"

    if any(k in t for k in [
        "tell me", "about", "profile", "details"
    ]):
        return "CUSTOMER_INFO"

    # 🔥 NEW — PRODUCT ADVISORY
    if any(k in t for k in [
        "what product", "which product", "suggest product",
        "can be sold", "recommend product", "offer to this user"
    ]):
        return "PRODUCT_ADVISORY"

    return "GENERIC_CHAT"

## recommended products based on simple rules
def recommend_products(products_df, cust_row, top_k: int = 3):
    scored = []

    bal = float(cust_row.get("avg_monthly_balance") or 0)
    lifecycle = cust_row.get("lifecycle_stage", "")
    risk = cust_row.get("risk_profile", "")

    for _, r in products_df.iterrows():
        score = 0
        rules = r.get("eligibility_rules")

        # Safe rule parsing
        if isinstance(rules, str):
            try:
                import json
                rules = json.loads(rules)
            except Exception:
                rules = {}
        if not isinstance(rules, dict):
            rules = {}

        min_bal = float(rules.get("min_balance", 0) or 0)
        allowed_risk = rules.get("risk", [])

        # Basic eligibility
        if bal < min_bal:
            continue

        # Scoring
        score += min(bal / 50000, 5)

        if lifecycle == "active":
            score += 2
        elif lifecycle == "dormant":
            score += 1

        if allowed_risk:
            score += 2 if risk in allowed_risk else -1

        category = (r.get("category") or "").lower()
        if category == "savings":
            score += 1
        elif category in ("loan", "credit"):
            score += 2 if bal > 75000 else 0
        elif category == "insurance":
            score += 1 if bal > 50000 else 0

        scored.append((score, r.to_dict()))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [p for _, p in scored[:top_k]]

def recommend_top_products(products_df, cust_row, top_k=3):
    scored = []

    bal = float(cust_row.get("avg_monthly_balance") or 0)
    lifecycle = cust_row.get("lifecycle_stage", "")
    risk = cust_row.get("risk_profile", "")

    for _, r in products_df.iterrows():
        score = 0
        rules = r.get("eligibility_rules", {})

        # Parse rules safely
        if isinstance(rules, str):
            try:
                import json
                rules = json.loads(rules)
            except Exception:
                rules = {}

        min_bal = float(rules.get("min_balance", 0) or 0)
        allowed_risk = rules.get("risk", [])

        if bal < min_bal:
            continue

        # ---------- SCORING ----------
        score += min(bal / 50000, 5)

        if lifecycle == "active":
            score += 2
        if lifecycle == "winback":
            score += 1

        if allowed_risk:
            score += 2 if risk in allowed_risk else -1

        category = (r.get("category") or "").lower()
        if category == "savings":
            score += 1
        elif category in ("loan", "credit"):
            score += 2 if bal > 75000 else 0
        elif category == "insurance":
            score += 1 if bal > 50000 else 0

        scored.append((score, r.to_dict()))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]
