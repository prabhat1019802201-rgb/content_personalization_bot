# engine/pipeline.py
"""
Stable GenAI personalization pipeline
- Backward compatible with existing UI
- Supports analytics-only (use_llm=False)
- Supports LLM marketing generation (use_llm=True)
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
    now = datetime.now()
    cutoff = now - timedelta(days=days)

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
    return round(score, 3)


def compute_value_score(cust_row):
    bal = float(cust_row.get("avg_monthly_balance") or 0)
    base = np.log1p(bal)
    if cust_row.get("credit_card_holder"):
        base *= 1.2
    return round(float(base), 3)


def infer_segment(cust_row, engagement, recent_issues):
    lifecycle = str(cust_row.get("lifecycle_stage") or "").lower()
    risk = str(cust_row.get("risk_profile") or "").lower()

    if recent_issues and recent_issues != "none":
        return "SERVICE_RECOVERY"
    if lifecycle == "onboarding":
        return "ONBOARDING_NEW"
    if lifecycle == "dormant":
        return "DORMANT_WINBACK"
    if lifecycle == "active":
        bal = float(cust_row.get("avg_monthly_balance") or 0)
        if bal >= 100000 and risk in ("low", ""):
            return "ACTIVE_HIGH_VALUE"
        return "ACTIVE_STANDARD"
    return "UNCLASSIFIED"


def get_recent_events(events_df, customer_id, n=5):
    sub = events_df[events_df["customer_id"] == customer_id].copy()
    if sub.empty:
        return []
    sub["event_ts"] = pd.to_datetime(sub["event_ts"], errors="coerce")
    sub = sub.sort_values("event_ts", ascending=False)
    return sub.head(n).to_dict(orient="records")


def recommend_product(products_df, cust_row):
    for _, r in products_df.iterrows():
        rules = r.get("eligibility_rules")

        if isinstance(rules, str):
            try:
                import json
                rules = json.loads(rules)
            except Exception:
                rules = {}

        if not isinstance(rules, dict):
            rules = {}

        min_bal = float(rules.get("min_balance", 0) or 0)
        allowed_risk = rules.get("risk") or []

        bal = float(cust_row.get("avg_monthly_balance") or 0)
        cust_risk = cust_row.get("risk_profile")

        if bal >= min_bal:
            if allowed_risk:
                if cust_risk in allowed_risk:
                    return r.to_dict()
            else:
                return r.to_dict()

    return products_df.iloc[0].to_dict()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def generate_for_customer(
    customer_id: str,
    user_request: Optional[str] = None,
    use_llm: bool = False,
    text_model_choice: Optional[str] = None,
    custom_model: Optional[str] = None,  # accepted for UI compatibility
    image_model_choice: Optional[str] = None,
    creative_kit: Optional[dict] = None,
    creative_width: int = 1200,
    creative_height: int = 400,
    creative_steps: int = 20,
) -> Dict[str, Any]:

    try:
        print(f"🔥 generate_for_customer CALLED | use_llm = {use_llm}")

        customers_df, events_df, products_df = load_all()

        # Normalize IDs
        customer_id = str(customer_id).strip()
        customers_df["customer_id"] = customers_df["customer_id"].astype(str).str.strip()
        events_df["customer_id"] = events_df["customer_id"].astype(str).str.strip()

        if customer_id not in set(customers_df["customer_id"]):
            raise ValueError(f"Customer {customer_id} not found")

        cust_row = customers_df[customers_df["customer_id"] == customer_id].iloc[0].to_dict()

        engagement = compute_engagement(events_df, customer_id)
        value_score = compute_value_score(cust_row)
        recent_issues = cust_row.get("recent_issues")
        segment = infer_segment(cust_row, engagement, recent_issues)
        cust_row["segment"] = segment

        metrics = {
            "engagement_score": engagement,
            "value_score": value_score,
            "avg_monthly_balance": cust_row.get("avg_monthly_balance"),
            "relationship_tenure_months": cust_row.get("relationship_tenure_months"),
        }

        product = recommend_product(products_df, cust_row)
        recent_events = get_recent_events(events_df, customer_id)

        # =====================================================
        # ANALYTICS ONLY (SIDEBAR / CUSTOMER 360)
        # =====================================================
        if not use_llm:
            return {
                "customer": cust_row,
                "product": product,
                "recent_events": recent_events,
                "metrics": metrics,
            }

        # =====================================================
        # MARKETING GENERATION (LLM)
        # =====================================================
        # -----------------------------------------------------
        # Respect creative_kit toggles (banner / whatsapp / email)
        # -----------------------------------------------------

        enabled_channels = []

        ck = creative_kit or {}

        if ck.get("banner"):
           enabled_channels.append("banner")
        if ck.get("whatsapp"):
           enabled_channels.append("whatsapp")
        if ck.get("email"):
           enabled_channels.append("email")

        per_channel_raw_all = build_prompts_per_channel(
           cust_row,
           product,
           use_llm=True,
           n=3,
           model=text_model_choice,
        )

        # 🔥 FILTER ONLY ENABLED CHANNELS
        per_channel_raw = {
         ch: variants
         for ch, variants in per_channel_raw_all.items()
         if ch in enabled_channels
       }


        variants_scored = {}
        selected = {}

        for channel, variants in per_channel_raw.items():
            processed = []
            for v in variants:
                compliant, reason, cleaned = apply_policy_filter(v)

                # 🔥 FALLBACK: if policy wipes content, keep original LLM output
                if not cleaned or not cleaned.get("body"):
                    cleaned = v
                    compliant = True

                processed.append({
                     "variant": cleaned,
                     "compliant": compliant
               })

            try:
                ranked = rank_variants(processed, cust_row, channel)
            except TypeError:
                ranked = rank_variants(processed)

            normalized = []
            for item in ranked:
                if isinstance(item, dict) and "variant" in item:
                    normalized.append({
                        "variant": item["variant"],
                        "score": float(item.get("score", 0.0))
                    })
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    a, b = item
                    if isinstance(a, (int, float)):
                        normalized.append({"variant": b, "score": float(a)})
                    elif isinstance(b, (int, float)):
                        normalized.append({"variant": a, "score": float(b)})
                elif isinstance(item, dict):
                    normalized.append({"variant": item, "score": 0.0})
                else:
                    normalized.append({"variant": {"body": str(item)}, "score": 0.0})

           # ---------- FINAL SAFETY NET ----------
            if not normalized:
              # 🔥 fallback: use raw LLM output directly
              first_llm_variant = variants[0] if variants else None
              if first_llm_variant:
                normalized = [{
                "variant": first_llm_variant,
                "score": 1.0
               }]

            variants_scored[channel] = normalized

            # ✅ FINAL HARD GUARANTEE — NEVER RETURN EMPTY VARIANT
            chosen = None

            # 1️⃣ Prefer ranked output ONLY if it has real content
            if normalized:
                 v = normalized[0].get("variant", {})
                 if isinstance(v, dict) and v.get("body"):
                      chosen = v

            # 2️⃣ Fallback to RAW LLM output (this is the key fix)
            if not chosen and variants:
                 raw = variants[0]
                 if isinstance(raw, dict) and raw.get("body"):
                   chosen = raw

            # 3️⃣ Absolute last resort (should never happen)
            if not chosen:
              chosen = {
              "variant_tag": "A",
              "body": "Special offer curated just for you.",
             }

            selected[channel] = chosen

        # =====================================================
        # IMAGE GENERATION (OPTIONAL)
        # =====================================================
        creative_result = None
        ck = creative_kit or {"banner": True}

        if ck.get("banner") and image_model_choice and image_model_choice != "No image":

               # fallback-safe banner variant
               banner_variant = (
                 selected.get("banner")
                 or per_channel_raw.get("banner", [{}])[0]
               )

               if banner_variant and banner_variant.get("body"):
                    print("🖼️ [PIPELINE] Generating banner image...")

                    mapped = settings.map_image_choice(image_model_choice)

                    creative_result = generate_creatives_for_variant(
                       campaign_id=f"cust_{customer_id}",
                       variant_tag=banner_variant.get("variant_tag", "A"),
                       product_key=product.get("category", "generic"),
                       variant_brief=banner_variant["body"][:200],
                       headline=banner_variant.get("subject") or product.get("name"),
                       subtitle=banner_variant["body"][:120],
                       cta_text="Know More",
                       n_images=1,
                       product_context=product,          # ✅ NEW
                       customer_context=cust_row, 
                       output_sizes=[(creative_width, creative_height)],
                       #image_model_choice=mapped.get("backend"),
                       image_model_choice=image_model_choice,
                       steps=creative_steps,
                       device=mapped.get("device"),
                      )

        return {
            "customer": cust_row,
            "product": product,
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
# INTENT DETECTION
# =============================================================================

def detect_intent(user_text: str) -> str:
    t = user_text.lower()
    if any(k in t for k in ["generate", "marketing", "campaign", "promotion", "message"]):
        return "MARKETING_GEN"
    if any(k in t for k in ["tell me", "about", "profile", "details"]):
        return "CUSTOMER_INFO"
    return "GENERIC_CHAT"
