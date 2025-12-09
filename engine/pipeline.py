from typing import Dict, Any, List, Tuple
import json

from engine.prompting import build_prompts_for_customer
from engine.llm_client import generate_variants
from engine.policy import apply_compliance_rules
from engine.ranking import rank_variants


def _build_dummy_variants(customer: Dict, product: Dict) -> List[Dict]:
    """
    Fallback variants when LLM is not available.
    Very simple templates just to keep the pipeline working.
    """
    first_name = (customer.get("name") or "").split()[0]
    city = customer.get("city") or ""
    product_name = product.get("name") or "our product"

    base = (
        f"Hi {first_name}, we have a {product_name} offer for you in {city}. "
        f"Check details in the Union Bank app."
    )

    return [
        {
            "variant_tag": "A",
            "subject": f"{product_name} just for you",
            "body": base + " Apply now to get started.",
            "disclaimers": [
                "Subject to eligibility and KYC.",
                "Rates subject to change. T&C apply.",
            ],
        },
        {
            "variant_tag": "B",
            "subject": f"{product_name} | Quick access",
            "body": base + " View more details in your app dashboard.",
            "disclaimers": [
                "Subject to eligibility and KYC.",
            ],
        },
        {
            "variant_tag": "C",
            "subject": f"Explore {product_name} benefits",
            "body": base + " Explore benefits and decide at your convenience.",
            "disclaimers": [
                "Subject to eligibility and KYC.",
            ],
        },
    ]


def generate_for_customer(customer_id: str, use_llm: bool = True) -> Dict[str, Any]:
    """
    Main pipeline:

    - Build prompts for a given customer.
    - Try to generate A/B/C variants using LLM (Ollama).
      If LLM fails or use_llm=False, fall back to dummy variants.
    - Apply compliance rules.
    - Rank variants and pick the best.
    - Return a structured result for the UI.
    """
    system_prompt, user_prompt, customer, product = build_prompts_for_customer(customer_id)

    # 1) Generate variants (LLM or fallback)
    if use_llm:
        try:
            raw_variants = generate_variants(system_prompt, user_prompt)
        except Exception as e:
            # Log / print for debugging; for now just fallback
            print(f"[WARN] LLM generation failed for {customer_id}: {e}")
            raw_variants = _build_dummy_variants(customer, product)
    else:
        raw_variants = _build_dummy_variants(customer, product)

    # 2) Compliance filtering
    default_disclaimers = json.loads(product["disclaimers"])
    processed_variants = []
    for v in raw_variants:
        v_clean, rejected = apply_compliance_rules(v, default_disclaimers)
        v_clean["rejected_by_policy"] = rejected
        processed_variants.append(v_clean)

    # 3) Ranking
    channel = customer.get("primary_channel", "sms")
    scored = rank_variants(processed_variants, customer, channel=channel)

    # 4) Selected variant (top score, but prefer compliant)
    # scored is sorted descending; we pick first compliant if possible
    selected_variant = None
    for s, v in scored:
        if v.get("compliant", True) and not v.get("rejected_by_policy", False):
            selected_variant = v
            break
    if selected_variant is None and scored:
        # fallback to top-scoring even if non-compliant (for debug)
        selected_variant = scored[0][1]

    result = {
        "customer": {
            "customer_id": customer["customer_id"],
            "name": customer["name"],
            "city": customer["city"],
            "segment": customer["segment"],
            "preferred_language": customer["preferred_language"],
            "primary_channel": channel,
        },
        "product": {
            "product_id": product["product_id"],
            "name": product["name"],
            "category": product["category"],
        },
        "variants_scored": [
            {
                "score": s,
                "variant": v,
            }
            for s, v in scored
        ],
        "selected": selected_variant,
    }

    return result


if __name__ == "__main__":
    # Quick manual test without LLM
    out = generate_for_customer("C00001", use_llm=False)
    print("Customer:", out["customer"])
    print("Product:", out["product"])
    print("Selected variant tag:", out["selected"].get("variant_tag"))
    print("Selected body:", out["selected"].get("body"))
