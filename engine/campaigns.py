from typing import List, Dict
import json

import pandas as pd

from engine.data_loader import load_all
from engine.features_segmenter import build_customer_feature_table


# Simple segment priority: lower number = higher priority
SEGMENT_PRIORITY = {
    "SEG_D_SERVICE_RECOVERY": 0,
    "SEG_B_DORMANT_WINBACK": 1,
    "SEG_A_ACTIVE_VALUE_LOW_RISK": 2,
    "SEG_C_ONBOARDING_NEW_TO_BANK": 3,
    "SEG_X_GENERAL": 4,
}


def _is_customer_eligible_for_product(cust_row: pd.Series, prod_row: pd.Series) -> bool:
    """
    Reuse the same JSON eligibility rules logic used when choosing a product per customer.
    """
    rules = json.loads(prod_row["eligibility_rules"])
    min_balance = rules.get("min_balance", 0)
    allowed_risks = rules.get("risk", [])

    balance = float(cust_row.get("avg_monthly_balance", 0.0))
    risk = cust_row.get("risk_profile")

    if balance < min_balance:
        return False

    if allowed_risks and risk not in allowed_risks:
        return False

    return True


def _score_customer_for_product(cust_row: pd.Series) -> float:
    """
    A simple scoring formula for ranking customers for a campaign:
    - Segment priority (service recovery / winback / high value)
    - Value score (how profitable)
    - Engagement score (how active)
    """
    seg = cust_row.get("segment", "SEG_X_GENERAL")
    seg_prio = SEGMENT_PRIORITY.get(seg, 5)

    value_score = float(cust_row.get("value_score", 0.0))
    engagement_score = float(cust_row.get("engagement_score", 0.0))

    # Lower segment priority is better, so subtract it
    score = 0.0
    score += (5 - seg_prio) * 2.0  # strong weight on segment
    score += value_score * 3.0      # weight on value
    score += engagement_score * 1.0 # weight on engagement

    return score


def get_top_customers_for_product(
    product_id: str,
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, object]:
    """
    For a given product_id, return a page of the best customers:

    - Filter by eligibility_rules.
    - Score customers for this product.
    - Sort descending by score.
    - Apply offset + limit.

    Returns:
        {
            "product": {...},
            "total_eligible": int,
            "customers": [ {customer fields + score}, ... ]
        }
    """
    customers, events, products = load_all()
    # Ensure features/segments are present
    features = build_customer_feature_table()

    prod_row = products[products["product_id"] == product_id]
    if prod_row.empty:
        raise ValueError(f"Product {product_id} not found")

    product = prod_row.iloc[0]

    # Filter eligible customers
    eligible_mask = features.apply(
        lambda row: _is_customer_eligible_for_product(row, product),
        axis=1,
    )
    eligible_customers = features[eligible_mask].copy()

    if eligible_customers.empty:
        return {
            "product": product.to_dict(),
            "total_eligible": 0,
            "customers": [],
        }

    # Score them
    eligible_customers["campaign_score"] = eligible_customers.apply(
        _score_customer_for_product, axis=1
    )

    # Sort by score
    eligible_customers = eligible_customers.sort_values(
        "campaign_score", ascending=False
    )

    total_eligible = len(eligible_customers)

    # Apply pagination
    page = eligible_customers.iloc[offset : offset + limit].copy()

    # Convert to simple dicts for UI / API
    customers_out: List[Dict] = []
    for _, row in page.iterrows():
        customers_out.append(
            {
                "customer_id": row["customer_id"],
                "name": row["name"],
                "city": row["city"],
                "age": int(row.get("age", 0)),
                "segment": row.get("segment"),
                "lifecycle_stage": row.get("lifecycle_stage"),
                "risk_profile": row.get("risk_profile"),
                "avg_monthly_balance": float(row.get("avg_monthly_balance", 0.0)),
                "engagement_score": float(row.get("engagement_score", 0.0)),
                "value_score": float(row.get("value_score", 0.0)),
                "campaign_score": float(row.get("campaign_score", 0.0)),
            }
        )

    return {
        "product": product.to_dict(),
        "total_eligible": total_eligible,
        "customers": customers_out,
    }


if __name__ == "__main__":
    # Simple manual test
    out = get_top_customers_for_product("P001", limit=20, offset=0)
    print("Product:", out["product"]["name"])
    print("Total eligible:", out["total_eligible"])
    print("Top customers:")
    for c in out["customers"][:5]:
        print(" -", c["customer_id"], c["name"], "| score:", c["campaign_score"])
