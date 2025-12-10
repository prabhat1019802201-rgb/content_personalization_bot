from datetime import timedelta

import pandas as pd

from engine.data_loader import load_all


def compute_engagement_scores(customers: pd.DataFrame,
                              events: pd.DataFrame,
                              days_window: int = 30) -> pd.DataFrame:
    """
    For each customer, compute a simple engagement_score
    based on events in the last N days.
    """
    if events.empty:
        # no events -> zero engagement
        customers["engagement_score"] = 0.0
        return customers

    # Take only recent events (last N days from max timestamp)
    max_ts = events["event_ts"].max()
    window_start = max_ts - timedelta(days=days_window)
    recent = events[events["event_ts"] >= window_start].copy()

    # Define weights per event type
    weights = {
        "app_login": 1.0,
        "fund_transfer": 2.0,
        "bill_pay": 1.5,
        "card_payment": 1.5,
        "fd_creation": 3.0,
        "declined_txn": -1.0,
    }

    # Map event_type to a numeric weight (unknown types -> 0)
    recent["event_weight"] = recent["event_type"].map(weights).fillna(0.0)

    # Aggregate per customer_id
    engagement = (
        recent.groupby("customer_id")["event_weight"]
        .sum()
        .rename("engagement_score")
        .reset_index()
    )

    # Merge back to customers; missing => 0
    customers = customers.merge(engagement, on="customer_id", how="left")
    customers["engagement_score"] = customers["engagement_score"].fillna(0.0)

    return customers


def compute_value_scores(customers: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a simple value_score based on avg_monthly_balance and credit_card_holder.
    This is intentionally simple, just for demo.
    """
    # Normalize balance: higher balance -> higher base value
    # Divide by 100000 so typical values are around 0–2
    base_value = customers["avg_monthly_balance"] / 100000.0

    # Small boost if customer has a credit card
    cc_boost = customers["credit_card_holder"].astype(int) * 0.3

    customers["value_score"] = (base_value + cc_boost).round(3)
    return customers


def assign_segment(row: pd.Series) -> str:
    """
    Decide a segment label based on lifecycle, issues, risk, and scores.

    Some simple rules:
    - Any recent issue -> SEG_D_SERVICE_RECOVERY
    - Onboarding customers -> SEG_C_ONBOARDING_NEW_TO_BANK
    - Dormant/winback -> SEG_B_DORMANT_WINBACK
    - Active + high value + low risk -> SEG_A_ACTIVE_VALUE_LOW_RISK
    - Otherwise -> SEG_X_GENERAL
    """
    lifecycle = row["lifecycle_stage"]
    issues = row["recent_issues"]
    risk = row["risk_profile"]
    value_score = row.get("value_score", 0.0)

    if issues != "none":
        return "SEG_D_SERVICE_RECOVERY"

    if lifecycle == "onboarding":
        return "SEG_C_ONBOARDING_NEW_TO_BANK"

    if lifecycle in ("dormant", "winback"):
        return "SEG_B_DORMANT_WINBACK"

    if lifecycle == "active" and value_score > 1.0 and risk == "low":
        return "SEG_A_ACTIVE_VALUE_LOW_RISK"

    return "SEG_X_GENERAL"


def add_segments(customers: pd.DataFrame) -> pd.DataFrame:
    """
    Apply assign_segment to every row and return the DataFrame with a new 'segment' column.
    """
    customers = customers.copy()
    customers["segment"] = customers.apply(assign_segment, axis=1)
    return customers


def build_customer_feature_table() -> pd.DataFrame:
    """
    Full pipeline:
    - Load customers, events, products
    - Compute engagement and value scores
    - Assign segments

    Returns a customers DataFrame with new columns:
    - engagement_score
    - value_score
    - segment
    """
    customers, events, _ = load_all()

    customers = compute_engagement_scores(customers, events)
    customers = compute_value_scores(customers)
    customers = add_segments(customers)

    return customers


if __name__ == "__main__":
    # Quick manual test: show a few rows with features and segments
    df = build_customer_feature_table()
    print(df[["customer_id", "avg_monthly_balance",
              "engagement_score", "value_score",
              "lifecycle_stage", "recent_issues", "segment"]].head(10))
