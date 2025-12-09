import json
from typing import Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape
import pandas as pd

from engine.data_loader import load_all
from engine.features_segmenter import build_customer_feature_table

# Setup Jinja2 environment to load templates from ../templates
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")

env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(enabled_extensions=("j2",)),
    trim_blocks=True,
    lstrip_blocks=True,
)

SYSTEM_PROMPT_PATH = os.path.join(TEMPLATES_DIR, "prompt_system.txt")


def load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def summarize_recent_events(customer_id: str, events: pd.DataFrame, limit: int = 5) -> str:
    """
    Take the last few events for a customer and turn them into a short summary string.
    """
    ev = events[events["customer_id"] == customer_id].sort_values("event_ts", ascending=False).head(limit)
    if ev.empty:
        return "no recent activity"

    # Example summary: "3x app_login, 1x fund_transfer, 1x bill_pay"
    counts = ev["event_type"].value_counts()
    parts = [f"{cnt}x {etype}" for etype, cnt in counts.items()]
    return ", ".join(parts)


def choose_product_for_customer(customer_row: pd.Series, products: pd.DataFrame) -> Dict:
    """
    Very simple product selection:
    - Filter by risk profile if present in eligibility_rules.
    - Just pick the first matching product for now.
    """
    risk = customer_row["risk_profile"]
    balance = customer_row["avg_monthly_balance"]

    def is_eligible(prod_row) -> bool:
        rules = json.loads(prod_row["eligibility_rules"])
        min_balance = rules.get("min_balance", 0)
        allowed_risks = rules.get("risk", [])
        return (balance >= min_balance) and (risk in allowed_risks)

    eligible = products[products.apply(is_eligible, axis=1)]
    if eligible.empty:
        # fallback: just pick the first product
        return products.iloc[0].to_dict()
    return eligible.iloc[0].to_dict()


def build_prompts_for_customer(customer_id: str):
    """
    High-level helper:
    - Load data and feature/segment table
    - Find customer row
    - Summarize recent events
    - Pick a product
    - Build system + user prompt strings
    """
    customers, events, products = load_all()
    features = build_customer_feature_table()

    row = features[features["customer_id"] == customer_id]
    if row.empty:
        raise ValueError(f"Customer {customer_id} not found")

    cust = row.iloc[0]
    recent_summary = summarize_recent_events(customer_id, events)
    product = choose_product_for_customer(cust, products)

    # Prepare values for template
    user_template = env.get_template("prompt_user.j2")
    benefits = json.loads(product["benefit_bullets"])
    disclaimers = json.loads(product["disclaimers"])

    user_prompt = user_template.render(
        name=cust["name"],
        preferred_language=cust["preferred_language"],
        city=cust["city"],
        segment=cust["segment"],
        primary_channel=cust["primary_channel"],
        recent_events=recent_summary,
        product_name=product["name"],
        product_category=product["category"],
        benefits=benefits,
        disclaimers=disclaimers,
    )

    system_prompt = load_system_prompt()

    return system_prompt, user_prompt, cust.to_dict(), product


if __name__ == "__main__":
    # Quick test
    sp, up, cust, prod = build_prompts_for_customer("C00001")
    print("=== SYSTEM PROMPT ===")
    print(sp[:400], "...\n")
    print("=== USER PROMPT ===")
    print(up)
    print("\nCustomer segment:", cust["segment"])
    print("Product chosen :", prod["name"])
