import os
import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

# Where to save CSV files (../data relative to this script)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# How many rows to generate
NUM_CUSTOMERS = 10000
NUM_EVENTS = 200000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def generate_customers(n: int) -> pd.DataFrame:
    """
    Create synthetic customers with fields similar to the email spec.
    """
    cities = [
        "Hyderabad", "Hyderabad", "Hyderabad",  # bias towards Hyderabad
        "Mumbai", "Delhi", "Bengaluru", "Chennai",
        "Kolkata", "Pune", "Ahmedabad", "Visakhapatnam",
    ]
    languages = ["en", "hi", "te", "bn", "ta", "mr"]
    lifecycle_choices = ["onboarding", "active", "dormant", "winback"]
    risk_profiles = ["low", "medium", "high"]
    primary_channels = ["sms", "email", "app"]

    rows = []
    for i in range(1, n + 1):
        customer_id = f"C{i:05d}"
        name = f"Customer{i:05d}"

        age = int(np.clip(np.random.normal(38, 10), 18, 75))
        city = random.choice(cities)

        # More weight to English, then Hindi/Telugu etc.
        preferred_language = random.choices(
            languages, weights=[0.6, 0.15, 0.1, 0.05, 0.05, 0.05]
        )[0]

        # Tenure in months (0–20 years roughly)
        relationship_tenure_months = int(
            np.clip(np.random.exponential(24), 1, 240)
        )

        # Average monthly balance
        avg_monthly_balance = round(abs(np.random.normal(50000, 60000)), 2)

        risk_profile = random.choices(
            risk_profiles, weights=[0.6, 0.3, 0.1]
        )[0]

        lifecycle_stage = random.choices(
            lifecycle_choices, weights=[0.05, 0.7, 0.18, 0.07]
        )[0]

        primary_channel = random.choices(
            primary_channels, weights=[0.2, 0.3, 0.5]
        )[0]

        credit_card_holder = random.random() < 0.25

        loan_holder = random.choices(
            ["none", "personal", "auto", "home", "SME"],
            weights=[0.7, 0.1, 0.07, 0.08, 0.05],
        )[0]

        recent_issues = random.choices(
            ["none", "app_crash", "txn_decline", "duplicate_debit"],
            weights=[0.9, 0.05, 0.04, 0.01],
        )[0]

        rows.append(
            {
                "customer_id": customer_id,
                "name": name,
                "age": age,
                "city": city,
                "preferred_language": preferred_language,
                "relationship_tenure_months": relationship_tenure_months,
                "avg_monthly_balance": avg_monthly_balance,
                "risk_profile": risk_profile,
                "lifecycle_stage": lifecycle_stage,
                "primary_channel": primary_channel,
                "credit_card_holder": credit_card_holder,
                "loan_holder": loan_holder,
                "recent_issues": recent_issues,
            }
        )

    return pd.DataFrame(rows)


def generate_events(customers: pd.DataFrame, n_events: int, days: int = 90) -> pd.DataFrame:
    """
    Create synthetic events (logins, payments, transfers etc.) for last N days.
    """
    event_types = [
        "fund_transfer",
        "bill_pay",
        "app_login",
        "declined_txn",
        "card_payment",
        "fd_creation",
    ]

    end = datetime.utcnow()
    start = end - timedelta(days=days)

    rows = []
    cust_ids = customers["customer_id"].tolist()

    for _ in tqdm(range(n_events), desc="Generating events"):
        cid = random.choice(cust_ids)

        # Random timestamp in last N days
        ts = start + timedelta(
            seconds=random.randint(0, int((end - start).total_seconds()))
        )

        event_type = random.choices(
            event_types,
            weights=[0.2, 0.15, 0.4, 0.05, 0.18, 0.02],
        )[0]

        amount = None
        if event_type in ["fund_transfer", "bill_pay", "card_payment", "fd_creation"]:
            amount = round(abs(np.random.normal(2000, 8000)), 2)

        channel = random.choices(
            ["app", "web", "branch"], weights=[0.8, 0.15, 0.05]
        )[0]

        rows.append(
            {
                "customer_id": cid,
                "event_ts": ts.isoformat(),
                "event_type": event_type,
                "amount": amount,
                "channel": channel,
            }
        )

    return pd.DataFrame(rows)


def generate_products() -> pd.DataFrame:
    """
    Create a small catalog of products with basic eligibility and disclaimers.
    """
    product_specs = [
        ("Easy Savings", "deposits"),
        ("Max Credit Card", "payments"),
        ("Home Loan Plus", "loans"),
        ("Personal Loan Flex", "loans"),
        ("SME Working Cap", "loans"),
        ("Secure FD", "deposits"),
        ("Travel Insurance", "insurance"),
    ]

    rows = []
    for i, (name, category) in enumerate(product_specs, start=1):
        product_id = f"P{i:03d}"

        if category == "deposits":
            rules = {"min_balance": 5000, "risk": ["low", "medium", "high"]}
        elif category == "payments":
            rules = {"min_balance": 20000, "risk": ["low", "medium"]}
        elif category == "loans":
            rules = {"min_balance": 0, "risk": ["low", "medium"]}
        else:  # insurance
            rules = {"min_balance": 0, "risk": ["low", "medium", "high"]}

        benefit_bullets = [
            f"Benefit {j+1} for {name}" for j in range(3)
        ]

        disclaimers = [
            f"{name} is subject to eligibility and KYC.",
            "Rates subject to change. Terms & conditions apply.",
        ]

        cross_sell_targets = [
            "SEG_A_ACTIVE_VALUE_LOW_RISK",
            "SEG_B_DORMANT_WINBACK",
        ]

        rows.append(
            {
                "product_id": product_id,
                "name": name,
                "category": category,
                "eligibility_rules": json.dumps(rules),
                "benefit_bullets": json.dumps(benefit_bullets),
                "disclaimers": json.dumps(disclaimers),
                "cross_sell_targets": json.dumps(cross_sell_targets),
            }
        )

    return pd.DataFrame(rows)


def main():
    print(f"Saving data to: {DATA_DIR}")

    customers = generate_customers(NUM_CUSTOMERS)
    customers_path = os.path.join(DATA_DIR, "customers.csv")
    customers.to_csv(customers_path, index=False)
    print(f"customers.csv -> {customers_path} (rows: {len(customers)})")

    events = generate_events(customers, NUM_EVENTS)
    events_path = os.path.join(DATA_DIR, "events.csv")
    events.to_csv(events_path, index=False)
    print(f"events.csv -> {events_path} (rows: {len(events)})")

    products = generate_products()
    products_path = os.path.join(DATA_DIR, "products.csv")
    products.to_csv(products_path, index=False)
    print(f"products.csv -> {products_path} (rows: {len(products)})")

    print("Synthetic data generation complete.")


if __name__ == "__main__":
    main()


