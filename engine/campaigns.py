# -----------------------------------------
# engine/campaigns.py (FULL UPDATED VERSION)
# -----------------------------------------

import pandas as pd
from engine.data_loader import load_all
from engine.pipeline import compute_engagement, compute_value_score
from engine.data_loader import load_customers, load_products
from engine.ranking import rank_customers_for_product


def resolve_product_id(products_df, name_or_id):
    """Return product_id for either a name or an ID (defensive)."""
    name_or_id = str(name_or_id or "").strip()
    # direct id match
    if name_or_id in products_df["product_id"].astype(str).values:
        return name_or_id

    # name match (case-insensitive)
    name_lower = name_or_id.lower()
    mask = products_df['name'].astype(str).str.lower() == name_lower
    if mask.any():
        return str(products_df.loc[mask, "product_id"].iloc[0])

    # if looks like "P0001 — Name", try split
    if "—" in name_or_id or "-" in name_or_id:
        pid = name_or_id.split("—")[0].split("-")[0].strip()
        if pid and pid in products_df["product_id"].astype(str).values:
            return pid

    raise ValueError(f"Product '{name_or_id}' not found")

def get_top_customers_for_product(product_id: str, offset: int = 0, limit: int = 100):
    """
    Return paginated list of best customer IDs for a product.
    offset = page * limit
    """

    customers_df = load_customers()
    products_df = load_products()

    if product_id not in products_df["product_id"].astype(str).values:
        raise ValueError(f"Product {product_id} not found")

    # Ranking must return FULL sorted list (best → worst)
    ranked = rank_customers_for_product(product_id, customers_df)

    if not ranked:
        return []

    # ranked can be list of IDs or list of dicts
    customer_ids = []
    for r in ranked:
        if isinstance(r, dict):
            customer_ids.append(r.get("customer_id"))
        else:
            customer_ids.append(r)

    return customer_ids[offset : offset + limit]