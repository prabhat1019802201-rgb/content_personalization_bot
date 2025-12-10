import os
import pandas as pd

# Resolve path to ../data relative to this file
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def load_customers() -> pd.DataFrame:
    """
    Load customers.csv as a DataFrame.
    """
    path = os.path.join(DATA_DIR, "customers.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"customers.csv not found at {path}")
    return pd.read_csv(path)


def load_events() -> pd.DataFrame:
    """
    Load events.csv as a DataFrame (parses event_ts as datetime).
    """
    path = os.path.join(DATA_DIR, "events.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"events.csv not found at {path}")
    return pd.read_csv(path, parse_dates=["event_ts"])


def load_products() -> pd.DataFrame:
    """
    Load products.csv as a DataFrame.
    """
    path = os.path.join(DATA_DIR, "products.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"products.csv not found at {path}")
    return pd.read_csv(path)


def load_all():
    """
    Convenience helper: load all three tables at once.
    Returns (customers_df, events_df, products_df).
    """
    customers = load_customers()
    events = load_events()
    products = load_products()
    return customers, events, products


if __name__ == "__main__":
    # Quick manual test
    c, e, p = load_all()
    print("Customers:", c.shape)
    print("Events   :", e.shape)
    print("Products :", p.shape)
