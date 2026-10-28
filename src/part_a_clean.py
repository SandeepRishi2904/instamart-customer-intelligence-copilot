"""Part A, tasks 3-5: clean customers and orders, write CSVs to data/clean/ and tables to data/instamart.db.
Run from the repository root: python src/part_a_clean.py"""
import os
import sqlite3

import pandas as pd

RAW, CLEAN, DB = "data/raw/", "data/clean/", "data/instamart.db"
GENDER_MAP = {"male": "male", "m": "male", "female": "female", "f": "female"}


def clean_customers(raw):
    c = raw.copy()
    c["city"] = c["city"].str.strip().str.title()
    c["gender"] = c["gender"].fillna("").astype(str).str.strip().str.lower().map(GENDER_MAP).fillna("unknown")
    bad_age = (c["age"] < 10) | (c["age"] > 100)
    print(f"customers: ages outside 10-100 set to missing: {int(bad_age.sum())}")
    c.loc[bad_age, "age"] = float("nan")
    n_missing = int(c["age"].isna().sum())
    c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
    print(f"customers: missing ages (blank + impossible) filled with the city median: {n_missing}")
    c["signup_date"] = pd.to_datetime(c["signup_date"])
    return c


def clean_orders(raw, products):
    o = raw.copy()
    print(f"orders: raw rows {len(o):,}")
    # 1. timestamps
    o["order_ts"] = pd.to_datetime(o["order_ts"])
    # 2. exact duplicates
    before = len(o)
    o = o.drop_duplicates()
    print(f"orders: exact duplicate rows removed: {before - len(o):,}")
    # 3. one record per order_id: keep the latest order_ts (the final record after the earlier 'pending' one)
    before = len(o)
    o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
    print(f"orders: superseded earlier records removed (pending): {before - len(o):,}")
    # 4. amounts: unparseable -> missing -> recomputed from price x quantity x (1 - discount)
    o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
    o = o.merge(products[["product_id", "category", "price_inr"]], on="product_id", how="left")
    miss = o["order_amount"].isna()
    o.loc[miss, "order_amount"] = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)[miss]
    print(f"orders: unparseable order_amount values filled from the product price: {int(miss.sum())}")
    # 5. delivery minutes
    neg = o["delivery_minutes"] < 0
    o.loc[neg, "delivery_minutes"] = float("nan")
    print(f"orders: negative delivery_minutes set to missing: {int(neg.sum())}")
    comp = o["status"] == "completed"
    med = o[comp].groupby("category")["delivery_minutes"].transform("median")
    to_fill = comp & o["delivery_minutes"].isna()
    o.loc[comp, "delivery_minutes"] = o.loc[comp, "delivery_minutes"].fillna(med)
    print(f"orders: missing delivery_minutes of completed orders filled with category median: {int(to_fill.sum())}")
    print(f"orders: cancelled orders left with missing delivery_minutes: {int((o['status'] == 'cancelled').sum())}")
    o = o.sort_values("order_ts").reset_index(drop=True)
    return o.drop(columns=["category", "price_inr"])


def main():
    os.makedirs(CLEAN, exist_ok=True)
    customers = clean_customers(pd.read_csv(RAW + "customers.csv"))
    products = pd.read_csv(RAW + "products.csv")
    orders = clean_orders(pd.read_csv(RAW + "orders.csv"), products)

    cust_out, ord_out = customers.copy(), orders.copy()
    cust_out["signup_date"] = cust_out["signup_date"].dt.strftime("%Y-%m-%d")
    ord_out["order_ts"] = ord_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cust_out.to_csv(CLEAN + "customers.csv", index=False)
    products.to_csv(CLEAN + "products.csv", index=False)
    ord_out.to_csv(CLEAN + "orders.csv", index=False)

    if os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    cust_out.to_sql("customers", con, index=False)
    products.to_sql("products", con, index=False)
    ord_out.to_sql("orders", con, index=False)
    con.close()
    print(f"written: {CLEAN}customers.csv, products.csv, orders.csv and {DB}")


if __name__ == "__main__":
    main()
