"""Part A, task 2: audit the raw exports. Run from the repository root: python src/part_a_audit.py"""
import pandas as pd

RAW = "data/raw/"


def table_report(name, df):
    print(f"\n=== {name}: shape {df.shape} ===")
    rep = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
    })
    print(rep.to_string())


def main():
    customers = pd.read_csv(RAW + "customers.csv")
    products = pd.read_csv(RAW + "products.csv")
    orders = pd.read_csv(RAW + "orders.csv")
    for name, df in [("customers", customers), ("products", products), ("orders", orders)]:
        table_report(name, df)

    # amounts must be inspected as text: pandas silently turns the string "N/A" into NaN
    orders_txt = pd.read_csv(RAW + "orders.csv", dtype=str, keep_default_na=False)
    amount_num = pd.to_numeric(orders_txt["order_amount"], errors="coerce")
    deliv = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    age = pd.to_numeric(customers["age"], errors="coerce")
    gender_raw = pd.read_csv(RAW + "customers.csv", dtype=str, keep_default_na=False)["gender"]

    audit = {
        "orders raw rows": len(orders),
        "exact duplicate order rows": int(orders.duplicated().sum()),
        "rows with status pending": int((orders["status"] == "pending").sum()),
        "distinct order_id": int(orders["order_id"].nunique()),
        "non-numeric order_amount entries": int(amount_num.isna().sum()),
        "negative delivery_minutes": int((deliv < 0).sum()),
        "blank delivery_minutes": int(deliv.isna().sum()),
        "customers: blank age": int(age.isna().sum()),
        "customers: age outside 10-100": int(((age < 10) | (age > 100)).sum()),
        "customers: distinct raw city spellings": int(customers["city"].nunique()),
        "customers: distinct non-blank raw gender spellings": int(gender_raw[gender_raw != ""].nunique()),
        "customers: blank gender": int((gender_raw == "").sum()),
    }
    print("\n=== Audit figures ===")
    for k, v in audit.items():
        print(f"{k}: {v:,}")


if __name__ == "__main__":
    main()
