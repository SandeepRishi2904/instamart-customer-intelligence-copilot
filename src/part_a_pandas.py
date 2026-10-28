"""Part A, task 7: pandas cross-check of Q2 and exploratory statistics of completed order_amount.
Run from the repository root: python src/part_a_pandas.py"""
import sqlite3

import pandas as pd

o = pd.read_csv("data/clean/orders.csv")
c = pd.read_csv("data/clean/customers.csv")
p = pd.read_csv("data/clean/products.csv")
done = o[o["status"] == "completed"].merge(c[["customer_id", "city"]], on="customer_id") \
                                    .merge(p[["product_id", "category"]], on="product_id")

# (a) pivot table versus SQL Q2
pivot = pd.pivot_table(done, values="order_amount", index="city", columns="category",
                       aggfunc="sum", margins=True, margins_name="Total").round(2)
print("Pivot table (INR):\n", pivot.to_string())
con = sqlite3.connect("data/instamart.db")
sql_total = con.execute("""SELECT ROUND(SUM(o.order_amount),2) FROM orders o
                           JOIN customers c ON c.customer_id=o.customer_id
                           JOIN products p ON p.product_id=o.product_id
                           WHERE o.status='completed'""").fetchone()[0]
con.close()
pivot_total = round(float(pivot.loc["Total", "Total"]), 2)
print(f"\npivot grand total {pivot_total:,.2f} == SQL total {sql_total:,.2f}: {pivot_total == sql_total}")

by_city = done.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"),
                                   orders=("order_amount", "count")).round(2)
print("\nPer-city completed order_amount:\n", by_city.to_string())

# (b) exploratory statistics
x = done["order_amount"]
q1, q3 = x.quantile(0.25), x.quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("\nCompleted order_amount statistics (INR):")
print(f"mean {x.mean():.2f}, median {x.median():.2f}, skewness {x.skew():.2f}")
print(f"Q1 {q1:.2f}, Q3 {q3:.2f}, IQR {iqr:.2f}")
print(f"lower fence {lo:.2f} -> {int((x < lo).sum())} outliers; upper fence {hi:.2f} -> {int((x > hi).sum())} outliers")
corr = done[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr().round(3)
print("\nCorrelation matrix:\n", corr.to_string())
cols = list(corr.columns)
pairs = {(a, b): corr.loc[a, b] for i, a in enumerate(cols) for b in cols[i + 1:]}
best = max(pairs, key=lambda k: abs(pairs[k]))
print(f"largest |correlation|: {best[0]} vs {best[1]} = {pairs[best]:.3f}")
