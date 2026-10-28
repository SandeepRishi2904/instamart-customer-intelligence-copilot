"""Part A validation: asserts the reference checkpoints of the customer and order cleaning.
Run from the repository root: python src/validate.py"""
import sqlite3

import pandas as pd

c = pd.read_csv("data/clean/customers.csv")
o = pd.read_csv("data/clean/orders.csv")
done = o[o["status"] == "completed"]
canc = o[o["status"] == "cancelled"]

cities = c["city"].value_counts().to_dict()
genders = c["gender"].value_counts().to_dict()

checks = [
    ("six cities with the reference counts",
     cities == {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}),
    ("genders female 610 / male 560 / unknown 30", genders == {"female": 610, "male": 560, "unknown": 30}),
    ("0 missing ages", int(c["age"].isna().sum()) == 0),
    ("every age between 18 and 60", bool(c["age"].between(18, 60).all())),
    ("12,309 order rows, one per order_id", len(o) == 12309 and o["order_id"].is_unique),
    ("11,110 completed and 1,199 cancelled", len(done) == 11110 and len(canc) == 1199),
    ("no pending rows", int((o["status"] == "pending").sum()) == 0),
    ("0 missing order_amount", int(o["order_amount"].isna().sum()) == 0),
    ("0 missing delivery_minutes among completed", int(done["delivery_minutes"].isna().sum()) == 0),
    ("1,199 missing delivery_minutes among cancelled", int(canc["delivery_minutes"].isna().sum()) == 1199),
    ("no negative delivery_minutes", int((o["delivery_minutes"] < 0).sum()) == 0),
    ("completed order_amount sum is INR 2,998,668.80", round(float(done["order_amount"].sum()), 2) == 2998668.80),
]

con = sqlite3.connect("data/instamart.db")
tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
db_sum = con.execute("select round(sum(order_amount),2) from orders where status='completed'").fetchone()[0]
con.close()
checks.append(("database has tables customers, products, orders", {"customers", "products", "orders"} <= tables))
checks.append(("database completed revenue is INR 2,998,668.80", db_sum == 2998668.80))

failed = [name for name, ok in checks if not ok]
for name, ok in checks:
    print(("PASS  " if ok else "FAIL  ") + name)
assert not failed, f"failed checkpoints: {failed}"
print("ALL CHECKPOINTS PASSED")
