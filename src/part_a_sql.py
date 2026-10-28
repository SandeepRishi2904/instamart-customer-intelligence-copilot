"""Part A, task 6: run the seven queries of sql/queries.sql against data/instamart.db and print the stated results.
Run from the repository root: python src/part_a_sql.py"""
import re
import sqlite3

import pandas as pd

DB, SQL = "data/instamart.db", "sql/queries.sql"


def load_queries(path=SQL):
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"^-- (Q\d+):.*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def main():
    con = sqlite3.connect(DB)
    q = {name: pd.read_sql_query(sql, con) for name, sql in load_queries().items()}
    con.close()

    print(f"Q1 customers: {len(q['Q1'])}")
    q2 = q["Q2"]
    top = q2.iloc[0]
    print(f"Q2 rows: {len(q2)}; revenue total: {q2['revenue'].sum():,.2f}; "
          f"largest cell: {top['city']} / {top['category']} = {top['revenue']:,.2f}")
    print(f"Q3 products above the average price: {len(q['Q3'])}")
    print(f"Q4 customers without a completed order: {len(q['Q4'])}")
    q5 = q["Q5"]
    print(f"Q5 customers above average: {len(q5)}; average spend: {q5['avg_total'].iloc[0]:,.2f}")
    q6 = q["Q6"]
    print("Q6 top customer per city:")
    for _, r in q6[q6["row_num"] == 1].iterrows():
        print(f"   {r['city']}: {r['customer_id']} ({r['total_spend']:,.2f})")
    print(f"Q6 rows with RANK() != ROW_NUMBER(): {int((q6['rnk'] != q6['row_num']).sum())}")
    q7 = q["Q7"]
    print(f"Q7 daily rows: {len(q7)}; last cumulative revenue: {q7['cumulative_revenue'].iloc[-1]:,.2f}")


if __name__ == "__main__":
    main()
