"""Part B, tasks 1-3: RFM table, K-means, segment profile and names, PCA, outputs/customer_segments.csv.
Run from the repository root: python src/part_b_segments.py"""
import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4


def build_rfm():
    o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    done = o[o["status"] == "completed"]
    g = done.groupby("customer_id")
    rfm = pd.DataFrame({
        "recency": (SNAPSHOT - g["order_ts"].max()).dt.days,
        "frequency": g["order_id"].nunique(),
        "monetary": g["order_amount"].sum().round(2),
    }).reset_index()
    return rfm


def name_segments(profile):
    """Rule-based names so they do not depend on arbitrary cluster numbers."""
    names = {}
    premium = profile["monetary"].idxmax()
    names[premium] = "Premium Big Spenders"
    rest = profile.drop(index=premium)
    loyal = rest["frequency"].idxmax()
    names[loyal] = "Loyal Regulars"
    rest = rest.drop(index=loyal)
    dormant = rest["recency"].idxmax()
    names[dormant] = "Drifting / At Risk"
    for k in rest.index:
        names.setdefault(k, "Occasional Shoppers")
    return names


ACTIONS = {
    "Premium Big Spenders": "Early access to electronics launches and a dedicated premium desk.",
    "Loyal Regulars": "Subscription / bundle offers to protect the high order frequency.",
    "Drifting / At Risk": "Win-back coupon timed before the next expected order.",
    "Occasional Shoppers": "Light-touch nudges and first-order-of-the-week discounts.",
}


def main():
    rfm = build_rfm()
    print(f"RFM customers: {len(rfm)}")
    print(f"recency median {rfm['recency'].median()}, min {rfm['recency'].min()}, max {rfm['recency'].max()}")
    print(f"frequency mean {rfm['frequency'].mean():.2f}, max {rfm['frequency'].max()}")
    print(f"monetary mean {rfm['monetary'].mean():.2f}, max {rfm['monetary'].max():.2f}")

    X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append({"K": k, "inertia": round(km.inertia_, 1), "silhouette": round(silhouette_score(X, km.labels_), 3)})
    print("\nK-means table:\n", pd.DataFrame(rows).to_string(index=False))

    km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
    rfm["segment_id"] = km.labels_
    profile = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                            frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
    names = name_segments(profile)
    profile["segment_name"] = profile.index.map(names)
    profile["action"] = profile["segment_name"].map(ACTIONS)
    print(f"\nSegment profile for K={FINAL_K}:\n", profile.to_string())

    rfm["segment_name"] = rfm["segment_id"].map(names)
    premium_id = profile["monetary"].idxmax()
    rfm["is_premium"] = rfm["segment_id"] == premium_id
    pca = PCA(n_components=2).fit(X)
    evr = pca.explained_variance_ratio_
    print(f"\nPCA explained_variance_ratio_: {evr[0]:.3f}, {evr[1]:.3f}, total {evr.sum():.3f}")

    os.makedirs("outputs", exist_ok=True)
    rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]] \
        .to_csv("outputs/customer_segments.csv", index=False)
    print(f"premium customers: {int(rfm['is_premium'].sum())}; written outputs/customer_segments.csv")


if __name__ == "__main__":
    main()
