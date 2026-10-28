"""Part B, tasks 4-8: lapse problem, leakage-free preprocessing, tuning, evaluation, balancing, persistence.
Needs outputs/customer_segments.csv (run src/part_b_segments.py first).
Run from the repository root: python src/part_b_lapse.py"""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF = pd.Timestamp("2025-09-01")
NUM = ["recency", "frequency", "monetary", "avg_discount_pct", "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]


def build_dataset():
    o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    c = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
    p = pd.read_csv("data/clean/products.csv")
    before = o[o["order_ts"] < CUTOFF].merge(p[["product_id", "category"]], on="product_id")   # nothing from 2025-09-01 on
    done = before[before["status"] == "completed"]
    g = done.groupby("customer_id")
    fav = done.groupby("customer_id")["category"].agg(lambda s: sorted(s.value_counts().index[s.value_counts() == s.value_counts().max()])[0])
    f = pd.DataFrame({
        "recency": (CUTOFF - g["order_ts"].max()).dt.days,
        "frequency": g["order_id"].nunique(),
        "monetary": g["order_amount"].sum(),
        "avg_discount_pct": g["discount_pct"].mean(),
        "avg_delivery_minutes": g["delivery_minutes"].mean(),
        "fav_category": fav,
    })
    final = before[before["status"].isin(["completed", "cancelled"])]
    f["cancellation_rate"] = (final["status"] == "cancelled").groupby(final["customer_id"]).mean()
    f = f.reset_index().merge(c, on="customer_id")
    f["tenure_days"] = (CUTOFF - f["signup_date"]).dt.days
    # label: no completed order in September 2025
    sept = o[(o["status"] == "completed") & (o["order_ts"] >= CUTOFF) & (o["order_ts"] < pd.Timestamp("2025-10-01"))]
    f["lapsed"] = (~f["customer_id"].isin(sept["customer_id"])).astype(int)
    return f


def make_pipeline(model):
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", OneHotEncoder(drop="first"), CAT)])
    return Pipeline([("prep", pre), ("clf", model)])


def lapsed_metrics(y_true, pred):
    return {"precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0)}


def show(name, y_true, pred):
    m = lapsed_metrics(y_true, pred)
    print(f"{name}: precision {m['precision']:.3f}, recall {m['recall']:.3f}, F1 {m['f1']:.3f}")
    print("   confusion matrix [[TN FP] [FN TP]]:", confusion_matrix(y_true, pred).tolist())
    return m


def main():
    data = build_dataset()
    X, y = data[NUM + CAT], data["lapsed"]
    print(f"cohort {len(data)}; lapsed {int(y.sum())} ({y.mean() * 100:.2f}%)")
    assert not X.isna().any().any(), "a feature column has missing values"
    print("no feature column contains a missing value: confirmed")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"train {len(Xtr)}, test {len(Xte)}")
    base = np.zeros(len(yte), dtype=int)
    print(f"baseline (always not lapsed): accuracy {accuracy_score(yte, base) * 100:.1f}%, "
          f"lapsed F1 {f1_score(yte, base, zero_division=0):.2f}")

    # --- tuning (preprocessing sits inside the Pipeline, so every fold refits encoder and scaler on its own training part)
    lr_gs = GridSearchCV(make_pipeline(LogisticRegression(max_iter=1000)),
                         {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(Xtr, ytr)
    rf_gs = GridSearchCV(make_pipeline(RandomForestClassifier(n_estimators=100, random_state=42)),
                         {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(Xtr, ytr)
    print(f"\nLogistic regression best {lr_gs.best_params_}, CV F1 {lr_gs.best_score_:.3f}")
    print(f"Random forest best {rf_gs.best_params_}, CV F1 {rf_gs.best_score_:.3f}")

    # --- leakage check: scaler mean of frequency vs the full-cohort mean
    sc = lr_gs.best_estimator_.named_steps["prep"].named_transformers_["num"]
    print(f"\nscaler mean of frequency (train rows only) {sc.mean_[NUM.index('frequency')]:.3f}"
          f" vs full-cohort mean {data['frequency'].mean():.3f}")

    # --- test evaluation
    print("\nTest set, lapsed class:")
    lr_best, rf_best = lr_gs.best_estimator_, rf_gs.best_estimator_
    m_lr = show("Logistic regression", yte, lr_best.predict(Xte))
    m_rf = show("Random forest", yte, rf_best.predict(Xte))
    chosen_name = "random forest" if m_rf["f1"] >= m_lr["f1"] else "logistic regression"
    chosen = rf_best if chosen_name == "random forest" else lr_best
    print(f"selected model: {chosen_name}")

    # --- class-balance experiment: oversample lapsed rows of the TRAINING set only
    n_extra = int((ytr == 0).sum() - (ytr == 1).sum())
    extra = ytr[ytr == 1].sample(n=n_extra, replace=True, random_state=42).index
    Xbal, ybal = pd.concat([Xtr, Xtr.loc[extra]]), pd.concat([ytr, ytr.loc[extra]])
    print(f"\nbalanced training set: {int((ybal == 0).sum())} not lapsed / {int((ybal == 1).sum())} lapsed")
    lr_bal = clone(lr_best).fit(Xbal, ybal)
    print("Test set, original rows, logistic regression:")
    m0 = show("un-balanced", yte, lr_best.predict(Xte))
    m1 = show("over-sampled", yte, lr_bal.predict(Xte))
    print(f"change: precision {m1['precision'] - m0['precision']:+.3f}, recall {m1['recall'] - m0['recall']:+.3f}, "
          f"F1 {m1['f1'] - m0['f1']:+.3f}")

    # --- interpretation
    names = rf_best.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf_best.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
    print("\nTop 5 random-forest importances:\n", imp.head(5).round(3).to_string())
    coef = pd.Series(lr_best.named_steps["clf"].coef_[0], index=lr_best.named_steps["prep"].get_feature_names_out())
    print("\nLogistic regression coefficients (standardised features):\n", coef.round(3).sort_values().to_string())

    # --- persistence
    joblib.dump(chosen, "outputs/lapse_model.joblib")
    reloaded = joblib.load("outputs/lapse_model.joblib")
    same = bool((reloaded.predict(Xte) == chosen.predict(Xte)).all())
    print(f"\nreloaded model reproduces the test predictions exactly: {same}")
    assert same

    # --- retention list: Premium customers in the test set ranked by lapse probability
    seg = pd.read_csv("outputs/customer_segments.csv")[["customer_id", "is_premium"]]
    te = data.loc[Xte.index, ["customer_id", "lapsed"]].copy()
    te["lapse_probability"] = chosen.predict_proba(Xte)[:, 1].round(4)
    te = te.merge(seg, on="customer_id", how="left")
    prem = te[te["is_premium"] == True].sort_values("lapse_probability", ascending=False).head(10)
    out = prem.rename(columns={"lapsed": "actually_lapsed"})[["customer_id", "lapse_probability", "actually_lapsed"]]
    out.to_csv("outputs/retention_list.csv", index=False)
    print(f"retention list: {len(out)} Premium test customers, {int(out['actually_lapsed'].sum())} truly lapsed")


if __name__ == "__main__":
    main()
