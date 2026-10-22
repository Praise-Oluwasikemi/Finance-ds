"""P03 Fraud detection: which transactions should the fraud team review?

Fraud is rare (about 2%), so accuracy is meaningless. The model is judged on
average precision and on how much fraud the team catches within a fixed daily
review budget.

Run: ``python -m projects.p03_fraud_detection.analysis``
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.generate_data import make_transactions
from fintools import metrics
from fintools.plotting import plot_precision_recall

NUMERIC = ["log_amount", "night", "distance_from_home_km", "is_foreign", "txns_last_24h", "card_present"]
CATEGORICAL = ["merchant_category"]


def add_features(df):
    df = df.copy()
    df["log_amount"] = np.log1p(df["amount"])
    df["night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    return df


def build_models():
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return {
        "logistic_balanced": Pipeline(
            [("prep", prep), ("model", LogisticRegression(class_weight="balanced", max_iter=1000))]
        ),
        "random_forest": Pipeline(
            [
                ("prep", prep),
                ("model", RandomForestClassifier(
                    n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample",
                    n_jobs=-1, random_state=0,
                )),
            ]
        ),
    }


def review_budget_table(y_true, score, amounts, budgets=(50, 100, 200, 400)):
    """Fraud caught if the team reviews only the top-``k`` scored transactions."""
    y_true, amounts = np.asarray(y_true), np.asarray(amounts)
    order = np.argsort(np.asarray(score))[::-1]
    rows = []
    for k in budgets:
        top = order[:k]
        rows.append(
            {
                "reviews": k,
                "precision": y_true[top].mean(),
                "recall": y_true[top].sum() / y_true.sum(),
                "fraud_value_caught": amounts[top][y_true[top] == 1].sum(),
            }
        )
    return pd.DataFrame(rows)


def run(seed=42, out_dir=None):
    df = add_features(make_transactions(seed=seed))
    X, y = df.drop(columns=["txn_id", "is_fraud"]), df["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

    scores, results = {}, {}
    for name, model in build_models().items():
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        scores[name] = p
        results[name] = {
            "roc_auc": roc_auc_score(y_te, p),
            "avg_precision": average_precision_score(y_te, p),
            "precision_at_100": metrics.precision_at_k(y_te, p, 100),
        }

    best = max(results, key=lambda k: results[k]["avg_precision"])
    table = review_budget_table(y_te, scores[best], X_te["amount"])

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        plot_precision_recall(y_te, scores[best], label=best, path=out / "p03_pr_curve.png")
        table.to_csv(out / "p03_review_budget.csv", index=False)

    return {
        "fraud_rate": float(y.mean()),
        "models": results,
        "best_model": best,
        "review_budget": table,
    }


if __name__ == "__main__":
    res = run(out_dir="outputs")
    print(f"Fraud rate: {res['fraud_rate']:.2%}")
    print(pd.DataFrame(res["models"]).T.round(3))
    print(f"\nReview budget using {res['best_model']}:")
    print(res["review_budget"].round(3).to_string(index=False))
