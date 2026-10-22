"""P01 Credit risk: which loan applicants are likely to default?

Fits a logistic regression scorecard and a gradient boosting model, compares
them on AUC, Gini and KS, and turns the scorecard into an approval cut-off
with an expected-loss estimate.

Run: ``python -m projects.p01_credit_risk.analysis``
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.generate_data import make_loans
from fintools import metrics
from fintools.plotting import plot_roc

NUMERIC = [
    "age", "income", "loan_amount", "term_months", "credit_score",
    "employment_years", "debt_to_income", "num_delinquencies",
]
CATEGORICAL = ["home_ownership", "purpose"]
LGD = 0.45  # assumed loss given default for unsecured lending


def add_features(df):
    df = df.copy()
    df["loan_to_income"] = df["loan_amount"] / df["income"]
    df["monthly_payment_share"] = df["loan_amount"] / df["term_months"] / (df["income"] / 12)
    return df


def build_models():
    numeric = NUMERIC + ["loan_to_income", "monthly_payment_share"]
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return {
        "logistic": Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000))]),
        "gradient_boosting": Pipeline(
            [("prep", prep), ("model", HistGradientBoostingClassifier(max_depth=3, random_state=0))]
        ),
    }


def approval_table(y_true, pd_hat, ead, cutoffs=(0.05, 0.10, 0.15, 0.20, 0.30)):
    """Approval rate, bad rate and expected loss for each PD cut-off."""
    rows = []
    for c in cutoffs:
        approved = pd_hat < c
        rows.append(
            {
                "pd_cutoff": c,
                "approval_rate": approved.mean(),
                "bad_rate_approved": y_true[approved].mean() if approved.any() else np.nan,
                "expected_loss": metrics.expected_loss(pd_hat[approved], LGD, ead[approved]).sum(),
            }
        )
    return pd.DataFrame(rows)


def run(seed=42, out_dir=None):
    df = add_features(make_loans(seed=seed))
    X, y = df.drop(columns=["loan_id", "default"]), df["default"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

    scores, results = {}, {}
    for name, model in build_models().items():
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        scores[name] = p
        results[name] = {
            "auc": roc_auc_score(y_te, p),
            "gini": metrics.gini(y_te, p),
            "ks": metrics.ks_statistic(y_te, p),
        }

    best = max(results, key=lambda k: results[k]["auc"])
    table = approval_table(y_te.to_numpy(), scores[best], X_te["loan_amount"].to_numpy())

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        plot_roc(y_te, scores[best], label=best, path=out / "p01_roc.png")
        table.to_csv(out / "p01_approval_table.csv", index=False)

    return {
        "default_rate": float(y.mean()),
        "models": results,
        "best_model": best,
        "approval_table": table,
        "psi_train_vs_test": metrics.population_stability_index(
            X_tr["credit_score"], X_te["credit_score"]
        ),
    }


if __name__ == "__main__":
    res = run(out_dir="outputs")
    print(f"Default rate: {res['default_rate']:.1%}")
    print(pd.DataFrame(res["models"]).T.round(3))
    print(f"\nApproval cut-offs using {res['best_model']}:")
    print(res["approval_table"].round(3).to_string(index=False))
