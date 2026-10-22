"""Project 3 - Card Fraud Detection on imbalanced transaction data.

Business question: which transactions should we block or send for review,
given that fraud is <1% of volume and analyst time is limited?

Skills shown: time-based feature engineering (per-customer behaviour),
groupby/rolling windows, imbalanced classification, precision-recall,
choosing an alert threshold by review capacity, time-ordered validation.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from fintools.plotting import SERIES, apply_style, plt, save  # noqa: E402

DAILY_REVIEW_CAPACITY = 3    # alerts an analyst team can work per day


def build_features(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.sort_values(["customer_id", "timestamp"]).copy()
    g = tx.groupby("customer_id")
    tx["hour"] = tx["timestamp"].dt.hour
    tx["is_night"] = tx["hour"].between(0, 5).astype(int)
    tx["log_amount"] = np.log1p(tx["amount"])
    # Customer's history BEFORE this transaction (shift avoids target leakage)
    tx["cust_mean_amt"] = g["amount"].transform(lambda s: s.shift().expanding().mean())
    tx["amt_vs_cust_mean"] = tx["amount"] / tx["cust_mean_amt"]
    tx["cust_tx_count"] = g.cumcount()
    tx["secs_since_last"] = g["timestamp"].diff().dt.total_seconds()
    tx = pd.get_dummies(tx, columns=["category"], dtype=int)
    return tx.sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    apply_style()
    tx = pd.read_csv(ROOT / "data" / "transactions.csv", parse_dates=["timestamp"])
    print(f"Transactions: {len(tx):,} | fraud rate {tx['is_fraud'].mean():.2%} "
          f"| fraud $ share {tx.loc[tx.is_fraud == 1, 'amount'].sum() / tx['amount'].sum():.1%}")
    print(tx.groupby("category")["is_fraud"].mean().sort_values(ascending=False).map("{:.2%}".format)
          .to_string(), "\n")

    df = build_features(tx)
    features = [c for c in df.columns if c not in ("tx_id", "customer_id", "timestamp", "is_fraud", "amount")]

    # Time-ordered split: train on Jan-Sep, test on Oct-Dec (no peeking at the future)
    split = pd.Timestamp("2025-10-01")
    train, test = df[df["timestamp"] < split], df[df["timestamp"] >= split]
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                           random_state=0)
    model.fit(train[features], train["is_fraud"])
    score = model.predict_proba(test[features])[:, 1]

    ap = average_precision_score(test["is_fraud"], score)
    print(f"Average precision {ap:.3f} (random baseline {test['is_fraud'].mean():.3f})")

    # Threshold by operational capacity: top-N scores per day get reviewed
    n_days = test["timestamp"].dt.date.nunique()
    budget = DAILY_REVIEW_CAPACITY * n_days
    flagged = test.assign(score=score).nlargest(budget, "score")
    caught = flagged["is_fraud"].sum()
    total = test["is_fraud"].sum()
    saved = flagged.loc[flagged.is_fraud == 1, "amount"].sum()
    print(f"Reviewing {DAILY_REVIEW_CAPACITY}/day ({budget:,} alerts): precision {caught / budget:.1%}, "
          f"recall {caught / total:.1%}, fraud $ intercepted ${saved:,.0f} of "
          f"${test.loc[test.is_fraud == 1, 'amount'].sum():,.0f}")

    prec, rec, _ = precision_recall_curve(test["is_fraud"], score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color=SERIES[0], label=f"Gradient boosting (AP {ap:.2f})")
    ax.axhline(test["is_fraud"].mean(), color="#b5b4ad", ls="--", lw=1, label="Random")
    ax.scatter(caught / total, caught / budget, color=SERIES[1], s=80, edgecolor="white", lw=2,
               label=f"{DAILY_REVIEW_CAPACITY} alerts/day", zorder=3)
    ax.set(title="Precision-recall - Q4 holdout", xlabel="Recall", ylabel="Precision")
    ax.legend(loc="center left")
    save(fig, "fraud_pr_curve.png")


if __name__ == "__main__":
    main()
