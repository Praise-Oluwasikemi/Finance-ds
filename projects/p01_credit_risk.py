"""Project 1 - Credit Risk: Probability-of-Default model for a retail loan book.

Business question: which applicants are likely to default, and what cut-off
score balances approval volume against expected credit losses?

Skills shown: data cleaning, feature engineering, Information Value screening,
logistic regression vs gradient boosting, AUC / Gini / KS, expected-loss
based cut-off selection.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from fintools import gini, information_value, ks_statistic, naira  # noqa: E402
from fintools.plotting import SERIES, apply_style, plt, save  # noqa: E402

LGD = 0.45          # loss given default (Basel foundation IRB unsecured)
MARGIN = 0.08       # net interest margin earned on a good loan over its life


def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "loans.csv")
    # Feature engineering - ratios lenders actually use
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"]
    df["has_delinquency"] = (df["num_delinquencies"] > 0).astype(int)
    return df


def main() -> None:
    apply_style()
    df = load()
    print(f"Loans: {len(df):,}  | default rate: {df['default'].mean():.1%}")
    print("Missing values:\n", df.isna().sum()[lambda s: s > 0], "\n")

    numeric = ["age", "annual_income", "loan_amount", "term_months", "credit_score", "dti",
               "employment_years", "num_delinquencies", "loan_to_income", "has_delinquency"]
    categorical = ["home_ownership", "purpose"]

    # --- Feature screening with Information Value (scorecard industry standard)
    iv = pd.Series({c: information_value(df[c].fillna(df[c].median()) if c in numeric else df[c],
                                         df["default"]) for c in numeric + categorical})
    print("Information Value (>0.1 = useful, >0.3 = strong):")
    print(iv.sort_values(ascending=False).round(3).to_string(), "\n")

    X, y = df[numeric + categorical], df["default"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    models = {
        "Logistic regression": Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))]),
        "Gradient boosting": Pipeline([("prep", prep), ("clf", HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=300, random_state=0))]),
    }

    fig, ax = plt.subplots(figsize=(6, 5))
    scores = {}
    for i, (name, model) in enumerate(models.items()):
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        scores[name] = p
        auc = roc_auc_score(y_te, p)
        print(f"{name:20s} AUC={auc:.3f}  Gini={gini(auc):.3f}  KS={ks_statistic(y_te, p):.3f}")
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, color=SERIES[i], label=f"{name} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], color="#b5b4ad", lw=1, ls="--")
    ax.set(title="ROC curve - default prediction", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right")
    save(fig, "credit_roc.png")

    # --- Interpretability: logistic coefficients = drivers of risk
    lr = models["Logistic regression"]
    names = lr.named_steps["prep"].get_feature_names_out()
    coefs = pd.Series(lr.named_steps["clf"].coef_[0], index=names).sort_values()
    print("\nTop risk drivers (standardised log-odds):")
    print(pd.concat([coefs.head(3), coefs.tail(3)]).round(3).to_string())

    # --- Business decision: choose an approval cut-off that maximises profit
    p = scores["Logistic regression"]
    book = X_te.assign(pd_=p, default=y_te.values)
    results = []
    for cutoff in np.arange(0.05, 0.61, 0.01):
        approved = book[book["pd_"] < cutoff]
        profit = (approved["loan_amount"] * np.where(approved["default"] == 1, -LGD, MARGIN)).sum()
        results.append((cutoff, len(approved) / len(book), approved["default"].mean(), profit))
    res = pd.DataFrame(results, columns=["cutoff", "approval_rate", "bad_rate", "profit"])
    best = res.loc[res["profit"].idxmax()]
    print(f"\nOptimal PD cut-off {best.cutoff:.2f}: approve {best.approval_rate:.1%}, "
          f"bad rate {best.bad_rate:.1%}, profit {naira(best.profit)} "
          f"(vs approve-all {naira(res['profit'].iloc[-1])} at cut-off 0.60)")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(res["cutoff"], res["profit"] / 1e6, color=SERIES[0])
    ax.axvline(best.cutoff, color="#52514e", lw=1, ls="--")
    ax.annotate(f"best cut-off {best.cutoff:.2f}", (best.cutoff, best.profit / 1e6),
                xytext=(10, -15), textcoords="offset points")
    ax.set(title="Portfolio profit by approval cut-off (test set)", xlabel="Max PD approved", ylabel="Profit (₦m)")
    save(fig, "credit_profit_curve.png")


if __name__ == "__main__":
    main()
