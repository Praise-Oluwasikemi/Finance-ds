"""Synthetic datasets for the four portfolio projects.

Every generator takes a ``seed`` so results are reproducible. The data is
made up, but the relationships between columns are realistic enough that the
models in ``projects/`` have real signal to find.

Run ``python -m data.generate_data`` to write CSVs to ``data/raw/``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent / "raw"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_loans(n=5000, seed=42):
    """Loan applications with a binary ``default`` target."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 70, n)
    income = np.round(rng.lognormal(mean=10.8, sigma=0.5, size=n), -2)
    loan_amount = np.round(income * rng.uniform(0.1, 0.8, n), -2)
    term_months = rng.choice([12, 24, 36, 48, 60], n)
    credit_score = np.clip(rng.normal(660, 70, n), 300, 850).round().astype(int)
    employment_years = np.clip(rng.exponential(5, n), 0, 40).round(1)
    debt_to_income = np.clip(rng.beta(2, 5, n) * 0.8, 0.01, 0.8).round(3)
    num_delinquencies = rng.poisson(0.4, n)
    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.45, 0.40, 0.15])
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "business", "car"], n
    )

    logit = (
        -2.2
        - 0.012 * (credit_score - 660)
        + 3.0 * (debt_to_income - 0.23)
        + 0.45 * num_delinquencies
        - 0.04 * employment_years
        + 0.9 * (loan_amount / income - 0.45)
        + 0.01 * (term_months - 36)
        + 0.25 * (home_ownership == "RENT")
        + 0.30 * (purpose == "business")
    )
    default = (rng.uniform(size=n) < _sigmoid(logit)).astype(int)

    return pd.DataFrame(
        {
            "loan_id": np.arange(1, n + 1),
            "age": age,
            "income": income,
            "loan_amount": loan_amount,
            "term_months": term_months,
            "credit_score": credit_score,
            "employment_years": employment_years,
            "debt_to_income": debt_to_income,
            "num_delinquencies": num_delinquencies,
            "home_ownership": home_ownership,
            "purpose": purpose,
            "default": default,
        }
    )


def make_prices(n_days=756, seed=42):
    """Daily closing prices for five assets plus a market index.

    Returns are drawn from a multivariate normal with a one-factor structure,
    so assets are correlated with the market and with each other.
    """
    rng = np.random.default_rng(seed)
    tickers = ["BANK_A", "BANK_B", "INSURER", "FINTECH", "BOND_ETF"]
    betas = np.array([1.1, 1.0, 0.8, 1.5, 0.1])
    alphas = np.array([0.0002, 0.0001, 0.0002, 0.0003, 0.0001])
    idio_vol = np.array([0.010, 0.011, 0.009, 0.020, 0.003])

    market = rng.normal(0.0003, 0.011, n_days)
    noise = rng.normal(0.0, 1.0, (n_days, len(tickers))) * idio_vol
    returns = alphas + market[:, None] * betas + noise

    dates = pd.bdate_range("2022-01-03", periods=n_days)
    start = np.array([50.0, 35.0, 80.0, 25.0, 100.0])
    prices = start * np.exp(np.cumsum(returns, axis=0))
    market_price = 4000.0 * np.exp(np.cumsum(market))

    df = pd.DataFrame(prices, index=dates, columns=tickers).round(2)
    df["MARKET"] = market_price.round(2)
    df.index.name = "date"
    return df


def make_transactions(n=20000, fraud_rate=0.02, seed=42):
    """Card transactions with a rare ``is_fraud`` target."""
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    is_fraud = np.zeros(n, dtype=int)
    is_fraud[rng.choice(n, n_fraud, replace=False)] = 1
    fraud = is_fraud == 1

    amount = np.where(
        fraud, rng.lognormal(4.3, 1.1, n), rng.lognormal(3.6, 1.0, n)
    ).round(2)
    hour = np.where(
        fraud, rng.choice(24, n, p=_hour_weights(night_heavy=True)),
        rng.choice(24, n, p=_hour_weights(night_heavy=False)),
    )
    distance_from_home_km = np.where(
        fraud, rng.exponential(60, n), rng.exponential(20, n)
    ).round(1)
    merchant_category = np.where(
        fraud,
        rng.choice(["electronics", "online", "travel", "grocery", "fuel"], n,
                   p=[0.2, 0.35, 0.15, 0.2, 0.1]),
        rng.choice(["electronics", "online", "travel", "grocery", "fuel"], n,
                   p=[0.08, 0.25, 0.07, 0.4, 0.2]),
    )
    is_foreign = (rng.uniform(size=n) < np.where(fraud, 0.2, 0.05)).astype(int)
    txns_last_24h = np.where(fraud, rng.poisson(3.5, n), rng.poisson(2, n))
    card_present = (rng.uniform(size=n) < np.where(fraud, 0.4, 0.65)).astype(int)

    return pd.DataFrame(
        {
            "txn_id": np.arange(1, n + 1),
            "amount": amount,
            "hour": hour,
            "distance_from_home_km": distance_from_home_km,
            "merchant_category": merchant_category,
            "is_foreign": is_foreign,
            "txns_last_24h": txns_last_24h,
            "card_present": card_present,
            "is_fraud": is_fraud,
        }
    )


def _hour_weights(night_heavy):
    hours = np.arange(24)
    if night_heavy:
        w = np.where((hours < 6) | (hours >= 22), 1.8, 1.0)
    else:
        w = np.exp(-((hours - 14) ** 2) / 30.0) + 0.05
    return w / w.sum()


def make_customers(n=3000, seed=42):
    """Retail bank customers drawn from four latent segments."""
    rng = np.random.default_rng(seed)
    # (age, income, balance, products, txns/month, digital share, tenure)
    profiles = {
        "young_digital": (27, 35000, 2500, 2, 60, 0.9, 2),
        "mass_market": (42, 45000, 6000, 3, 30, 0.5, 8),
        "affluent": (50, 140000, 85000, 5, 25, 0.6, 12),
        "retired_saver": (68, 30000, 40000, 3, 12, 0.15, 25),
    }
    names = list(profiles)
    segment = rng.choice(names, n, p=[0.3, 0.4, 0.15, 0.15])
    p = np.array([profiles[s] for s in segment], dtype=float)

    def jitter(col, scale):
        return p[:, col] * rng.lognormal(0, scale, n)

    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "age": np.clip(p[:, 0] + rng.normal(0, 8, n), 18, 95).round().astype(int),
            "income": jitter(1, 0.35).round(-2),
            "avg_balance": jitter(2, 0.7).round(2),
            "num_products": np.clip(p[:, 3] + rng.integers(-1, 2, n), 1, 8).astype(int),
            "monthly_txns": np.clip(jitter(4, 0.45), 1, None).round().astype(int),
            "digital_share": np.clip(p[:, 5] + rng.normal(0, 0.15, n), 0, 1).round(2),
            "tenure_years": np.clip(p[:, 6] + rng.normal(0, 5, n), 0, 60).round(1),
            "true_segment": segment,
        }
    )


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    make_loans().to_csv(RAW_DIR / "loans.csv", index=False)
    make_prices().to_csv(RAW_DIR / "prices.csv")
    make_transactions().to_csv(RAW_DIR / "transactions.csv", index=False)
    make_customers().to_csv(RAW_DIR / "customers.csv", index=False)
    print(f"Wrote datasets to {RAW_DIR}")


if __name__ == "__main__":
    main()
