"""Generate realistic synthetic banking & market datasets.

Everything is seeded, so results are reproducible and no external data or
API keys are needed. Swap in real data later (e.g. Kaggle "Give Me Some
Credit", yfinance prices) once the pipeline works.

Usage:  python data/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
RNG = np.random.default_rng(42)


def make_loans(n: int = 20_000) -> pd.DataFrame:
    """Loan applications with a default flag driven by plausible risk factors."""
    age = RNG.integers(21, 70, n)
    income = np.round(RNG.lognormal(mean=10.8, sigma=0.5, size=n), -2)
    loan_amount = np.round(income * RNG.uniform(0.1, 0.8, n), -2)
    term_months = RNG.choice([12, 24, 36, 48, 60], n, p=[0.1, 0.2, 0.35, 0.15, 0.2])
    credit_score = np.clip(RNG.normal(680, 70, n), 300, 850).round()
    dti = np.clip(RNG.normal(0.32, 0.12, n), 0.01, 0.95).round(3)
    employment_years = np.clip(RNG.exponential(6, n), 0, 40).round(1)
    num_delinquencies = RNG.poisson(0.4, n)
    home = RNG.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.45, 0.4, 0.15])
    purpose = RNG.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "business", "car", "other"],
        n, p=[0.4, 0.2, 0.12, 0.08, 0.1, 0.1],
    )

    # Latent log-odds of default
    logit = (
        -2.2
        - 0.012 * (credit_score - 680)
        + 3.0 * (dti - 0.32)
        + 0.9 * (loan_amount / income - 0.45)
        + 0.35 * num_delinquencies
        - 0.03 * employment_years
        + 0.01 * (term_months - 36)
        + np.where(home == "RENT", 0.25, 0.0)
        + np.where(purpose == "business", 0.4, 0.0)
        + RNG.normal(0, 0.5, n)
    )
    default = RNG.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame({
        "loan_id": np.arange(1, n + 1),
        "age": age, "annual_income": income, "loan_amount": loan_amount,
        "term_months": term_months, "credit_score": credit_score, "dti": dti,
        "employment_years": employment_years, "num_delinquencies": num_delinquencies,
        "home_ownership": home, "purpose": purpose, "default": default.astype(int),
    })
    # Real data is messy: inject ~3% missing values in two columns
    for col in ["employment_years", "annual_income"]:
        df.loc[RNG.random(n) < 0.03, col] = np.nan
    return df


def make_prices(start: str = "2019-01-01", years: int = 6, seed: int = 21) -> pd.DataFrame:
    """Daily close prices for 6 assets via correlated geometric Brownian motion."""
    tickers = ["BANK_A", "BANK_B", "TECH", "ENERGY", "BOND_ETF", "GOLD"]
    mu = np.array([0.08, 0.07, 0.15, 0.06, 0.03, 0.05])      # annual drift
    sigma = np.array([0.25, 0.28, 0.35, 0.30, 0.06, 0.15])  # annual vol
    corr = np.array([
        [1.0, 0.8, 0.5, 0.4, -0.1, 0.0],
        [0.8, 1.0, 0.5, 0.4, -0.1, 0.0],
        [0.5, 0.5, 1.0, 0.3, -0.2, 0.0],
        [0.4, 0.4, 0.3, 1.0, -0.1, 0.2],
        [-0.1, -0.1, -0.2, -0.1, 1.0, 0.3],
        [0.0, 0.0, 0.0, 0.2, 0.3, 1.0],
    ])
    dates = pd.bdate_range(start, periods=252 * years)
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    cov = np.outer(sigma, sigma) * corr * dt
    shocks = rng.multivariate_normal(np.zeros(len(tickers)), cov, len(dates))
    log_ret = (mu - 0.5 * sigma**2) * dt + shocks
    prices = 100 * np.exp(np.cumsum(log_ret, axis=0))
    return pd.DataFrame(prices.round(2), index=pd.Index(dates, name="date"), columns=tickers)


def make_transactions(n_customers: int = 3_000, n_tx: int = 120_000) -> pd.DataFrame:
    """Card transactions with ~0.8% fraud that behaves differently from normal spend."""
    customer_id = RNG.integers(1, n_customers + 1, n_tx)
    ts = pd.Timestamp("2025-01-01") + pd.to_timedelta(RNG.uniform(0, 365 * 24 * 3600, n_tx), unit="s")
    category = RNG.choice(
        ["grocery", "fuel", "restaurant", "online_retail", "travel", "electronics", "atm"],
        n_tx, p=[0.3, 0.15, 0.2, 0.15, 0.05, 0.05, 0.1],
    )
    amount = np.round(RNG.lognormal(3.5, 0.9, n_tx), 2)
    is_foreign = RNG.random(n_tx) < 0.04
    is_fraud = RNG.random(n_tx) < 0.008

    # Fraud: larger amounts, odd hours, foreign, electronics/online
    amount[is_fraud] = np.round(RNG.lognormal(5.5, 1.0, is_fraud.sum()), 2)
    is_foreign[is_fraud] = RNG.random(is_fraud.sum()) < 0.45
    category[is_fraud] = RNG.choice(["online_retail", "electronics", "atm", "travel"], is_fraud.sum())
    night = pd.to_timedelta(RNG.integers(0, 5, is_fraud.sum()), unit="h")
    ts = pd.Series(ts)
    ts[is_fraud] = ts[is_fraud].dt.normalize() + night.values

    df = pd.DataFrame({
        "tx_id": np.arange(1, n_tx + 1), "customer_id": customer_id, "timestamp": ts,
        "amount": amount, "category": category, "is_foreign": is_foreign.astype(int),
        "is_fraud": is_fraud.astype(int),
    })
    return df.sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    make_loans().to_csv(DATA_DIR / "loans.csv", index=False)
    make_prices().to_csv(DATA_DIR / "prices.csv")
    make_transactions().to_csv(DATA_DIR / "transactions.csv", index=False)
    print("Wrote loans.csv, prices.csv, transactions.csv to", DATA_DIR)
