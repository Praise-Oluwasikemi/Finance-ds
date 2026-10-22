"""Worked answers to ``practice.py``."""

import numpy as np
import pandas as pd


# 1. Compound interest ----------------------------------------------------------
def future_value(principal, annual_rate, years, compounds_per_year=12):
    r = annual_rate / compounds_per_year
    return principal * (1 + r) ** (compounds_per_year * years)


# 2. Loan amortization --------------------------------------------------------------
def monthly_payment(principal, annual_rate, years):
    n = years * 12
    r = annual_rate / 12
    if r == 0:
        return principal / n
    return principal * r / (1 - (1 + r) ** -n)


def amortization_schedule(principal, annual_rate, years):
    payment = monthly_payment(principal, annual_rate, years)
    r = annual_rate / 12
    balance = principal
    rows = []
    for month in range(1, years * 12 + 1):
        interest = balance * r
        principal_paid = payment - interest
        balance -= principal_paid
        rows.append(
            {
                "month": month,
                "payment": payment,
                "interest": interest,
                "principal": principal_paid,
                "balance": max(balance, 0.0),
            }
        )
    return pd.DataFrame(rows)


# 3. Returns with pandas ---------------------------------------------------------------
def monthly_returns(daily_prices):
    return daily_prices.resample("ME").last().pct_change().dropna()


def rolling_volatility(returns, window=21, periods=252):
    return returns.rolling(window).std() * np.sqrt(periods)


# 4. Data cleaning -------------------------------------------------------------------
def clean_transactions(df):
    out = df.drop_duplicates(subset="txn_id", keep="first").copy()
    out["amount"] = pd.to_numeric(
        out["amount"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"
    )
    out["date"] = pd.to_datetime(out["date"])
    out["category"] = out["category"].str.strip().str.lower().fillna("unknown")
    out = out[out["amount"] > 0]
    return out.reset_index(drop=True)


# 5. Groupby ------------------------------------------------------------------------------
def default_rate_by_band(loans, bands=(300, 580, 670, 740, 800, 851)):
    band = pd.cut(loans["credit_score"], bins=list(bands), right=False)
    return loans.groupby(band, observed=False)["default"].agg(
        n_loans="size", default_rate="mean"
    )


# 6. Numpy -------------------------------------------------------------------------------
def simulate_price_paths(s0, mu, sigma, days, n_paths, seed=0):
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    z = rng.standard_normal((days, n_paths))
    steps = (mu - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.vstack([np.zeros(n_paths), np.cumsum(steps, axis=0)])
    return s0 * np.exp(log_paths)
