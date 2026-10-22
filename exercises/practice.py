"""Practice problems: Python, pandas and numpy for finance.

Fill in each function, then check your work with:

    pytest tests/test_exercises.py --practice

Worked answers are in ``solutions.py``. Try each one before you look.
"""

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401


# 1. Compound interest ----------------------------------------------------------
def future_value(principal, annual_rate, years, compounds_per_year=12):
    """Value of ``principal`` after ``years`` with periodic compounding.

    future_value(1000, 0.05, 10) -> about 1647.01
    """
    raise NotImplementedError


# 2. Loan amortization --------------------------------------------------------------
def monthly_payment(principal, annual_rate, years):
    """Fixed monthly payment that repays a loan in full.

    Use P * r / (1 - (1 + r) ** -n) with monthly rate r and n payments.
    Handle a 0% rate. monthly_payment(200_000, 0.06, 30) -> about 1199.10
    """
    raise NotImplementedError


def amortization_schedule(principal, annual_rate, years):
    """DataFrame with one row per month: month, payment, interest, principal, balance."""
    raise NotImplementedError


# 3. Returns with pandas ---------------------------------------------------------------
def monthly_returns(daily_prices):
    """Month-end to month-end simple returns from a daily price Series with a DatetimeIndex.

    Hint: resample("ME").last(), then pct_change().
    """
    raise NotImplementedError


def rolling_volatility(returns, window=21, periods=252):
    """Annualized rolling standard deviation of daily returns."""
    raise NotImplementedError


# 4. Data cleaning -------------------------------------------------------------------
def clean_transactions(df):
    """Clean a raw transactions DataFrame with columns txn_id, amount, date, category.

    - drop duplicate txn_id rows (keep the first)
    - strip "$" and "," from amount and convert it to float
    - parse date to datetime
    - lower-case and strip category, filling missing values with "unknown"
    - drop rows whose amount is missing or not positive
    Return the result with a fresh 0..n-1 index.
    """
    raise NotImplementedError


# 5. Groupby ------------------------------------------------------------------------------
def default_rate_by_band(loans, bands=(300, 580, 670, 740, 800, 851)):
    """Default rate and loan count per credit-score band.

    ``loans`` has credit_score and default columns. Return a DataFrame indexed
    by band (from pd.cut with right=False) with columns n_loans and default_rate.
    """
    raise NotImplementedError


# 6. Numpy -------------------------------------------------------------------------------
def simulate_price_paths(s0, mu, sigma, days, n_paths, seed=0):
    """Geometric Brownian motion price paths, shape (days + 1, n_paths).

    Daily step: S_t+1 = S_t * exp((mu - sigma**2 / 2) * dt + sigma * sqrt(dt) * Z)
    with dt = 1/252. Row 0 is s0. Use np.random.default_rng(seed).
    """
    raise NotImplementedError
