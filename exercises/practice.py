"""Practice exercises - Python for finance & banking data science.
All money amounts are in Nigerian naira (₦).

Replace each `raise NotImplementedError` with your code, then run:

    pytest tests/test_exercises.py -x     # stops at the first failing exercise

Work top to bottom; difficulty rises by level. Solutions: exercises/solutions.py
"""
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401


# ============== Level 1: finance maths with plain Python / NumPy ==============
def compound_interest(principal, annual_rate, years, periods_per_year=12):
    """Future value of `principal` compounded `periods_per_year` times a year.
    e.g. ₦1m in an 18% fixed deposit for 5 years: compound_interest(1_000_000, 0.18, 5, 1) -> 2_287_757.76
    """
    raise NotImplementedError


def monthly_payment(principal, annual_rate, months):
    """Fixed monthly instalment of an amortising loan (handle a 0% rate!).
    e.g. ₦5m car loan at 24% over 3 years: monthly_payment(5_000_000, 0.24, 36) -> 196_164.26
    """
    raise NotImplementedError


def npv(rate, cashflows):
    """Net present value. cashflows[0] happens today (t=0), cashflows[1] in one period, ...
    e.g. npv(0.2, [-10_000_000, 6_000_000, 7_000_000]) -> -138_888.89  (negative: reject the project)
    """
    raise NotImplementedError


# ============== Level 2: pandas data wrangling (banking data) ==============
def clean_transactions(df):
    """Return a cleaned copy of a raw transactions DataFrame:
    - drop duplicate rows by tx_id (keep first)
    - convert `timestamp` to datetime
    - fill missing `amount` with the median amount of that row's `category`
    - sort by timestamp and reset the index
    Do not modify the input DataFrame.
    """
    raise NotImplementedError


def monthly_spend(df):
    """Pivot table: one row per customer_id, one column per month (pd.Period 'M'),
    values = total amount spent, missing months = 0.
    Hint: .dt.to_period("M") and pivot_table.
    """
    raise NotImplementedError


def top_customers(df, n=5):
    """Series of the n customers with the highest total spend, largest first (index = customer_id)."""
    raise NotImplementedError


def default_rate_by_band(loans, bins=(300, 580, 670, 740, 800, 851)):
    """Default rate per FICO-style credit band.
    Bands (left-inclusive): Poor [300,580), Fair [580,670), Good [670,740),
    Very good [740,800), Excellent [800,851). Return a Series indexed by band label.
    Hint: pd.cut(..., right=False, labels=[...]) then groupby.
    """
    raise NotImplementedError


# ============== Level 3: market & credit risk ==============
def daily_returns(prices):
    """Simple daily returns of a price DataFrame/Series, first (NaN) row dropped."""
    raise NotImplementedError


def cumulative_return(returns):
    """Total compounded return over the period, e.g. [0.1, -0.1] -> -0.01."""
    raise NotImplementedError


def rolling_volatility(returns, window=21):
    """Rolling standard deviation over `window` days, annualised with sqrt(252)."""
    raise NotImplementedError


def max_drawdown(returns):
    """Worst peak-to-trough fall of cumulative wealth, as a negative number (e.g. -0.35)."""
    raise NotImplementedError


def historical_var(returns, level=0.95):
    """1-day historical Value at Risk as a POSITIVE number: minus the (1-level) quantile."""
    raise NotImplementedError


def expected_loss(pd_, lgd, ead):
    """Basel expected loss = PD x LGD x EAD, e.g. expected_loss(0.02, 0.45, 10_000_000) -> ₦90,000. Must work on scalars and on NumPy arrays / Series."""
    raise NotImplementedError


def misclassification_cost(y_true, y_pred, fp_cost, fn_cost):
    """Total cost of a fraud/credit model's mistakes.
    False positive (pred 1, true 0) costs fp_cost; false negative (pred 0, true 1) costs fn_cost.
    """
    raise NotImplementedError
