"""Reference solutions. Try practice.py first - peek here only when stuck."""
import numpy as np
import pandas as pd


# ---------------- Level 1: finance maths with plain Python / NumPy ----------------
def compound_interest(principal, annual_rate, years, periods_per_year=12):
    return principal * (1 + annual_rate / periods_per_year) ** (periods_per_year * years)


def monthly_payment(principal, annual_rate, months):
    r = annual_rate / 12
    if r == 0:
        return principal / months
    return principal * r / (1 - (1 + r) ** -months)


def npv(rate, cashflows):
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))


# ---------------- Level 2: pandas data wrangling (banking data) ----------------
def clean_transactions(df):
    out = df.drop_duplicates(subset="tx_id").copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["amount"] = out["amount"].fillna(out.groupby("category")["amount"].transform("median"))
    return out.sort_values("timestamp").reset_index(drop=True)


def monthly_spend(df):
    month = pd.to_datetime(df["timestamp"]).dt.to_period("M")
    return df.assign(month=month).pivot_table(index="customer_id", columns="month", values="amount",
                                              aggfunc="sum", fill_value=0)


def top_customers(df, n=5):
    return df.groupby("customer_id")["amount"].sum().nlargest(n)


def default_rate_by_band(loans, bins=(300, 580, 670, 740, 800, 851)):
    labels = ["Poor", "Fair", "Good", "Very good", "Excellent"]
    band = pd.cut(loans["credit_score"], bins=list(bins), labels=labels, right=False)
    return loans.groupby(band, observed=False)["default"].mean()


# ---------------- Level 3: market & credit risk ----------------
def daily_returns(prices):
    return prices.pct_change().dropna()


def cumulative_return(returns):
    return (1 + returns).prod() - 1


def rolling_volatility(returns, window=21):
    return returns.rolling(window).std() * np.sqrt(252)


def max_drawdown(returns):
    wealth = (1 + returns).cumprod()
    return (wealth / wealth.cummax() - 1).min()


def historical_var(returns, level=0.95):
    return -np.quantile(returns, 1 - level)


def expected_loss(pd_, lgd, ead):
    return pd_ * lgd * ead


def misclassification_cost(y_true, y_pred, fp_cost, fn_cost):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return fp * fp_cost + fn * fn_cost
