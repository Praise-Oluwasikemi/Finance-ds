"""Reusable finance & risk metrics used across the projects."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------- Time value of money ----------
def loan_payment(principal: float, annual_rate: float, months: int) -> float:
    """Fixed monthly payment of an amortising loan."""
    r = annual_rate / 12
    if r == 0:
        return principal / months
    return principal * r / (1 - (1 + r) ** -months)


def amortization_schedule(principal: float, annual_rate: float, months: int) -> pd.DataFrame:
    """Month-by-month split of each payment into interest and principal."""
    pmt = loan_payment(principal, annual_rate, months)
    balance, rows = principal, []
    for m in range(1, months + 1):
        interest = balance * annual_rate / 12
        principal_paid = pmt - interest
        balance -= principal_paid
        rows.append((m, pmt, interest, principal_paid, max(balance, 0.0)))
    return pd.DataFrame(rows, columns=["month", "payment", "interest", "principal", "balance"])


def npv(rate: float, cashflows) -> float:
    """Net present value; cashflows[0] occurs today."""
    cf = np.asarray(cashflows, dtype=float)
    return float(np.sum(cf / (1 + rate) ** np.arange(len(cf))))


def irr(cashflows, lo: float = -0.99, hi: float = 10.0, tol: float = 1e-8) -> float:
    """Internal rate of return by bisection (assumes one sign change)."""
    f_lo = npv(lo, cashflows)
    for _ in range(500):
        mid = (lo + hi) / 2
        f_mid = npv(mid, cashflows)
        if abs(f_mid) < tol:
            break
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return mid


# ---------- Returns & performance ----------
def returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    r = np.log(prices / prices.shift(1)) if log else prices.pct_change()
    return r.dropna()


def annualized_return(r: pd.Series) -> float:
    growth = (1 + r).prod()
    return growth ** (TRADING_DAYS / len(r)) - 1


def annualized_vol(r: pd.Series) -> float:
    return r.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(r: pd.Series, risk_free: float = 0.0) -> float:
    excess = r - risk_free / TRADING_DAYS
    return excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)


def max_drawdown(r: pd.Series) -> float:
    """Largest peak-to-trough fall of the cumulative wealth curve (negative number)."""
    wealth = (1 + r).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


# ---------- Market risk ----------
def historical_var(r: pd.Series, level: float = 0.95) -> float:
    """1-day historical Value at Risk, reported as a positive loss."""
    return float(-np.quantile(r, 1 - level))


def historical_cvar(r: pd.Series, level: float = 0.95) -> float:
    """Expected shortfall: average loss beyond VaR."""
    cutoff = np.quantile(r, 1 - level)
    return float(-r[r <= cutoff].mean())


# ---------- Credit risk model evaluation ----------
def ks_statistic(y_true, y_score) -> float:
    """Max separation between cumulative score distributions of goods and bads."""
    df = pd.DataFrame({"y": y_true, "s": y_score}).sort_values("s")
    cum_bad = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    cum_good = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()
    return float((cum_bad - cum_good).abs().max())


def gini(auc: float) -> float:
    return 2 * auc - 1


def information_value(x: pd.Series, y: pd.Series, bins: int = 10) -> float:
    """IV of a feature vs a binary target — standard credit-scorecard screening."""
    binned = pd.qcut(x, bins, duplicates="drop") if x.dtype.kind in "if" and x.nunique() > bins else x
    tab = pd.crosstab(binned, y)
    good = (tab[0] + 0.5) / tab[0].sum()
    bad = (tab[1] + 0.5) / tab[1].sum()
    woe = np.log(good / bad)
    return float(((good - bad) * woe).sum())
