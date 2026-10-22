"""Finance and model-evaluation metrics used across the projects."""

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# --- Returns -----------------------------------------------------------------

def simple_returns(prices):
    """Period-over-period percentage change, first row dropped."""
    return prices.pct_change().dropna()


def log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def annualized_return(returns, periods=TRADING_DAYS):
    """Geometric annualized return from periodic simple returns."""
    returns = pd.Series(returns) if not isinstance(returns, (pd.Series, pd.DataFrame)) else returns
    growth = (1 + returns).prod()
    return growth ** (periods / len(returns)) - 1


def annualized_volatility(returns, periods=TRADING_DAYS):
    return np.std(returns, ddof=1, axis=0) * np.sqrt(periods)


def sharpe_ratio(returns, risk_free=0.0, periods=TRADING_DAYS):
    """Annualized Sharpe ratio. ``risk_free`` is an annual rate."""
    excess = np.asarray(returns) - risk_free / periods
    return excess.mean() / excess.std(ddof=1) * np.sqrt(periods)


def sortino_ratio(returns, risk_free=0.0, periods=TRADING_DAYS):
    """Like Sharpe, but only penalizes downside volatility."""
    excess = np.asarray(returns) - risk_free / periods
    downside = np.minimum(excess, 0)
    downside_dev = np.sqrt((downside ** 2).mean())
    return excess.mean() / downside_dev * np.sqrt(periods)


def max_drawdown(returns):
    """Largest peak-to-trough loss of the cumulative return path (negative)."""
    wealth = np.cumprod(1 + np.asarray(returns))
    peaks = np.maximum.accumulate(wealth)
    return (wealth / peaks - 1).min()


def beta(asset_returns, market_returns):
    a, m = np.asarray(asset_returns), np.asarray(market_returns)
    return np.cov(a, m, ddof=1)[0, 1] / np.var(m, ddof=1)


# --- Risk --------------------------------------------------------------------

def historical_var(returns, confidence=0.95):
    """Value at Risk as a positive loss, from the empirical distribution."""
    return -np.percentile(returns, 100 * (1 - confidence))


def parametric_var(returns, confidence=0.95):
    """Gaussian Value at Risk as a positive loss."""
    r = np.asarray(returns)
    z = stats.norm.ppf(1 - confidence)
    return -(r.mean() + z * r.std(ddof=1))


def conditional_var(returns, confidence=0.95):
    """Expected Shortfall: the average loss on days worse than VaR."""
    r = np.asarray(returns)
    cutoff = np.percentile(r, 100 * (1 - confidence))
    return -r[r <= cutoff].mean()


def portfolio_returns(returns, weights):
    weights = np.asarray(weights, dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must sum to 1")
    return returns @ weights


# --- Model evaluation ----------------------------------------------------------

def ks_statistic(y_true, y_score):
    """Max separation between the score CDFs of positives and negatives."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    return stats.ks_2samp(y_score[y_true == 1], y_score[y_true == 0]).statistic


def gini(y_true, y_score):
    """Gini coefficient, 2 * AUC - 1, the usual credit-scoring summary."""
    from sklearn.metrics import roc_auc_score

    return 2 * roc_auc_score(y_true, y_score) - 1


def population_stability_index(expected, actual, bins=10):
    """PSI between a baseline and a new sample of a score or feature.

    Rule of thumb: < 0.1 stable, 0.1 to 0.25 watch, > 0.25 shifted.
    """
    expected, actual = np.asarray(expected), np.asarray(actual)
    edges = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def precision_at_k(y_true, y_score, k):
    """Share of true positives among the ``k`` highest-scored cases."""
    y_true = np.asarray(y_true)
    top = np.argsort(np.asarray(y_score))[::-1][:k]
    return y_true[top].mean()


def expected_loss(pd_, lgd, ead):
    """Credit expected loss: probability of default x loss given default x exposure."""
    return np.asarray(pd_) * np.asarray(lgd) * np.asarray(ead)
