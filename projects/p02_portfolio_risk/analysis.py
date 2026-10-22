"""P02 Portfolio risk: how much could this portfolio lose, and is it well built?

Compares an equal-weight portfolio with the minimum-variance and max-Sharpe
portfolios, reports VaR, Expected Shortfall and drawdown, and backtests VaR.

Run: ``python -m projects.p02_portfolio_risk.analysis``
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data.generate_data import make_prices
from fintools import metrics
from fintools.plotting import plot_correlation_heatmap, plot_cumulative_returns, plot_drawdown


def optimize_weights(returns, objective="min_variance", max_weight=0.4):
    """Long-only weights that sum to 1, for ``min_variance`` or ``max_sharpe``.

    ``max_weight`` caps any single asset so the optimizer cannot put
    everything into one holding.
    """
    # Annualize so the objective is not tiny, which stalls SLSQP at the start point.
    mu = returns.mean().to_numpy() * 252
    cov = returns.cov().to_numpy() * 252
    n = len(mu)

    def variance(w):
        return w @ cov @ w

    def neg_sharpe(w):
        return -(w @ mu) / np.sqrt(w @ cov @ w)

    fn = {"min_variance": variance, "max_sharpe": neg_sharpe}[objective]
    res = minimize(
        fn,
        np.full(n, 1 / n),
        bounds=[(0, max_weight)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        method="SLSQP",
    )
    w = np.where(res.x < 1e-6, 0.0, res.x)
    return pd.Series(w / w.sum(), index=returns.columns)


def risk_summary(r, market, confidence=0.95):
    return {
        "annual_return": metrics.annualized_return(r),
        "annual_volatility": metrics.annualized_volatility(r),
        "sharpe": metrics.sharpe_ratio(r, risk_free=0.03),
        "sortino": metrics.sortino_ratio(r, risk_free=0.03),
        "max_drawdown": metrics.max_drawdown(r),
        "beta": metrics.beta(r, market),
        "var_hist": metrics.historical_var(r, confidence),
        "var_param": metrics.parametric_var(r, confidence),
        "cvar": metrics.conditional_var(r, confidence),
    }


def var_backtest(r, window=250, confidence=0.95):
    """Rolling historical VaR; returns the breach rate (should be near 1 - confidence)."""
    var = r.rolling(window).quantile(1 - confidence).shift(1).dropna()
    breaches = r.loc[var.index] < var
    return float(breaches.mean())


def run(seed=42, out_dir=None):
    prices = make_prices(seed=seed)
    returns = metrics.simple_returns(prices)
    market = returns.pop("MARKET")

    weights = {
        "equal_weight": pd.Series(1 / returns.shape[1], index=returns.columns),
        "min_variance": optimize_weights(returns, "min_variance"),
        "max_sharpe": optimize_weights(returns, "max_sharpe"),
    }
    port = pd.DataFrame({k: metrics.portfolio_returns(returns, w) for k, w in weights.items()})
    summary = pd.DataFrame({k: risk_summary(port[k], market) for k in port}).T

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        plot_cumulative_returns(port, title="Portfolio growth", path=out / "p02_growth.png")
        plot_drawdown(port["equal_weight"], title="Equal-weight drawdown", path=out / "p02_drawdown.png")
        plot_correlation_heatmap(returns, title="Asset return correlation", path=out / "p02_corr.png")

    return {
        "weights": pd.DataFrame(weights),
        "summary": summary,
        "var_breach_rate": var_backtest(port["equal_weight"]),
    }


if __name__ == "__main__":
    res = run(out_dir="outputs")
    print("Weights:\n", res["weights"].round(3))
    print("\nRisk summary:\n", res["summary"].round(4))
    print(f"\n95% VaR backtest breach rate: {res['var_breach_rate']:.1%} (target 5%)")
