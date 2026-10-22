"""Project 2 - Portfolio Analytics & Market Risk.

Business question: how have our six holdings performed on a risk-adjusted
basis, what is the optimal (max-Sharpe) mix, and how much could we lose on a
bad day?

Skills shown: return calculations, Sharpe, drawdown, rolling statistics,
Monte-Carlo efficient frontier, historical & parametric VaR / CVaR,
VaR backtesting.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from fintools import (TRADING_DAYS, annualized_return, annualized_vol, historical_cvar,  # noqa: E402
                      historical_var, max_drawdown, returns, sharpe_ratio)
from fintools.plotting import SERIES, apply_style, plt, save  # noqa: E402

RF = 0.03                # risk-free rate
PORTFOLIO_VALUE = 1_000_000


def main() -> None:
    apply_style()
    prices = pd.read_csv(ROOT / "data" / "prices.csv", index_col="date", parse_dates=True)
    r = returns(prices)

    summary = pd.DataFrame({
        "Ann. return": r.apply(annualized_return),
        "Ann. vol": r.apply(annualized_vol),
        "Sharpe": r.apply(sharpe_ratio, risk_free=RF),
        "Max drawdown": r.apply(max_drawdown),
        "VaR 95% (1d)": r.apply(historical_var),
    })
    print("Asset performance\n", summary.round(3).to_string(), "\n")
    print("Correlation matrix\n", r.corr().round(2).to_string(), "\n")

    # --- Efficient frontier via random portfolios (long-only)
    mu, cov = r.mean() * TRADING_DAYS, r.cov() * TRADING_DAYS
    rng = np.random.default_rng(0)
    w = rng.dirichlet(np.ones(len(mu)), 20_000)
    port_ret = w @ mu.values
    port_vol = np.sqrt(np.einsum("ij,jk,ik->i", w, cov.values, w))
    sharpe = (port_ret - RF) / port_vol
    best_w = pd.Series(w[sharpe.argmax()], index=mu.index)
    minvar_w = pd.Series(w[port_vol.argmin()], index=mu.index)
    print("Max-Sharpe weights\n", best_w.round(3).to_string())
    print(f"  -> return {port_ret[sharpe.argmax()]:.1%}, "
          f"vol {port_vol[sharpe.argmax()]:.1%}, Sharpe {sharpe.max():.2f}\n")
    print("Min-variance weights\n", minvar_w.round(3).to_string(), "\n")

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(port_vol, port_ret, c=sharpe, cmap="Blues", s=4, alpha=0.6)
    ax.scatter(port_vol[sharpe.argmax()], port_ret[sharpe.argmax()], color=SERIES[1], s=80,
               edgecolor="white", lw=2, label="Max Sharpe", zorder=3)
    ax.scatter(port_vol.min(), port_ret[port_vol.argmin()], color=SERIES[2], s=80,
               edgecolor="white", lw=2, label="Min variance", zorder=3)
    fig.colorbar(sc, label="Sharpe ratio")
    ax.set(title="Efficient frontier - 20,000 random portfolios", xlabel="Annualised volatility",
           ylabel="Annualised return")
    ax.legend()
    save(fig, "portfolio_frontier.png")

    # --- Market risk on the max-Sharpe portfolio
    pr = r @ best_w
    hist_var, hist_cvar = historical_var(pr), historical_cvar(pr)
    param_var = -(pr.mean() + norm.ppf(0.05) * pr.std())
    print(f"1-day 95% VaR on ${PORTFOLIO_VALUE:,}: historical ${hist_var * PORTFOLIO_VALUE:,.0f}, "
          f"parametric ${param_var * PORTFOLIO_VALUE:,.0f}; CVaR ${hist_cvar * PORTFOLIO_VALUE:,.0f}")

    # Backtest: rolling 1-year VaR, count breaches (expect ~5%)
    rolling_var = -pr.rolling(TRADING_DAYS).quantile(0.05).shift(1)
    test = pd.DataFrame({"ret": pr, "var": rolling_var}).dropna()
    breaches = (test["ret"] < -test["var"]).mean()
    print(f"VaR backtest: breach rate {breaches:.2%} (target 5.00%)")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(test.index, test["ret"], color="#b5b4ad", lw=0.8, label="Daily return")
    ax.plot(test.index, -test["var"], color=SERIES[0], label="Rolling 95% VaR")
    hit = test[test["ret"] < -test["var"]]
    ax.scatter(hit.index, hit["ret"], color=SERIES[7], s=10, label="Breach", zorder=3)
    ax.set(title="Rolling VaR backtest - max-Sharpe portfolio", ylabel="Daily return")
    ax.legend(ncol=3, loc="lower left")
    save(fig, "portfolio_var_backtest.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, col in enumerate(prices.columns):
        ax.plot(prices.index, prices[col] / prices[col].iloc[0] * 100, color=SERIES[i], label=col)
    ax.set(title="Growth of 100 invested", ylabel="Indexed value")
    ax.legend(ncol=3)
    save(fig, "portfolio_growth.png")


if __name__ == "__main__":
    main()
