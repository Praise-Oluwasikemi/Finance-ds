# P02 Portfolio risk

**Question:** How much could a portfolio of financial-sector assets lose on a bad day, and can a better allocation reduce that risk?

**Data:** Three years of synthetic daily prices for two banks, an insurer, a fintech, a bond ETF and a market index, generated with a one-factor model so assets move together realistically.

**Approach**
1. Compute daily returns and compare three long-only portfolios: equal weight, minimum variance and maximum Sharpe ratio, with each asset capped at 40%.
2. Report annual return and volatility, Sharpe and Sortino ratios, maximum drawdown and beta to the market.
3. Measure one-day 95% Value at Risk (historical and parametric) and Expected Shortfall.
4. Backtest a rolling 250-day historical VaR and count how often losses breached it.

**Result:** The minimum-variance portfolio cuts annual volatility from 18% to 12%, daily 95% VaR from 1.8% to 1.2% and beta from 0.93 to 0.56. The VaR model is breached on 5.7% of days, close to the 5% it promises. The max-Sharpe weights are fitted on the same data they are scored on, so its Sharpe ratio is optimistic; a walk-forward test is the natural next step.

**Run:** `python -m projects.p02_portfolio_risk.analysis`
