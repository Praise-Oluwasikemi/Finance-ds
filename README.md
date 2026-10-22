# Finance-ds
This repository contains four projects that answers practical business questions and my approach to answering them as a data scientist

Each project starts from a question a bank or investment team would actually ask, builds a model or analysis to answer it, and ends with a number someone could act on. The data is synthetic (generated in `data/generate_data.py`) so everything runs offline and is reproducible, but it is built with realistic relationships between variables.

## Projects

| # | Project | Business question | Methods | Headline result |
|---|---------|-------------------|---------|-----------------|
| 01 | [Credit risk](projects/p01_credit_risk/) | Which loan applicants are likely to default, and where should we set the approval cut-off? | Logistic regression scorecard, gradient boosting, AUC / Gini / KS, expected loss | AUC 0.76, Gini 0.51. A 10% PD cut-off approves 46% of applicants with a 5.6% bad rate, versus 14.7% overall |
| 02 | [Portfolio risk](projects/p02_portfolio_risk/) | How much could this portfolio lose on a bad day, and can we build a better one? | Mean-variance optimization, historical and parametric VaR, Expected Shortfall, drawdown, VaR backtest | Min-variance portfolio cuts volatility from 18% to 12% and 95% VaR from 1.8% to 1.2% a day. VaR backtest breaches 5.7% of days (target 5%) |
| 03 | [Fraud detection](projects/p03_fraud_detection/) | Which transactions should a fraud team with limited capacity review first? | Class-weighted logistic regression and random forest, precision-recall, precision at k | With 2% fraud, reviewing the top 50 alerts is 78% precise. Average precision 0.49, about 24x the base rate |
| 04 | [Customer segmentation](projects/p04_customer_segmentation/) | What distinct groups of customers does the bank have, and how should we treat them? | Feature scaling, K-means, silhouette analysis, PCA, segment profiling | Three actionable segments: digital everyday bankers, retired savers, and affluent multi-product customers |

## Repository layout

```
data/generate_data.py      synthetic loans, prices, transactions and customers
fintools/metrics.py        returns, Sharpe, Sortino, drawdown, VaR, CVaR, beta, KS, Gini, PSI
fintools/plotting.py       shared charts (ROC, PR curve, drawdown, correlation heatmap)
projects/p01..p04/         one analysis.py and README.md per project
exercises/practice.py      Python, pandas and numpy practice problems for finance
exercises/solutions.py     worked answers
tests/                     pytest suite for the tools, data, projects and exercises
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m data.generate_data                     # optional: write CSVs to data/raw/
python -m projects.p01_credit_risk.analysis      # each project prints results and saves charts to outputs/
python -m pytest                                 # run the test suite
```

## Practice exercises

`exercises/practice.py` has stub functions covering compound interest, loan amortization, resampling prices to monthly returns, rolling volatility, cleaning messy transaction data, groupby default rates and Monte Carlo price simulation. Fill them in and grade yourself with:

```bash
python -m pytest tests/test_exercises.py --practice
```
