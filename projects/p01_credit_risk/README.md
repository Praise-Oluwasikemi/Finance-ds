# P01 Credit risk

**Question:** Which loan applicants are likely to default, and where should the lender set its approval cut-off?

**Data:** 5,000 synthetic loan applications with income, loan amount, term, credit score, employment length, debt-to-income, past delinquencies, home ownership and purpose. About 15% default.

**Approach**
1. Engineer affordability features: loan-to-income and monthly payment as a share of monthly income.
2. Fit a logistic regression scorecard (interpretable, the industry standard) and a gradient boosting model as a challenger.
3. Evaluate on a held-out 30% with AUC, Gini (2 x AUC - 1) and the KS statistic, the metrics credit teams use.
4. Turn predicted probability of default (PD) into an approval policy: for each PD cut-off, report approval rate, bad rate among approved loans and expected loss (PD x LGD x exposure, with LGD assumed at 45%).
5. Check population stability (PSI) of credit score between train and test.

**Result:** The logistic scorecard (AUC 0.76, Gini 0.51, KS 0.40) matches the gradient boosting model, so the simpler, explainable model wins. Approving applicants with PD under 10% accepts 46% of applicants with a 5.6% bad rate, against 14.7% if everyone were approved.

**Run:** `python -m projects.p01_credit_risk.analysis`
