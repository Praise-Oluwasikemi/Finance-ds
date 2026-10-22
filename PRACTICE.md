# Practice Guide: one session at a time

This guide is for working through the repo slowly, one sitting at a time, and building real skill and confidence.
There are **12 sessions of about 60–90 minutes each**. At two or three sessions a week, that is about a month.
Each session builds on the last, so do them in order. Stopping in the middle of one is fine, because Git saves your progress.

What you'll practise and be able to show at the end:

| Skill | Where it shows up |
|---|---|
| **Python fluency** | 20 graded exercises, from simple maths to SQL |
| **Statistical thinking** | Confidence intervals, hypothesis tests, bootstrap, risk quantiles, and saying how sure you are |
| **Business intelligence** | KPIs, month-on-month trends, SQL, a dashboard, decision memos written for managers |
| **Confidence** | A learning log of what you can now do that you couldn't before |

---

## Session 0: Set up your machine (about 30 min, do this once)

1. Install **Python 3.12+** from [python.org](https://www.python.org/downloads/). On Windows, tick **"Add python.exe to PATH"**.
2. Install **Git** ([git-scm.com](https://git-scm.com/downloads)) and **VS Code** with the **Python** and **Jupyter** extensions.
3. Open a terminal (Git Bash on Windows) and run:

```bash
git clone https://github.com/Praise-Oluwasikemi/Finance-ds.git
cd Finance-ds
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pytest tests/test_exercises.py     # you should see "20 failed": that's your to-do list
```

4. Open the folder in VS Code (`code .`). Pick the `.venv` interpreter when VS Code asks.

> **No laptop handy?** In Google Colab, run `!git clone https://github.com/Praise-Oluwasikemi/Finance-ds.git`,
> then `%cd Finance-ds`, `!pip install -r requirements.txt` and `!pytest tests/test_exercises.py -k compound`.
> Edit `exercises/practice.py` in Colab's file panel. Colab doesn't save back to GitHub, so treat it as a backup option.

---

## How every session runs

| Step | Time | What you do |
|---|---|---|
| 1. Warm up | 5 min | Read your last entry in [`journal/LEARNING_LOG.md`](journal/LEARNING_LOG.md) |
| 2. Exercises | 30–40 min | Solve the session's exercises in `exercises/practice.py` and grade **one at a time**: `pytest tests/test_exercises.py -k npv` |
| 3. Explore | 20 min | Run a project, change **one** thing, and note what happens |
| 4. Think | 10 min | Answer the session's statistics or business question in your log, in plain English |
| 5. Save | 5 min | Commit and push (commands below) |

```bash
git add -A
git commit -m "Session 3: credit band default rates"
git push
```

Your commit history becomes public proof that you practise steadily, and recruiters do look at it.

### When you're stuck (in this order)
1. Read the error **from the bottom up**, because the last line says what went wrong.
2. `print()` the intermediate values. Most bugs are a wrong shape or type, or a NaN.
3. Re-read the docstring hint.
4. Give it **20 honest minutes**. Struggling is when the learning happens.
5. Peek at `exercises/solutions.py`, **close it**, then write the answer yourself. Do that exercise again from memory next session.
6. Ask an AI tutor for a **hint**, not the answer: *"Don't give me code. What concept am I missing?"*

---

## The 12 sessions

### Part A: Python foundations (Sessions 1–5)

**Session 1: The maths of money**
- Exercises: `compound_interest`, `monthly_payment`, `npv`
- Explore: in a notebook, run `from fintools import amortization_schedule` and then `amortization_schedule(5_000_000, 0.24, 36)`. How much interest is paid in total? Plot the balance over time.
- Think: *A customer asks for 60 months instead of 36. What happens to the monthly payment, and to the bank's total interest income? Why might the bank still say no?*

**Session 2: Cleaning real-world banking data**
- Exercises: `clean_transactions`, `monthly_spend`
- Explore: `pd.read_csv("data/transactions.csv")`, then `.info()`, `.describe()`, and `value_counts()` on `category`.
- Think: *Card spend has a long right tail. Compare its mean and median. Which one would you report to a manager as the "typical" transaction, and why?*

**Session 3: Customers and credit bands**
- Exercises: `top_customers`, `default_rate_by_band`
- Explore: run `python projects/p01_credit_risk.py` and read the Information Value table. Plot default rate by credit band as a bar chart.
- Think: *Default rate climbs as credit score falls. Is that proof that a low score **causes** default? What else could explain it?*

**Session 4: Returns and volatility**
- Exercises: `daily_returns`, `cumulative_return`, `rolling_volatility`
- Explore: run `projects/p02_portfolio_risk.py` and open `reports/figures/portfolio_growth.png`.
- Think: *Why is +10% followed by -10% a loss? What does that mean for a customer whose savings swing up and down?*

**Session 5: Measuring risk**
- Exercises: `max_drawdown`, `historical_var`, `expected_loss`, `misclassification_cost`
- Explore: in `p02`, change VaR from 95% to 99% (`historical_var(pr, 0.99)`). How much bigger is the worst-day loss?
- Think: *Explain "1-day 95% VaR of ₦2.96m" to a non-technical board member in two sentences. What does VaR NOT tell them?*

### Part B: Statistical thinking (Sessions 6–7)

**Session 6: Is the difference real?**
- Exercises: `mean_confidence_interval`, `two_proportion_ztest`
- Explore: in `data/loans.csv`, compare the default rate of RENT and MORTGAGE customers using your z-test. Is the gap statistically significant?
- Think: *Statistically significant is not the same as important for the business. With 20,000 loans even tiny gaps come out "significant", so how big would a gap need to be before you changed lending policy?*

**Session 7: Uncertainty without formulas**
- Exercises: `bootstrap_ci`, `loan_book_kpis`
- Explore: bootstrap the **median** income of defaulters and of non-defaulters. Do the two intervals overlap?
- Think: *A manager wants one number. Why is "median income ₦3.63m (95% CI ₦3.60m–₦3.67m)" a better answer than "₦3.63m"?*

### Part C: Business intelligence (Sessions 8–9)

**Session 8: SQL and trends**
- Exercises: `month_over_month_growth`, `category_share_sql`
- Explore: load `transactions.csv` into SQLite as you did in your TS Academy capstone, then write **three SQL queries of your own**. Example: fraud rate by hour of day.
- Think: *December spend is up 12% month on month. Is that good news? What would you compare it with first? (Hint: seasonality, last year.)*

**Session 9: Build a dashboard**
- Build a one-page dashboard from `loans.csv` and `transactions.csv` in **Power BI, Excel, or Streamlit** (whichever you know best).
- It needs 4 KPI cards (loans, ₦ disbursed, default rate, NPL ratio), default rate by credit band, monthly spend trend, and a filter for `purpose`.
- Save a screenshot to `reports/figures/` and link it in the README.

### Part D: Think like a data scientist in a bank (Sessions 10–12)

**Session 10: Own the credit project**
- Re-read `projects/p01_credit_risk.py` and write a comment in your own words above each block.
- Change `LGD` (0.45 → 0.60) and `MARGIN` (0.08 → 0.12). How does the best cut-off move? Make a small table.
- Write a **half-page decision memo** in `journal/`: what you recommend, the ₦ impact, the risks, and what you'd test next.

**Session 11: Trade-offs in fraud**
- In `p03`, change `DAILY_REVIEW_CAPACITY` to 1, 3, 5 and 10. Record precision, recall and ₦ intercepted each time.
- Think: *Each analyst costs ₦X a month. At what point does hiring another analyst pay for itself?* Make a reasonable estimate and show your working.

**Session 12: Ship it and tell people**
- Pick your favourite project and write a 150-word LinkedIn post: the problem, what you did, the ₦ result, and one lesson.
- Pin `Finance-ds` on your GitHub profile. Then apply the same polish to your capstone README by adding the results and metrics your group achieved.

---

## After session 12

Pick the next items from the "Practice roadmap" in the [README](README.md). Swapping in real data and building a scorecard are good next steps.
Then repeat the exercises you had to peek at. When you can solve all 20 without the solutions, you know this material.
