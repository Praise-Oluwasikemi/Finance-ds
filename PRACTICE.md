# Practice Guide

**Goal:** write the code yourself, think like a statistician, communicate like a business analyst, and become confident enough to show your work.

You've already been through the whole TS Academy curriculum once. This guide takes you through it a second time with **you at the keyboard**. Each build matches one or more TS modules and is set in a Nigerian bank. The last build moves outside finance, to show that your skills transfer.

---

## Where does the work happen?

| Place | What happens there |
|---|---|
| **Your laptop: Anaconda + Jupyter Notebook** | Where you **write and run all your code**. This is the main place. |
| **GitHub (this repo)** | Where your work is **saved and shown**. You push after every session, and the commit history becomes your public track record. |
| **Chat with Claude** | Your **tutor and reviewer**: explanations, hints, reviews of code you wrote, and help when you're stuck. Not where code gets written for you. |

VS Code is optional. Jupyter Notebook is all you need. You can try VS Code later, once you're comfortable.

---

## Using AI the right way: tutor, not ghostwriter

At first, having AI write your code feels like progress, but you can't defend that code in an interview or reuse it at work. These rules make sure the skill ends up in **your** head.

**Never ask for the code itself.** Ask for one of these three kinds of help, in this order:

| Level | When | Example prompt |
|---|---|---|
| 1. **Hint** | Stuck for 20+ minutes | *"I'm doing Build 1, step 4. I want the default rate per purpose. Don't give me code. Which pandas method should I look at?"* |
| 2. **Explain a concept** | You don't understand *why* | *"Explain the chi-square test as if I were a bank manager. When would I use it instead of a t-test?"* |
| 3. **Review my code** | After you have something that runs | *"Here's my code for step 5. Don't rewrite it. Tell me what's wrong or could be better, and why."* |

Paste this at the start of a chat to set the rule:

> I'm practising data science. Act as my tutor. Don't write solution code for me. Give hints, ask me guiding questions, explain concepts, and review code I've written. If I ask for code, remind me of this rule.

**Be honest about credit.** In interviews, say what you built yourself and where you had help. "I wrote it, then had it reviewed and improved X" is a strong answer.

---

## Session 0: Set up your laptop (about 45 min, once)

You already have Anaconda, so you need three more things.

**1. Install GitHub Desktop** ([desktop.github.com](https://desktop.github.com)) and sign in with your GitHub account. It's the easiest way to save work to GitHub without typing Git commands.

**2. Clone this repo.** In GitHub Desktop, choose **File → Clone repository**, select `Praise-Oluwasikemi/Finance-ds`, and pick a folder (e.g. `Documents\GitHub\Finance-ds`).

**3. Create a clean Python environment.** Open **Anaconda Prompt** and run:

```bash
conda create -n finance-ds python=3.12 pandas numpy scipy scikit-learn matplotlib seaborn statsmodels notebook pytest -y
conda activate finance-ds
pip install imbalanced-learn streamlit shap
```

**4. Open Jupyter in the repo folder:**

```bash
conda activate finance-ds
cd Documents\GitHub\Finance-ds
jupyter notebook
```

From now on, these three lines start every session.

**5.** In Jupyter, open **`notebooks/00_start_here.ipynb`** and run every cell. It checks your setup and teaches you the grader.

**Saving to GitHub after each session:** open GitHub Desktop. It lists the files you changed. Type a short summary (e.g. *"Workbook 1: finance maths done"*), click **Commit to main**, then **Push origin**.

> Before committing a notebook, run **Kernel → Restart & Run All** so the saved outputs are clean and in order.

---

## The rhythm of every session (60–90 min)

| Step | Time | What you do |
|---|---|---|
| 1. Warm up | 5 min | Read your last [learning-log](journal/LEARNING_LOG.md) entry. Redo one exercise from memory. |
| 2. Write code | 40–60 min | Work through the next workbook exercises or build steps. |
| 3. Think | 10 min | Answer the ✍️ questions in your own words. |
| 4. Log & push | 5–10 min | Add a log entry, tick off skills in the [checklist](journal/SKILLS_CHECKLIST.md), commit and push. |

Two or three sessions a week is plenty. Being consistent matters more than doing a lot in one go.

### When you're stuck (in this order)
1. Read the error **from the bottom up**. The last line tells you what went wrong.
2. `print()` the intermediate values. Check `.shape`, `.dtypes` and `.head()`.
3. Search the error message, or read the docs for the function.
4. Give it **20 honest minutes**.
5. Ask Claude for a **level 1 hint**.
6. Only then look at `exercises/solutions.py` (for workbooks). Close it, write the answer yourself, and redo it next session.

---

## The roadmap

### Phase 1: Fluency (Sessions 1–6)
Five workbooks in `notebooks/exercises/`. Each has short functions to write and a `check()` cell that grades you instantly.

| Session | Workbook | TS module |
|---|---|---|
| 1 | `1_finance_maths`: interest, loan instalments, NPV | 1 |
| 2–3 | `2_pandas_wrangling`: cleaning, pivots, grouping | 1 |
| 4 | `3_market_credit_risk`: returns, volatility, VaR, expected loss | 1–2 |
| 5 | `4_statistical_thinking`: confidence intervals, z-test, bootstrap | 2 |
| 6 | `5_business_intelligence`: KPIs, growth, SQL | 4 |

### Phase 2: Builds (one portfolio project at a time)
Each build takes 2–4 sessions. B1 and B2 have guided notebooks ready in `notebooks/builds/`. For the others, start a new notebook from the brief below, or ask Claude to create a guided notebook in the same style (questions and blank cells, no solutions).

**B1: Who defaults? EDA & statistics** · TS 1–2 · ready
Clean the data, explore it, run chi-square, Mann-Whitney and z-tests with confidence intervals, and write an executive summary.

**B2: Card spending with SQL + pandas** · TS 4 · ready
Use SELECT, GROUP BY/HAVING, INNER and LEFT JOINs, subqueries and time trends. Find when fraud happens, then write a memo to the Head of Cards.

**B3: Credit default model** · TS 3, 5, 7
- Baseline logistic regression, then a decision tree, then a random forest, all inside a scikit-learn `Pipeline`.
- 5-fold cross-validation and `GridSearchCV`. Use a learning curve to discuss bias versus variance.
- Metrics: confusion matrix, ROC-AUC and F1. Then the business metric: **₦ profit at each approval cut-off**.
- *Deliverable:* a recommended cut-off with its ₦ impact.

**B4: Fraud detection** · TS 6
- Fraud is only 0.8% of transactions. Show why accuracy is misleading.
- Compare KNN, logistic regression and random forest, then class weights versus SMOTE.
- Use a precision-recall curve to choose an alert threshold that matches the team's review capacity.
- *Deliverable:* "with N alerts a day we catch X% of fraud, ₦Y saved".

**B5: Customer segmentation** · TS 8
- Build customer features (recency, frequency, monetary value, category mix, state, app use).
- Run K-Means and hierarchical clustering. Choose k with the elbow method and silhouette scores. Visualise with PCA.
- *Deliverable:* named personas, each with a recommended product offer.

**B6: Spending forecast** · TS 9
- Daily or weekly card volume: trend, seasonality and residuals.
- Lag and rolling features, then ARIMA.
- Beat a naive baseline on MAE/RMSE.
- *Deliverable:* next quarter's forecast with an uncertainty band.

**B7: Explainable & fair credit model** · TS 11
- SHAP (global and per-applicant) and LIME on your B3 model.
- Check approval rates across age groups and discuss fairness.
- *Deliverable:* a one-page model card (purpose, data, performance, limits, fairness).

**B8: Loan decision app** · TS 10
- Save the B3 pipeline with `joblib`.
- A Streamlit app: enter applicant details and see the probability of default, the decision, and the top reasons.
- Deploy it free on Streamlit Community Cloud and link it in the README.
- This is also your **dashboard** skill: add a second page of loan-book KPI charts.

**B9: HR attrition, solo rebuild** · TS 12
Rebuild your capstone **alone, from a blank notebook**, using the IBM HR dataset. You know the problem; now prove you can solve it yourself. Then compare it with the group version. It shows employers your skills work outside finance too.

### After each build
1. Run **Restart & Run All**, then commit and push.
2. Compare your work with the matching file in `worked_examples/` (B3 → `p01`, B4 → `p03`, B5 → `p04`). Note one thing you did better and one thing worth adopting.
3. Update the status column in the README (⬜ → ✅).
4. Every second build, write a short LinkedIn post: the problem, what you did, the ₦ result, and one lesson.

At two or three sessions a week, the whole roadmap takes about **4 months**. That isn't slow. It's about how long it takes to genuinely learn this material.
