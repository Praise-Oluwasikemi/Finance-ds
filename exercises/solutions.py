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


# ---------------- Level 4: statistical thinking ----------------
def mean_confidence_interval(x, level=0.95):
    from scipy import stats
    x = np.asarray(x, dtype=float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    margin = stats.t.ppf((1 + level) / 2, len(x) - 1) * se
    return x.mean() - margin, x.mean() + margin


def two_proportion_ztest(successes_a, n_a, successes_b, n_b):
    from scipy import stats
    p_a, p_b = successes_a / n_a, successes_b / n_b
    pooled = (successes_a + successes_b) / (n_a + n_b)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_a - p_b) / se
    return z, 2 * stats.norm.sf(abs(z))


def bootstrap_ci(x, stat=np.median, n_boot=5_000, level=0.95, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    boots = np.array([stat(rng.choice(x, len(x), replace=True)) for _ in range(n_boot)])
    alpha = (1 - level) / 2
    return np.quantile(boots, alpha), np.quantile(boots, 1 - alpha)


# ---------------- Level 5: business intelligence ----------------
def loan_book_kpis(loans):
    return {
        "loans": len(loans),
        "total_disbursed": loans["loan_amount"].sum(),
        "avg_loan": loans["loan_amount"].mean(),
        "default_rate": loans["default"].mean(),
        "npl_ratio": loans.loc[loans["default"] == 1, "loan_amount"].sum() / loans["loan_amount"].sum(),
    }


def month_over_month_growth(tx):
    monthly = tx.groupby(pd.to_datetime(tx["timestamp"]).dt.to_period("M"))["amount"].sum()
    return monthly.pct_change()


def category_share_sql(conn):
    query = """
        SELECT category,
               SUM(amount) AS total,
               SUM(amount) * 1.0 / (SELECT SUM(amount) FROM transactions) AS share
        FROM transactions
        GROUP BY category
        ORDER BY total DESC
    """
    return pd.read_sql_query(query, conn)
