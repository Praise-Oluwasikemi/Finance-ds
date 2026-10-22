"""Auto-grader for exercises/practice.py.

    pytest tests/test_exercises.py -x                         # grade your answers
    EXERCISES=solutions pytest tests/test_exercises.py        # check the reference solutions
"""
import importlib
import os

import numpy as np
import pandas as pd
import pytest

ex = importlib.import_module(f"exercises.{os.environ.get('EXERCISES', 'practice')}")


@pytest.fixture
def raw_tx():
    return pd.DataFrame({
        "tx_id": [3, 1, 2, 2, 4],
        "customer_id": [10, 10, 20, 20, 20],
        "timestamp": ["2025-02-03", "2025-01-15", "2025-01-20", "2025-01-20", "2025-02-10"],
        "amount": [5_000.0, np.nan, 3_000.0, 3_000.0, 7_000.0],
        "category": ["grocery", "grocery", "fuel", "fuel", "grocery"],
    })


# ---- Level 1
def test_compound_interest():
    assert ex.compound_interest(1_000_000, 0.18, 5, 1) == pytest.approx(2_287_757.76, abs=0.01)
    assert ex.compound_interest(1_000_000, 0.18, 5) == pytest.approx(2_443_219.78, abs=0.01)


def test_monthly_payment():
    assert ex.monthly_payment(5_000_000, 0.24, 36) == pytest.approx(196_164.26, abs=0.01)
    assert ex.monthly_payment(1_200_000, 0.0, 12) == pytest.approx(100_000)


def test_npv():
    assert ex.npv(0.2, [-10_000_000, 6_000_000, 7_000_000]) == pytest.approx(-138_888.89, abs=0.01)


# ---- Level 2
def test_clean_transactions(raw_tx):
    before = raw_tx.copy()
    out = ex.clean_transactions(raw_tx)
    pd.testing.assert_frame_equal(raw_tx, before)  # input untouched
    assert list(out["tx_id"]) == [1, 2, 3, 4]
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])
    assert out.loc[out.tx_id == 1, "amount"].item() == pytest.approx(6_000.0)  # grocery median of 5,000 and 7,000


def test_monthly_spend(raw_tx):
    out = ex.monthly_spend(raw_tx.dropna().drop_duplicates("tx_id"))
    assert out.loc[20, pd.Period("2025-01", "M")] == 3_000
    assert out.loc[20, pd.Period("2025-02", "M")] == 7_000
    assert out.loc[10, pd.Period("2025-01", "M")] == 0


def test_top_customers(raw_tx):
    out = ex.top_customers(raw_tx.fillna(0), n=1)
    assert out.index.tolist() == [20] and out.iloc[0] == 13_000


def test_default_rate_by_band():
    loans = pd.DataFrame({"credit_score": [500, 579, 580, 700, 820, 820],
                          "default": [1, 1, 0, 0, 0, 1]})
    out = ex.default_rate_by_band(loans)
    assert out["Poor"] == 1.0 and out["Fair"] == 0.0 and out["Excellent"] == 0.5
    assert np.isnan(out["Very good"])


# ---- Level 3
PRICES = pd.Series([100, 110, 99, 120, 90.0])


def test_daily_returns():
    r = ex.daily_returns(PRICES)
    assert len(r) == 4 and r.iloc[0] == pytest.approx(0.10) and r.iloc[1] == pytest.approx(-0.10)


def test_cumulative_return():
    assert ex.cumulative_return(pd.Series([0.1, -0.1])) == pytest.approx(-0.01)


def test_rolling_volatility():
    r = pd.Series([0.01, -0.01] * 20)
    vol = ex.rolling_volatility(r, window=10)
    assert vol.isna().sum() == 9
    assert vol.iloc[-1] == pytest.approx(r.iloc[-10:].std() * np.sqrt(252))


def test_max_drawdown():
    r = PRICES.pct_change().dropna()
    assert ex.max_drawdown(r) == pytest.approx(90 / 120 - 1)


def test_historical_var():
    r = pd.Series(np.linspace(-0.05, 0.05, 101))
    assert ex.historical_var(r, 0.95) == pytest.approx(0.045)


def test_expected_loss():
    assert ex.expected_loss(0.02, 0.45, 10_000_000) == pytest.approx(90_000)
    np.testing.assert_allclose(ex.expected_loss(np.array([0.1, 0.2]), 0.5, np.array([100, 100])), [5, 10])


def test_misclassification_cost():
    assert ex.misclassification_cost([0, 1, 1, 0], [1, 0, 1, 0], fp_cost=5, fn_cost=100) == 105


# ---- Level 4
def test_mean_confidence_interval():
    lo, hi = ex.mean_confidence_interval([10, 12, 14, 16, 18])
    assert (lo, hi) == (pytest.approx(10.073, abs=0.001), pytest.approx(17.927, abs=0.001))


def test_two_proportion_ztest():
    z, p = ex.two_proportion_ztest(120, 1000, 150, 1000)
    assert z == pytest.approx(-1.963, abs=0.001) and p == pytest.approx(0.0496, abs=0.0001)


def test_bootstrap_ci():
    x = np.random.default_rng(1).lognormal(15, 0.5, 400)
    lo, hi = ex.bootstrap_ci(x, n_boot=2_000)
    assert lo < np.median(x) < hi
    assert (hi - lo) / np.median(x) < 0.2
    assert ex.bootstrap_ci(x, n_boot=500, seed=3) == ex.bootstrap_ci(x, n_boot=500, seed=3)


# ---- Level 5
def test_loan_book_kpis():
    loans = pd.DataFrame({"loan_amount": [1_000_000, 3_000_000, 2_000_000, 2_000_000], "default": [0, 1, 0, 0]})
    k = ex.loan_book_kpis(loans)
    assert k["loans"] == 4 and k["total_disbursed"] == 8_000_000 and k["avg_loan"] == 2_000_000
    assert k["default_rate"] == 0.25 and k["npl_ratio"] == pytest.approx(0.375)


def test_month_over_month_growth(raw_tx):
    g = ex.month_over_month_growth(raw_tx.dropna().drop_duplicates("tx_id"))
    assert np.isnan(g.iloc[0])
    assert g[pd.Period("2025-02", "M")] == pytest.approx(12_000 / 3_000 - 1)


def test_category_share_sql(raw_tx):
    import sqlite3
    conn = sqlite3.connect(":memory:")
    raw_tx.dropna().drop_duplicates("tx_id").to_sql("transactions", conn, index=False)
    out = ex.category_share_sql(conn)
    assert list(out.columns) == ["category", "total", "share"]
    assert out["category"].tolist() == ["grocery", "fuel"]
    assert out["share"].tolist() == [pytest.approx(0.8), pytest.approx(0.2)]
