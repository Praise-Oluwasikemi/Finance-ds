"""Checks for the exercises.

By default these run against ``solutions.py``. Run with ``--practice`` to grade
your own answers in ``practice.py``.
"""

import importlib

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def ex(request):
    name = "practice" if request.config.getoption("--practice") else "solutions"
    return importlib.import_module(f"exercises.{name}")


def test_future_value(ex):
    assert ex.future_value(1000, 0.05, 10) == pytest.approx(1647.01, abs=0.01)
    assert ex.future_value(1000, 0.05, 10, compounds_per_year=1) == pytest.approx(1628.89, abs=0.01)


def test_monthly_payment(ex):
    assert ex.monthly_payment(200_000, 0.06, 30) == pytest.approx(1199.10, abs=0.01)
    assert ex.monthly_payment(12_000, 0.0, 1) == pytest.approx(1000.0)


def test_amortization_schedule(ex):
    sched = ex.amortization_schedule(10_000, 0.05, 2)
    assert len(sched) == 24
    assert list(sched.columns) == ["month", "payment", "interest", "principal", "balance"]
    assert sched["balance"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    assert sched["principal"].sum() == pytest.approx(10_000)
    assert sched["interest"].is_monotonic_decreasing


def test_monthly_returns(ex):
    idx = pd.bdate_range("2024-01-01", "2024-03-31")
    prices = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
    r = ex.monthly_returns(prices)
    assert len(r) == 2
    assert r.iloc[0] == pytest.approx(prices["2024-02"].iloc[-1] / prices["2024-01"].iloc[-1] - 1)


def test_rolling_volatility(ex):
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100))
    vol = ex.rolling_volatility(r, window=21)
    assert vol.iloc[:20].isna().all()
    assert vol.iloc[20] == pytest.approx(r.iloc[:21].std() * np.sqrt(252))


def test_clean_transactions(ex):
    raw = pd.DataFrame(
        {
            "txn_id": [1, 2, 2, 3, 4, 5],
            "amount": ["$1,200.50", "30", "30", None, "-5", "$15"],
            "date": ["2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"],
            "category": [" Grocery", "FUEL ", "FUEL ", "travel", "online", None],
        }
    )
    out = ex.clean_transactions(raw)
    assert out["txn_id"].tolist() == [1, 2, 5]
    assert out["amount"].tolist() == [1200.50, 30.0, 15.0]
    assert out["category"].tolist() == ["grocery", "fuel", "unknown"]
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert out.index.tolist() == [0, 1, 2]


def test_default_rate_by_band(ex):
    loans = pd.DataFrame({"credit_score": [500, 550, 600, 700, 820], "default": [1, 0, 1, 0, 0]})
    out = ex.default_rate_by_band(loans)
    assert out["n_loans"].tolist() == [2, 1, 1, 0, 1]
    assert out["default_rate"].iloc[0] == pytest.approx(0.5)


def test_simulate_price_paths(ex):
    paths = ex.simulate_price_paths(100, 0.08, 0.2, days=252, n_paths=2000, seed=1)
    assert paths.shape == (253, 2000)
    assert np.all(paths[0] == 100)
    assert (paths > 0).all()
    # Mean terminal price should be near 100 * e^0.08
    assert paths[-1].mean() == pytest.approx(100 * np.exp(0.08), rel=0.03)
