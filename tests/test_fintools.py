import numpy as np
import pandas as pd
import pytest

from fintools import (amortization_schedule, historical_cvar, historical_var, irr, ks_statistic,
                      loan_payment, max_drawdown, npv, sharpe_ratio)


def test_amortization_pays_off_loan():
    sched = amortization_schedule(10_000, 0.07, 24)
    assert sched["balance"].iloc[-1] == pytest.approx(0, abs=1e-6)
    assert sched["principal"].sum() == pytest.approx(10_000)
    assert sched["payment"].iloc[0] == pytest.approx(loan_payment(10_000, 0.07, 24))


def test_irr_zeroes_npv():
    cf = [-1000, 300, 400, 500]
    assert npv(irr(cf), cf) == pytest.approx(0, abs=1e-6)


def test_cvar_at_least_var():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 5000))
    assert historical_cvar(r) >= historical_var(r) > 0


def test_max_drawdown_and_sharpe():
    r = pd.Series([0.1, -0.5, 0.2])
    assert max_drawdown(r) == pytest.approx(-0.5)
    assert sharpe_ratio(pd.Series([0.01, 0.02, 0.0])) > 0


def test_ks_perfect_separation():
    assert ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
