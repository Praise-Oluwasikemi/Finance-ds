import numpy as np
import pandas as pd
import pytest

from fintools import metrics


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0005, 0.01, 1000))


def test_simple_and_log_returns():
    prices = pd.Series([100.0, 110.0, 99.0])
    assert metrics.simple_returns(prices).tolist() == pytest.approx([0.10, -0.10])
    assert metrics.log_returns(prices).iloc[0] == pytest.approx(np.log(1.1))


def test_annualized_return_constant_growth():
    r = pd.Series([0.01] * 12)
    assert metrics.annualized_return(r, periods=12) == pytest.approx(1.01**12 - 1)


def test_volatility_scales_with_sqrt_time(returns):
    assert metrics.annualized_volatility(returns) == pytest.approx(returns.std() * np.sqrt(252))


def test_sharpe_and_sortino_positive_for_positive_drift(returns):
    assert metrics.sharpe_ratio(returns) > 0
    assert metrics.sortino_ratio(returns) > metrics.sharpe_ratio(returns)


def test_max_drawdown_known_path():
    # 100 -> 120 -> 90 -> 108: worst drop is 120 to 90, i.e. -25%
    r = [0.20, -0.25, 0.20]
    assert metrics.max_drawdown(r) == pytest.approx(-0.25)


def test_beta_of_scaled_series():
    m = np.random.default_rng(1).normal(0, 0.01, 500)
    assert metrics.beta(1.5 * m, m) == pytest.approx(1.5)


def test_var_ordering(returns):
    var = metrics.historical_var(returns, 0.95)
    assert var > 0
    assert metrics.conditional_var(returns, 0.95) >= var
    assert metrics.historical_var(returns, 0.99) > var
    assert metrics.parametric_var(returns, 0.95) == pytest.approx(var, rel=0.15)


def test_portfolio_returns_requires_full_weights():
    r = pd.DataFrame({"a": [0.01, 0.02], "b": [0.03, -0.01]})
    assert metrics.portfolio_returns(r, [0.5, 0.5]).tolist() == pytest.approx([0.02, 0.005])
    with pytest.raises(ValueError):
        metrics.portfolio_returns(r, [0.5, 0.4])


def test_ks_and_gini_perfect_separation():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert metrics.ks_statistic(y, s) == pytest.approx(1.0)
    assert metrics.gini(y, s) == pytest.approx(1.0)


def test_psi_detects_shift():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 5000)
    assert metrics.population_stability_index(base, rng.normal(0, 1, 5000)) < 0.1
    assert metrics.population_stability_index(base, rng.normal(1, 1, 5000)) > 0.25


def test_precision_at_k():
    y = [1, 0, 1, 0]
    s = [0.9, 0.8, 0.7, 0.1]
    assert metrics.precision_at_k(y, s, 2) == pytest.approx(0.5)


def test_expected_loss():
    assert metrics.expected_loss(0.02, 0.45, 10_000) == pytest.approx(90.0)
