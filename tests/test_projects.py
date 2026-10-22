"""Smoke tests: each project runs end to end and finds real signal."""

import numpy as np
import pytest

from projects.p01_credit_risk import analysis as p01
from projects.p02_portfolio_risk import analysis as p02
from projects.p03_fraud_detection import analysis as p03
from projects.p04_customer_segmentation import analysis as p04


def test_credit_risk(tmp_path):
    res = p01.run(out_dir=tmp_path)
    for scores in res["models"].values():
        assert scores["auc"] > 0.7
        assert scores["gini"] == pytest.approx(2 * scores["auc"] - 1)
    table = res["approval_table"]
    assert table["approval_rate"].is_monotonic_increasing
    assert table["bad_rate_approved"].iloc[0] < res["default_rate"]
    assert (tmp_path / "p01_roc.png").exists()


def test_portfolio_risk(tmp_path):
    res = p02.run(out_dir=tmp_path)
    weights = res["weights"]
    assert np.allclose(weights.sum(), 1.0)
    assert (weights >= 0).all().all()
    summary = res["summary"]
    assert summary.loc["min_variance", "annual_volatility"] <= summary["annual_volatility"].min() + 1e-9
    assert summary.loc["max_sharpe", "sharpe"] >= summary["sharpe"].max() - 1e-6
    assert 0.02 < res["var_breach_rate"] < 0.10
    assert (tmp_path / "p02_growth.png").exists()


def test_fraud_detection(tmp_path):
    res = p03.run(out_dir=tmp_path)
    best = res["models"][res["best_model"]]
    assert best["avg_precision"] > 5 * res["fraud_rate"]
    budget = res["review_budget"]
    assert budget["recall"].is_monotonic_increasing
    assert (tmp_path / "p03_pr_curve.png").exists()


def test_customer_segmentation(tmp_path):
    res = p04.run(out_dir=tmp_path)
    assert 2 <= res["best_k"] <= 8
    assert res["profile"]["customers"].sum() == 3000
    assert res["adjusted_rand_index"] > 0.3
    assert (tmp_path / "p04_segments.png").exists()
