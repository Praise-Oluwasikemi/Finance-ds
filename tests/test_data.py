import pandas as pd

from data import generate_data as gd


def test_loans_shape_and_target():
    df = gd.make_loans(n=1000)
    assert len(df) == 1000
    assert set(df["default"].unique()) <= {0, 1}
    assert 0.05 < df["default"].mean() < 0.35
    assert df["credit_score"].between(300, 850).all()
    assert df.isna().sum().sum() == 0


def test_generators_are_reproducible():
    pd.testing.assert_frame_equal(gd.make_loans(n=200, seed=1), gd.make_loans(n=200, seed=1))
    pd.testing.assert_frame_equal(gd.make_customers(n=200, seed=1), gd.make_customers(n=200, seed=1))


def test_prices_are_positive_business_days():
    prices = gd.make_prices(n_days=100)
    assert prices.shape == (100, 6)
    assert (prices > 0).all().all()
    assert prices.index.dayofweek.max() < 5


def test_transactions_fraud_rate():
    df = gd.make_transactions(n=5000, fraud_rate=0.02)
    assert df["is_fraud"].sum() == 100
    assert df["hour"].between(0, 23).all()


def test_customers_columns():
    df = gd.make_customers(n=500)
    assert df["digital_share"].between(0, 1).all()
    assert df["true_segment"].nunique() == 4
