"""Project 4 - Customer Segmentation (RFM + K-Means) for a retail bank.

Business question: how do we group cardholders so marketing can target
offers (premium card upgrade, re-activation, retention)?

Skills shown: aggregating transactions to customer level, RFM analysis,
scaling, choosing k with silhouette score, profiling and naming segments.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from matplotlib.ticker import FuncFormatter  # noqa: E402

from fintools import naira  # noqa: E402
from fintools.plotting import SERIES, apply_style, plt, save  # noqa: E402


def rfm_table(tx: pd.DataFrame) -> pd.DataFrame:
    snapshot = tx["timestamp"].max() + pd.Timedelta(days=1)
    rfm = tx.groupby("customer_id").agg(
        recency_days=("timestamp", lambda s: (snapshot - s.max()).days),
        frequency=("tx_id", "count"),
        monetary=("amount", "sum"),
        travel_share=("category", lambda s: (s == "travel").mean()),
        foreign_share=("is_foreign", "mean"),
    )
    return rfm


def main() -> None:
    apply_style()
    tx = pd.read_csv(ROOT / "data" / "transactions.csv", parse_dates=["timestamp"])
    tx = tx[tx["is_fraud"] == 0]  # segment genuine behaviour only

    # Give customers different activity patterns (synthetic data is too uniform otherwise)
    rng = np.random.default_rng(1)
    activity = pd.Series(rng.lognormal(0, 0.8, tx["customer_id"].max() + 1))
    tx = tx[rng.random(len(tx)) < np.clip(activity[tx["customer_id"]].values / 3, 0.05, 1)]

    rfm = rfm_table(tx)
    features = ["recency_days", "frequency", "monetary", "travel_share", "foreign_share"]
    X = StandardScaler().fit_transform(np.log1p(rfm[features]))

    sil = {k: silhouette_score(X, KMeans(k, n_init=10, random_state=0).fit_predict(X)) for k in range(2, 8)}
    k = max(sil, key=sil.get)
    print("Silhouette by k:", {key: round(v, 3) for key, v in sil.items()}, "-> k =", k)

    rfm["segment"] = KMeans(k, n_init=10, random_state=0).fit_predict(X)
    profile = rfm.groupby("segment").agg(customers=("frequency", "size"),
                                         **{f: (f, "median") for f in features})
    profile["share_of_spend"] = rfm.groupby("segment")["monetary"].sum() / rfm["monetary"].sum()
    print("\nSegment profile (medians)\n", profile.round(3).to_string())

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (seg, d) in enumerate(rfm.groupby("segment")):
        ax.scatter(d["frequency"], d["monetary"], s=10, color=SERIES[i], alpha=0.7, label=f"Segment {seg}")
    ax.set(title="Customer segments - frequency vs spend", xlabel="Transactions per year",
           ylabel="Annual spend", yscale="log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: naira(v, 0)))
    ax.legend()
    save(fig, "segments_scatter.png")


if __name__ == "__main__":
    main()
