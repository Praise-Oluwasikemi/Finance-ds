"""P04 Customer segmentation: what groups of customers does the bank have?

Clusters customers with K-means on scaled behaviour features, picks the number
of clusters by silhouette score, and profiles each segment so marketing can act
on it.

Run: ``python -m projects.p04_customer_segmentation.analysis``
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from data.generate_data import make_customers

FEATURES = [
    "age", "log_income", "log_balance", "num_products",
    "monthly_txns", "digital_share", "tenure_years",
]


def add_features(df):
    df = df.copy()
    df["log_income"] = np.log1p(df["income"])
    df["log_balance"] = np.log1p(df["avg_balance"])
    return df


def choose_k(X, k_range=range(2, 9), seed=0):
    """Silhouette score for each k; returns (best_k, scores)."""
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    return max(scores, key=scores.get), scores


def profile_segments(df, labels):
    cols = ["age", "income", "avg_balance", "num_products", "monthly_txns", "digital_share", "tenure_years"]
    prof = df.assign(segment=labels).groupby("segment")[cols].median()
    prof.insert(0, "customers", pd.Series(labels).value_counts().sort_index())
    return prof


def run(seed=42, out_dir=None):
    df = add_features(make_customers(seed=seed))
    X = StandardScaler().fit_transform(df[FEATURES])

    best_k, sil_scores = choose_k(X)
    labels = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit_predict(X)
    profile = profile_segments(df, labels)

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        xy = PCA(n_components=2, random_state=0).fit_transform(X)
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels, s=6, cmap="tab10")
        ax.legend(*sc.legend_elements(), title="segment")
        ax.set_title("Customer segments (PCA projection)")
        fig.savefig(out / "p04_segments.png", dpi=120, bbox_inches="tight")
        profile.to_csv(out / "p04_segment_profiles.csv")

    return {
        "best_k": best_k,
        "silhouette_scores": sil_scores,
        "profile": profile,
        # The data has known segments, so we can check how well we recovered them.
        "adjusted_rand_index": adjusted_rand_score(df["true_segment"], labels),
    }


if __name__ == "__main__":
    res = run(out_dir="outputs")
    print(f"Chosen k: {res['best_k']}")
    print({k: round(v, 3) for k, v in res["silhouette_scores"].items()})
    print("\nSegment profiles (medians):")
    print(res["profile"].round(2))
    print(f"\nAgreement with true segments (ARI): {res['adjusted_rand_index']:.2f}")
