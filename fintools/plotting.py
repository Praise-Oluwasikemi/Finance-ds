"""Shared chart helpers. Each function returns the matplotlib Figure."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path):
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def plot_cumulative_returns(returns, title="Growth of 1", path=None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    (1 + returns).cumprod().plot(ax=ax, linewidth=1.4)
    ax.set_title(title)
    ax.set_ylabel("Value")
    return _save(fig, path)


def plot_drawdown(returns, title="Drawdown", path=None):
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.4)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    return _save(fig, path)


def plot_return_distribution(returns, var=None, title="Daily returns", path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(returns, bins=60, alpha=0.75)
    if var is not None:
        ax.axvline(-var, linestyle="--", color="black", label=f"VaR = {var:.2%}")
        ax.legend()
    ax.set_title(title)
    return _save(fig, path)


def plot_roc(y_true, y_score, label="model", path=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle=":", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()
    return _save(fig, path)


def plot_precision_recall(y_true, y_score, label="model", path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=label)
    ax.axhline(np.mean(y_true), linestyle=":", color="grey", label="base rate")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.legend()
    return _save(fig, path)


def plot_correlation_heatmap(df, title="Correlation", path=None):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    return _save(fig, path)
