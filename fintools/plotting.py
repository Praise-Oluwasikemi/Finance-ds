"""Consistent, accessible chart styling for every project figure."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

# Colorblind-validated categorical order - always assign in this order, never cycle.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "bold", "legend.frameon": False,
    })


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
