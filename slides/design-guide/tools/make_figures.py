"""Build the example charts used in the design guide.

All numbers come from research/mstudy_20261117/SLIDES.md.
Run from slides/design-guide:  python3 tools/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures" / "guide"
OUT.mkdir(parents=True, exist_ok=True)

ORANGE = "#BF5700"
INK = "#1B1F24"
SLATE = "#333F48"
GRAY = "#9CADB7"
LIGHT = "#D6D2C4"

HORIZONS = ["0.1s", "0.5s", "1s", "10s"]
R2 = {
    "LightGBM, handcrafted": [0.232, 0.152, 0.114, 0.015],
    "Transformer, 75 min": [0.233, 0.154, 0.116, 0.018],
    "Transformer, 75+90 min": [0.244, 0.158, 0.119, 0.017],
    "GRU, 60 min": [0.262, 0.164, 0.123, 0.018],
}


def bad_chart():
    """Deliberately poor version: defaults, rainbow, boxed legend, no message."""
    with plt.style.context("classic"):
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        x = range(len(HORIZONS))
        w = 0.2
        colors = ["red", "lime", "blue", "magenta"]
        for i, (name, vals) in enumerate(R2.items()):
            ax.bar([xi + i * w for xi in x], vals, w, label=name, color=colors[i])
        ax.set_xticks([xi + 1.5 * w for xi in x])
        ax.set_xticklabels(HORIZONS)
        ax.set_title("results")
        ax.set_ylabel("r2")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8)
        fig.savefig(OUT / "bad_chart.pdf")
        plt.close(fig)


def good_chart():
    """Same data: one highlight, direct labels, title states the finding."""
    with plt.style.context(HERE / "ucf.mplstyle"):
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        x = list(range(len(HORIZONS)))
        for name, vals in R2.items():
            hi = name.startswith("GRU")
            ax.plot(
                x, vals,
                color=ORANGE if hi else GRAY,
                lw=2.6 if hi else 1.4,
                marker="o", ms=5 if hi else 3.5,
                zorder=3 if hi else 2,
            )
            ax.annotate(
                name, (x[0], vals[0]),
                xytext=(-8, 0), textcoords="offset points",
                ha="right", va="center", fontsize=9,
                color=ORANGE if hi else "#6B7178",
                fontweight="bold" if hi else "normal",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(HORIZONS)
        ax.set_xlim(-1.9, 3.2)
        ax.set_ylim(0, 0.29)
        ax.set_ylabel("Test R²")
        ax.set_xlabel("Prediction horizon")
        fig.savefig(OUT / "good_chart.pdf")
        plt.close(fig)


def r2_bar_short():
    """Bar chart for the 0.1s horizon only, sorted, best highlighted."""
    with plt.style.context(HERE / "ucf.mplstyle"):
        fig, ax = plt.subplots(figsize=(5.6, 2.6))
        items = sorted(((v[0], k) for k, v in R2.items()))
        names = [k for _, k in items]
        vals = [v for v, _ in items]
        cols = [ORANGE if n.startswith("GRU") else LIGHT for n in names]
        ax.barh(names, vals, color=cols, height=0.62)
        for i, v in enumerate(vals):
            ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9.5,
                    color=INK if cols[i] == ORANGE else "#6B7178",
                    fontweight="bold" if cols[i] == ORANGE else "normal")
        ax.set_xlim(0.20, 0.272)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
        fig.savefig(OUT / "r2_bar_short.pdf")
        plt.close(fig)


def toxicity_auc():
    with plt.style.context(HERE / "ucf.mplstyle"):
        fig, ax = plt.subplots(figsize=(5.0, 2.8))
        h = ["0.1s", "0.5s", "1s", "10s"]
        auc = [0.927, 0.892, 0.870, 0.785]
        ax.bar(h, auc, color=[ORANGE, LIGHT, LIGHT, LIGHT], width=0.6)
        for i, v in enumerate(auc):
            ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9.5,
                    color=INK if i == 0 else "#6B7178")
        ax.axhline(0.5, color=SLATE, lw=1, ls=(0, (3, 3)))
        ax.text(3.35, 0.51, "chance", fontsize=8.5, color=SLATE, ha="right")
        ax.set_ylim(0.45, 1.0)
        ax.set_ylabel("AUC (toxic vs retail)")
        ax.set_xlabel("Markout horizon")
        fig.savefig(OUT / "toxicity_auc.pdf")
        plt.close(fig)


def family_auc():
    with plt.style.context(HERE / "ucf.mplstyle"):
        fig, ax = plt.subplots(figsize=(4.6, 2.2))
        fams = ["Perps", "Crypto", "Sports"]
        auc = [0.77, 0.82, 0.92]
        ax.barh(fams, auc, color=[LIGHT, LIGHT, ORANGE], height=0.6)
        for i, v in enumerate(auc):
            ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=10,
                    color=INK if i == 2 else "#6B7178")
        ax.set_xlim(0.5, 1.0)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Toxicity AUC at 1s")
        fig.savefig(OUT / "family_auc.pdf")
        plt.close(fig)


def separability():
    with plt.style.context(HERE / "ucf.mplstyle"):
        fig, ax = plt.subplots(figsize=(5.0, 2.8))
        x = ["16 events", "1024 events"]
        ax.plot(x, [0.906, 0.986], color=ORANGE, marker="o", lw=2.6, label="Coarse (3 families)")
        ax.plot(x, [0.68, 0.86], color=SLATE, marker="o", lw=1.8, label="Fine (13 classes)")
        for xi, v in zip(x, [0.906, 0.986]):
            ax.text(xi, v + 0.025, f"{v:.3f}", ha="center", fontsize=9.5, color=ORANGE, fontweight="bold")
        for xi, v in zip(x, [0.68, 0.86]):
            ax.text(xi, v - 0.06, f"{v:.2f}", ha="center", fontsize=9.5, color=SLATE)
        ax.set_ylim(0.5, 1.07)
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel("Balanced accuracy")
        ax.legend(loc="lower right")
        fig.savefig(OUT / "separability.pdf")
        plt.close(fig)


if __name__ == "__main__":
    bad_chart()
    good_chart()
    r2_bar_short()
    toxicity_auc()
    family_auc()
    separability()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
