"""
make_charts.py
README charts from the research results and the big-move model. Writes a light and a
dark PNG of each chart to reports/figures/ (the README picks one by the viewer's theme).

Run from the project root, after `python -m src.research` and `python -m src.vol_model`:
    python scripts/make_charts.py
Needs matplotlib (requirements-dev.txt).
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model_registry import load_artifacts  # noqa: E402

OUT = os.path.join("reports", "figures")

# Validated palette (dataviz reference palette, slots 1-2; both modes pass CVD/contrast checks)
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "s1": "#3987e5", "s2": "#d95926"},
}
SERIES = {"crypto": ("Crypto", "s1"), "stock": ("Stocks", "s2")}
QUESTIONS = [("dir5", "Up or down in 5 days?"), ("dir20", "Up or down in 20 days?"),
             ("excess5", "Beat the others over 5 days?"), ("bigmove5", "Big move in 5 days?")]


def _style(ax, t, fig):
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _title(fig, t, title, subtitle):
    fig.text(0.02, 0.97, title, color=t["ink"], fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 0.905, subtitle, color=t["ink2"], fontsize=9.5, va="top")


def _legend(ax, t, handles):
    leg = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])


def chart_auc(res: pd.DataFrame, mode: str):
    """Median out-of-sample ROC-AUC across every ML experiment, per question and asset class."""
    t = THEMES[mode]
    ml = res[(res["model"].isin(["logreg", "hgb", "rf"])) & (res["asset_class"].isin(SERIES))]
    med = ml.groupby(["asset_class", "target"])["mean_auc"].median()
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    _style(ax, t, fig)
    y = np.arange(len(QUESTIONS))[::-1]
    h = 0.36
    handles = []
    for k, (ac, (label, slot)) in enumerate(SERIES.items()):
        vals = [med.get((ac, q), np.nan) for q, _ in QUESTIONS]
        offs = y + (h / 2 + 0.01 if k == 0 else -h / 2 - 0.01)
        bars = ax.barh(offs, np.array(vals) - 0.5, left=0.5, height=h, color=t[slot], label=label)
        handles.append(bars)
        for yy, v in zip(offs, vals):
            ax.text(max(v, 0.5) + 0.002, yy, f"{v:.3f}", va="center", fontsize=8.5, color=t["ink2"])
    ax.axvline(0.5, color=t["ink2"], linewidth=1.2)
    ax.text(0.5, y.max() + 0.62, "0.5 = coin flip", color=t["ink2"], fontsize=8.5, ha="center")
    ax.set_yticks(y, [q for _, q in QUESTIONS], color=t["ink"], fontsize=10)
    ax.set_xlim(0.48, 0.62)
    ax.set_xlabel("ROC-AUC on unseen data (median of all models and feature sets)", color=t["muted"], fontsize=9)
    _legend(ax, t, handles)
    _title(fig, t, "Only volatility is predictable",
           "Direction scores sit at the coin-flip line. 'Big move' clears it for both asset classes.")
    fig.subplots_adjust(left=0.27, right=0.97, top=0.8, bottom=0.14)
    return fig


def _folds(s: str):
    pts = [p.split(":") for p in s.split()]
    return [pd.Timestamp(d) for d, _ in pts], [float(a) for _, a in pts]


def chart_folds(res: pd.DataFrame, mode: str):
    """Per-test-period AUC for the stock +events HGB model: big move vs 5-day direction."""
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=160)
    _style(ax, t, fig)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.grid(axis="x", visible=False)
    rows = res[(res["asset_class"] == "stock") & (res["features"] == "+events") & (res["model"] == "hgb")]
    for target, label, slot in [("bigmove5", "Big move", "s2"), ("dir5", "Up or down", "s1")]:
        r = rows[rows["target"] == target].iloc[0]
        d, a = _folds(r["folds"])
        ax.plot(d, a, color=t[slot], linewidth=2, marker="o", markersize=6,
                markeredgecolor=t["surface"], markeredgewidth=2, label=label)
        ax.text(d[-1] + pd.Timedelta(days=40), a[-1], label, color=t["ink2"], fontsize=9, va="center")
    ax.axhline(0.5, color=t["ink2"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(ax.get_xlim()[0], 0.497, " coin flip", color=t["ink2"], fontsize=8.5, va="top")
    ax.set_ylim(0.4, 0.65)
    ax.set_ylabel("ROC-AUC", color=t["muted"], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    _legend(ax, t, ax.get_lines()[:2])
    _title(fig, t, "Stocks: the big-move signal holds in every test period",
           "Each point is one ~1-year period the model never saw (gradient boosting, all stock features).")
    fig.subplots_adjust(left=0.09, right=0.88, top=0.8, bottom=0.1)
    return fig


def chart_levels(mode: str):
    """Big-move hit rate by model level (hold-out test), crypto vs stocks."""
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=160)
    _style(ax, t, fig)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    levels = ["Low", "Below normal", "Normal", "Above normal", "High"]
    x = np.arange(len(levels))
    w = 0.38
    handles = []
    for k, (ac, (label, slot)) in enumerate(SERIES.items()):
        b = load_artifacts(ac)
        if not b or not b.get("vol_metrics"):
            continue
        actual = [c["actual"] for c in b["vol_metrics"]["calibration"]]
        xs = x + (-w / 2 - 0.01 if k == 0 else w / 2 + 0.01)
        bars = ax.bar(xs, actual, width=w, color=t[slot], label=label, zorder=2)
        handles.append(bars)
        for xx, v in zip(xs, actual):  # label inside the bar top so it never meets the 50% line
            ax.text(xx, v - 0.012, f"{v:.0%}", ha="center", va="top", fontsize=8.5, color="#ffffff")
    ax.axhline(0.5, color=t["ink2"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)  # behind the bars
    ax.text(-0.6, 0.508, "normal ≈ 50%", color=t["ink2"], fontsize=8.5, ha="left", va="bottom")
    ax.set_xticks(x, levels, color=t["ink"], fontsize=9.5)
    ax.set_ylim(0, 0.75)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Big move happened", color=t["muted"], fontsize=9)
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False, fontsize=9, ncol=2)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    _title(fig, t, "When the app says 'High', big moves really are more common",
           "Hold-out test (Dec 2024 – Nov 2026): share of days at each level followed by a bigger-than-usual 5-day move.")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.8, bottom=0.1)
    return fig


if __name__ == "__main__":
    plt.rcParams["font.family"] = ["Segoe UI", "Arial", "DejaVu Sans"]
    os.makedirs(OUT, exist_ok=True)
    res = pd.read_csv(os.path.join("reports", "signal_research.csv"))
    for mode in THEMES:
        for name, fig in [("auc_by_question", chart_auc(res, mode)),
                          ("stock_periods", chart_folds(res, mode)),
                          ("bigmove_levels", chart_levels(mode))]:
            path = os.path.join(OUT, f"{name}-{mode}.png")
            fig.savefig(path, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(path)
