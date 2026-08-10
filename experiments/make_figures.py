"""Generate the paper's main figure from the experiment CSVs.

Reads results/ rather than hardcoding numbers, so the figure cannot drift out of
sync with the experiments.

Run:  python -m experiments.make_figures
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Fixed categorical order, validated for CVD separation (worst adjacent pair
# dE 11.4 protan) against a light surface. Colour follows the method, never its
# rank, so a method keeps its hue across both panels.
METHODS = [
    ("anchored", "Anchored Shapley", "#0072B2"),
    ("shapley", "Shapley", "#009E73"),
    ("loo", "Leave-one-out", "#E69F00"),
    ("drift", "Per-node drift", "#CC79A7"),
    ("stagewise", "Stagewise (practice)", "#D55E00"),
]

BINS = [(0.0, 0.5, "$r < 0.5$"), (0.5, 1.0, "$0.5 \\leq r < 1$"), (1.0, 1e18, "$r \\geq 1$")]


def load(path: Path, valid_key: str) -> List[Dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if r.get(valid_key, "True") != "True":
            continue
        r["ratio"] = float(r["ratio"])
        for key, _, _ in METHODS:
            r[f"{key}_p@1"] = float(r[f"{key}_p@1"])
        out.append(r)
    return out


def binned(rows: List[Dict]):
    labels, counts, series = [], [], {k: [] for k, _, _ in METHODS}
    for lo, hi, label in BINS:
        sel = [r for r in rows if lo <= r["ratio"] < hi]
        if not sel:
            continue
        labels.append(label)
        counts.append(len(sel))
        for key, _, _ in METHODS:
            series[key].append(sum(r[f"{key}_p@1"] for r in sel) / len(sel))
    return labels, counts, series


def panel(ax, rows, title):
    labels, counts, series = binned(rows)
    n = len(labels)
    width = 0.16
    xs = range(n)

    for i, (key, name, colour) in enumerate(METHODS):
        offs = [x + (i - 2) * width for x in xs]
        bars = ax.bar(offs, series[key], width * 0.88, label=name,
                      color=colour, linewidth=0)
        # Direct value labels: the required secondary encoding for the two hues
        # that sit below 3:1 contrast against the surface.
        for b, v in zip(bars, series[key]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=5.2, color="#333333",
                    rotation=90)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{l}\n$n={c}$" for l, c in zip(labels, counts)], fontsize=7.5)
    ax.set_ylim(0, 1.28)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_ylabel("precision@1", fontsize=8)
    ax.set_title(title, fontsize=8.5, pad=6)

    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
        ax.spines[s].set_linewidth(0.6)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    synthetic = load(RESULTS / "severity.csv", "measurable")
    real_path = RESULTS / "real_data_tlc_8v13.csv"
    if not real_path.exists():
        raise SystemExit(f"missing {real_path}; run experiments.real_data --real --days 8,13,10")
    real = load(real_path, "valid")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
    panel(axes[0], synthetic, "(a) Synthetic pipeline, 7 operators")
    panel(axes[1], real, "(b) NYC taxi, 9 operators, Mon vs Sat")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    targets = [FIGURES / f"main_result.{e}" for e in ("pdf", "png")]
    # paper/ is kept self-contained so it can be uploaded to Overleaf as-is.
    targets.append(ROOT / "paper" / "main_result.pdf")
    for out in targets:
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")

    print(f"\nsynthetic: {len(synthetic)} valid cases; real: {len(real)} valid cases")


if __name__ == "__main__":
    main()
