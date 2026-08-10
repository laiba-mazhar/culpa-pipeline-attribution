"""Additional paper figures: pipeline topology, order dependence, severity, cost.

Every figure reads from results/ or from the live Pipeline objects, so none of
the numbers are transcribed by hand.

Imported by experiments.make_figures; also runnable on its own.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

HARM = "#D55E00"
HELP = "#0072B2"
ANCHOR = "#7D8BA1"
PLAIN = "#F2F4F7"
INK = "#333333"


def _save(fig, stem: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIGURES / f"{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


# -- 1. pipeline topology -------------------------------------------------


def _layers(pipe) -> Dict[str, int]:
    """Longest-path layer of each operator, so edges always point rightwards."""
    depth: Dict[str, int] = {}
    for name in pipe.order:                      # already topological
        parents = pipe.operators[name].parents
        depth[name] = 1 + max((depth[p] for p in parents), default=-1)
    return depth


def pipeline_figure(pipe, culprits, stem: str, title: str) -> None:
    depth = _layers(pipe)
    by_layer: Dict[int, List[str]] = {}
    for name, d in depth.items():
        by_layer.setdefault(d, []).append(name)

    pos: Dict[str, tuple] = {}
    for d, names in by_layer.items():
        for i, name in enumerate(sorted(names)):
            y = -(i - (len(names) - 1) / 2)
            pos[name] = (d * 2.25, y * 1.25)

    sources = pipe.exogenous()
    fig, ax = plt.subplots(figsize=(7.0, 2.4))

    for name, op in pipe.operators.items():
        for parent in op.parents:
            x0, y0 = pos[parent]
            x1, y1 = pos[name]
            ax.add_patch(FancyArrowPatch(
                (x0 + 0.82, y0), (x1 - 0.82, y1),
                arrowstyle="-|>", mutation_scale=8,
                color="#b6bcc6", linewidth=0.9,
                connectionstyle="arc3,rad=0.06", zorder=1))

    for name, (x, y) in pos.items():
        if name in culprits:
            face, edge, lw, tc = "#FBE6D9", HARM, 1.8, "#7A3400"
        elif name in sources:
            face, edge, lw, tc = "#EEF2F7", ANCHOR, 1.1, "#243447"
        else:
            face, edge, lw, tc = PLAIN, "#C2C8D0", 1.0, "#4A5568"
        ax.add_patch(FancyBboxPatch(
            (x - 0.82, y - 0.30), 1.64, 0.60,
            boxstyle="round,pad=0.02,rounding_size=0.09",
            facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))
        ax.text(x, y, name, ha="center", va="center", fontsize=6.4,
                color=tc, zorder=3)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
    ax.set_ylim(min(ys) - 0.85, max(ys) + 0.85)
    ax.axis("off")
    ax.set_title(title, fontsize=8.5, pad=4)

    handles = [
        FancyBboxPatch((0, 0), 1, 1, boxstyle="round", facecolor="#EEF2F7",
                       edgecolor=ANCHOR, label="exogenous source (anchored)"),
        FancyBboxPatch((0, 0), 1, 1, boxstyle="round", facecolor=PLAIN,
                       edgecolor="#C2C8D0", label="pipeline operator"),
        FancyBboxPatch((0, 0), 1, 1, boxstyle="round", facecolor="#FBE6D9",
                       edgecolor=HARM, linewidth=1.8, label="injected fault"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.4,
              frameon=False, bbox_to_anchor=(0.5, -0.16))
    _save(fig, stem)


# -- 2. order dependence --------------------------------------------------


def order_dependence_figure() -> None:
    src = RESULTS / "order_dependence.csv"
    if not src.exists():
        print("skipping order-dependence figure; run experiments.order_dependence")
        return
    with src.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    scen = "unit_change"
    sel = [r for r in rows if r["scenario"] == scen]
    sel.sort(key=lambda r: float(r["stagewise_min"]))
    names = [r["operator"] for r in sel]
    lo = [float(r["stagewise_min"]) for r in sel]
    hi = [float(r["stagewise_max"]) for r in sel]
    anch = [float(r["anchored"]) for r in sel]
    n_orders = sel[0]["n_orders"]

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ys = range(len(names))

    for y, (a, b, r) in enumerate(zip(lo, hi, sel)):
        flips = r["sign_flips"] == "True"
        colour = HARM if flips else "#B9C0CA"
        ax.plot([a, b], [y, y], color=colour,
                linewidth=5.5 if flips else 3.5, solid_capstyle="round",
                zorder=2, alpha=0.95)

    ax.scatter(anch, list(ys), s=34, color=HELP, zorder=4,
               label="anchored Shapley (order-independent)")
    ax.scatter(lo, list(ys), s=9, color="#6b7280", zorder=3)
    ax.scatter(hi, list(ys), s=9, color="#6b7280", zorder=3)

    ax.axvline(0, color="#666666", linewidth=0.8, zorder=1)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("blame assigned to the operator (AUC)", fontsize=8)
    ax.set_title(
        f"Range of blame stagewise diagnosis assigns across all {n_orders} "
        f"topological orders", fontsize=8.5, pad=6)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(axis="x", color="#e8e8e8", linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["bottom"].set_linewidth(0.6)

    from matplotlib.lines import Line2D
    # Legend sits below the axes: placed inside, it covered the two rows that
    # carry the result.
    ax.legend(handles=[
        Line2D([0], [0], color=HARM, linewidth=5.5,
               label="stagewise range — blame changes sign"),
        Line2D([0], [0], color="#B9C0CA", linewidth=3.5,
               label="stagewise range — same sign"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=HELP,
               markersize=6, label="anchored Shapley (one value, any order)"),
    ], loc="upper center", ncol=3, fontsize=6.8, frameon=False,
        bbox_to_anchor=(0.5, -0.20))
    fig.tight_layout()
    _save(fig, "order_dependence")


# -- 3. severity ladder ---------------------------------------------------


def severity_figure() -> None:
    src = RESULTS / "severity.csv"
    if not src.exists():
        return
    with src.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_type: Dict[str, List[dict]] = {}
    for r in rows:
        by_type.setdefault(r["fault_type"], []).append(r)

    palette = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#D55E00"]
    fig, ax = plt.subplots(figsize=(7.0, 2.9))

    for (ftype, rs), colour in zip(by_type.items(), palette):
        xs = list(range(1, len(rs) + 1))
        ys = [abs(float(r["fault_alone"])) for r in rs]
        ax.plot(xs, ys, marker="o", markersize=4.5, linewidth=1.6,
                color=colour, label=ftype.replace("_", " "))

    ax.axhline(0.002, color="#888888", linewidth=0.9, linestyle=(0, (4, 3)))
    ax.annotate("measurability threshold:\nno culprit to find below here",
                xy=(6.0, 0.002), xytext=(5.55, 0.0135),
                fontsize=6.3, color="#666666", ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color="#999999", linewidth=0.7))

    ax.set_yscale("log")
    ax.set_xlabel("severity level (increasing)", fontsize=8)
    ax.set_ylabel("damage the fault does alone (AUC, log)", fontsize=8)
    ax.set_title("Not every injected fault degrades the model", fontsize=8.5, pad=6)
    ax.set_xticks(range(1, 7))
    ax.tick_params(labelsize=7.5)
    ax.grid(color="#ececec", linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
        ax.spines[s].set_linewidth(0.6)
    ax.legend(fontsize=6.8, frameon=False, ncol=3, loc="lower right",
              bbox_to_anchor=(1.0, -0.02))
    _save(fig, "severity_ladder")


# -- 4. cost scaling ------------------------------------------------------


def cost_figure() -> None:
    src = RESULTS / "scale.csv"
    if not src.exists():
        return
    with src.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n = [int(r["n"]) for r in rows]
    lattice = [float(r["lattice"]) for r in rows]
    fits = [float(r["model_fits"]) for r in rows]
    frontier = [int(r["frontier"]) for r in rows]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    ax.plot(n, lattice, marker="o", markersize=4.5, linewidth=1.8,
            color="#CC79A7", label="coalitions in the lattice  $2^n$")
    ax.plot(n, fits, marker="s", markersize=4.5, linewidth=1.8,
            color=HELP, label="model fits actually run")
    ax.set_yscale("log")
    ax.set_xlabel("operators in the DAG  ($n$)", fontsize=8)
    ax.set_ylabel("count (log)", fontsize=8)
    ax.set_title("(a) Replays avoided", fontsize=8.5, pad=6)
    ax.legend(fontsize=6.8, frameon=False, loc="upper left")

    ax2.plot(n, n, marker="", linewidth=1.2, color="#b6bcc6",
             linestyle=(0, (4, 3)), label="$|V|$  (all operators)")
    ax2.plot(n, frontier, marker="o", markersize=4.5, linewidth=1.8,
             color=HARM, label="$|F|$  active frontier")
    ax2.set_xlabel("operators in the DAG  ($n$)", fontsize=8)
    ax2.set_ylabel("operators", fontsize=8)
    ax2.set_title("(b) Cost tracks the frontier, not the DAG", fontsize=8.5, pad=6)
    ax2.legend(fontsize=6.8, frameon=False, loc="upper left")

    for a in (ax, ax2):
        a.tick_params(labelsize=7.5)
        a.grid(color="#ececec", linewidth=0.5)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color("#999999")
            a.spines[s].set_linewidth(0.6)

    fig.tight_layout()
    _save(fig, "cost_scaling")


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })
    from culpa.taxi_pipeline import taxi_scenario
    from experiments.demo import INCIDENT_FAULTS, REF, INC

    pipe, culprits = taxi_scenario(INCIDENT_FAULTS, REF, INC, fixture=False)
    pipeline_figure(pipe, culprits, "pipeline_taxi",
                    "The nine-operator NYC taxi pipeline, with the two injected faults")
    order_dependence_figure()
    severity_figure()
    cost_figure()


if __name__ == "__main__":
    main()
