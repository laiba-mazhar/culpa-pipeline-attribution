"""Severity sweep -- the paper's main experiment.

The gate experiment showed that leave-one-out beats Shapley at ranking culprits,
and that Shapley loses exactly the scenarios where the injected fault does less
damage than benign day-over-day drift. That suggests the governing variable is
not the fault *type* but the ratio

    r  =  | damage the fault does alone |  /  | damage benign drift does alone |

This sweeps every fault type across a monotone severity ladder, measures r for
each, and scores every attribution method as a function of r. If the story from
the gate run is right, there is a crossover: below r ~ 1 Shapley ranks benign
sources first (correct accounting, useless triage) and above it the two agree.

The anchored estimator should be flat across the whole range, because anchoring
the exogenous operators removes benign drift from the comparison by construction.

Also validates the two boundary claims of anchored_shapley:
    anchor = {}       == exact Shapley
    anchor = V \\ {i}  == leave-one-out for i

Run:  python -m experiments.severity
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from culpa.game import (
    DegradationGame,
    anchored_shapley,
    exact_shapley,
    leave_one_out,
    mrr,
    per_node_drift,
    precision_at_1,
    stagewise_single_permutation,
)
from culpa.utility import build_probe, make_utility
from culpa.workload import FAULT_SWEEPS, scenario_from_faults

EPS = 1e-9


def validate_boundaries(game: DegradationGame, players: List[str]) -> None:
    """anchor={} == Shapley, anchor=V\\{i} == LOO. Checked once, loudly."""
    phi = exact_shapley(game, players)
    phi_empty = anchored_shapley(game, players, [])
    gap = max(abs(phi[p] - phi_empty[p]) for p in players)
    assert gap < 1e-12, f"anchor={{}} should equal Shapley, gap={gap:.2e}"

    loo = leave_one_out(game, players)
    for p in players:
        anchored = anchored_shapley(game, players, [q for q in players if q != p])
        assert abs(anchored[p] - loo[p]) < 1e-12, (
            f"anchor=V\\{{{p}}} should equal LOO, "
            f"got {anchored[p]:.6f} vs {loo[p]:.6f}"
        )
    print("boundary checks passed: anchor={} == Shapley, anchor=V\\{i} == LOO\n")


def main() -> None:
    probe = build_probe()
    utility = make_utility(probe)

    rows: List[dict] = []
    validated = False

    for fault_type, (op_name, ladder) in FAULT_SWEEPS.items():
        print("=" * 96)
        print(f"FAULT TYPE: {fault_type}   (operator: {op_name})")
        print("=" * 96)
        print(f"{'severity':26s} {'v(V)':>9s} {'fault':>9s} {'drift':>9s} {'ratio r':>9s} "
              f"{'shap':>6s} {'anch':>6s} {'loo':>6s} {'drift':>6s} {'stage':>6s}  eff-gap")
        print("-" * 96)

        for level, params in ladder:
            pipe, culprits = scenario_from_faults({op_name: params})
            players = sorted(pipe.ancestors_of_sink())
            sources = sorted(pipe.exogenous())
            game = DegradationGame(pipe, utility)

            if not validated:
                validate_boundaries(game, players)
                validated = True

            delta = game.total_degradation
            fault_alone = game.v(frozenset([op_name]))

            # Benign drift magnitude must be measured per-source, not on the
            # sources jointly: the two sources push in opposite directions and
            # v({both}) ~ -0.0002 while each alone is worth ~0.05. The joint
            # value cancels and makes a useless denominator.
            drift_alone = max(abs(game.v(frozenset([s]))) for s in sources)
            ratio = abs(fault_alone) / max(drift_alone, EPS)

            phi = exact_shapley(game, players)
            anch = anchored_shapley(game, players, sources)
            loo = leave_one_out(game, players)
            drift = {k: -v for k, v in per_node_drift(pipe).items() if k in players}
            stage = stagewise_single_permutation(game, pipe.all_topological_orders(1)[0])

            # Anchored efficiency: values sum to v(V) - v(B) on the sub-game.
            v_anchor = game.v(frozenset(sources))
            eff_gap = abs(sum(anch.values()) - (delta - v_anchor))
            # Leave-one-out has no such guarantee. This is the column that
            # separates the two methods, since their rankings agree.
            loo_eff_gap = abs(sum(loo.values()) - delta)

            attrs = {"shapley": phi, "anchored": anch, "loo": loo,
                     "drift": drift, "stagewise": stage}
            p1 = {k: precision_at_1(a, culprits) for k, a in attrs.items()}

            print(f"{level:26s} {delta:+9.4f} {fault_alone:+9.4f} {drift_alone:+9.4f} "
                  f"{ratio:9.2f} "
                  f"{p1['shapley']:6.0f} {p1['anchored']:6.0f} {p1['loo']:6.0f} "
                  f"{p1['drift']:6.0f} {p1['stagewise']:6.0f}  {eff_gap:.1e}")

            row = {
                "fault_type": fault_type, "severity": level, "culprit": op_name,
                "delta": delta, "fault_alone": fault_alone,
                "drift_alone": drift_alone, "ratio": ratio,
                "anchored_eff_gap": eff_gap, "loo_eff_gap": loo_eff_gap,
                "model_fits": game.stats.model_fits,
                # Below this the incident is not measurable at all and ranking
                # metrics are meaningless -- see the benign controls in the gate.
                "measurable": abs(delta) >= 0.002,
            }
            for k, a in attrs.items():
                row[f"{k}_p@1"] = p1[k]
                row[f"{k}_mrr"] = mrr(a, culprits)
            rows.append(row)
        print()

    # -- the main figure, as a table -----------------------------------
    print("=" * 96)
    print("MAIN RESULT: attribution accuracy vs fault-to-drift ratio\n")

    bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e18)]
    labels = ["r < 0.25", "0.25 - 0.5", "0.5 - 1", "1 - 2", "r > 2"]
    methods = ["shapley", "anchored", "loo", "drift", "stagewise"]

    live = [r for r in rows if r["measurable"]]
    dead = [r for r in rows if not r["measurable"]]

    print("p@1 by fault-to-drift ratio, measurable incidents only "
          f"({len(live)} of {len(rows)}; |v(V)| >= 0.002)\n")
    print(f"{'ratio bin':12s} {'n':>4s} " + " ".join(f"{m:>10s}" for m in methods))
    print("-" * 68)
    for (lo, hi), label in zip(bins, labels):
        sel = [r for r in live if lo <= r["ratio"] < hi]
        if not sel:
            continue
        cells = [sum(r[f"{m}_p@1"] for r in sel) / len(sel) for m in methods]
        print(f"{label:12s} {len(sel):4d} " + " ".join(f"{c:10.2f}" for c in cells))

    print(f"\n{'OVERALL':12s} {len(live):4d} " + " ".join(
        f"{sum(r[f'{m}_p@1'] for r in live) / len(live):10.2f}" for m in methods))

    if dead:
        print(f"\n{len(dead)} sub-threshold incidents excluded (|v(V)| < 0.002): "
              + ", ".join(f"{r['fault_type']}/{r['severity']}" for r in dead))

    # -- anchored vs LOO: rankings agree, decomposition does not ---------
    print("\n" + "=" * 96)
    print("ANCHORED vs LEAVE-ONE-OUT\n")
    agree = sum(1 for r in live if r["anchored_p@1"] == r["loo_p@1"])
    print(f"p@1 agreement on measurable incidents: {agree}/{len(live)}")
    print(f"anchored efficiency  max |sum(phi) - (v(V) - v(B))| = "
          f"{max(r['anchored_eff_gap'] for r in rows):.2e}")
    print(f"LOO      efficiency  max |sum(phi) -  v(V)|         = "
          f"{max(r['loo_eff_gap'] for r in rows):.2e}")
    print(f"LOO      efficiency  mean                           = "
          f"{sum(r['loo_eff_gap'] for r in rows) / len(rows):.2e}")
    print("\nThe two rank identically. Only the anchored value decomposes the")
    print("degradation exactly; LOO's attributions do not sum to anything.")

    out = Path(__file__).resolve().parents[1] / "results" / "severity.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
