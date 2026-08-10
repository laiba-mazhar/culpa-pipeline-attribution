"""How much does stagewise diagnosis depend on the order you walk the DAG in?

The paper claims current practice is the marginal-contribution vector of a single
permutation, and is therefore arbitrary. That claim was previously supported by
quoting two topological orders that disagreed. This enumerates *every* valid
topological order and records the full spread of blame each operator can receive,
which is the honest version of the same claim and a much stronger one.

Run:  python -m experiments.order_dependence
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
    blame_ranking,
    stagewise_single_permutation,
)
from culpa.utility import build_probe, make_utility
from culpa.workload import scenario

SCENARIOS = [
    ["unit_change"],
    ["predicate_flip"],
    ["schema_drift"],
    ["unit_change", "predicate_flip"],
]
MAX_ORDERS = 5000


def main() -> None:
    utility = make_utility(build_probe())
    rows: List[Dict] = []

    for fault_names in SCENARIOS:
        pipe, culprits = scenario(fault_names)
        players = sorted(pipe.ancestors_of_sink())
        sources = sorted(pipe.exogenous())
        game = DegradationGame(pipe, utility)

        orders = pipe.all_topological_orders(limit=MAX_ORDERS)
        anch = anchored_shapley(game, players, sources)

        # Blame each operator receives under every valid topological order.
        per_op: Dict[str, List[float]] = {p: [] for p in players}
        top_blamed: Dict[str, int] = {}
        for order in orders:
            st = stagewise_single_permutation(game, order)
            for p in players:
                per_op[p].append(st.get(p, 0.0))
            winner = blame_ranking(st)[0]
            top_blamed[winner] = top_blamed.get(winner, 0) + 1

        label = "+".join(fault_names)
        print("=" * 84)
        print(f"SCENARIO {label}   culprits={culprits}")
        print(f"{len(orders)} distinct topological orders enumerated")
        print("-" * 84)
        print(f"{'operator':>18s} {'stagewise min':>14s} {'max':>10s} {'spread':>9s} "
              f"{'sign flips?':>12s} {'anchored':>10s}")
        for p in players:
            vals = per_op[p]
            lo, hi = min(vals), max(vals)
            flips = lo < -1e-9 and hi > 1e-9
            print(f"{p:>18s} {lo:>14.4f} {hi:>10.4f} {hi - lo:>9.4f} "
                  f"{('YES' if flips else '-'):>12s} {anch[p]:>10.4f}")
            rows.append({
                "scenario": label, "operator": p, "is_culprit": p in culprits,
                "is_source": p in sources, "n_orders": len(orders),
                "stagewise_min": lo, "stagewise_max": hi, "spread": hi - lo,
                "sign_flips": flips, "anchored": anch[p],
            })

        print()
        print("  which operator stagewise blames most, by order:")
        for name, count in sorted(top_blamed.items(), key=lambda kv: -kv[1]):
            hit = "correct" if name in culprits else "WRONG"
            print(f"    {name:>18s}  {count:>5d}/{len(orders)} orders   ({hit})")
        print(f"  anchored value blames: {blame_ranking(anch)[0]}  "
              f"({'correct' if blame_ranking(anch)[0] in culprits else 'WRONG'}), "
              f"and does not depend on order")
        print()

    flipping = [r for r in rows if r["sign_flips"]]
    print("=" * 84)
    print(f"{len(flipping)} of {len(rows)} operator/scenario pairs receive blame of "
          f"BOTH signs\ndepending only on the topological order chosen:")
    for r in flipping:
        print(f"  {r['scenario']:>28s} / {r['operator']:<18s} "
              f"{r['stagewise_min']:+.4f} .. {r['stagewise_max']:+.4f}")

    out = Path(__file__).resolve().parents[1] / "results" / "order_dependence.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
