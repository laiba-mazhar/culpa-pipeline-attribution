"""The week-2 gate experiment.

Success criterion from the proposal: on a pipeline with benign day-over-day
drift *plus* one injected fault, Shapley attribution must concentrate blame on
the injected operator while the baselines do not. If that fails, the idea is
wrong and it is much better to learn it now.

Run:  python -m experiments.gate
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from culpa.game import (
    DegradationGame,
    blame_mass_on_culprits,
    blame_ranking,
    exact_shapley,
    leave_one_out,
    mc_shapley,
    mrr,
    per_node_drift,
    precision_at_1,
    stagewise_single_permutation,
)
from culpa.utility import build_probe, make_utility
from culpa.workload import FAULTS, scenario

SCENARIOS: List[List[str]] = [
    # -- harmful single faults
    ["unit_change"],
    ["schema_drift"],
    ["predicate_flip"],
    ["join_fanout_biased"],
    ["null_spike_mnar"],
    # -- benign single faults: loud constraint violations that do not hurt the
    #    model. Correct behaviour is near-zero blame everywhere.
    ["join_fanout_uniform"],
    ["null_spike_mcar"],
    # -- compound, independent branches
    ["unit_change", "predicate_flip"],
    ["null_spike_mnar", "schema_drift"],
    # -- compound, superadditive: neither alone is fatal, together they are.
    #    The regime where leave-one-out double-counts.
    ["starve_rows_a", "starve_rows_b"],
]

BENIGN = {"join_fanout_uniform", "null_spike_mcar"}


def fmt(d: Dict[str, float], order: List[str]) -> str:
    return "  ".join(f"{n}={d.get(n, 0.0):+.4f}" for n in order)


def main() -> None:
    probe = build_probe()
    utility = make_utility(probe)
    print(f"probe set: {len(probe)} rows, {probe['label'].mean():.3f} positive rate\n")

    summary = []

    for fault_names in SCENARIOS:
        pipe, culprits = scenario(fault_names)
        players = sorted(pipe.ancestors_of_sink())
        game = DegradationGame(pipe, utility)

        frontier = pipe.active_frontier()
        t0 = time.time()
        phi = exact_shapley(game, players)
        t_exact = time.time() - t0

        phi_mc = mc_shapley(game, players, n_permutations=60, seed=1)
        loo = leave_one_out(game, players)
        orders = pipe.all_topological_orders(limit=32)
        stage_a = stagewise_single_permutation(game, orders[0])
        stage_b = stagewise_single_permutation(game, orders[-1])
        drift = per_node_drift(pipe)

        delta = game.total_degradation
        shapley_sum = sum(phi.values())
        mc_err = max(abs(phi[p] - phi_mc[p]) for p in players)

        print("=" * 78)
        print(f"SCENARIO {'+'.join(fault_names)}   culprits={culprits}")
        print("-" * 78)
        print(f"observed degradation  v(V) = {delta:+.4f} AUC")
        print(f"sum of Shapley values      = {shapley_sum:+.4f}   "
              f"(efficiency gap {abs(shapley_sum - delta):.2e})")
        print(f"active frontier ({len(frontier)}/{len(players)}): {sorted(frontier)}")
        print(f"cost: {2**len(players)} coalitions -> "
              f"{game.stats.model_fits} model fits, "
              f"{game.stats.distinct_sink_outputs} distinct sinks, "
              f"operator cache hit rate {pipe.stats.cache_hit_rate:.1%}, "
              f"{t_exact:.1f}s")
        print(f"distinct topological orders enumerated: {len(orders)}")
        print()
        print(f"  shapley (exact)  {fmt(phi, players)}")
        print(f"  shapley (MC,60)  {fmt(phi_mc, players)}   max err {mc_err:.4f}")
        print(f"  leave-one-out    {fmt(loo, players)}")
        print(f"  stagewise ord.A  {fmt(stage_a, players)}")
        print(f"  stagewise ord.B  {fmt(stage_b, players)}")
        print(f"  per-node drift   {fmt({k: -v for k, v in drift.items()}, players)}")
        print()

        row = {
            "scenario": "+".join(fault_names),
            "benign": bool(set(fault_names) & BENIGN),
            "delta": delta,
            "max_abs_phi": max(abs(x) for x in phi.values()),
        }
        for label, attr in [
            ("shapley", phi),
            ("shapley_mc", phi_mc),
            ("loo", loo),
            ("stagewise_A", stage_a),
            ("stagewise_B", stage_b),
            ("drift", {k: -v for k, v in drift.items() if k in players}),
        ]:
            row[f"{label}_p@1"] = precision_at_1(attr, culprits)
            row[f"{label}_mrr"] = mrr(attr, culprits)
            row[f"{label}_mass"] = blame_mass_on_culprits(attr, culprits)
            print(f"  {label:12s} top-blame={blame_ranking(attr)[0]:18s} "
                  f"p@1={row[f'{label}_p@1']:.0f}  "
                  f"MRR={row[f'{label}_mrr']:.2f}  "
                  f"mass={row[f'{label}_mass']:.2f}")
        summary.append(row)
        print()

    methods = ["shapley", "shapley_mc", "loo", "stagewise_A", "stagewise_B", "drift"]

    def table(title: str, rows: List[dict]) -> None:
        if not rows:
            return
        print(f"\n{title} ({len(rows)} scenarios)")
        print(f"{'method':14s} {'p@1':>6s} {'MRR':>6s} {'blame mass':>11s}")
        for m in methods:
            p = sum(r[f"{m}_p@1"] for r in rows) / len(rows)
            rr = sum(r[f"{m}_mrr"] for r in rows) / len(rows)
            mass = sum(r[f"{m}_mass"] for r in rows) / len(rows)
            print(f"{m:14s} {p:6.2f} {rr:6.2f} {mass:11.2f}")

    harmful = [r for r in summary if not r["benign"]]
    print("=" * 78)
    print("AGGREGATE")
    table("all harmful scenarios", harmful)
    table("harmful compound only", [r for r in harmful if "+" in r["scenario"]])

    benign = [r for r in summary if r["benign"]]
    if benign:
        print("\nbenign scenarios -- loud constraint violations, no model damage.")
        print("correct behaviour is |phi| ~ 0 everywhere; ranking metrics are")
        print("meaningless here and are omitted.")
        print(f"{'scenario':24s} {'v(V)':>10s} {'max|phi|':>10s}")
        for r in benign:
            print(f"{r['scenario']:24s} {r['delta']:+10.4f} {r['max_abs_phi']:10.4f}")


if __name__ == "__main__":
    main()
