"""The real-data experiment: same method, NYC taxi pipeline, real day-over-day drift.

This is the experiment that answers the most serious limitation in PAPER.md
section 8. The synthetic workload draws its benign drift from a stationary
distribution, which is far kinder than reality. Here the reference and incident
runs read two different real days of New York taxi trips, so the fault-to-drift
ratio that drives the main result is measured against genuine background
movement.

Runs against the fixture by default so it works before the real download:

    python -m experiments.real_data              # fixture
    python -m experiments.real_data --real       # real TLC data, see DATA.md
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
    exact_shapley,
    leave_one_out,
    mrr,
    per_node_drift,
    precision_at_1,
    stagewise_single_permutation,
)
from culpa.nyctaxi import DataMissing
from culpa.taxi_pipeline import FAULT_SWEEPS, build_probe, taxi_scenario
from culpa.utility import make_utility

def _days() -> tuple:
    """--days REF,INC,PROBE  (default 8,9,10 -- Mon/Tue/Wed of 2024-01)."""
    for i, a in enumerate(sys.argv):
        if a == "--days" and i + 1 < len(sys.argv):
            d = [int(x) for x in sys.argv[i + 1].split(",")]
            if len(d) != 3:
                raise SystemExit("--days needs exactly REF,INC,PROBE")
            return tuple(d)
    return (8, 9, 10)


REF_DAY, INC_DAY, PROBE_DAY = _days()
EPS = 1e-9

# Below this, benign day-over-day drift is too small for the anchoring question
# to even arise: with no drift to confound it, plain Shapley and the anchored
# value must agree, and a comparison between them says nothing.
DRIFT_FLOOR = 0.002

# A configuration is diagnosable only if the incident run is actually *worse*
# than the reference run by a margin. Requiring |v(V)| >= eps is not enough: with
# real day-over-day drift a fault can be offset by a favourable partition, and
# v(V) comes out positive. There is then no degradation to attribute and every
# method is ranking noise. Excluding these is not cherry-picking -- attribution
# is undefined when nothing got worse -- but the count must be reported.
DEGRADED = -0.002

# A configuration is a valid test case only if the injected fault ALSO damages
# the model in isolation. Several do not: duplicating rows in the zone lookup
# produces textbook join fan-out, but because a left join on DOLocationID
# reproduces the same borough value it merely reweights the training set
# uniformly and costs nothing. Scoring an attribution method for failing to
# blame an operator that caused no harm penalises it for being right.
#
# This is the same lesson that bit us twice on the synthetic workloads: an
# injected fault is a hypothesis about damage, and it has to be checked.
FAULT_HARMFUL = -0.002


def main() -> None:
    fixture = "--real" not in sys.argv
    label = "FIXTURE (synthetic, real schema)" if fixture else "REAL NYC TLC DATA"
    print(f"{'=' * 96}\n{label}")
    print(f"reference day = 2024-01-{REF_DAY:02d}   incident day = 2024-01-{INC_DAY:02d}   "
          f"probe day = 2024-01-{PROBE_DAY:02d}\n")

    try:
        probe = build_probe(PROBE_DAY, fixture=fixture)
    except DataMissing as e:
        print(f"cannot run:\n{e}")
        sys.exit(1)

    utility = make_utility(probe)
    print(f"probe: {len(probe):,} trips, {probe['label'].mean():.3f} generous-tip rate, "
          f"{len(probe.columns) - 1} features\n")

    rows: List[Dict] = []

    for fault_type, (op_name, ladder) in FAULT_SWEEPS.items():
        print("=" * 96)
        print(f"FAULT: {fault_type}   (operator: {op_name})")
        print(f"{'severity':28s} {'v(V)':>9s} {'fault':>9s} {'drift':>9s} {'r':>7s} "
              f"{'shap':>5s} {'anch':>5s} {'loo':>5s} {'drift':>5s} {'stage':>5s}   top-blame")
        print("-" * 96)

        for level, params in ladder:
            pipe, culprits = taxi_scenario({op_name: params}, REF_DAY, INC_DAY, fixture)
            players = sorted(pipe.ancestors_of_sink())
            sources = sorted(pipe.exogenous())
            game = DegradationGame(pipe, utility)

            delta = game.total_degradation
            fault_alone = game.v(frozenset([op_name]))
            drift_alone = max(abs(game.v(frozenset([s]))) for s in sources)
            ratio = abs(fault_alone) / max(drift_alone, EPS)

            phi = exact_shapley(game, players)
            anch = anchored_shapley(game, players, sources)
            loo = leave_one_out(game, players)
            drift = {k: -v for k, v in per_node_drift(pipe).items() if k in players}
            stage = stagewise_single_permutation(game, pipe.all_topological_orders(1)[0])

            attrs = {"shapley": phi, "anchored": anch, "loo": loo,
                     "drift": drift, "stagewise": stage}
            p1 = {k: precision_at_1(a, culprits) for k, a in attrs.items()}
            degraded = delta <= DEGRADED
            harmful = fault_alone <= FAULT_HARMFUL
            valid = degraded and harmful

            flag = ""
            if not harmful:
                flag = "  (fault does no harm)"
            elif not degraded:
                flag = "  (no net degradation)"
            print(f"{level:28s} {delta:+9.4f} {fault_alone:+9.4f} {drift_alone:+9.4f} "
                  f"{ratio:7.2f} "
                  f"{p1['shapley']:5.0f} {p1['anchored']:5.0f} {p1['loo']:5.0f} "
                  f"{p1['drift']:5.0f} {p1['stagewise']:5.0f}   "
                  f"{blame_ranking(anch)[0]}{flag}")

            row = {
                "fault_type": fault_type, "severity": level, "culprit": op_name,
                "delta": delta, "fault_alone": fault_alone, "drift_alone": drift_alone,
                "ratio": ratio, "degraded": degraded, "harmful": harmful, "valid": valid,
                "n_operators": len(players),
                "lattice": 2 ** len(players), "model_fits": game.stats.model_fits,
                "cache_hit_rate": pipe.stats.cache_hit_rate,
                "frontier": len(pipe.active_frontier()),
                "anchored_eff_gap": abs(sum(anch.values())
                                        - (delta - game.v(frozenset(sources)))),
                "loo_eff_gap": abs(sum(loo.values()) - delta),
            }
            for k, a in attrs.items():
                row[f"{k}_p@1"] = p1[k]
                row[f"{k}_mrr"] = mrr(a, culprits)
            rows.append(row)
        print()

    # -- summary -------------------------------------------------------
    methods = ["shapley", "anchored", "loo", "drift", "stagewise"]
    live = [r for r in rows if r["valid"]]
    dead = [r for r in rows if not r["valid"]]

    print("=" * 96)
    print(f"SUMMARY -- {len(live)} valid test cases of {len(rows)} configurations\n")
    print(f"{'method':14s} {'p@1':>7s} {'MRR':>7s}")
    print("-" * 30)
    for m in methods:
        p = sum(r[f"{m}_p@1"] for r in live) / len(live)
        rr = sum(r[f"{m}_mrr"] for r in live) / len(live)
        print(f"{m:14s} {p:7.2f} {rr:7.2f}")

    bins = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e18)]
    labels = ["r < 0.5", "0.5 - 1", "1 - 2", "r > 2"]
    print(f"\np@1 by fault-to-drift ratio\n")
    print(f"{'bin':10s} {'n':>3s} " + " ".join(f"{m:>10s}" for m in methods))
    print("-" * 64)
    for (lo, hi), lab in zip(bins, labels):
        sel = [r for r in live if lo <= r["ratio"] < hi]
        if sel:
            print(f"{lab:10s} {len(sel):3d} " + " ".join(
                f"{sum(r[f'{m}_p@1'] for r in sel) / len(sel):10.2f}" for m in methods))

    print(f"\nanchored efficiency  max = {max(r['anchored_eff_gap'] for r in rows):.2e}")
    print(f"LOO      efficiency  max = {max(r['loo_eff_gap'] for r in rows):.2e}   "
          f"mean = {sum(r['loo_eff_gap'] for r in rows) / len(rows):.2e}")

    r0 = rows[0]
    print(f"\ncost: n = {r0['n_operators']} operators, {r0['lattice']} coalitions -> "
          f"{min(r['model_fits'] for r in rows)}-{max(r['model_fits'] for r in rows)} model fits, "
          f"cache hit {min(r['cache_hit_rate'] for r in rows):.1%}-"
          f"{max(r['cache_hit_rate'] for r in rows):.1%}, "
          f"frontier {min(r['frontier'] for r in rows)}-{max(r['frontier'] for r in rows)}")

    no_harm = [r for r in rows if not r["harmful"]]
    no_drop = [r for r in rows if r["harmful"] and not r["degraded"]]
    if no_harm:
        print(f"\nexcluded -- the injected fault causes no damage in isolation "
              f"(v(culprit) > {FAULT_HARMFUL}), so there is no culprit to find:\n  "
              + ", ".join(f"{r['fault_type']}/{r['severity']}" for r in no_harm))
    if no_drop:
        print(f"\nexcluded -- fault is harmful but benign drift offset it "
              f"(v(V) > {DEGRADED}), so nothing got worse overall:\n  "
              + ", ".join(f"{r['fault_type']}/{r['severity']}" for r in no_drop))

    tag = "fixture" if fixture else "tlc"
    name = f"real_data_{tag}_{REF_DAY}v{INC_DAY}.csv"
    out = Path(__file__).resolve().parents[1] / "results" / name
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")

    max_drift = max(r["drift_alone"] for r in rows)
    if max_drift < DRIFT_FLOOR:
        print("\n" + "!" * 96)
        print(f"BENIGN DRIFT IS NEGLIGIBLE: max |v(source)| = {max_drift:.5f} "
              f"< {DRIFT_FLOOR}")
        print("")
        print("The two partitions are near-identical in distribution, so there is no")
        print("confounding drift for anchoring to remove. Plain Shapley and the anchored")
        print("value necessarily agree here, and the p@1 comparison above carries no")
        print("information about which estimator is better. It shows only that the code")
        print("path runs end to end on this schema.")
        print("")
        print("To make this experiment informative, run it on partitions that genuinely")
        print("differ -- a weekday against a weekend, or either side of a holiday:")
        print("    python -m experiments.real_data --real --days 8,13,10")
        print("!" * 96)

    if fixture:
        print("\nNOTE: this ran on the fixture. The benign drift is synthetic, which is")
        print("exactly the weakness this experiment exists to remove. Re-run with")
        print("--real once the TLC files are in data/ (see DATA.md).")


if __name__ == "__main__":
    main()
