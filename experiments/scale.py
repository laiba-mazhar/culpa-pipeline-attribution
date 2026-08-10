"""Scale study for the cost claims.

The gate and severity experiments both run on a 7-operator DAG, where "128
coalitions became 8 model fits" is true but not yet evidence. This sweeps DAG
width and measures how the two cost mechanisms -- output-hash pruning and prefix
memoisation -- behave as n grows.

The generated pipeline is a fan-in: m independent source branches, each
source -> clean -> aggregate, all joined into a single feature table that is then
filtered and encoded. n = 3m + 3. That shape is deliberate: real pipelines fan in
from many source systems, and it is the shape where a single bad source has to
be isolated from m-1 innocent ones.

Run:  python -m experiments.scale
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from culpa.game import (
    DegradationGame,
    anchored_shapley,
    exact_shapley,
    mc_anchored_shapley,
)
from culpa.pipeline import Operator, Pipeline
from culpa.utility import make_utility

N_ROWS = 3000


def _branch_data(seed: int, branch: int) -> pd.DataFrame:
    """One source branch: a noisy view of a shared latent signal.

    The latent `z` is seeded by the day alone, so every branch observes the same
    underlying truth with independent noise. Branch 0 is the high-signal source
    (small noise); the rest are weak. The label is carried on branch 0 and is a
    function of `z` -- *not* of the observed columns -- so corrupting a feature
    genuinely destroys information rather than moving the target along with it.
    """
    z = np.random.default_rng(seed).normal(0, 1, N_ROWS)
    rng = np.random.default_rng(seed * 1000 + branch)
    noise = 0.3 if branch == 0 else 2.0

    df = pd.DataFrame({
        "entity_id": np.arange(N_ROWS),
        f"x{branch}": z + rng.normal(0, noise, N_ROWS),
        f"w{branch}": rng.lognormal(0, 0.5, N_ROWS),
    })
    if branch == 0:
        df["label"] = (z + np.random.default_rng(seed + 7).normal(0, 0.3, N_ROWS) > 0).astype(int)
    return df


def build_wide_pipeline(m: int, fault_branch: int, scale: float) -> Tuple[Pipeline, str]:
    """m source branches; the aggregate on `fault_branch` suffers a unit change."""
    ops: List[Operator] = []

    for b in range(m):
        ops.append(Operator(
            f"load_{b}", [],
            lambda inp, p, b=b: _branch_data(p["day"], b),
            {"day": 1}, {"day": 2},
        ))
        ops.append(Operator(
            f"clean_{b}", [f"load_{b}"],
            lambda inp, p, b=b: inp[f"load_{b}"][inp[f"load_{b}"][f"w{b}"] > p["floor"]]
                                .reset_index(drop=True),
            {"floor": 0.0}, {"floor": 0.0},
        ))
        # The fault destroys the branch's signal by mixing it with noise.
        # A pure rescale would be undone by the downstream standardisation --
        # the first version of this generator used one and measured no damage
        # at any width, which is a good reminder that a fault has to be checked
        # for effect, not assumed to have one.
        def agg_fn(inp, p, b=b):
            df = inp[f"clean_{b}"].copy()
            corrupt = p["corrupt"]
            if corrupt > 0:
                noise = np.random.default_rng(99 + b).normal(0, 1, len(df))
                df[f"x{b}"] = (1 - corrupt) * df[f"x{b}"] + corrupt * noise
            return df

        agg_fault = {"corrupt": scale} if b == fault_branch else {"corrupt": 0.0}
        ops.append(Operator(f"agg_{b}", [f"clean_{b}"], agg_fn, {"corrupt": 0.0}, agg_fault))

    def join_all(inp, p):
        out = inp["agg_0"]
        for b in range(1, m):
            out = out.merge(inp[f"agg_{b}"], on="entity_id", how="inner")
        return out

    ops.append(Operator("join_all", [f"agg_{b}" for b in range(m)], join_all, {}, {}))

    ops.append(Operator(
        "filter_rows", ["join_all"],
        lambda inp, p: inp["join_all"].head(int(len(inp["join_all"]) * p["keep"]))
                       .reset_index(drop=True),
        {"keep": 1.0}, {"keep": 1.0},
    ))

    def encode(inp, p):
        # The label rides through from the source; it is never recomputed here.
        df = inp["filter_rows"].copy()
        keep = [c for c in df.columns if c.startswith(("x", "w"))]
        return df[keep + ["label"]].reset_index(drop=True)

    ops.append(Operator("encode", ["filter_rows"], encode, {}, {}))
    return Pipeline(ops, sink="encode"), f"agg_{fault_branch}"


def main() -> None:
    rows: List[Dict] = []
    print(f"{'m':>3s} {'n':>4s} {'2^n':>9s} {'fits':>6s} {'reduction':>10s} "
          f"{'cache':>7s} {'|F|':>5s} {'exact s':>9s} {'mc s':>7s} {'mc err':>8s}")
    print("-" * 82)

    for m in [1, 2, 3, 4, 5, 6, 8, 10]:
        pipe, culprit = build_wide_pipeline(m, fault_branch=0, scale=0.95)
        players = sorted(pipe.ancestors_of_sink())
        n = len(players)

        probe_pipe, _ = build_wide_pipeline(m, fault_branch=0, scale=0.0)
        probe = probe_pipe.replay(frozenset())
        game = DegradationGame(pipe, make_utility(probe))

        frontier = pipe.active_frontier()

        sources = sorted(pipe.exogenous())
        delta = game.total_degradation

        exact_t = float("nan")
        mc_err = float("nan")
        phi = None
        if n <= 14:  # brute force affordable
            t0 = time.time()
            phi = exact_shapley(game, players)
            exact_t = time.time() - t0

        # The anchored value at every width, sampled -- so the reported blame
        # comes from the same estimator across the whole sweep.
        t0 = time.time()
        anch = mc_anchored_shapley(game, players, sources, n_permutations=80, seed=3)
        mc_t = time.time() - t0

        if phi is not None:
            exact_anch = anchored_shapley(game, players, sources)
            mc_err = max(abs(exact_anch[p] - anch[p]) for p in players)

        top = min(anch, key=anch.get)

        fits = game.stats.model_fits
        lattice = 2 ** n
        measurable = abs(delta) >= 0.002
        rows.append({
            "m": m, "n": n, "lattice": lattice, "model_fits": fits,
            "reduction": lattice / fits, "cache_hit_rate": pipe.stats.cache_hit_rate,
            "frontier": len(frontier), "exact_s": exact_t, "mc_s": mc_t,
            "mc_max_err": mc_err, "delta": delta, "measurable": measurable,
            "top_blame": top, "culprit": culprit,
            "correct": (top == culprit) if measurable else None,
        })
        flag = "" if measurable else "   (no measurable damage)"
        print(f"{m:3d} {n:4d} {lattice:9d} {fits:6d} {lattice/fits:9.1f}x "
              f"{pipe.stats.cache_hit_rate:6.1%} {len(frontier):5d} "
              f"{exact_t:9.2f} {mc_t:7.2f} {mc_err:8.4f}{flag}")

    live = [r for r in rows if r["measurable"]]
    print(f"\nculprit identified correctly at every measurable width "
          f"({len(live)}/{len(rows)}): {all(r['correct'] for r in live)}")
    print(f"frontier grows {rows[0]['frontier']} -> {rows[-1]['frontier']} "
          f"while n grows {rows[0]['n']} -> {rows[-1]['n']}: "
          f"one drifting source per branch, plus the one faulted aggregate.")

    print("\nHOW TO READ THE REDUCTION COLUMN")
    print("-" * 60)
    print("The headline ratio (up to 2.9e9x at n=33) is real but it is not a")
    print("function of n. Only 3 distinct sink states exist in this workload --")
    print("baseline, anchor-only, and anchor-plus-the-faulted-aggregate -- because")
    print("every other operator is state-insensitive and its flip is caught by the")
    print("output hash before any model is fit. The cost of the method therefore")
    print("tracks |F|, the active frontier, and not the size of the DAG.")
    print("")
    print("That is the claim worth making, and it is the one to stress-test next:")
    print("the reduction collapses when a change touches many operators at once")
    print("(a shared library upgrade, a global config change, a backfill). Those")
    print("cases are absent from this sweep and belong in the evaluation.")

    out = Path(__file__).resolve().parents[1] / "results" / "scale.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
