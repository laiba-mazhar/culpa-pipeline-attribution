"""One concrete incident, end to end -- the thing to show someone first.

Takes a real NYC taxi pipeline, injects two independent faults into the same
run, and prints the blame CULPA assigns. Everything printed here is computed,
not narrated.

Run:  python -m experiments.demo [--fixture]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from culpa.game import DegradationGame, anchored_shapley
from culpa.taxi_pipeline import build_probe, taxi_scenario
from culpa.utility import make_utility

REF, INC, PROBE = 8, 13, 10

# A compound incident: two independent faults land in the same run, on different
# branches of the DAG. This is the case that separates decomposition from
# ranking -- a method that only ranks tells you which is worse, but not how the
# lost AUC divides between them, and that division is what decides whether you
# fix one or both.
INCIDENT_FAULTS = {
    # A filter predicate flips and silently drops the short-trip stratum.
    "filter_valid": {"min_fare": 20.0},
    # An upstream rename goes unnoticed and three feature columns disappear.
    "encode": {"drop_columns": ["trip_distance", "duration", "fare_amount"]},
}


def bar(value: float, scale: float, width: int = 28) -> str:
    """A blame bar anchored at zero: harm left, help right."""
    n = int(round(abs(value) / scale * width)) if scale else 0
    if abs(value) < 1e-9:
        return " " * width + "|" + " " * width
    if value < 0:
        return " " * (width - n) + "#" * n + "|" + " " * width
    return " " * width + "|" + "#" * n + " " * (width - n)


def main() -> None:
    fixture = "--fixture" in sys.argv
    pipe, culprits = taxi_scenario(INCIDENT_FAULTS, REF, INC, fixture)
    players = sorted(pipe.ancestors_of_sink())
    sources = sorted(pipe.exogenous())
    game = DegradationGame(pipe, make_utility(build_probe(PROBE, fixture)))

    delta = game.total_degradation
    anch = anchored_shapley(game, players, sources)
    v_anchor = game.v(frozenset(sources))

    print()
    print("  INCIDENT".ljust(64))
    print("  " + "-" * 62)
    print(f"  pipeline          NYC yellow taxi, {len(players)} operators")
    print(f"  reference run     2024-01-{REF:02d}  (Monday)")
    print(f"  incident run      2024-01-{INC:02d}  (Saturday)")
    print(f"  model             generous-tip classifier")
    print(f"  what changed      nothing was deployed")
    print()
    print(f"  AUC dropped by    {abs(delta):.4f}")
    print()

    print("  BLAME".ljust(64))
    print("  " + "-" * 62)
    scale = max(abs(v) for v in anch.values()) or 1.0
    ranked = sorted(anch.items(), key=lambda kv: kv[1])
    for name, val in ranked:
        mark = "  <-- culprit" if name in culprits else ""
        anchor_mark = "  (anchored: exogenous)" if name in sources else ""
        print(f"  {name:>18s} {val:+8.4f}  {bar(val, scale)}{mark}{anchor_mark}")

    print()
    print(f"  {'sum':>18s} {sum(anch.values()):+8.4f}")
    print(f"  {'to explain':>18s} {delta - v_anchor:+8.4f}   "
          f"(v(V) minus the anchored drift)")
    print(f"  {'gap':>18s} {abs(sum(anch.values()) - (delta - v_anchor)):.2e}")
    print()

    print("  COST".ljust(64))
    print("  " + "-" * 62)
    print(f"  coalitions in the lattice   {2 ** len(players):,}")
    print(f"  model fits actually run     {game.stats.model_fits}")
    print(f"  operator cache hit rate     {pipe.stats.cache_hit_rate:.1%}")
    print(f"  active frontier             {len(pipe.active_frontier())} of {len(players)} operators")
    print()

    out = Path(__file__).resolve().parents[1] / "results" / "demo_attribution.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "fixture": fixture, "ref_day": REF, "inc_day": INC, "probe_day": PROBE,
        "faults": {k: str(v) for k, v in INCIDENT_FAULTS.items()},
        "culprits": culprits, "delta": delta,
        "anchor_value": v_anchor, "attribution": anch,
        "operators": players, "sources": sources,
        "lattice": 2 ** len(players), "model_fits": game.stats.model_fits,
        "cache_hit_rate": pipe.stats.cache_hit_rate,
        "frontier": sorted(pipe.active_frontier()),
    }, indent=2), encoding="utf-8")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
