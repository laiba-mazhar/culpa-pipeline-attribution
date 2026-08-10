"""The cooperative game, its Shapley value, and the baselines we compare to.

Players are pipeline operators. The coalition S is the set of operators running
in incident state. The value v(S) = u(S) - u(empty) is the change in downstream
model utility, so v(V) is exactly the observed degradation and -- by Shapley
efficiency -- the attributions sum to it.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Sequence

import numpy as np

from .pipeline import Pipeline, frame_hash

UtilityFn = Callable[["object"], float]


@dataclass
class GameStats:
    utility_calls: int = 0
    model_fits: int = 0
    distinct_sink_outputs: int = 0


class DegradationGame:
    """v(S) with two layers of memoisation.

    1. coalition -> value, the obvious one.
    2. sink content hash -> value. This is the payoff of output-stability
       pruning: distinct coalitions that produce a byte-identical sink share a
       utility, so the expensive model fit happens once. Every hit here is a
       model fit that the naive 2^n enumeration would have paid for.
    """

    def __init__(self, pipeline: Pipeline, utility: UtilityFn):
        self.pipeline = pipeline
        self.utility = utility
        self.stats = GameStats()
        self._by_coalition: Dict[FrozenSet[str], float] = {}
        self._by_sink_hash: Dict[str, float] = {}
        self._baseline = self._raw_utility(frozenset())

    def _raw_utility(self, coalition: FrozenSet[str]) -> float:
        self.stats.utility_calls += 1
        sink = self.pipeline.replay(coalition)
        h = frame_hash(sink)
        if h in self._by_sink_hash:
            return self._by_sink_hash[h]
        self.stats.model_fits += 1
        val = self.utility(sink)
        self._by_sink_hash[h] = val
        self.stats.distinct_sink_outputs = len(self._by_sink_hash)
        return val

    def u(self, coalition: FrozenSet[str]) -> float:
        if coalition not in self._by_coalition:
            self._by_coalition[coalition] = self._raw_utility(coalition)
        return self._by_coalition[coalition]

    def v(self, coalition: FrozenSet[str]) -> float:
        return self.u(coalition) - self._baseline

    @property
    def total_degradation(self) -> float:
        """v(V): the whole drop we are decomposing. Negative when the incident
        run is worse, which is the case of interest."""
        return self.v(frozenset(self.pipeline.operators))


# -- attribution methods -------------------------------------------------


def exact_shapley(game: DegradationGame, players: Sequence[str]) -> Dict[str, float]:
    """Brute-force Shapley over 2^n coalitions. Tractable to about n = 14 and
    used as ground truth for the Monte-Carlo estimator."""
    players = list(players)
    n = len(players)
    phi = {p: 0.0 for p in players}

    for i, player in enumerate(players):
        others = [p for p in players if p != player]
        for size in range(n):
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for combo in itertools.combinations(others, size):
                S = frozenset(combo)
                phi[player] += weight * (game.v(S | {player}) - game.v(S))
    return phi


def mc_shapley(
    game: DegradationGame,
    players: Sequence[str],
    n_permutations: int = 200,
    seed: int = 0,
) -> Dict[str, float]:
    """Permutation sampling. Each permutation walks the coalition from empty to
    full, charging every player its marginal contribution at the moment it is
    added -- so one permutation costs n+1 utility evaluations, most of which hit
    the memo."""
    rng = np.random.default_rng(seed)
    players = list(players)
    totals = {p: 0.0 for p in players}

    for _ in range(n_permutations):
        perm = list(rng.permutation(players))
        S: FrozenSet[str] = frozenset()
        prev = game.v(S)
        for p in perm:
            S = S | {p}
            cur = game.v(S)
            totals[p] += cur - prev
            prev = cur

    return {p: t / n_permutations for p, t in totals.items()}


def anchored_shapley(
    game: DegradationGame, players: Sequence[str], anchor: Sequence[str]
) -> Dict[str, float]:
    """Incident-anchored Shapley value.

    Hold the operators in `anchor` at incident state in every coalition, and
    take the Shapley value of the resulting sub-game on the remaining players:

        v_B(T) = v(T u B) - v(B)

    This is exactly the estimator the gate experiment called for. It is a single
    knob spanning the two things practitioners actually compute:

        anchor = {}          ->  standard Shapley       (accounting)
        anchor = V \\ {i}     ->  leave-one-out          (triage)

    Efficiency survives on the sub-game: the values sum to v(V) - v(B), i.e.
    they decompose exactly the part of the degradation that is *not* already
    explained by the anchored operators.

    The principled default is to anchor the **exogenous** operators -- sources,
    whose state differs because the world moved, not because the pipeline did.
    You cannot "fix" the fact that Tuesday's data is not Monday's, so charging
    blame to it is accurate accounting and useless triage.
    """
    anchor = frozenset(anchor)
    free = [p for p in players if p not in anchor]
    m = len(free)
    phi = {p: 0.0 for p in free}
    if m == 0:
        return phi

    for player in free:
        others = [p for p in free if p != player]
        for size in range(m):
            weight = math.factorial(size) * math.factorial(m - size - 1) / math.factorial(m)
            for combo in itertools.combinations(others, size):
                S = frozenset(combo) | anchor
                phi[player] += weight * (game.v(S | {player}) - game.v(S))

    for p in anchor:
        phi[p] = 0.0  # anchored: held fixed, not attributed
    return phi


def mc_anchored_shapley(
    game: DegradationGame,
    players: Sequence[str],
    anchor: Sequence[str],
    n_permutations: int = 200,
    seed: int = 0,
) -> Dict[str, float]:
    """Permutation-sampled `anchored_shapley`, for DAGs past the brute-force
    regime. Each permutation starts from the anchor set rather than the empty
    coalition and walks the free players in random order."""
    rng = np.random.default_rng(seed)
    anchor = frozenset(anchor)
    free = [p for p in players if p not in anchor]
    totals = {p: 0.0 for p in free}

    for _ in range(n_permutations):
        S = anchor
        prev = game.v(S)
        for p in rng.permutation(free):
            S = S | {p}
            cur = game.v(S)
            totals[p] += cur - prev
            prev = cur

    out = {p: t / n_permutations for p, t in totals.items()}
    for p in anchor:
        out[p] = 0.0
    return out


def leave_one_out(game: DegradationGame, players: Sequence[str]) -> Dict[str, float]:
    """Baseline B5. 'How much damage would freezing this one operator have
    avoided?' Costs n+1 evaluations, and is the method Shapley must beat.

    Note it does not sum to the total degradation -- with interacting faults it
    can badly under- or over-count.
    """
    full = frozenset(game.pipeline.operators)
    v_full = game.v(full)
    return {p: v_full - game.v(full - {p}) for p in players}


def stagewise_single_permutation(
    game: DegradationGame, order: Sequence[str]
) -> Dict[str, float]:
    """Baseline B4: what engineers actually do. Walk the DAG in one topological
    order, swap in incident data one stage at a time, watch where the metric
    moves. This is the marginal-contribution vector of a single permutation --
    i.e. one sample of the average that Shapley computes."""
    S: FrozenSet[str] = frozenset()
    prev = game.v(S)
    out: Dict[str, float] = {}
    for name in order:
        S = S | {name}
        cur = game.v(S)
        out[name] = cur - prev
        prev = cur
    return out


def per_node_drift(pipeline: Pipeline) -> Dict[str, float]:
    """Baseline B3: run the pipeline fully-reference and fully-incident, and
    score every node by how much its own output moved. This is what a drift
    monitor attached to each task reports.

    Its structural flaw is the point of the experiment: benign upstream change
    propagates, so every downstream node lights up and the ranking is dominated
    by depth rather than by blame.
    """
    ref = pipeline.replay_all_nodes(frozenset())
    inc = pipeline.replay_all_nodes(frozenset(pipeline.operators))

    scores: Dict[str, float] = {}
    for name in pipeline.order:
        a, b = ref[name], inc[name]
        shared = [c for c in a.columns if c in b.columns]
        num = [c for c in shared if np.issubdtype(a[c].dtype, np.number)]
        if not num:
            scores[name] = float(set(a.columns) != set(b.columns))
            continue
        # Normalised mean shift across shared numeric columns, plus a schema and
        # a cardinality term -- a deliberately generous stand-in for PSI/KS.
        parts = []
        for c in num:
            av, bv = a[c].astype(float), b[c].astype(float)
            scale = av.std() or 1.0
            parts.append(abs(av.mean() - bv.mean()) / scale)
        shift = float(np.mean(parts)) if parts else 0.0
        card = abs(len(a) - len(b)) / max(len(a), 1)
        schema = float(set(a.columns) != set(b.columns))
        scores[name] = shift + card + schema
    return scores


# -- scoring -------------------------------------------------------------


def blame_ranking(attribution: Dict[str, float]) -> List[str]:
    """Rank by magnitude of harm. Degradation is negative, so the most harmful
    operator is the most negative; we rank by -phi and break ties by name."""
    return [k for k, _ in sorted(attribution.items(), key=lambda kv: (kv[1], kv[0]))]


def precision_at_1(attribution: Dict[str, float], culprits: Sequence[str]) -> float:
    return float(blame_ranking(attribution)[0] in culprits)


def mrr(attribution: Dict[str, float], culprits: Sequence[str]) -> float:
    ranking = blame_ranking(attribution)
    for i, name in enumerate(ranking, start=1):
        if name in culprits:
            return 1.0 / i
    return 0.0


def blame_mass_on_culprits(
    attribution: Dict[str, float], culprits: Sequence[str]
) -> float:
    """Fraction of total *harmful* attribution mass landing on true culprits.
    The metric that matters for compound faults."""
    harmful = {k: -v for k, v in attribution.items() if v < 0}
    total = sum(harmful.values())
    if total <= 0:
        return 0.0
    return sum(v for k, v in harmful.items() if k in culprits) / total
