"""Pipeline representation and the hybrid-replay engine.

A pipeline is a DAG of operators. Every operator has two *states*:

  'R'  reference  -- how it behaved in the last known-good run
  'I'  incident   -- how it behaved in the degraded run

A *hybrid replay* executes the whole DAG with an arbitrary subset S of operators
in incident state and the rest in reference state. S = {} reproduces the good
run, S = V reproduces the bad one, and everything in between interpolates.

The engine memoises operator outputs on (operator, state, input hashes), so the
2^n replays of the coalition lattice share all their common prefixes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Sequence

import pandas as pd
from pandas.util import hash_pandas_object

REFERENCE = "R"
INCIDENT = "I"


def frame_hash(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame: schema + row values, order-sensitive.

    Order sensitivity is deliberate. Two operators that emit the same rows in a
    different order are *not* interchangeable downstream once a head/limit or a
    non-commutative aggregation is involved. Canonicalising row order for
    genuinely order-insensitive operators is listed as open question 5 in the
    proposal.
    """
    h = hashlib.sha256()
    h.update(repr(list(df.columns)).encode())
    h.update(repr([str(t) for t in df.dtypes]).encode())
    if len(df):
        h.update(hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()


@dataclass
class Operator:
    """One node of the pipeline.

    `fn(inputs, params) -> DataFrame` is the operator body, identical in both
    states. What differs between states is `ref_params` vs `inc_params`, which
    captures both causes of degradation uniformly: a changed source partition
    (params point at different data) and a changed config or code path (params
    change a threshold, a unit, a join key).
    """

    name: str
    parents: List[str]
    fn: Callable[[Dict[str, pd.DataFrame], dict], pd.DataFrame]
    ref_params: dict = field(default_factory=dict)
    inc_params: dict = field(default_factory=dict)

    def params_for(self, state: str) -> dict:
        return self.ref_params if state == REFERENCE else self.inc_params

    @property
    def state_sensitive(self) -> bool:
        """Whether this operator's two states differ at all."""
        return self.ref_params != self.inc_params


@dataclass
class ReplayStats:
    """Instrumentation for the cost claims in the paper."""

    operator_executions: int = 0
    operator_cache_hits: int = 0
    sink_hashes: Dict[str, FrozenSet[str]] = field(default_factory=dict)

    @property
    def cache_hit_rate(self) -> float:
        total = self.operator_executions + self.operator_cache_hits
        return self.operator_cache_hits / total if total else 0.0


class Pipeline:
    def __init__(self, operators: Sequence[Operator], sink: str):
        self.operators = {op.name: op for op in operators}
        self.sink = sink
        if sink not in self.operators:
            raise ValueError(f"sink {sink!r} is not an operator")
        self.order = self._topological_order()
        self._cache: Dict[tuple, tuple] = {}
        self.stats = ReplayStats()

    # -- structure -------------------------------------------------------

    def _topological_order(self) -> List[str]:
        """Kahn's algorithm, ties broken by name for determinism."""
        indeg = {n: 0 for n in self.operators}
        children: Dict[str, List[str]] = {n: [] for n in self.operators}
        for op in self.operators.values():
            for p in op.parents:
                if p not in self.operators:
                    raise ValueError(f"{op.name} references unknown parent {p!r}")
                indeg[op.name] += 1
                children[p].append(op.name)

        ready = sorted(n for n, d in indeg.items() if d == 0)
        order: List[str] = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for c in sorted(children[n]):
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
            ready.sort()

        if len(order) != len(self.operators):
            raise ValueError("pipeline contains a cycle")
        return order

    def all_topological_orders(self, limit: int = 64) -> List[List[str]]:
        """Enumerate distinct topological orders, up to `limit`.

        Used to demonstrate that single-permutation stagewise diagnosis -- what
        engineers do by hand today -- is order-dependent.
        """
        indeg = {n: 0 for n in self.operators}
        children: Dict[str, List[str]] = {n: [] for n in self.operators}
        for op in self.operators.values():
            for p in op.parents:
                indeg[op.name] += 1
                children[p].append(op.name)

        results: List[List[str]] = []

        def rec(order: List[str], indeg: Dict[str, int], avail: List[str]) -> None:
            if len(results) >= limit:
                return
            if not avail:
                results.append(list(order))
                return
            for n in list(avail):
                nxt_indeg = dict(indeg)
                nxt_avail = [a for a in avail if a != n]
                for c in children[n]:
                    nxt_indeg[c] -= 1
                    if nxt_indeg[c] == 0:
                        nxt_avail.append(c)
                order.append(n)
                rec(order, nxt_indeg, sorted(nxt_avail))
                order.pop()
                if len(results) >= limit:
                    return

        rec([], indeg, sorted(n for n, d in indeg.items() if d == 0))
        return results

    def ancestors_of_sink(self) -> FrozenSet[str]:
        """Operators that can reach the sink. Everything else is a null player
        by ancestral sufficiency (proposal section 4.1)."""
        keep = {self.sink}
        changed = True
        while changed:
            changed = False
            for name in self.operators:
                if name in keep:
                    continue
                if any(c in keep for c in self._children(name)):
                    keep.add(name)
                    changed = True
        return frozenset(keep)

    def exogenous(self) -> FrozenSet[str]:
        """Source operators -- no parents. Their state differs because the world
        moved, not because the pipeline did, so they are the principled default
        anchor set for `anchored_shapley`."""
        return frozenset(n for n, op in self.operators.items() if not op.parents)

    def _children(self, name: str) -> List[str]:
        return [op.name for op in self.operators.values() if name in op.parents]

    # -- execution -------------------------------------------------------

    def replay(self, coalition: FrozenSet[str]) -> pd.DataFrame:
        """Execute the DAG with every operator in `coalition` in incident state."""
        results: Dict[str, pd.DataFrame] = {}
        hashes: Dict[str, str] = {}

        for name in self.order:
            op = self.operators[name]
            state = INCIDENT if name in coalition else REFERENCE
            key = (name, state, tuple(hashes[p] for p in op.parents))

            if key in self._cache:
                self.stats.operator_cache_hits += 1
                out, out_hash = self._cache[key]
            else:
                self.stats.operator_executions += 1
                inputs = {p: results[p] for p in op.parents}
                out = op.fn(inputs, op.params_for(state))
                out_hash = frame_hash(out)
                self._cache[key] = (out, out_hash)

            results[name] = out
            hashes[name] = out_hash

        self.stats.sink_hashes.setdefault(hashes[self.sink], frozenset(coalition))
        return results[self.sink]

    def replay_all_nodes(self, coalition: FrozenSet[str]) -> Dict[str, pd.DataFrame]:
        """Same as `replay` but returns every intermediate. Used by the
        per-node drift baseline."""
        results: Dict[str, pd.DataFrame] = {}
        hashes: Dict[str, str] = {}
        for name in self.order:
            op = self.operators[name]
            state = INCIDENT if name in coalition else REFERENCE
            inputs = {p: results[p] for p in op.parents}
            out = op.fn(inputs, op.params_for(state))
            results[name] = out
            hashes[name] = frame_hash(out)
        return results

    def active_frontier(self) -> FrozenSet[str]:
        """Operators whose own state flip changes their output hash, holding
        inputs at the all-reference configuration.

        This is the cheap, sound-in-practice approximation of the active
        frontier from proposal section 4.2: it needs n+1 executions rather than
        2^n, and it is exact for operators whose behaviour does not depend on
        the input configuration. Operators outside it are candidates for null
        players and can be pruned before any model is trained.
        """
        base = self.replay_all_nodes(frozenset())
        frontier = set()
        for name in self.order:
            op = self.operators[name]
            if not op.state_sensitive:
                continue
            inputs = {p: base[p] for p in op.parents}
            flipped = op.fn(inputs, op.params_for(INCIDENT))
            if frame_hash(flipped) != frame_hash(base[name]):
                frontier.add(name)
        return frozenset(frontier)
