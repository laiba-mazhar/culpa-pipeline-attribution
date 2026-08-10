# Gate experiment findings

> **Resolved.** The reformulation proposed in §4 below was implemented and it
> works: the incident-anchored Shapley value reaches p@1 = 1.00 across every
> severity regime while keeping exact decomposition. See [PAPER.md](PAPER.md)
> §6.3–6.4. This document is kept as the record of how the negative result was
> found and what it forced.


Date: August 2026
Run log: `results/gate_run.txt` — reproduce with `python -m experiments.gate`

Setup: 7-operator churn feature pipeline, 4000 customers/day, benign
day-over-day source drift in *every* scenario, one or two injected faults on
top. 10 scenarios, 6 attribution methods, exact Shapley as ground truth.

**Verdict: the mechanism works, the framing was wrong.** Proceed, with the
reformulation in §4.

---

## 1. What held up

### 1.1 Efficiency is exact

Across all 10 scenarios, `Σφ_i − v(V)` is between `3e-18` and `6e-17` — floating
point noise. The attributions provably and numerically sum to the observed
degradation.

No baseline has this property. Leave-one-out overshoots badly: in
`starve_rows_a+starve_rows_b` its values sum to `−0.121` against an actual
degradation of `−0.0955` — a 27% over-attribution, because it double-counts
interacting faults. This is the one thing only Shapley gives you, and it should
be the headline, not the ranking accuracy.

### 1.2 Pruning is worth what the proposal claimed

| | value |
|---|---|
| coalitions in the lattice | 128 |
| model fits actually performed | **8** (single fault) / **16** (compound) |
| operator cache hit rate | 91.6% – 96.0% |
| active frontier | 3/7 or 4/7 operators |
| exact Shapley wall-clock | 0.2 – 0.6 s |

A **16× reduction** in model fits from output-hash pruning alone, on a
deliberately small DAG. The frontier is exactly the operators that were faulted
plus the two sources — the theory predicted this and it holds. This is the
systems result and it is solid.

### 1.3 Monte-Carlo converges cheaply

60 permutations gives max per-operator error of `0.005` AUC against exact
Shapley, on degradations ranging from `0.003` to `0.23`. Good enough, and it
means the method scales past the brute-force regime.

### 1.4 Current practice is provably and *visibly* arbitrary

This is the most quotable result in the run. Same incident, same DAG, two
different topological orders:

```
stagewise ord.A   load_customers=-0.0556   load_transactions=+0.0554
stagewise ord.B   load_customers=+0.0405   load_transactions=-0.0407
```

**The sign flips.** Two engineers running the identical manual procedure on the
identical incident reach opposite conclusions about which source is to blame,
purely from the order they happened to walk the DAG in. Across harmful
scenarios, stagewise scores p@1 = 0.38 against Shapley's 0.75.

This validates the Proposition in proposal §3.3 empirically, and it is a strong
motivating figure for the paper's first page.

### 1.5 Per-node drift monitoring fails exactly as predicted

In `null_spike_mnar`, per-node drift ranks `encode_features` (−1.48) and
`filter_active` (−0.56) above the true culprit `clean_customers`. Damage
propagates downstream, so the deepest node always looks worst. Drift magnitude
measures *depth in the DAG*, not blame. Confirmed.

---

## 2. The negative result

**Leave-one-out beats Shapley at ranking the culprit.**

| method | p@1 | MRR | blame mass |
|---|---|---|---|
| **leave-one-out** | **1.00** | **1.00** | **0.87** |
| drift | 0.88 | 0.94 | 0.51 |
| shapley (exact) | 0.75 | 0.88 | 0.62 |
| shapley (MC, 60) | 0.75 | 0.88 | 0.58 |
| stagewise A / B | 0.38 | 0.69 | 0.41 / 0.44 |

*(8 harmful scenarios)*

I predicted this as a risk in proposal §10 and assumed compound faults would
rescue it. They did not — on compound faults both hit p@1 = 1.00 and LOO still
leads on blame mass (0.72 vs 0.68).

### Why — and this is the actual discovery

Shapley loses precisely the two scenarios where the fault is *smaller than the
benign day-over-day drift*:

| scenario | v(V) | shapley p@1 | loo p@1 |
|---|---|---|---|
| `join_fanout_biased` | −0.0016 | 0 | 1 |
| `null_spike_mnar` | −0.0028 | 0 | 1 |

Benign source drift is worth ±0.005–0.008 AUC on its own. In these two
scenarios the injected fault does *less* damage than the natural churn in the
data. Shapley therefore ranks `load_customers` above the culprit — and it is
**not wrong to do so**. As accounting, that is the correct answer.

The two estimators answer different questions:

- **LOO** evaluates the marginal effect at the *actual incident configuration*,
  where every other operator is already in incident state. Benign drift is held
  fixed and cancels out of the comparison. It answers *"what do I fix now?"*
- **Shapley** averages over all `2^n` configurations, including many where the
  benign drift has not happened. Benign drift therefore receives its own real,
  nonzero share. It answers *"how is the total degradation composed?"*

When the fault dominates benign drift the two agree. When it does not, Shapley
is right as accounting and wrong as triage.

### Two more sub-findings worth keeping

**Not every data-quality violation degrades the model.** Uniform join fan-out
(35% duplicate keys) and MCAR nullness (55% of rows dropped) produce
`v(V) = +0.0001` and `+0.0006` — no damage at all, and CULPA correctly assigns
near-zero blame (`max|φ| ≤ 0.0085`). A constraint monitor fires loudly on both.
This is a direct argument for utility-grounded attribution over constraint
checking, and the two benign scenarios stay in the benchmark as a control.

**Row-count faults are much weaker than unit and schema faults.** Unit change
costs 0.115 AUC and schema drift 0.031, while duplication and nullness cost
<0.003 even when biased and severe. A well-specified linear model is robust to
losing rows and fragile to losing or rescaling columns. Worth reporting; it
means the benchmark needs a calibrated severity dial rather than hand-tuned
fault parameters.

---

## 3. What this costs the paper

The claim *"Shapley attribution identifies the culprit better than existing
methods"* is **not supported** and must be dropped. Do not write it.

What survives, and is enough:

1. Exact decomposition of degradation — unique to Shapley, verified to 1e-17.
2. 16× fit reduction from output-hash pruning; 91–96% cache hit rate.
3. Current stagewise practice is order-dependent to the point of sign reversal.
4. Constraint-based data quality flags faults that do not matter and misses the
   ranking of ones that do.
5. A characterisation of when marginal (LOO) and averaged (Shapley) attribution
   diverge, with the benign-drift mechanism as the explanation.

Item 5 was not in the original proposal. It is now the most interesting thing
here.

---

## 4. The reformulation

The gap between LOO and Shapley is a *conditioning* choice, and the space
between them is unexplored. Propose an **incident-anchored Shapley value**:
average marginal contributions only over coalitions in which a designated set
`B` of benign operators is held at incident state.

```
φ_i^anchor = Σ_{S ⊆ V∖(B ∪ {i})} w(|S|) · [ v(S ∪ B ∪ {i}) − v(S ∪ B) ]
```

- `B = ∅` recovers standard Shapley (pure accounting).
- `B = V∖{i}` recovers leave-one-out (pure triage).
- Intermediate `B` gives a tunable estimator, and `B` can be *chosen* — e.g. all
  operators whose state change is attributable to external source movement
  rather than to pipeline code.

This should: keep efficiency on the sub-game over `V∖B`, inherit LOO's ranking
accuracy, and give a principled account of what to hold fixed when you diagnose.
Deriving its axioms and the exact relationship to the conjunctive permission
value of proposal §3.2 is now the theoretical core.

**This is a better paper than the one I proposed.** It has a real negative
result driving a real reformulation, which is what referees want to see.

---

## 5. Immediate next steps

- [ ] Implement `φ^anchor`; verify it interpolates LOO and Shapley as `B` grows.
- [ ] Prove efficiency on the sub-game and characterise the axioms it satisfies.
- [ ] Build a calibrated severity dial so every fault type spans
      0.005 → 0.20 AUC; re-run the whole grid as a function of
      fault-severity ÷ benign-drift ratio. **That ratio is the x-axis of the
      paper's main figure** — the crossover point where LOO and Shapley diverge.
- [ ] Add a genuinely superadditive fault pair. `starve_rows` was designed to be
      one and was not: LOO still got p@1 = 1.00, though it did claim
      `clean_customers` *helped* by +0.04 when its true φ is +0.005.
- [ ] Scale study on the synthetic DAG generator: n = 10…40, measure how
      frontier size and cache hit rate move with depth and branching.
- [ ] Move off synthetic data — NYC Taxi has real day-over-day partitions and
      real benign drift, which is the whole point.
- [ ] Only then: Airflow integration. It is packaging, not science, and it can
      wait until the estimator is settled.
