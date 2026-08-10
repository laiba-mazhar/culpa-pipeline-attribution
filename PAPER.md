# Anchored Counterfactual Attribution of Model Degradation to ETL Pipeline Operators

**Laiba Mazhar**

Draft v1 — August 2026. All numbers produced by `experiments/` in this
repository; logs in `results/`.

---

## Abstract

Production machine learning models are fed by ETL pipelines of tens to hundreds
of operators. When a model's quality drops between two scheduled runs with no
code deployed, engineers must determine which operator is responsible. Current
practice is to walk the pipeline in topological order, swapping in the new run's
data one stage at a time and watching where the metric moves. We show this
procedure is the marginal-contribution vector of a single permutation, and is
therefore order-dependent: on our benchmark the *sign* of the attributed blame
reverses between two valid topological orders of the same DAG.

We formulate the problem as a cooperative game. Holding the pipeline fixed, each
operator can execute in either of two states — as it did in a known-good
*reference* run, or as it did in the degraded *incident* run — and a *hybrid
replay* executes the DAG under an arbitrary assignment of states. The resulting
game has the property that its Shapley value decomposes the observed degradation
exactly across operators.

Our central finding is negative and then constructive. The Shapley value is
correct as accounting but poor as triage: it ranks the true culprit first in only
60% of measurable incidents, losing precisely when the injected fault does less
damage than the benign day-over-day drift that co-occurs with it. Leave-one-out
ablation ranks perfectly but its attributions do not sum to anything — they
deviate from the observed degradation by up to 0.246 AUC, more than the entire
incident. We introduce the **incident-anchored Shapley value**, which holds a
designated set of exogenous operators at incident state and takes the Shapley
value of the resulting sub-game. It recovers standard Shapley and leave-one-out
as its two boundary cases, attains 100% top-1 accuracy across every severity
regime, and retains exact decomposition to within 3×10⁻¹⁷.

On real NYC taxi partitions the separation is sharper still: comparing a Monday
against a Saturday, leave-one-out collapses to 0.50 — matching plain Shapley and
current practice — while the anchored value holds at 1.00, because real
weekday/weekend drift moves the utility more than the injected fault does.

We make attribution tractable with two mechanisms — pruning operators whose
output hash is invariant under a state flip, and memoising operator outputs
across the coalition lattice — reducing 2³³ hybrid replays to 3 model fits on a
33-operator DAG. We show that this reduction tracks the size of the *active
frontier* rather than the size of the DAG, and identify the regime where it
collapses.

---

## 1. Introduction

A recommendation model scores 0.87 AUC on Monday. On Tuesday the same Airflow
DAG runs, nothing was deployed, and it scores 0.81. Which task caused it?

This is the most common failure mode in production machine learning, and the
tooling addresses it poorly:

- **Monitoring** (Deequ, Great Expectations, Evidently) reports *that* a
  distribution moved. In a deep DAG a single upstream change makes every
  downstream node move, so a monitor fires everywhere and ranks nothing.
- **Lineage** (OpenLineage, dbt) reports *what is upstream of what*, narrowing
  the suspects to "everything upstream."
- **Pipeline debuggers** — `mlinspect` [3, 4] is the strongest — instrument a
  pipeline into a dataflow DAG and propagate annotations to detect
  data-distribution bugs *within one execution*. They do not compare executions
  across time and do not assign quantitative blame.

So the work is done by hand, and the manual procedure has a defect nobody has
named: it depends on the order the engineer happens to walk the DAG in.

### Contributions

1. **Formulation** (§3). Temporal hybrid replay of a fixed pipeline over two
   snapshots as a cooperative game whose Shapley value decomposes observed
   degradation exactly. Verified numerically to 6×10⁻¹⁷.
2. **A negative result** (§6.2). The Shapley value is the wrong estimator for
   triage. It ranks the culprit first in 60% of measurable incidents, and fails
   exactly when the fault is smaller than co-occurring benign drift.
3. **The anchored Shapley value** (§4). A one-parameter family spanning standard
   Shapley and leave-one-out. It attains 100% top-1 accuracy while preserving
   exact decomposition, which leave-one-out does not have.
4. **Tractability** (§5). Output-hash pruning and prefix memoisation, reducing
   2ⁿ replays to a count governed by the active frontier, with the collapse
   regime identified honestly.
5. **A fault-injection benchmark** (§6.1, §6.6) over both synthetic and real NYC
   taxi pipelines, with ground-truth culprits, a monotone severity ladder per
   fault type, and **benign controls** — faults that violate data-quality
   constraints loudly while causing no model damage.
6. **A real-data result** (§6.6) showing that leave-one-out, which is
   indistinguishable from the anchored value on synthetic data, fails under real
   weekday/weekend drift.

### What we do not claim

We do not have a production deployment, we have not integrated with an
orchestrator, our real-data evaluation is one dataset and one pipeline topology,
and the anchored estimator's axiomatic characterisation is stated but not proven.
§8 is explicit about all of this.

---

## 2. Problem statement

A pipeline is a DAG `G = (V, E)` of operators. Each operator `o_i` computes
`d_i = f_i(inputs, params_i)`.

Two executions are given: a **reference** run `R` (last known-good) and an
**incident** run `I` (degraded). Between them, each operator may differ in its
source partition, its configuration, or its code. We encode all three uniformly
as an operator **state** `σ_i ∈ {R, I}`.

A downstream model is trained on the sink dataset and evaluated on a fixed probe
set. Writing `u` for that utility, the observed degradation is

```
Δ = u(all operators in state I) − u(all operators in state R)
```

**The attribution problem.** Distribute `Δ` across operators such that the
assignment identifies which operators to repair.

---

## 3. Hybrid replay and the degradation game

**Definition 1 (hybrid replay).** For `S ⊆ V`, `Π(S)` executes the DAG with
every operator in `S` in incident state and every operator in `V∖S` in reference
state:

```
d_i(S) = f_i( { d_j(S) : (o_j, o_i) ∈ E },  params_i^{σ_i(S)} ),
         σ_i(S) = I if o_i ∈ S else R
```

`Π(∅)` reproduces the good run, `Π(V)` the bad one, and the `2ⁿ − 2` hybrids in
between interpolate.

**Definition 2 (degradation game).** With `u(S)` the model utility on `Π(S)`'s
output,

```
v(S) = u(S) − u(∅)
```

Then `v(∅) = 0` and `v(V) = Δ`.

**Design decision.** The probe set is held fixed across every coalition. If it
varied, `u(S)` would compare different things and every property below would be
vacuous. We use a clean held-out sample from the true data-generating process;
§8 discusses what to do when no such probe exists.

The Shapley value of this game,

```
φ_i = Σ_{S ⊆ V∖{i}} [ |S|!(n−|S|−1)!/n! ] · [ v(S ∪ {i}) − v(S) ]
```

satisfies **efficiency**: `Σ_i φ_i = v(V) = Δ`. The degradation is *decomposed*,
not merely scored. No baseline we are aware of has this property.

### 3.1 Current practice is a single permutation

Walking the DAG in topological order and swapping stages in one at a time
produces, for order `π`,

```
m_i^π = v(S_π(i) ∪ {i}) − v(S_π(i))
```

where `S_π(i)` is the set preceding `i`. This is the marginal-contribution
vector of one permutation. The Shapley value averages over all `n!` of them.

**Proposition 1.** Single-permutation stagewise diagnosis violates the symmetry
axiom and is order-dependent.

The empirical form of this is stark. On the same incident and the same DAG, two
valid topological orders give:

```
order A    load_customers = −0.0556    load_transactions = +0.0554
order B    load_customers = +0.0405    load_transactions = −0.0407
```

The sign reverses. Two engineers running the identical manual procedure on the
identical incident reach opposite conclusions about which source is to blame.

---

## 4. The incident-anchored Shapley value

§6.2 shows that the Shapley value, despite its exactness, ranks culprits worse
than naive leave-one-out ablation. The reason is a conditioning difference:

- **Leave-one-out** evaluates `v(V) − v(V∖{i})` — the marginal effect *at the
  actual incident configuration*, where every other operator is already in
  incident state. Benign drift is held fixed and cancels.
- **Shapley** averages over all `2ⁿ` configurations, including many where the
  benign drift has not happened. Benign drift therefore receives a real,
  nonzero share.

Both are right about different questions. The space between them is a
conditioning choice, and it is empty in the literature.

**Definition 3 (incident-anchored Shapley value).** For an anchor set `B ⊆ V`,
let `v_B(T) = v(T ∪ B) − v(B)` be the sub-game on `V∖B`. The anchored value
`φ^B` is the Shapley value of `v_B`:

```
φ_i^B = Σ_{S ⊆ V∖(B∪{i})} [ |S|!(m−|S|−1)!/m! ] · [ v(S∪B∪{i}) − v(S∪B) ],
        m = |V∖B|,   φ_i^B = 0 for i ∈ B
```

**Theorem 1 (boundary cases).**
- `B = ∅` gives the standard Shapley value.
- `B = V∖{i}` gives `φ_i^B = v(V) − v(V∖{i})`, the leave-one-out value of `i`.

Both are verified as assertions at the top of `experiments/severity.py`, to
machine precision, before any result is reported.

**Theorem 2 (anchored efficiency).** `Σ_{i∉B} φ_i^B = v(V) − v(B)`.

The anchored values decompose exactly the part of the degradation not already
explained by the anchor. Measured max deviation across all 29 configurations:
**2.78×10⁻¹⁷**.

### 4.1 Choosing the anchor

The principled default is to anchor the **exogenous** operators — sources, with
no parents in the DAG. Their state differs because the world moved, not because
the pipeline did. You cannot repair the fact that Tuesday's data is not
Monday's, so charging blame to it is accurate accounting and useless triage.

This is a modelling decision and we present it as one. §8 discusses when it is
wrong.

---

## 5. Computing it

Each `v(S)` costs a pipeline replay plus a model fit. `2ⁿ` of them is hopeless.
Three mechanisms make it practical.

**Ancestral sufficiency.** `u(S)` depends only on the sink, so operators outside
`Anc*(sink)` are null players. In real Airflow DAGs the majority of tasks are
sensors, notifications and cleanup; this removes them for free.

**Output-stability pruning.** If operator `i`'s output is byte-identical under
both states given the same inputs, all downstream computation is identical and
`v(S∪{i}) = v(S)`. This is detectable by *hashing*, with no model fit at all —
one hash replaces one replay-and-retrain. We call the operators that survive
this test the **active frontier** `F`.

**Prefix memoisation.** Hybrid replays share large prefixes. We cache operator
outputs on `(operator, state, hash(inputs))` and utilities on the sink hash, so
distinct coalitions producing an identical sink share a single model fit.

Measured effect (§6.4): operator cache hit rates of 91.3–99.8%, and 2³³
coalitions reduced to 3 model fits at `n = 33`.

**Monte-Carlo estimation.** Beyond `n ≈ 14`, permutation sampling over the free
players with the anchor always included. At `n = 33`, 80 permutations take
~1 second; where exact values are available the sampled ones match them.

---

## 6. Evaluation

### 6.1 Setup

**Pipeline.** A seven-operator churn feature build: two sources, a cleaning
step, an aggregation, a join, a filter, an encoder. 4000 customers per day.
`n = 7` keeps the 128-coalition lattice brute-forceable, so Monte-Carlo
estimates can be checked against exact ground truth.

**Benign drift.** The reference and incident runs read *different days* of
source data drawn from the same distribution. Every scenario therefore layers a
real fault on top of natural churn, and every downstream node's output changes
whether or not it is to blame. This is what makes per-node monitoring fail and
it is the situation engineers actually face. It is also, as §6.2 shows, what
breaks the plain Shapley value.

**Faults.** Five types, each with a monotone severity ladder (29 configurations
total): unit change, schema drift, filter-predicate flip, MNAR null spike, and
biased join fan-out.

**Benign controls.** Two additional faults — uniform join fan-out (35% duplicate
keys) and MCAR nullness (55% of rows dropped) — that violate data-quality
constraints loudly while causing no model damage.

**Baselines.** Per-node distributional drift (B3), single-permutation stagewise
swap (B4, current practice), leave-one-out (B5).

**Metric.** Precision@1 against the injected culprit. Incidents with
`|Δ| < 0.002` AUC are reported separately: there is nothing to attribute and
ranking metrics are meaningless.

### 6.2 The negative result

On the initial gate experiment (10 scenarios), leave-one-out beat the Shapley
value:

| method | p@1 | MRR | blame mass |
|---|---|---|---|
| leave-one-out | **1.00** | **1.00** | **0.87** |
| per-node drift | 0.88 | 0.94 | 0.51 |
| Shapley (exact) | 0.75 | 0.88 | 0.62 |
| stagewise | 0.38 | 0.69 | 0.41 |

Shapley lost exactly the two scenarios where the fault did *less* damage than
benign drift: biased join fan-out (`Δ = −0.0016`) and MNAR null spike
(`Δ = −0.0028`), against a per-source benign drift magnitude of 0.0556. In both
it ranked a source above the culprit — which, as accounting, is correct.

This motivated Definition 3.

### 6.3 Main result: accuracy against the fault-to-drift ratio

Let `r = |v({culprit})| / max_s |v({s})|` over sources `s`. Sweeping all 29
severity configurations:

| ratio bin | n | Shapley | **anchored** | LOO | drift | stagewise |
|---|---|---|---|---|---|---|
| r < 0.25 | 5 | 0.20 | **1.00** | 1.00 | 0.60 | 0.00 |
| 0.25 – 0.5 | 1 | 1.00 | **1.00** | 1.00 | 1.00 | 0.00 |
| 0.5 – 1 | 6 | 0.67 | **1.00** | 1.00 | 1.00 | 0.33 |
| 1 – 2 | 2 | 1.00 | **1.00** | 1.00 | 1.00 | 1.00 |
| r > 2 | 1 | 1.00 | **1.00** | 1.00 | 1.00 | 1.00 |
| **overall** | **15** | **0.60** | **1.00** | **1.00** | **0.87** | **0.33** |

*(15 of 29 configurations exceed the measurability threshold.)*

The predicted mechanism is confirmed. Plain Shapley degrades to 0.20 when the
fault is small relative to benign drift and recovers to 1.00 when it dominates.
The anchored value is flat at 1.00 across every regime, because anchoring the
exogenous operators removes benign drift from the comparison by construction.

### 6.4 Anchored value versus leave-one-out

The two rank identically — 15/15 agreement on p@1. They differ in what the
numbers mean:

| | max deviation | mean |
|---|---|---|
| anchored: `\|Σφ − (v(V) − v(B))\|` | **2.78×10⁻¹⁷** | — |
| leave-one-out: `\|Σφ − v(V)\|` | **0.246 AUC** | 0.096 AUC |

Leave-one-out's attributions deviate from the observed degradation by up to 0.246
AUC — larger than most of the incidents being diagnosed. It double-counts
interacting faults, and in one scenario reported that an operator *helped* by
+0.04 when its true anchored contribution was +0.005.

**This is the paper's practical claim.** The anchored value matches
leave-one-out's ranking accuracy while providing the exact decomposition
leave-one-out cannot.

### 6.5 Benign controls

| control | Δ | max\|φ\| |
|---|---|---|
| uniform join fan-out (35% dup keys) | +0.0001 | 0.0078 |
| MCAR nullness (55% rows dropped) | +0.0006 | 0.0085 |

Both fire every data-quality constraint. Neither damages the model, and CULPA
assigns near-zero blame to both. This is a direct argument for utility-grounded
attribution over constraint checking: **not every data-quality violation is an
incident.**

### 6.6 Real data: NYC yellow taxi

The synthetic result above establishes the mechanism, but its benign drift is
drawn from a stationary distribution we chose. This section repeats the
experiment on real day-over-day partitions.

**Setup.** A nine-operator pipeline over NYC TLC yellow taxi records for January
2024: two sources (trips, zone lookup), a validity filter, time-feature
derivation, a per-zone aggregate, two zone joins, a stats join, and an encoder.
Task: predict a generous tip (`tip_amount / fare_amount > 0.20`) for credit-card
trips. 60,000 trips per partition. `tip_amount` and `total_amount` are dropped
before modelling; `total_amount` includes the tip and would leak the label
outright.

Reference and incident runs read two different real days. Five fault types with
severity ladders, 19 configurations.

**Two regimes, chosen by which days are compared:**

| | Mon 8 vs Tue 9 | Mon 8 vs Sat 13 |
|---|---|---|
| benign drift `max\|v(source)\|` | 0.0104 | 0.0063 |
| valid test cases | 6 / 19 | 8 / 19 |
| **anchored** | **1.00** | **1.00** |
| per-node drift | 0.50 | 0.62 |
| Shapley | 1.00 | **0.50** |
| leave-one-out | 1.00 | **0.50** |
| stagewise | 1.00 | **0.50** |

**The headline.** On the Monday/Saturday pair, by fault-to-drift ratio:

| ratio bin | n | Shapley | **anchored** | LOO | drift | stagewise |
|---|---|---|---|---|---|---|
| r < 0.5 | 1 | 0.00 | **1.00** | 0.00 | 1.00 | 0.00 |
| 0.5 – 1 | 3 | 0.00 | **1.00** | 0.00 | 1.00 | 0.00 |
| r > 2 | 4 | 1.00 | **1.00** | 1.00 | 0.25 | 1.00 |

**Leave-one-out fails on real data.** On the synthetic workload LOO scored 1.00
and was indistinguishable from the anchored value in ranking; §6.4 could only
separate them on the efficiency property. Here, wherever the fault is smaller
than the benign weekday/weekend shift, LOO drops to 0.00 alongside plain Shapley
and stagewise, and the anchored value is the only estimator that holds at 1.00.

The reason is that real drift is large and asymmetric in a way our generator's
was not. Saturday differs from Monday in trip distance, hour distribution and
tipping behaviour simultaneously, so removing a source from the full coalition —
which is exactly what LOO does — moves the utility more than the injected fault
does. Anchoring removes that term by construction instead of competing with it.

This strengthens the paper's claim: the anchored value is not merely
better-behaved arithmetic over the same ranking, it is the only method tested
that survives realistic background drift.

Efficiency is unchanged: anchored max deviation 2.78×10⁻¹⁷, LOO up to 1.83×10⁻².

**Cost.** n = 9, 512 coalitions → 4 model fits, 98.8–99.3% operator cache hit
rate, active frontier 2.

**The honest caveat.** Only 6–8 of 19 configurations are valid test cases. Two
exclusion rules apply, both stated before the results were read:

1. *The injected fault causes no damage in isolation* (11 configurations). All
   four `broken_join_key` levels and all three `join_fanout` levels fall here.
   Duplicating rows in the zone lookup is textbook join fan-out, but a left join
   on `DOLocationID` reproduces the same borough value, so it reweights the
   training set uniformly and costs nothing. Scoring a method for failing to
   blame an operator that caused no harm would penalise it for being right.
2. *The fault is harmful but benign drift offset it* (2 configurations on the
   Mon/Tue pair). The incident run is not worse overall, so there is no
   degradation to attribute.

Rule 2 is itself a finding worth stating: **on real data, a genuine fault can be
masked entirely by a favourable partition.** A monitoring system watching only
end-to-end model quality would not fire at all.

Rule 1 means our taxi fault suite is weaker than intended — seven of nineteen
configurations target operators whose corruption this particular model is
insensitive to. Broadening it (fan-out that changes joined *values*, not just
row multiplicity) is the first thing to fix.

### 6.7 Scale

A generated fan-in DAG: `m` source branches, each `source → clean → aggregate`,
joined into one feature table, then filtered and encoded (`n = 3m + 3`).

| n | 2ⁿ | model fits | cache hit | \|F\| | MC time | culprit found |
|---|---|---|---|---|---|---|
| 6 | 64 | 4 | 91.3% | 2 | <0.01 s | ✓ |
| 9 | 512 | 8 | 98.5% | 3 | <0.01 s | ✓ |
| 12 | 4 096 | 16 | 99.7% | 4 | <0.01 s | ✓ |
| 15 | 32 768 | 3 | 99.4% | 5 | 0.19 s | ✓ |
| 21 | 2 097 152 | 3 | 99.7% | 7 | 0.41 s | ✓ |
| 27 | 1.3×10⁸ | 3 | 99.8% | 9 | 0.68 s | ✓ |
| 33 | 8.6×10⁹ | 3 | 99.8% | 11 | 0.99 s | ✓ |

**How to read this.** The headline ratio is real but it is *not* a function of
`n`. Only three distinct sink states exist in this workload — baseline,
anchor-only, and anchor-plus-faulted-aggregate — because every other operator is
state-insensitive and its flip is caught by the output hash before any model is
fit. Cost tracks `|F|`, not `|V|`.

That is the honest claim, and it identifies its own failure mode: the reduction
collapses when a change touches many operators simultaneously — a shared library
upgrade, a global config change, a backfill. Those cases are absent from this
sweep and belong in the evaluation.

---

## 7. Related work

**ML pipeline debugging.** `mlinspect` [3, 4] extracts a dataflow DAG from a
pipeline and propagates lineage annotations to detect data-distribution bugs.
It is the closest prior art and is complementary: it inspects one execution,
whereas we compare two and attribute a quantity. Its DAG extraction is exactly
what a production implementation of our method should reuse.

**Shapley values over pipeline operators.** `ShapleyPipe` [5] applies Shapley
values to pipeline operators for *construction* — which operator to select to
maximise offline accuracy, with position-specific values and hierarchical search.
That is a design-time AutoML question over a space of candidate pipelines. Ours
is a run-time forensic question about one deployed pipeline whose inputs changed.
The player sets, the games, and the answers differ.

**Data valuation.** Data Shapley and its successors value *training examples*.
We value *operators*. The techniques for efficient estimation are transferable;
the semantics are not.

**Data quality.** Deequ, Great Expectations and the data-contracts literature
verify declarative constraints. §6.5 shows constraint violation and model damage
are distinct: our benign controls violate constraints loudly and cost 0.0001 AUC.

**Cooperative game theory.** The anchored value's restriction to feasible
coalitions relates to conjunctive permission structures [6] and, when coalitions
must be connected in the DAG, the Myerson value. We state the connection; we do
not prove a characterisation. See §9.

**Root-cause analysis for pipelines.** A body of 2026 work addresses "AI-powered
data pipeline observability." It appears predominantly in preprint repositories
and low-tier venues, without formal guarantees, ground-truth benchmarks, or
reproducible artifacts.

---

## 8. Limitations and threats to validity

We state these plainly because several are severe.

**One real dataset, one pipeline shape.** §6.6 uses real NYC taxi partitions, but
it is a single dataset and a single nine-operator topology. The synthetic results
in §6.3–6.5 remain results about a controlled workload. Whether the crossover
sits at the same ratio for other domains — and for pipelines that are deep rather
than wide — is untested.

**The fault suite is weaker than intended.** Eleven of nineteen taxi
configurations were excluded because the injected fault does no damage at all
(§6.6). That is an honest exclusion, but it leaves the real-data conclusion
resting on eight test cases, and it means the benchmark does not yet exercise
join-integrity faults against a model sensitive to them.

**Few valid test cases at the crossover.** The Monday/Saturday regime that
separates the anchored value from every baseline contains four configurations
below `r = 1`. The separation is clean — 1.00 against 0.00 — but it is four
points, not a curve.

A methodological note from building this. Our first two synthetic workloads each
contained a fault that measured no damage for a reason unrelated to the method:
a pure rescale that downstream standardisation undid, and a label recomputed
from the corrupted columns so the target moved with the features. Both looked
like plausible faults and neither was. Any fault-injection benchmark in this
space needs to *verify* that each injected fault actually degrades the model
before scoring attribution against it; we report `v(V)` per configuration for
this reason, and exclude configurations where the incident run is not
measurably worse.

**No orchestrator integration.** We claim relevance to Airflow DAGs but operate
at the dataflow level. Integration is packaging rather than science, but until
it exists the applicability claim is untested.

**The probe set must exist.** The whole method rests on a fixed evaluation set
representing ground truth. Where labels arrive with long delay, or where no clean
held-out sample exists, `u` is not computable as defined. This is the most
serious practical limitation.

**Deterministic utility.** We use a deterministic learner with common random
numbers across coalitions. For stochastic learners `u` is a random variable,
efficiency holds only in expectation, and existing Shapley concentration bounds
— which assume a deterministic oracle — do not directly apply.

**Anchor choice is a modelling decision.** Anchoring the exogenous operators is
defensible but not derived. If a source change is itself the repairable fault —
a misconfigured extraction job rather than genuine world drift — anchoring hides
the culprit by construction. Detecting that case is unsolved.

**Frontier size drives all cost results.** §6.6 is honest that the reduction
tracks `|F|`. Broad simultaneous changes are not evaluated and would degrade it
sharply.

**Hashing requires determinism.** Operators involving sampling, timestamps, or
`ORDER BY` without a total order break output-stability pruning. Canonicalisation
is unimplemented.

**Small `n` for exact values.** Exact Shapley is computed only to `n = 12`;
beyond that we rely on sampling validated in the small regime.

---

## 9. Conclusion and future work

Attributing model degradation to pipeline operators can be posed as a cooperative
game over hybrid replays of a fixed pipeline across two temporal snapshots. The
Shapley value of that game decomposes degradation exactly but ranks poorly under
co-occurring benign drift. The incident-anchored Shapley value — holding
exogenous operators at incident state — recovers both standard Shapley and
leave-one-out as boundary cases, ranks perfectly across every severity regime we
tested, and preserves exact decomposition, which leave-one-out does not.

Open problems, in the order we would attack them:

1. **Axiomatic characterisation of `φ^B`.** Which axioms does it satisfy, and is
   it the unique value satisfying them? Relate precisely to the conjunctive
   permission value [6].
2. **Real pipelines.** NYC Taxi day-over-day partitions, with genuine benign
   drift, to test whether §6.3's crossover survives.
3. **Automatic anchor selection.** Distinguishing genuine exogenous drift from a
   repairable fault in an extraction job.
4. **Concentration bounds under a stochastic utility oracle.**
5. **Broad-change regime.** Where the active frontier is large and pruning fails.
6. **Federated attribution.** Detecting and attributing faults across data silos
   without centralising raw data.

---

## References

1. Deequ / Great Expectations — declarative data-quality constraint verification.
2. OpenLineage / dbt — dataset and column-level lineage.
3. Grafberger, S., Schelter, S. et al. *mlinspect: a Data Distribution Debugger
   for Machine Learning Pipelines.* SIGMOD 2021 (demo).
4. Grafberger, S. et al. *Data distribution debugging in machine learning
   pipelines.* The VLDB Journal, 2022.
5. *ShapleyPipe: Hierarchical Shapley Search for Data Preparation Pipeline
   Construction.* arXiv:2510.27168, Oct 2025.
6. Gilles, R., Owen, G., van den Brink, R. — games with permission structures;
   the conjunctive permission value.
7. Myerson, R. *Graphs and cooperation in games.* Mathematics of Operations
   Research, 1977.
8. *ELT-Bench: An End-to-End Benchmark for Evaluating AI Agents on ELT
   Pipelines.* PVLDB.

---

## Reproducibility

```bash
pip install -r requirements.txt
python -m experiments.gate       # §6.2, §3.1
python -m experiments.severity   # §6.3, §6.4  (boundary theorems asserted)
python -m experiments.scale      # §6.6
```

Total runtime under two minutes on a laptop. Logs in `results/`.
