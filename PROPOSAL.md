# CULPA: Counterfactual Utility-Loss Pipeline Attribution

**Attributing machine-learning model degradation to individual ETL pipeline operators
via temporal counterfactual replay.**

Author: Laiba Mazhar
Status: research proposal / working document
Started: August 2026

---

## 1. The problem

A production ML system is fed by an ETL pipeline — in practice an Airflow DAG of
tens to hundreds of tasks. On Monday the pipeline runs and the model scores 0.87
AUC. On Tuesday the same DAG runs, no code was deployed, and the model scores 0.81.

**Which task caused it?**

Today this is answered by hand. An engineer opens the lineage graph, eyeballs
row counts and null rates at each stage, forms a hypothesis, re-runs a branch,
and iterates. It takes hours to days, and it is the single most common failure
mode in production ML — upstream data change, not model change.

The state of tooling:

- **Monitoring** (Evidently, Arize, Deequ, Great Expectations) tells you *that*
  a distribution moved. It fires at every node that moved, and in a real pipeline
  a single upstream change makes twenty downstream nodes move. It does not rank,
  and it does not separate cause from consequence.
- **Lineage** (OpenLineage, Marquez, dbt) tells you *what is upstream of what*.
  It narrows the suspect set to "everything upstream," which in a deep DAG is
  nearly everything.
- **Debugging tools** (`mlinspect`, SIGMOD'21 / VLDBJ'22) instrument a pipeline
  into a dataflow DAG and propagate lineage annotations to detect data-distribution
  bugs. This is the closest prior art and it is strong, but it *detects* bugs
  within a single pipeline execution. It does not compare two executions over
  time, and it does not assign quantitative blame.
- **Pipeline Shapley work** (`ShapleyPipe`, arXiv 2510.27168, Oct 2025) uses
  Shapley values over pipeline operators — but for **construction**: which
  operator should I *select* to maximise offline accuracy. That is a design-time
  AutoML question. Ours is a run-time forensic question about a fixed, deployed
  pipeline whose *inputs* changed.

There is a body of 2026 writing on "AI-powered data pipeline observability and
root cause analysis," but it sits in preprint repositories, low-tier journals and
patents. **No rigorous, formal, reproducible treatment exists.** That is the gap.

---

## 2. The idea

Let the pipeline be fixed. Let the *inputs and configuration* differ between a
known-good **reference run** `R` (Monday) and a degraded **incident run** `I`
(Tuesday).

Each operator therefore has two possible **states**: it can execute as it did in
the reference run, or as it did in the incident run. (This subsumes both causes
of degradation: the operator's own source data changed, or its code/config
changed.)

For a subset `S` of operators, define the **hybrid replay** `Π(S)`: execute the
whole DAG, with every operator in `S` in its incident state and every operator
outside `S` in its reference state.

There are `2^n` such hybrids. They interpolate continuously between Monday's
pipeline (`S = ∅`) and Tuesday's (`S = V`).

Now define utility `u(S)` = downstream model quality when trained on (or served
from) the output of `Π(S)`, and the cooperative game

```
v(S) = u(S) − u(∅)
```

with players = operators. Then `v(∅) = 0` and `v(V) = u(V) − u(∅) = Δ`, the
**exact observed degradation**.

The Shapley value of this game,

```
φ_i = Σ_{S ⊆ V∖{i}}  [ |S|! (n−|S|−1)! / n! ] · [ v(S ∪ {i}) − v(S) ]
```

is then a blame assignment with a property no existing tool has:

> **Σ_i φ_i = Δ.** The observed degradation is *exactly decomposed* across
> operators. Not scored, not ranked — decomposed. "Task `join_customer_dim` is
> responsible for 4.1 of the 6.0 AUC points you lost."

That is the headline claim.

---

## 3. Two attribution semantics

An immediate objection: is a hybrid where a downstream operator is "incident"
while its ancestor is "reference" even meaningful?

Yes — and both answers are useful, which gives us two values rather than one.
This is the theoretical core of the paper.

### 3.1 Interventional value `φ^int`

Unconstrained Shapley over all `2^n` coalitions. A coalition where operator `o`
is incident but its parent is reference means "the parent's change was somehow
prevented, but `o`'s own change still happened." This is exactly the **repair**
question:

> *If I had frozen this one operator, how much damage would I have avoided?*

This is what an on-call engineer wants: it ranks candidate interventions.

### 3.2 Precedence-constrained value `φ^perm`

Restrict to **feasible** coalitions — `S` is feasible iff for every `o ∈ S`, all
ancestors of `o` are in `S`. This is a *conjunctive permission structure* in the
sense of Gilles, Owen and van den Brink, and the corresponding solution concept
is the **conjunctive permission value**: the Shapley value of the restricted game
`v^r(S) = v(σ(S))`, where `σ(S)` is the largest feasible subset of `S`.

This answers the **forensic** question:

> *As change propagated through the DAG along real causal paths, how much did
> each operator contribute?*

A theorem relating `φ^int` and `φ^perm`, plus a characterisation of when they
disagree, is a genuine contribution. Conjecture to prove or refute early: they
coincide iff the DAG is a chain and the game is monotone along it.

### 3.3 Why current practice is provably arbitrary

The manual procedure engineers actually use — walk the DAG in topological order,
swap in Tuesday's data one stage at a time, watch where the metric drops — is
exactly the **marginal-contribution vector of a single topological permutation**.

Shapley averages over all `n!` orderings. The permission value averages over the
feasible ones. A single permutation is one sample from that average.

**Proposition (to state and prove).** Single-permutation stagewise diagnosis is
order-dependent and violates the symmetry axiom; for a DAG with `m` distinct
topological orders it can assign the entire degradation to any of up to `m`
different operators depending on the order chosen. Its deviation from `φ` is
bounded by the interaction terms (Shapley–Owen interaction indices) between
operators.

This is a strong framing: **the paper explains why current practice fails, and
the fix falls out of the same formalism.** It also makes for a compelling
experiment — construct a real DAG where two different topological orders blame
two different tasks, and show Shapley resolves it correctly.

> ⚠️ Note to self: I initially conjectured that `φ^perm` *reduces* to the
> telescoping stagewise differences on a chain. **This is false.** Hand-check on
> a 2-operator chain: `φ^perm_1 = ½v({1}) + ½v({1,2})`, whereas telescoping gives
> `v({1})`. Do not put this in the paper. The correct statement is the
> single-permutation framing above.

---

## 4. Making it computable — the systems contribution

`2^n` hybrid replays, each requiring a pipeline execution plus a model retrain,
is hopeless. Making it cheap *is* the systems half of the paper.

### 4.1 Ancestral sufficiency

`u(S)` depends only on the sink dataset, and the sink depends only on operators
that can reach it. Operators outside `Anc*(sink)` are null players: `φ_i = 0`.
Trivial, but it prunes orchestration-only tasks (sensors, notifications, cleanup)
which are the majority of tasks in real Airflow DAGs.

### 4.2 Output-stability pruning — the important one

**Lemma.** If `d_i(S ∪ {i}) = d_i(S)` — operator `i`'s output is *byte-identical*
whether it runs in reference or incident state, given the same inputs — then all
downstream computation is identical, hence `v(S ∪ {i}) = v(S)` and the marginal
contribution is zero.

Crucially this is detectable by **hashing the operator's output**, with no model
retraining at all. One hash replaces one full replay-and-retrain.

**Definition (active frontier).** The set `F` of operators whose output hash
actually changes under a state flip, for at least one reachable input
configuration.

**Theorem (quiescent collapse) — to prove.** Operators outside `F` are null
players, and the game restricted to `F` has identical Shapley values for members
of `F`. Cost drops from `2^n` to `2^|F|`.

In real incidents `|F| ≪ n`: a single bad upstream partition typically activates
a handful of operators along one path. This is the result that makes the method
practical, and it is empirically verifiable — measuring `|F|/n` across the
benchmark is a headline figure.

### 4.3 Prefix memoisation

Hybrid replays share enormous prefixes. Cache each operator's output keyed by
`(operator_id, state, hash(inputs))`. Across the coalition lattice the number of
*distinct* operator executions is far below `|F| · 2^|F|`.

Measuring the cache hit rate across lattice traversal orders — and choosing a
traversal order that maximises it — is a concrete, publishable systems result.

### 4.4 Monte-Carlo estimation with paired sampling

Permutation sampling for `φ`, with:
- output-stability pruning as an exact control variate (known-zero marginals),
- **common random numbers** across coalitions: same training seed, same CV folds,
  same initialisation for every `u(S)` evaluation.

Paired sampling matters more than usual here because `u` involves *training a
model* and is therefore stochastic. Two coalitions differing by one operator may
differ in utility by less than the seed-to-seed variance of the learner. Without
variance control the estimate is noise. Give a Hoeffding-type bound on the number
of permutations needed for `ε`-accuracy with probability `1−δ`.

> This is the point reviewers will attack hardest. Address it head-on in the
> paper, with an explicit "utility is a random variable" subsection and an
> experiment showing attribution stability across seeds.

### 4.5 Two utility modes

| Mode | `u(S)` definition | Diagnoses |
|---|---|---|
| **Training-time** | retrain model on `Π(S)` output, evaluate on a *fixed clean probe set* | bad training data |
| **Serving-time** | freeze the reference-trained model, score `Π(S)` output | training/serving skew |

Both are real production failure modes. Supporting both is a breadth win at
almost no extra cost.

**Design decision, must be stated explicitly:** the evaluation probe set is held
fixed across all coalitions. Otherwise `u(S)` compares different things and the
efficiency property is meaningless.

---

## 5. Evaluation

### 5.1 Fault-injection benchmark

Ground truth is the injected culprit, so precision/recall are exactly measurable.
This benchmark is itself a contribution — nothing comparable is public.

Single faults:

1. Schema drift — column renamed or dropped
2. Silent unit change — cm→m, USD→local currency
3. Null spike in a high-importance feature
4. Join fan-out — duplicate keys multiply rows
5. Timezone / date-parsing shift — off-by-one-day windows
6. Encoding change — cp1252 vs utf-8 mangling categorical values
7. Categorical vocabulary drift — unseen categories at serving
8. Label leakage introduced by a join
9. Filter predicate flip — `>=` vs `>` silently dropping a stratum
10. Late-arriving / partial partition
11. Aggregation window change
12. Type coercion — int → float → string
13. Deduplication key change
14. Sampling-rate change

**Compound faults** (two or three simultaneous, on different branches) are where
the method should decisively beat every baseline, because interactions are
exactly what single-node monitors cannot represent. Prioritise these.

### 5.2 Baselines

| ID | Baseline | Represents |
|---|---|---|
| B1 | Uniform blame over all ancestors | lineage-only tooling |
| B2 | Constraint violation per node (Deequ / Great Expectations style) | data-quality monitoring |
| B3 | Per-node distributional distance (KS / PSI / MMD) | drift detectors at every node |
| B4 | Single-permutation stagewise swap | **what engineers actually do today** |
| B5 | Leave-one-out (freeze one operator at a time) | the obvious ablation, Shapley's closest rival |
| B6 | Feature-level SHAP mapped back to producing operator | the "just use SHAP" reflex |
| B7 | LLM-as-diagnostician, given DAG + summary stats | the 2026 reflex |

B4 and B5 are the ones that matter. B7 is a good foil and cheap to run.

### 5.3 Metrics

- **Attribution quality:** precision@1, MRR, NDCG against injected culprits;
  for compound faults, fraction of total blame mass landing on true culprits.
- **Cost:** number of replays, number of model fits, wall-clock, cache hit rate,
  `|F|/n` frontier compression.
- **Fidelity:** estimation error against *exact* Shapley on small DAGs
  (`n ≤ 12`), where brute force is affordable.
- **Stability:** variance of `φ` across training seeds and across MC samples.

### 5.4 Pipelines and data

- Adult / Census Income — small, classic, comparable to `mlinspect`
- NYC Taxi — real, large, genuinely temporal (natural day-over-day snapshots)
- Airline on-time performance — temporal, multi-source joins
- UCI Online Retail II — authentically dirty
- A **parameterised synthetic DAG generator** for scaling experiments
  (vary `n`, depth, branching factor, frontier size)

Implement the pipelines as real Airflow DAGs so the orchestration claim is
genuine, but keep the replay engine at the dataflow level (pandas/SQL) so it is
portable and so experiments run on a laptop.

---

## 6. Claimed contributions

1. **Formulation.** Temporal counterfactual replay of a fixed pipeline over two
   snapshots as a cooperative game whose Shapley value *exactly decomposes*
   observed model degradation across operators.
2. **Theory.** Two attribution semantics (interventional / precedence-constrained),
   their relationship, and a proof that current stagewise practice is a
   single-permutation sample and therefore order-dependent.
3. **Algorithms.** Output-stability pruning and the quiescent-collapse theorem
   reducing `2^n` to `2^|F|`; prefix memoisation; paired-sampling MC estimation
   with concentration bounds under a stochastic utility.
4. **System.** `CULPA`, an Airflow-integrated replay-and-attribution engine.
5. **Benchmark.** An open fault-injection suite over real pipelines with ground-
   truth culprits — the first of its kind.

Contributions 1 and 3 are the load-bearing ones. 5 is what gets the work used
and cited.

---

## 7. Where this maps at KAUST

| Group | Fit |
|---|---|
| **Panos Kalnis** — Big Data, distributed systems, *Systems for Machine Learning* | Direct. Closest single match. |
| **Marco Canini — SANDS Lab** — systems support for AI/ML | Strong. Their PICO work won ISC HPC 2026 best paper; same "measure and optimise the ML plumbing" instinct. |
| **Di Wang** — trustworthy / interpretable ML | Attribution and interpretability angle; natural extension to the federated variant. |
| **Center of Excellence for Generative AI** (Ghanem) / AI Initiative (Schmidhuber) | Secondary, via the B7 LLM-diagnostician baseline. |
| **SDAIA–KAUST AI Center** | Funding 20 AI projects over 3 years — relevant context for a statement of purpose. |

**Follow-on paper (deliberately kept out of scope here):** federated CULPA —
attributing data-quality faults across silos *without centralising raw data*.
That is Di Wang + Canini territory and a natural "future work" paragraph that
doubles as a PhD proposal.

---

## 8. Venue strategy

Be realistic: a solo first-time author going straight to VLDB Journal is a long
shot. Sequence it.

1. **arXiv preprint + public artifact** — immediately on completion. This is what
   an admissions committee can actually see and click.
2. **DEEM @ SIGMOD** (*Data Management for End-to-End Machine Learning*) — the
   exact venue for this work; where `mlinspect` and its lineage live. Workshop,
   so tractable solo, and gives real reviewer feedback.
3. **EDBT** or **CIKM** — full conference, mid-tier-hard, realistic.
4. **Extend to Q1 journal** with the added experiments reviewers ask for:
   - *The VLDB Journal* (best fit)
   - *IEEE TKDE*
   - *Information Systems* (Elsevier)
   - *Journal of Big Data* (open access, faster turnaround)

The honest path to Q1 runs *through* step 2, not around it.

---

## 9. Plan (≈3 months, part-time)

**Weeks 1–2 — kill the idea if it's dead.**
Build the minimal replay engine and run exact Shapley on a 5-operator pipeline
with one injected fault. Success criterion: Shapley concentrates blame on the
injected operator and leave-one-out does not. *If this doesn't hold, stop and
re-scope.*

**Weeks 3–4 — related work, properly.**
Read and tabulate: `mlinspect` (VLDBJ'22), `ShapleyPipe`, Data Shapley and
successors, Deequ, `ELT-Bench`, the permission-value literature (Gilles/Owen/
van den Brink), Myerson value. Produce a positioning table. Confirm nobody has
done temporal hybrid replay. **Do this before writing any more code.**

**Weeks 5–7 — the real system.**
Frontier detection, hash-based pruning, prefix memoisation, MC estimation with
paired sampling. Airflow integration.

**Weeks 8–10 — benchmark and experiments.**
All 14 fault types, compound faults, all 7 baselines, 4 pipelines, scaling study.

**Weeks 11–12 — write.**
Paper draft, artifact cleanup, reproducibility packaging, arXiv.

**Gate at week 2 and week 7.** If the week-2 criterion fails, the idea is wrong
and it is much better to know in two weeks than in three months.

---

## 10. Honest risks

| Risk | Mitigation |
|---|---|
| Utility noise swamps marginal contributions | Paired sampling / common random numbers; report seed variance explicitly; this is a first-class subsection, not a footnote |
| Leave-one-out (B5) performs nearly as well for single faults | Lead with **compound** faults, where interactions are the whole point; report single-fault parity honestly |
| `|F|` is not small in practice | Measure it and report it. If frontiers are large the paper becomes about *approximating* them — still publishable, different framing |
| "This is engineering, not research" | The theory section (two semantics + order-dependence proposition) is the answer; keep it strong |
| Reviewers want a real production deployment | Cannot get one solo. Compensate with breadth of realistic pipelines + an open benchmark |
| Someone publishes this first | Move fast, arXiv early. The space is hot but the published work is low-rigour |

---

## 11. Open questions to resolve

- [ ] Prove or refute the quiescent-collapse theorem (§4.2). Load-bearing.
- [ ] Relationship between `φ^int` and `φ^perm` — when do they disagree?
- [ ] Correct concentration bound for Shapley estimation under a *stochastic*
      utility oracle. Existing bounds assume deterministic `v`.
- [ ] Does the efficiency property survive when the model retrain is stochastic?
      (Probably only in expectation — state this precisely.)
- [ ] Frontier detection when operators are non-deterministic (sampling,
      timestamps, `ORDER BY` without a total order). Hashing breaks. Needs
      canonicalisation.
