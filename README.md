# CULPA

**Counterfactual Utility-Loss Pipeline Attribution** — attributing machine
learning model degradation to individual ETL pipeline operators.

Your Airflow DAG ran fine on Monday. Tuesday's model is 6 AUC points worse and
nobody deployed anything. *Which task caused it?*

CULPA answers that with a number per operator, and the numbers sum exactly to
the degradation you observed.

## How

The pipeline is fixed; its inputs and config differ between a known-good
**reference** run and a degraded **incident** run. Each operator can therefore
execute in either state. For a subset `S` of operators, the *hybrid replay*
`Π(S)` runs the whole DAG with `S` in incident state and the rest in reference
state — interpolating between Monday's pipeline and Tuesday's.

With `u(S)` the downstream model's utility on a frozen probe set, the game
`v(S) = u(S) − u(∅)` has `v(V)` equal to the observed degradation, and its
Shapley value decomposes that degradation exactly across operators.

`2^n` replays would be hopeless, so the engine prunes: operators whose output
hash is unchanged under a state flip are null players, and operator outputs are
memoised on `(operator, state, input hashes)` so the lattice shares its prefixes.
On the 7-operator benchmark that turns 128 coalitions into **8 model fits**, at
a 93% operator cache hit rate.

## The result

Plain Shapley decomposes degradation exactly but ranks culprits poorly — 60%
top-1 — because it charges blame to the benign day-over-day drift that co-occurs
with every real incident. Leave-one-out ranks perfectly but its attributions
don't sum to anything, deviating from the observed degradation by up to 0.246
AUC.

The **incident-anchored Shapley value** holds the exogenous operators at incident
state and takes the Shapley value of the sub-game. It recovers standard Shapley
and leave-one-out as its two boundary cases, hits **100% top-1 across every
severity regime**, and keeps exact decomposition to 3×10⁻¹⁷.

| method | p@1 synthetic | p@1 real (Mon vs Sat) | decomposes exactly |
|---|---|---|---|
| **anchored Shapley** | **1.00** | **1.00** | **yes (2.8e-17)** |
| leave-one-out | 1.00 | 0.50 | no (0.246 AUC off) |
| per-node drift | 0.87 | 0.62 | no |
| Shapley | 0.60 | 0.50 | yes |
| stagewise (current practice) | 0.33 | 0.50 | no |

On real NYC taxi data the gap widens: leave-one-out, indistinguishable from the
anchored value on synthetic data, collapses under real weekday/weekend drift
because removing a source moves the utility more than the injected fault does.

## Documents

- **[PAPER.md](PAPER.md)** — the paper draft. Start here.
- **[DATA.md](DATA.md)** — the two files to download for the real-data
  experiment, and how to pick day pairs that make it informative
- [FINDINGS.md](FINDINGS.md) — the gate experiment's negative result and how it
  drove the reformulation
- [PROPOSAL.md](PROPOSAL.md) — original research plan. Predates the experiments;
  its "Shapley ranks better" claim is dead. Kept for the related-work and venue
  analysis.
- `results/` — full run logs and CSVs

## Run it

```bash
pip install -r requirements.txt && python -m experiments.gate && python -m experiments.severity && python -m experiments.scale && python -m experiments.real_data --real --days 8,13,10
```

The last one needs the two downloads in [DATA.md](DATA.md); without them it runs
against a generated fixture and says so.

Under two minutes on a laptop. `experiments/severity.py` asserts both boundary
theorems (anchor=∅ equals Shapley, anchor=V∖{i} equals leave-one-out) to machine
precision before reporting anything.

## Layout

| file | what |
|---|---|
| `culpa/pipeline.py` | operator/DAG model, hybrid replay engine, memoisation, frontier detection |
| `culpa/game.py` | the cooperative game, exact + Monte-Carlo Shapley, all baselines, scoring |
| `culpa/workload.py` | the 7-operator benchmark pipeline and the fault injectors |
| `culpa/utility.py` | `u(S)` — train on the replayed table, score on a frozen probe |
| `experiments/gate.py` | fault scenarios, order-dependence of current practice, benign controls |
| `experiments/severity.py` | severity ladder, the main accuracy-vs-ratio result, boundary theorem checks |
| `experiments/scale.py` | fan-in DAG generator, cost study to n = 33 |
| `culpa/nyctaxi.py` | NYC TLC loading, schema contract, fixture generator |
| `culpa/taxi_pipeline.py` | nine-operator taxi ETL pipeline and its fault injectors |
| `experiments/real_data.py` | the real-data experiment (runs on the fixture until you download) |

## Baselines implemented

| id | method | represents |
|---|---|---|
| B3 | per-node distributional drift | drift monitors on every task |
| B4 | single-permutation stagewise swap | what engineers do by hand today |
| B5 | leave-one-out | the strongest competitor |

B1 (uniform lineage), B2 (constraint violation), B6 (feature SHAP) and B7
(LLM diagnostician) are not implemented yet.
