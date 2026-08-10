"""A realistic small ETL pipeline, plus the fault injectors.

The pipeline is a churn-prediction feature build of the shape that shows up in
every production stack: two sources, a cleaning step, an aggregation, a join, a
filter, an encoder. Seven operators, so the 2^n coalition lattice is 128 and
exact Shapley is affordable -- which is what we want for the gate experiment,
where the Monte-Carlo estimator must be checked against ground truth.

The critical design choice is *benign* day-over-day drift. The reference and
incident runs read different days of source data, drawn from the same
distribution. So a real fault is superimposed on natural churn in the data, and
every downstream node's output changes whether or not it is to blame. That is
what makes per-node drift monitoring fail, and it is the situation an on-call
engineer actually faces.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .pipeline import Operator, Pipeline

N_CUSTOMERS = 4000
REGIONS = ["north", "south", "east", "west"]


# -- data generating process --------------------------------------------


def _generate_day(day_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One day's customer and transaction partitions.

    The label depends on age, income and true spend, so corrupting the spend
    aggregate genuinely damages the model rather than just perturbing it.
    """
    rng = np.random.default_rng(day_seed)

    customer_id = np.arange(N_CUSTOMERS)
    age = rng.integers(18, 80, N_CUSTOMERS)
    income = rng.lognormal(10.5, 0.5, N_CUSTOMERS)
    region = rng.choice(REGIONS, N_CUSTOMERS, p=[0.4, 0.3, 0.2, 0.1])
    tenure_days = rng.integers(1, 2000, N_CUSTOMERS)

    n_txn = rng.poisson(5, N_CUSTOMERS)
    txn_customer, txn_amount = [], []
    for cid, k in zip(customer_id, n_txn):
        if k == 0:
            continue
        txn_customer.append(np.full(k, cid))
        txn_amount.append(rng.lognormal(3.5, 0.8, k))

    transactions = pd.DataFrame(
        {
            "customer_id": np.concatenate(txn_customer),
            "amount": np.concatenate(txn_amount),
        }
    )

    true_total = (
        transactions.groupby("customer_id")["amount"].sum().reindex(customer_id).fillna(0.0)
    )

    # Churn risk: young, low income, low spend, short tenure.
    z = (
        -0.030 * (age - 48)
        - 1.10 * (np.log(income) - 10.5)
        - 0.85 * (np.log1p(true_total.values) - 4.4)
        - 0.0010 * (tenure_days - 1000)
    )
    p = 1.0 / (1.0 + np.exp(-z))
    label = (rng.random(N_CUSTOMERS) < p).astype(int)

    customers = pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "income": income,
            "region": region,
            "tenure_days": tenure_days,
            "label": label,
        }
    )
    return customers, transactions


# -- operator bodies -----------------------------------------------------


def op_load_customers(inputs, params):
    return _generate_day(params["day_seed"])[0].copy()


def op_load_transactions(inputs, params):
    return _generate_day(params["day_seed"])[1].copy()


def op_clean_customers(inputs, params):
    """Drop rows with missing income, clip implausible ages.

    Fault surface: `null_rate` simulates an upstream nullness defect that this
    cleaner then faithfully removes -- silently shrinking the training set.

    `null_mode` controls whether the nullness is MCAR or MNAR. The first run of
    the gate experiment showed that MCAR nullness at 55% does essentially no
    damage: you lose half your rows and the model barely notices. MNAR is the
    fault that actually hurts and the one that actually happens -- an upstream
    export fails for one segment, here the high earners, and the training set is
    silently truncated to a biased sub-population.
    """
    df = inputs["load_customers"].copy()
    null_rate = params.get("null_rate", 0.0)
    if null_rate > 0:
        rng = np.random.default_rng(params.get("null_seed", 7))
        draw = rng.random(len(df))
        if params.get("null_mode", "mcar") == "mnar":
            # "the CRM export failed for enterprise accounts"
            segment = df["income"] > df["income"].median()
            mask = segment & (draw < null_rate)
        else:
            mask = draw < null_rate
        df.loc[mask, "income"] = np.nan
    df = df[df["income"].notna()]
    df["age"] = df["age"].clip(18, 90)
    return df.reset_index(drop=True)


def op_agg_transactions(inputs, params):
    """Per-customer spend aggregate.

    Fault surface: `amount_scale` is the classic silent unit change -- an
    upstream system starts reporting cents instead of dollars. Nothing errors,
    nothing is null, row counts are unchanged, and the model quietly degrades.
    """
    df = inputs["load_transactions"].copy()
    df["amount"] = df["amount"] * params.get("amount_scale", 1.0)
    agg = (
        df.groupby("customer_id")["amount"]
        .agg(total_amount="sum", txn_count="count", avg_amount="mean")
        .reset_index()
    )
    return agg


def op_join_features(inputs, params):
    """Left-join the aggregate onto customers.

    Fault surface: `duplicate_rate` injects duplicate keys into the right-hand
    side, producing join fan-out -- rows silently multiply and the training set
    is reweighted towards whichever customers duplicated.

    `duplicate_bias` decides *which* keys duplicate. Uniform fan-out turned out
    to be harmless in the gate run (+0.0001 AUC): duplicating a random third of
    the rows leaves the training distribution intact. Fan-out concentrated on
    one stratum -- here the top spenders, which is what a broken partition key
    or a late-arriving retry actually produces -- reweights the training set and
    does real damage.
    """
    cust = inputs["clean_customers"]
    agg = inputs["agg_transactions"]

    dup_rate = params.get("duplicate_rate", 0.0)
    if dup_rate > 0:
        rng = np.random.default_rng(params.get("dup_seed", 11))
        eligible = agg
        if params.get("duplicate_bias") == "high_spend":
            eligible = agg[agg["total_amount"] >= agg["total_amount"].quantile(0.75)]
        dup = eligible[rng.random(len(eligible)) < dup_rate]
        # A real fan-out multiplies, it does not merely add one copy.
        agg = pd.concat([agg] + [dup] * params.get("duplicate_factor", 1), ignore_index=True)

    out = cust.merge(agg, on="customer_id", how="left")
    out[["total_amount", "txn_count", "avg_amount"]] = out[
        ["total_amount", "txn_count", "avg_amount"]
    ].fillna(0.0)
    return out


def op_filter_active(inputs, params):
    """Keep customers with enough activity.

    Fault surface: `min_txn` is a predicate flip. Raising the threshold drops a
    whole stratum of the population -- the model never sees low-activity
    customers again, and its ranking on them collapses.
    """
    df = inputs["join_features"]
    return df[df["txn_count"] >= params.get("min_txn", 1)].reset_index(drop=True)


def op_encode_features(inputs, params):
    """One-hot the region, select the modelling columns.

    Fault surface: `drop_columns` is schema drift -- a rename upstream that this
    step's explicit column list silently fails to pick up.
    """
    df = inputs["filter_active"].copy()
    dummies = pd.get_dummies(df["region"], prefix="region")
    for r in REGIONS:
        col = f"region_{r}"
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[[f"region_{r}" for r in REGIONS]].astype(int)

    feats = pd.concat(
        [df[["age", "income", "tenure_days", "total_amount", "txn_count", "avg_amount"]],
         dummies,
         df[["label"]]],
        axis=1,
    )
    for col in params.get("drop_columns", []):
        if col in feats.columns:
            feats = feats.drop(columns=[col])
    return feats.reset_index(drop=True)


# -- pipeline assembly ---------------------------------------------------

REF_DAY = 101
INC_DAY = 202
PROBE_DAY = 303

FAULTS: Dict[str, Tuple[str, dict]] = {
    "unit_change":    ("agg_transactions", {"amount_scale": 0.01}),
    "schema_drift":   ("encode_features",  {"drop_columns": ["income", "total_amount"]}),
    "predicate_flip": ("filter_active",    {"min_txn": 6}),

    # Benign-by-construction variants. These violate data-quality constraints
    # loudly -- row counts move, null rates spike -- but do not degrade the
    # model. Keeping them in the benchmark is deliberate: a good attribution
    # method must report near-zero blame for them, and a constraint monitor
    # cannot tell them apart from the harmful ones. See FINDINGS.md.
    "join_fanout_uniform": ("join_features",   {"duplicate_rate": 0.35}),
    "null_spike_mcar":     ("clean_customers", {"null_rate": 0.55, "null_mode": "mcar"}),

    # Harmful variants of the same two faults: the damage comes from *which*
    # rows are affected, not how many.
    "join_fanout_biased": ("join_features", {
        "duplicate_rate": 0.9, "duplicate_bias": "high_spend", "duplicate_factor": 5}),
    "null_spike_mnar":    ("clean_customers", {
        "null_rate": 0.9, "null_mode": "mnar"}),

    # Row-starvation pair: each halves the training set and is survivable
    # alone; together the training set collapses. Designed to be superadditive,
    # which is the regime where leave-one-out double-counts and Shapley does not.
    "starve_rows_a": ("clean_customers", {"null_rate": 0.80, "null_mode": "mcar"}),
    "starve_rows_b": ("filter_active",   {"min_txn": 8}),
}


def build_pipeline(faults: Dict[str, dict] | None = None, probe: bool = False) -> Pipeline:
    """Assemble the DAG.

    `faults` maps operator name -> extra incident params. Sources always differ
    between states (benign day-over-day drift); everything else differs only
    where a fault was injected.
    """
    faults = faults or {}
    day_ref = PROBE_DAY if probe else REF_DAY
    day_inc = PROBE_DAY if probe else INC_DAY

    def params(op_name: str, base: dict) -> Tuple[dict, dict]:
        return dict(base), {**base, **faults.get(op_name, {})}

    specs = [
        ("load_customers", [], op_load_customers, {"day_seed": day_ref}, {"day_seed": day_inc}),
        ("load_transactions", [], op_load_transactions, {"day_seed": day_ref}, {"day_seed": day_inc}),
    ]

    ops: List[Operator] = [
        Operator(name, parents, fn, ref, inc) for name, parents, fn, ref, inc in specs
    ]

    for name, parents, fn, base in [
        ("clean_customers", ["load_customers"], op_clean_customers, {}),
        ("agg_transactions", ["load_transactions"], op_agg_transactions, {}),
        ("join_features", ["clean_customers", "agg_transactions"], op_join_features, {}),
        ("filter_active", ["join_features"], op_filter_active, {"min_txn": 1}),
        ("encode_features", ["filter_active"], op_encode_features, {"drop_columns": []}),
    ]:
        ref, inc = params(name, base)
        ops.append(Operator(name, parents, fn, ref, inc))

    return Pipeline(ops, sink="encode_features")


# Severity sweeps. The gate experiment showed that hand-tuned fault parameters
# produce wildly different damage -- a unit change cost 0.115 AUC while a severe
# biased fan-out cost 0.0016 -- which makes cross-fault comparison meaningless.
# Each fault instead gets a monotone severity ladder, so every method can be
# scored as a function of how hard the fault actually hits rather than of which
# fault it happens to be.
FAULT_SWEEPS: Dict[str, Tuple[str, List[Tuple[str, dict]]]] = {
    "unit_change": ("agg_transactions", [
        ("0.9x", {"amount_scale": 0.9}),
        ("0.7x", {"amount_scale": 0.7}),
        ("0.5x", {"amount_scale": 0.5}),
        ("0.2x", {"amount_scale": 0.2}),
        ("0.05x", {"amount_scale": 0.05}),
        ("0.01x", {"amount_scale": 0.01}),
    ]),
    "schema_drift": ("encode_features", [
        ("drop avg_amount", {"drop_columns": ["avg_amount"]}),
        ("drop tenure", {"drop_columns": ["tenure_days"]}),
        ("drop total_amount", {"drop_columns": ["total_amount"]}),
        ("drop income", {"drop_columns": ["income"]}),
        ("drop income+total", {"drop_columns": ["income", "total_amount"]}),
        ("drop income+total+tenure",
         {"drop_columns": ["income", "total_amount", "tenure_days"]}),
    ]),
    "predicate_flip": ("filter_active", [
        ("min_txn=2", {"min_txn": 2}),
        ("min_txn=3", {"min_txn": 3}),
        ("min_txn=4", {"min_txn": 4}),
        ("min_txn=6", {"min_txn": 6}),
        ("min_txn=8", {"min_txn": 8}),
        ("min_txn=10", {"min_txn": 10}),
    ]),
    "null_spike_mnar": ("clean_customers", [
        ("20%", {"null_rate": 0.2, "null_mode": "mnar"}),
        ("40%", {"null_rate": 0.4, "null_mode": "mnar"}),
        ("60%", {"null_rate": 0.6, "null_mode": "mnar"}),
        ("80%", {"null_rate": 0.8, "null_mode": "mnar"}),
        ("95%", {"null_rate": 0.95, "null_mode": "mnar"}),
        ("100%", {"null_rate": 1.0, "null_mode": "mnar"}),
    ]),
    "join_fanout_biased": ("join_features", [
        ("x1", {"duplicate_rate": 0.9, "duplicate_bias": "high_spend", "duplicate_factor": 1}),
        ("x2", {"duplicate_rate": 0.9, "duplicate_bias": "high_spend", "duplicate_factor": 2}),
        ("x5", {"duplicate_rate": 0.9, "duplicate_bias": "high_spend", "duplicate_factor": 5}),
        ("x10", {"duplicate_rate": 0.9, "duplicate_bias": "high_spend", "duplicate_factor": 10}),
        ("x20", {"duplicate_rate": 0.9, "duplicate_bias": "high_spend", "duplicate_factor": 20}),
    ]),
}


def scenario_from_faults(faults: Dict[str, dict]) -> Tuple[Pipeline, List[str]]:
    """Build a pipeline from an explicit operator -> incident-params map."""
    return build_pipeline(faults), sorted(faults)


def scenario(fault_names: List[str]) -> Tuple[Pipeline, List[str]]:
    """Build a pipeline with the named faults injected; return it and the
    ground-truth culprit operators."""
    faults: Dict[str, dict] = {}
    culprits: List[str] = []
    for fname in fault_names:
        op_name, cfg = FAULTS[fname]
        faults.setdefault(op_name, {}).update(cfg)
        culprits.append(op_name)
    return build_pipeline(faults), culprits
