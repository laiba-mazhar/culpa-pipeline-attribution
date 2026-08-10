"""A nine-operator ETL pipeline over NYC yellow taxi data.

Task: predict whether a credit-card trip receives a generous tip
(tip_amount / fare_amount > 0.20).

Two runs of this pipeline differ by which *day partition* they read. That is the
whole point -- the benign drift between reference and incident is real
day-over-day movement in New York, not noise we generated. Faults are then
injected on top of it.

Label hygiene: `tip_amount` and `total_amount` are dropped before modelling.
total_amount includes the tip, so leaving it in would leak the label outright
and every fault would look harmless.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .nyctaxi import BOROUGHS, load_day, load_zones
from .pipeline import Operator, Pipeline

TIP_THRESHOLD = 0.20


# -- operator bodies -----------------------------------------------------


def op_load_trips(inputs, params):
    return load_day(params["year"], params["month"], params["day"],
                    fixture=params["fixture"])


def op_load_zones(inputs, params):
    return load_zones(fixture=params["fixture"])


def op_filter_valid(inputs, params):
    """Keep credit-card trips with sane fares and distances.

    Fault surface: `min_fare` is a predicate flip. Raising it silently drops the
    short-trip stratum, which is exactly the population that tips differently.
    """
    df = inputs["load_trips"]
    return df[
        (df["payment_type"] == 1)
        & (df["fare_amount"] > params.get("min_fare", 0.0))
        & (df["trip_distance"] > 0)
        & (df["trip_distance"] < 100)
        & (df["passenger_count"].notna())
    ].reset_index(drop=True)


def op_derive_time(inputs, params):
    """Trip duration and calendar features.

    Fault surface: `duration_unit` is the classic silent unit change -- an
    upstream refactor starts emitting seconds where the model was trained on
    minutes. Nothing errors and no constraint fires.
    """
    df = inputs["filter_valid"].copy()
    pickup = pd.to_datetime(df["tpep_pickup_datetime"])
    dropoff = pd.to_datetime(df["tpep_dropoff_datetime"])

    df["duration"] = (dropoff - pickup).dt.total_seconds() / params.get("duration_unit", 60.0)
    df["hour"] = pickup.dt.hour
    df["dow"] = pickup.dt.dayofweek
    return df[df["duration"].between(0, 1e5)].reset_index(drop=True)


def op_zone_stats(inputs, params):
    """Per-pickup-zone aggregate features, computed from the same partition."""
    df = inputs["derive_time"]
    return (
        df.groupby("PULocationID")
        .agg(zone_mean_fare=("fare_amount", "mean"),
             zone_trip_count=("fare_amount", "size"))
        .reset_index()
    )


def op_join_pu_zone(inputs, params):
    """Attach the pickup borough.

    Fault surface: `corrupt_ids` maps a fraction of PULocationIDs to values with
    no match in the lookup, so the join silently produces nulls -- a broken
    upstream key mapping, one of the most common real ETL faults.
    """
    df = inputs["derive_time"].copy()
    zones = inputs["load_zones"]

    frac = params.get("corrupt_ids", 0.0)
    if frac > 0:
        rng = np.random.default_rng(params.get("corrupt_seed", 5))
        mask = rng.random(len(df)) < frac
        df.loc[mask, "PULocationID"] = 9999

    z = zones.rename(columns={"LocationID": "PULocationID", "Borough": "pu_borough"})
    return df.merge(z[["PULocationID", "pu_borough"]], on="PULocationID", how="left")


def op_join_do_zone(inputs, params):
    """Attach the dropoff borough.

    Fault surface: `duplicate_zones` duplicates rows in the lookup, producing
    join fan-out -- trips silently multiply and the training set is reweighted.
    """
    df = inputs["join_pu_zone"]
    zones = inputs["load_zones"]
    z = zones.rename(columns={"LocationID": "DOLocationID", "Borough": "do_borough"})
    z = z[["DOLocationID", "do_borough"]]

    dup_frac = params.get("duplicate_zones", 0.0)
    if dup_frac > 0:
        rng = np.random.default_rng(params.get("dup_seed", 13))
        z = pd.concat([z, z[rng.random(len(z)) < dup_frac]], ignore_index=True)

    return df.merge(z, on="DOLocationID", how="left")


def op_join_stats(inputs, params):
    df = inputs["join_do_zone"]
    stats = inputs["zone_stats"]
    out = df.merge(stats, on="PULocationID", how="left")
    out[["zone_mean_fare", "zone_trip_count"]] = out[
        ["zone_mean_fare", "zone_trip_count"]
    ].fillna(0.0)
    return out


def op_encode(inputs, params):
    """Select modelling columns, one-hot the boroughs, emit the label.

    Fault surface: `drop_columns` is schema drift.

    The borough dummies are built against a fixed vocabulary rather than from
    whatever appears in this partition, so the feature matrix has a stable
    schema across days. Without that, ordinary day-over-day variation would
    change the column set and be indistinguishable from a real schema fault.
    """
    df = inputs["join_stats"].copy()

    df["label"] = (df["tip_amount"] / df["fare_amount"].replace(0, np.nan)
                   > TIP_THRESHOLD).fillna(False).astype(int)

    feats = df[["trip_distance", "duration", "hour", "dow", "passenger_count",
                "fare_amount", "zone_mean_fare", "zone_trip_count"]].copy()

    for col, prefix in [("pu_borough", "pu"), ("do_borough", "do")]:
        for b in BOROUGHS:
            feats[f"{prefix}_{b.replace(' ', '_')}"] = (df[col] == b).astype(int)

    for col in params.get("drop_columns", []):
        if col in feats.columns:
            feats = feats.drop(columns=[col])

    feats["label"] = df["label"].to_numpy()
    return feats.dropna().reset_index(drop=True)


# -- assembly ------------------------------------------------------------

FAULTS: Dict[str, Tuple[str, dict]] = {
    "unit_change":    ("derive_time",  {"duration_unit": 1.0}),      # seconds, not minutes
    "predicate_flip": ("filter_valid", {"min_fare": 20.0}),
    "broken_join_key": ("join_pu_zone", {"corrupt_ids": 0.6}),
    "join_fanout":    ("join_do_zone", {"duplicate_zones": 0.8}),
    "schema_drift":   ("encode",       {"drop_columns": ["trip_distance", "duration"]}),
}

FAULT_SWEEPS: Dict[str, Tuple[str, List[Tuple[str, dict]]]] = {
    "unit_change": ("derive_time", [
        ("/30", {"duration_unit": 30.0}), ("/10", {"duration_unit": 10.0}),
        ("/2", {"duration_unit": 2.0}), ("seconds", {"duration_unit": 1.0}),
    ]),
    "predicate_flip": ("filter_valid", [
        ("fare>5", {"min_fare": 5.0}), ("fare>10", {"min_fare": 10.0}),
        ("fare>20", {"min_fare": 20.0}), ("fare>35", {"min_fare": 35.0}),
    ]),
    "broken_join_key": ("join_pu_zone", [
        ("20%", {"corrupt_ids": 0.2}), ("50%", {"corrupt_ids": 0.5}),
        ("80%", {"corrupt_ids": 0.8}), ("100%", {"corrupt_ids": 1.0}),
    ]),
    "join_fanout": ("join_do_zone", [
        ("20%", {"duplicate_zones": 0.2}), ("50%", {"duplicate_zones": 0.5}),
        ("100%", {"duplicate_zones": 1.0}),
    ]),
    "schema_drift": ("encode", [
        ("drop hour", {"drop_columns": ["hour"]}),
        ("drop distance", {"drop_columns": ["trip_distance"]}),
        ("drop distance+duration", {"drop_columns": ["trip_distance", "duration"]}),
        ("drop distance+duration+fare",
         {"drop_columns": ["trip_distance", "duration", "fare_amount"]}),
    ]),
}


def build_taxi_pipeline(
    ref_day: int, inc_day: int, year: int = 2024, month: int = 1,
    faults: Dict[str, dict] | None = None, fixture: bool = True,
) -> Pipeline:
    faults = faults or {}
    base = {"year": year, "month": month, "fixture": fixture}

    def split(name: str, p: dict) -> Tuple[dict, dict]:
        return dict(p), {**p, **faults.get(name, {})}

    ops: List[Operator] = [
        Operator("load_trips", [], op_load_trips,
                 {**base, "day": ref_day}, {**base, "day": inc_day}),
        Operator("load_zones", [], op_load_zones, dict(base), dict(base)),
    ]

    for name, parents, fn, p in [
        ("filter_valid", ["load_trips"], op_filter_valid, {"min_fare": 0.0}),
        ("derive_time", ["filter_valid"], op_derive_time, {"duration_unit": 60.0}),
        ("zone_stats", ["derive_time"], op_zone_stats, {}),
        ("join_pu_zone", ["derive_time", "load_zones"], op_join_pu_zone, {}),
        ("join_do_zone", ["join_pu_zone", "load_zones"], op_join_do_zone, {}),
        ("join_stats", ["join_do_zone", "zone_stats"], op_join_stats, {}),
        ("encode", ["join_stats"], op_encode, {"drop_columns": []}),
    ]:
        ref, inc = split(name, p)
        ops.append(Operator(name, parents, fn, ref, inc))

    return Pipeline(ops, sink="encode")


def taxi_scenario(
    fault_specs: Dict[str, dict], ref_day: int, inc_day: int, fixture: bool = True
) -> Tuple[Pipeline, List[str]]:
    return (
        build_taxi_pipeline(ref_day, inc_day, faults=fault_specs, fixture=fixture),
        sorted(fault_specs),
    )


def build_probe(probe_day: int, fixture: bool = True) -> pd.DataFrame:
    """A clean feature table from a third, held-out day."""
    return build_taxi_pipeline(probe_day, probe_day, fixture=fixture).replay(frozenset())
