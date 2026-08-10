"""NYC TLC yellow taxi trip records: loading, schema contract, and a fixture.

Why this dataset. The whole method rests on comparing two runs of the same
pipeline over two temporal partitions, and on the *benign* drift between them
being realistic. Synthetic drift drawn from a stationary distribution is much
kinder than reality. NYC taxi data has genuine day-over-day partitions --
weekday/weekend, weather, holidays, tourism -- so the fault-to-drift ratio that
drives the main result is measured against real background movement rather than
against noise we generated ourselves.

Data is NOT bundled. See DATA.md for what to download and where to put it.
Everything here works against a generated fixture with the identical schema, so
the pipeline can be developed and tested before the real files arrive, and the
same code path runs unchanged once they do.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Columns the pipeline actually consumes. The real files carry ~19; we assert
# only on what we use, so a schema change elsewhere in the file does not break
# the run for no reason.
REQUIRED_TRIP_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "tip_amount",
    "total_amount",
]

REQUIRED_ZONE_COLUMNS = ["LocationID", "Borough", "Zone", "service_zone"]

BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"]


class DataMissing(RuntimeError):
    """Raised with actionable instructions rather than a bare FileNotFoundError."""


def trips_path(year: int, month: int, fixture: bool = False) -> Path:
    stem = "fixture_tripdata" if fixture else "yellow_tripdata"
    return DATA_DIR / f"{stem}_{year:04d}-{month:02d}.parquet"


def zones_path(fixture: bool = False) -> Path:
    return DATA_DIR / ("fixture_zone_lookup.csv" if fixture else "taxi_zone_lookup.csv")


# -- loading -------------------------------------------------------------


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """TLC has shipped `airport_fee` and `Airport_fee` in different months, and
    casing drifts elsewhere too. Map defensively to the canonical names we use."""
    canon = {c.lower(): c for c in REQUIRED_TRIP_COLUMNS}
    renames = {c: canon[c.lower()] for c in df.columns if c.lower() in canon and c not in canon.values()}
    return df.rename(columns=renames) if renames else df


def load_month(year: int, month: int, fixture: bool = False) -> pd.DataFrame:
    path = trips_path(year, month, fixture)
    if not path.exists():
        raise DataMissing(
            f"missing {path.name} in {DATA_DIR}\n"
            f"  download: https://d37ci6vzurychx.cloudfront.net/trip-data/"
            f"yellow_tripdata_{year:04d}-{month:02d}.parquet\n"
            f"  or generate a fixture:  python -m culpa.nyctaxi --make-fixture\n"
            f"  see DATA.md"
        )
    df = _normalise_columns(pd.read_parquet(path))
    missing = [c for c in REQUIRED_TRIP_COLUMNS if c not in df.columns]
    if missing:
        raise DataMissing(f"{path.name} is missing required columns: {missing}")
    return df[REQUIRED_TRIP_COLUMNS]


def load_zones(fixture: bool = False) -> pd.DataFrame:
    path = zones_path(fixture)
    if not path.exists():
        raise DataMissing(
            f"missing {path.name} in {DATA_DIR}\n"
            f"  download: https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv\n"
            f"  or generate a fixture:  python -m culpa.nyctaxi --make-fixture\n"
            f"  see DATA.md"
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_ZONE_COLUMNS if c not in df.columns]
    if missing:
        raise DataMissing(f"{path.name} is missing required columns: {missing}")
    return df[REQUIRED_ZONE_COLUMNS]


def load_day(
    year: int, month: int, day: int, fixture: bool = False, max_rows: Optional[int] = 60_000
) -> pd.DataFrame:
    """One day's partition, which is the unit two pipeline runs differ by.

    The TLC files are monthly, and they contain a small number of records whose
    pickup timestamp falls outside the nominal month -- a real data-quality
    quirk. Filtering by date rather than trusting the filename is the correct
    handling and is what a production pipeline would do.
    """
    df = load_month(year, month, fixture)
    ts = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    sel = df[(ts.dt.year == year) & (ts.dt.month == month) & (ts.dt.day == day)]
    if max_rows is not None and len(sel) > max_rows:
        # Deterministic head, not a random sample: the replay engine hashes
        # operator outputs, so any nondeterminism here would defeat pruning.
        sel = sel.head(max_rows)
    return sel.reset_index(drop=True)


# -- data availability check --------------------------------------------


def check(year: int, month: int, days: List[int], fixture: bool = False) -> bool:
    """Report what is present and what is not, without raising."""
    label = "fixture" if fixture else "real TLC"
    print(f"checking {label} data in {DATA_DIR}\n")
    ok = True

    for path in (trips_path(year, month, fixture), zones_path(fixture)):
        if path.exists():
            mb = path.stat().st_size / 1e6
            print(f"  [ok]      {path.name}  ({mb:.1f} MB)")
        else:
            print(f"  [MISSING] {path.name}")
            ok = False

    if not ok:
        print("\nsee DATA.md for download links")
        return False

    for d in days:
        n = len(load_day(year, month, d, fixture))
        print(f"  {year}-{month:02d}-{d:02d}: {n:,} trips")
        if n < 1000:
            print(f"    warning: thin partition, results will be noisy")
    return True


# -- fixture -------------------------------------------------------------


def make_fixture(year: int = 2024, month: int = 1, n_per_day: int = 40_000) -> None:
    """Generate a stand-in with the real schema and plausible relationships.

    This exists so the pipeline can be built and tested before the real download,
    and so CI has something to run against. It is NOT a substitute for the real
    data in the paper: its day-over-day drift is synthetic, which is exactly the
    weakness the real dataset is meant to remove.
    """
    DATA_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(20240101)

    # Zones are generated FIRST so tipping can depend on the pickup borough.
    # An earlier version assigned boroughs to location IDs at random and made
    # tips a function of the raw ID, which meant the borough columns the
    # pipeline computes carried no signal -- and the faults that target the
    # borough join were consequently untestable. If the pipeline derives a
    # feature, the fixture has to make that feature matter.
    zones = pd.DataFrame({
        "LocationID": np.arange(1, 264),
        "Borough": rng.choice(BOROUGHS, 263, p=[0.45, 0.2, 0.2, 0.1, 0.04, 0.01]),
        "Zone": [f"Zone {i}" for i in range(1, 264)],
        "service_zone": rng.choice(["Yellow Zone", "Boro Zone", "Airports"], 263),
    })
    zones.to_csv(zones_path(fixture=True), index=False)

    borough_tip_effect = {
        "Manhattan": 0.55, "Brooklyn": 0.10, "Queens": -0.05,
        "Bronx": -0.35, "Staten Island": -0.45, "EWR": 0.30,
    }
    id_to_effect = zones.set_index("LocationID")["Borough"].map(borough_tip_effect)

    frames = []

    for day in range(1, 15):
        n = n_per_day
        # Weekends run longer, cheaper-tipping trips. Gives real-ish drift
        # between adjacent days without pretending to be the real thing.
        weekend = day % 7 in (0, 6)
        base_dist = 3.4 if weekend else 2.6

        distance = rng.lognormal(np.log(base_dist), 0.7, n)
        duration_min = distance * rng.uniform(2.5, 5.0, n) + rng.exponential(3, n)
        hour = rng.integers(0, 24, n)
        pickup = (
            pd.Timestamp(year=year, month=month, day=day)
            + pd.to_timedelta(hour, unit="h")
            + pd.to_timedelta(rng.integers(0, 3600, n), unit="s")
        )
        fare = 3.0 + 2.6 * distance + 0.35 * duration_min + rng.normal(0, 1.5, n)
        fare = np.clip(fare, 2.5, None)

        pu = rng.integers(1, 264, n)
        do = rng.integers(1, 264, n)

        # Tip generosity: shorter trips, daytime, Manhattan-ish zones tip more.
        # The intercept is calibrated so the label lands near balanced. An
        # earlier version used 0.9 and produced a 92.5% positive rate, which
        # made AUC unstable and every effect indistinguishable from noise.
        z = (
            0.30
            - 0.30 * np.log1p(distance)
            + 0.45 * ((hour >= 7) & (hour <= 19)).astype(float)
            - 0.25 * weekend
            + id_to_effect.reindex(pu).to_numpy()
            + rng.normal(0, 0.40, n)
        )
        tip_pct = np.clip(0.18 + 0.10 * z, 0.0, 0.6)
        payment_type = rng.choice([1, 2], n, p=[0.72, 0.28])
        tip = np.where(payment_type == 1, fare * tip_pct, 0.0)

        frames.append(pd.DataFrame({
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": pickup + pd.to_timedelta(duration_min, unit="m"),
            "passenger_count": rng.choice([1, 1, 1, 2, 2, 3, 4], n).astype("float64"),
            "trip_distance": distance,
            "PULocationID": pu.astype("int64"),
            "DOLocationID": do.astype("int64"),
            "payment_type": payment_type.astype("int64"),
            "fare_amount": fare,
            "tip_amount": tip,
            "total_amount": fare + tip + 1.0,
        }))

    trips = pd.concat(frames, ignore_index=True)
    trips.to_parquet(trips_path(year, month, fixture=True), index=False)

    print(f"wrote {trips_path(year, month, fixture=True).name}  "
          f"({len(trips):,} trips over 14 days)")
    print(f"wrote {zones_path(fixture=True).name}  ({len(zones)} zones)")


if __name__ == "__main__":
    import sys

    if "--make-fixture" in sys.argv:
        make_fixture()
    else:
        real = "--fixture" not in sys.argv
        check(2024, 1, [8, 9, 15], fixture=not real)
