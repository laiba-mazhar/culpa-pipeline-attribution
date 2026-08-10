# Data you need to download

Everything in this repo runs today against a generated fixture. The fixture
validates that the code works end to end on the real NYC taxi schema — but it
**cannot produce the paper's real-data result**, because its day-over-day drift
is synthetic, and realistic benign drift is the entire point of the experiment.
Running `experiments/real_data.py` on the fixture prints a warning saying so.

Two files. No account, no API key, no signup. Both are public and direct.

---

## 1. Yellow taxi trips — January 2024

**URL**
```
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
```

**Size** ~48 MB
**Save as** `C:\Users\laiba\Documents\CULPA\data\yellow_tripdata_2024-01.parquet`

Keep the filename exactly as-is — the loader builds the path from the year and
month.

## 2. Taxi zone lookup

**URL**
```
https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv
```

**Size** ~12 KB
**Save as** `C:\Users\laiba\Documents\CULPA\data\taxi_zone_lookup.csv`

This is the `LocationID → Borough / Zone` table the two join operators use.

---

## Optional: more months

More months means more choices of reference/incident day pair, which matters
because the fault-to-drift ratio is what the main result is plotted against.
Worth grabbing if the January pairs turn out to have too little drift:

```
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-07.parquet
```

July is useful specifically because summer tourism traffic looks quite different
from January's.

The official landing page, if a link ever breaks, is
[TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
Files are published monthly with roughly a two-month lag.

---

## If you'd rather not click through a browser

From the project root:

```bash
mkdir -p data && curl -L -o data/yellow_tripdata_2024-01.parquet https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet && curl -L -o data/taxi_zone_lookup.csv https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv
```

---

## Once the files are in place

**Verify they loaded and the partitions are populated:**

```bash
python -m culpa.nyctaxi
```

Expect `[ok]` for both files and a few tens of thousands of trips per day. A
partition under 1,000 trips triggers a warning — results would be too noisy.

**Then run the experiment:**

```bash
python -m experiments.real_data --real
```

**Then pick day pairs with real drift.** This is the part that needs judgement,
not just a download. January 2024:

| pair | `--days REF,INC,PROBE` | what it tests |
|---|---|---|
| Mon vs Tue | `8,9,10` | minimal drift — the easy regime |
| Mon vs Sat | `8,13,10` | weekday/weekend — substantial drift |
| Jan 1 vs Jan 15 | `1,15,10` | New Year's Day against an ordinary Monday |

The experiment prints `BENIGN DRIFT IS NEGLIGIBLE` if the two partitions are too
similar for the comparison to mean anything. If that fires, pick a more
distinct pair. The result worth reporting comes from pairs where benign drift is
comparable to, or larger than, the injected fault — that is where plain Shapley
and the anchored value diverge, and it is the whole argument of the paper.

---

## What is *not* needed

- No cloud account, credentials, or paid API
- No GPU
- `pyarrow` is already installed (24.0.0), so parquet reads work
- The fixture stays in the repo and keeps working; the real files sit alongside
  it under different names, so nothing is overwritten
