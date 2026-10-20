# Golden tests

The golden tests pin the outputs of the library to the outputs of the reference implementation it was
refactored from. A golden is a JSON file of flat keys and values: hashes of arrays, floats as `float.hex`,
counts and score statistics. A refactoring step passes only if every value is reproduced bit for bit.

## What each golden pins

| Golden | Data | What it pins |
|---|---|---|
| `lib/synth.json` | three seeded synthetic days (`recipes/synth_stream.py`) | ITCH parsing (messages, counts, chunking, byte limit), the build log and the Parquet tables, the order-book chain of three stocks (flow-branch cleaning under both sign conventions and dedupe settings, labels, sample masks, minute-branch change events and buckets, scalers, sufficient statistics, ridge for five λ, evaluation scores, block t-test, first principal component) |
| `lib/synth_halfday.json` | a synthetic full day and half day | Δt with each day's regular trading time, the half-day session, labels and masks |
| `lib/real_smoke.json` | two stocks on five real days | the same chain on real books |
| `lib/real_build_outputs.json` | the stored build logs of the real days | input and output hashes and statuses; the Parquet files of the smoke days hashed again |
| `lib/lobster_samples.json` | the public LOBSTER samples | both reconciliation modes |
| `train/cache.json` (T1) | the smoke split's order-book tables | the feature cache rebuilt from the tables: every array, scaler and JSON file of the stored cache |
| `train/batches.json` (T2) | the stored smoke cache | the training days, the 64 first batches of the seed-0 pass (in-process, with loader processes, from batch 10, and from the compact backend), the pass hashes of five seeds |
| `train/init.json` (T3) | none | the initial weights of the six catalogue models for three seeds |
| `train/fits.json` (T4) | the stored smoke cache | short fits of three seeds (per-check losses, weights, configuration hash), the same fit again, deterministic algorithms, a fit stopped at its second check and continued |
| `train/ridge.json` (T5) | the stored smoke cache | the two linear references: λ, validation losses, weights |
| `train/scores_test.json` (T6) | the stored smoke cache | the test days scored once: cells, the stock-days the corporate-action file leaves out, price ratios |
| `train/scores_valid.json` (T7) | the stored smoke cache | the validation cells and per-stock losses of the three seed-0 fits, on the whole validation days |
| `results_totals.json` | the stored score files | cell counts and exact sums of SSE and SST per experiment and record kind |
| `reports/<name>.json` (local) | the stored report and the stored score files | every number of the report's score and fit tables, printed and as float hex |
| `reports/earlier.json` (local) | the earlier stage's stored report and score files, and the decay measurement's units | the earlier stage's score tables and the table of its second scoring; when it is recorded, 24 of its R² values are checked: 23 against the stored report, 1 against the stored linear file |

Real-data fixtures are local only and are read from `DOF_GOLDEN_DATA_DIR`.

Only the synthetic goldens and `results_totals.json` are committed. A golden holds no market data, no path
and no machine fact.

## How a golden is recorded

The recipe code in `recipes/` turns library outputs into flat values. It imports nothing of `dof`; it is run
once on the reference implementation, through a backend that only routes calls, to record a golden, and run
again on `dof`, through `dof_backend.py`, to test it. A backend never computes, casts, rounds, sorts or filters
a returned value. Every golden records `recipe_sha256`, the hash of `recipes/*.py`; when the recipe changes, the
golden tests stop with "recipe changed since recording" until the goldens are recorded again.

Each golden is recorded with one BLAS thread (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`), `PYTHONHASHSEED=0` and the library versions written in its `env` block. The tests refuse
to compare under other settings or versions.

## How to compare

```
DOF_GOLDENS=1 PYTHONPATH=<repo>/src python -m pytest tests/goldens -p no:cacheprovider
```

Tests of real-data goldens also need `DOF_GOLDEN_DATA_DIR` and the data root they read
(`DOF_GOLDEN_BOOK_ROOT`, `DOF_GOLDEN_LOBSTER_DIR` or `DOF_GOLDEN_ITCH_DIR`); without them they are skipped.

The training goldens (`test_dof_golden_train.py`) read:

| Variable | What |
|---|---|
| `DOF_GOLDEN_DATA_DIR` | the folder of the real-data goldens (`train/*.json`) |
| `DOF_GOLDEN_SMOKE_CACHE` | the stored smoke cache (`train/` and `eval/`), read only: a fixture checks that no file of it changes |
| `DOF_GOLDEN_BOOK_ROOT` | the order-book tables of the smoke days (T1 rebuilds the cache from them) |
| `DOF_GOLDEN_FULL_DIR` | optional: the full-form goldens, for clear values in the report of a difference |
| `DOF_GOLDEN_RESULTS_DIR` | the result tables the report goldens (`test_dof_golden_reports.py`) rebuild the reports from |

The training goldens run on the CPU with two torch threads; the test of the batches forks loader processes, so it
comes first.

The report goldens (`reports/<name>.json` in `DOF_GOLDEN_DATA_DIR`) and their recorder name every value with
`report_keys.py`; each report golden records its sha256, and the test refuses a golden recorded with another one.
