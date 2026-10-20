# deep-order-flow

**Deep order flow on Nasdaq: forecast power across models and across years**

Steven Shi · 2026-10-20 · a replication of Kolm, Turiel and Westray (2023) on Nasdaq TotalView-ITCH 5.0 data

See [Limitations and disclosures](#limitations-and-disclosures) for the data licence, the scope and the validity of every number below.

## Summary

**Networks forecast better than a linear reference on the 2019 test days.** The earlier stage of this work trained one pooled model per network for the 115 tickers on three days of 2019. On 2019-10-18, 2019-10-30 and 2019-12-30, pooled networks reach an out-of-sample R² of 1.136 % (CNN-LSTM; s.e. 0.108), 1.114 % (LSTM-MLP; s.e. 0.105) and 1.044 % (LSTM; s.e. 0.106) on 115 stocks, mean of the horizons k = 1..10. A pooled ridge regression on the same inputs reaches 0.241 % (s.e. 0.049; 115 stocks; k = 1..10).

**These 2019 numbers are optimistic.**

- These three test days were read many times while settings were tried: the earlier stage scored them for linear models on 20 tickers under two sign rules, one model per stock, a pooled model on 20 tickers, learning-rate and epsilon variants of the CNN-LSTM, and the models above. The numbers describe the models; they are not a first-look test.
- The days were fixed beforehand; the models and settings were not. No protocol document fixed the earlier stage's models, settings and order of work before the test days were read, and the test-day reads of its 20-stock runs sit in records that are not in this repository's tables, so the order of work cannot be audited.
- The CNN-LSTM's 3e-5/1e-7 setting was one of three picked on the validation day by the 20-stock learning-rate map (the lowest validation loss among the cells with epsilon 1e-7). All three 20-stock fits, this setting included, were scored on these test days. On 115 stocks, the setting with learning rate 0.3 and epsilon 1 was fitted first and scored 0.609 % on these test days (115 stocks; k = 1..10); its weight steps were about three times larger than on 20 stocks, so the 3e-5/1e-7 fit was added after that score had been read. The 0.609 % fit is not in these tables.

**Forecast power holds for months.** On the four days scored from 2.6 to 6.0 months after training, the same CNN-LSTM and LSTM-MLP stay between 0.916 % and 1.449 % (113 to 115 stocks; k = 1..10). Only 2020-01-30 had not been scored as a held-out day before the decay run. The values do not fall in a straight line.

**Forecast power falls over years, for a cause not tested.** From 2021-07-13 on, both networks are below zero when every stock that the decay run could score is included: from -0.328 % to -5.504 % (98 to 113 stocks; k = 1..10). The fall is not separated from the change in price levels since 2019.

**Retrained networks forecast weakly in 2026.** The main run, the pre-registered stage of this work, trained the networks again on 14 days from 2019-01-30 to 2025-12-10. On 2026-05-15, 2026-05-18 and 2026-06-12, five to six months after its last training day, the CNN-LSTM reaches 0.268 % (s.e. 0.059; 96 stocks; k = 1..10). The pooled ridge regression is at -0.194 % (s.e. 0.047; 96 stocks; k = 1..10).

The sections below give the models compared on 2019, the decay to 2026, the retrained networks, the adjacent training days, the method and its set-up caveats, the data and split, and the limitations and disclosures. The code follows.

## Results by model

A first look is a scoring of a day that no model of this feature family (order flow of the ten best levels) had been scored on before; a second look is any later scoring.

The three statements on the 2019 test days in the Summary apply to Table 1: the days were read many times, the order of work cannot be audited, and the CNN-LSTM's 115-stock fit was added after other fits had been scored on these days. The values below are therefore optimistic.

**Table 1: Models ranked by out-of-sample R², 2019 test days**

| rank | model | test days | stocks | R² %, k = 1..10 | s.e. | R² %, k = 10 | s.e. |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `CNN-LSTM-OF-lr3e-5-eps1e-7` | 2019-10-18, 2019-10-30, 2019-12-30 | 115 | 1.136 | 0.108 | 1.372 | 0.103 |
| 2 | `LSTM-MLP-OF` | 2019-10-18, 2019-10-30, 2019-12-30 | 115 | 1.114 | 0.105 | 1.327 | 0.108 |
| 3 | `LSTM-OF` | 2019-10-18, 2019-10-30, 2019-12-30 | 115 | 1.044 | 0.106 | 1.280 | 0.110 |
| 4 | `Ridge-OF` (pooled ridge) | 2019-10-18, 2019-10-30, 2019-12-30 | 115 | 0.241 | 0.049 | 0.330 | 0.047 |

Source: `results/reports/earlier.md`, section "Result". Trained on 2019-01-30, 2019-03-27 and 2019-07-30; validated on 2019-08-30; mean of the three test days; s.e. across stocks.

The CNN-LSTM ranks first at 1.136 % (s.e. 0.108; 115 stocks; k = 1..10), and all three networks are above the pooled ridge regression at 0.241 % (s.e. 0.049; 115 stocks; k = 1..10). The three networks lie within one standard error of each other, from 1.044 % to 1.136 % (115 stocks; k = 1..10), and the order rests on test days that were read many times; this study gives no reason why one architecture leads. A per-stock ridge regression of the earlier stage (one model per stock) scored higher than the pooled ridge regression on the same stocks and test days; it is not in these tables. Stocks move together, so these standard errors are too small.

## Decay

**Table 2: The 2019 networks by distance from training**

`CNN-LSTM-OF-lr3e-5-eps1e-7` and `LSTM-MLP-OF` of Table 1, with the same saved weights and each stock's 2019 scalers and Δt; k = 1..10. `LSTM-OF` and the ridge reference were not scored again. Row 'all': every stock-day that the decay run could score (it leaves out a reused ticker, a stock without a usable scaling group and a stock-day without a sample). Row 'inside': stock-days whose median mid price is inside 0.6 to 1.6 times its 2019-07-30 value. Months count from 2019-07-30, the last training day.

| day | months after 2019-07-30 | 'all': stocks | 'all': `CNN-LSTM-OF-lr3e-5-eps1e-7` R² % | 'all': `LSTM-MLP-OF` R² % | 'inside': stocks | 'inside': `CNN-LSTM-OF-lr3e-5-eps1e-7` R² % | 'inside': `LSTM-MLP-OF` R² % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2019-10-18 | 2.6 | 115 | 1.428 | 1.449 | 115 | 1.428 | 1.449 |
| 2019-10-30 | 3.0 | 115 | 0.928 | 0.959 | 115 | 0.928 | 0.959 |
| 2019-12-30 | 5.0 | 115 | 1.052 | 0.932 | 113 | 1.030 | 0.916 |
| 2020-01-30 | 6.0 | 115 | 1.030 | 1.021 | 114 | 1.038 | 1.029 |
| 2021-07-13 | 23.5 | 113 | -0.328 | -0.538 | 64 | 0.884 | 0.955 |
| 2021-08-13 | 24.5 | 112 | -1.003 | -1.684 | 68 | 0.332 | 0.317 |
| 2025-11-28 (half day) | 76.0 | 99 | -3.435 | -4.754 | 44 | -0.624 | -0.675 |
| 2025-12-08 | 76.3 | 99 | -3.244 | -4.578 | 44 | -0.821 | -0.760 |
| 2025-12-09 | 76.3 | 99 | -3.113 | -4.446 | 44 | -0.233 | -0.126 |
| 2025-12-10 | 76.4 | 99 | -1.867 | -2.731 | 44 | -0.051 | -0.016 |
| 2025-12-11 | 76.4 | 99 | -2.508 | -3.460 | 43 | -1.017 | -0.525 |
| 2025-12-12 | 76.4 | 99 | -2.633 | -3.944 | 44 | -0.262 | -0.251 |
| 2026-05-15 | 81.5 | 98 | -4.035 | -5.504 | 43 | -0.234 | -0.286 |
| 2026-05-18 | 81.6 | 98 | -2.208 | -2.953 | 44 | -0.132 | -0.162 |
| 2026-06-12 | 82.4 | 98 | -1.949 | -2.704 | 42 | 0.226 | 0.078 |

Source: `results/reports/decay.md`, tables "k = 1..10", basis (i), built by `python -m dof.experiments report --results results --all --out-dir results/reports`; the report also gives the s.e. of every cell.

The first three rows are the test days of Table 1, scored again with the same saved weights; `results/reports/earlier.md` compares the two scorings. From 2.6 to 6.0 months, both networks stay between 0.916 % and 1.449 % (113 to 115 stocks). At 6.0 months (1.030 %, 1.021 %; 115 stocks) the values are above those at 3.0 months (0.928 %, 0.959 %; 115 stocks), so the decline is not a straight line. From 2021-07-13 on, every 'all' row is below zero. The 'inside' rows reach 0.884 % and 0.955 % on 2021-07-13 (64 stocks), 0.332 % and 0.317 % on 2021-08-13 (68 stocks), and from -1.017 % to 0.226 % in 2025 and 2026 (42 to 44 stocks).

With scalers and Δt refitted on 2025-11-28 to 2025-12-10, the two networks reach 0.278 % (98 stocks) and 0.102 % (98 stocks) on 2025-12-12 (k = 1..10); `results/reports/decay.md` gives this refreshed basis for every later day.

The fall over years is not separated from the change in price levels. On 2026-05-15 the price ratio against 2019-07-30 was below 0.6 on 20 stock-days and above 1.6 on 35 stock-days, out of 98 stock-days scored. The 'inside' rows hold fewer stocks, and the stock set changes from day to day. This run does not test the cause. Every day of Table 2 is a second look for this feature family.

## Retrained networks on the 2026 test days

**Table 3: The 14-day models ranked by out-of-sample R², 2026 test days**

| rank | model | test days | stocks | R² %, k = 1..10 | s.e. | R² %, k = 10 | s.e. |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `CNN-LSTM-OF-alldays` | 2026-05-15, 2026-05-18, 2026-06-12 | 96 | 0.268 | 0.059 | 0.428 | 0.080 |
| 2 | `LSTM-MLP-OF-alldays` | 2026-05-15, 2026-05-18, 2026-06-12 | 96 | 0.196 | 0.059 | 0.359 | 0.087 |
| 3 | `Ridge-OF-alldays` | 2026-05-15, 2026-05-18, 2026-06-12 | 96 | -0.194 | 0.047 | -0.159 | 0.064 |

Source: `results/reports/alldays.md`, section "Result". Trained on the 14 days from 2019-01-30 to 2025-12-10; validated on 2025-12-11 and 2025-12-12; primary analysis; s.e. across stocks.

The CNN-LSTM ranks first again, at 0.268 % (s.e. 0.059; 96 stocks; k = 1..10). Stocks move together, so these standard errors are too small. 2026-05-15 is the weakest test day for both networks: 0.058 % and 0.020 % (96 stocks; k = 1..10). The secondary analysis (price ratio inside 0.6 to 1.6 of the last training day) gives 0.288 %, 0.206 % and -0.196 % (87 stocks; same test days; k = 1..10). Validation numbers chose the stopping check of each network and the λ of the ridge regression; they are not results, and `results/reports/alldays.md` gives them in its model sections.

## Adjacent training days

**Adjacent training days (one half day).** An `LSTM-MLP-OF` trained on five adjacent days (2025-11-28, a half day, and 2025-12-08 to 2025-12-11) and stopped on the morning of 2025-12-12 reaches 0.255 % on the afternoon of 2025-12-12 (s.e. 0.206; 98 stocks; k = 1..10; mean of three training runs). The same weights reach -0.002 % on the three 2026 test days (s.e. 0.069; 96 stocks; k = 1..10; primary analysis). One half day cannot show that adjacent training days forecast better in general; `results/reports/adjacent.md` gives the paired difference and its uncertainty.

## Method and set-up caveats

- **Data.** Nasdaq TotalView-ITCH 5.0 messages are rebuilt into a 10-level limit order book for each of the 115 tickers in `configs/universe_115.csv`, the universe of Kolm, Turiel and Westray (2023). A stock-day is used only when its book never hit an integrity error.
- **Inputs.** A sample is the last 100 book updates of one stock inside the trimmed session (09:40 to 15:50; 12:50 on a half day). Each update gives 20 numbers: the order flow at the 10 best bid levels and at the 10 best ask levels, with the sign convention of Cont, Kukanov and Stoikov (2014).
- **Labels.** The change of the mid price from t + 10 ms to t + h_k, for ten horizons h_k = (k / 5) · Δt, k = 1..10. Δt is the regular trading time of the training days over their number of mid-price changes (Kolm, Turiel and Westray 2023, eq. 16). Inputs and labels are scaled per stock with statistics of the training days only.
- **Earlier stage, 2019.** Each network is one model trained on the training rows of the 115 tickers: `LSTM-OF`, `LSTM-MLP-OF` and `CNN-LSTM-OF-lr3e-5-eps1e-7`. The pooled `Ridge-OF` is one closed-form ridge fit on the same rows, with λ chosen on the validation day. Training: 2019-01-30, 2019-03-27 and 2019-07-30. Validation: 2019-08-30. Test: 2019-10-18, 2019-10-30 and 2019-12-30. `LSTM-OF` and `LSTM-MLP-OF` were trained with Adam (learning rate 1e-5, epsilon 1e-7), the CNN-LSTM with Adam in the Keras form (learning rate 3e-5, epsilon 1e-7). The validation loss was read after each pass over the training rows; a fit stopped after 5 passes without a new lowest loss and kept the weights of its best pass.
- **Decay measurement.** `CNN-LSTM-OF-lr3e-5-eps1e-7` and `LSTM-MLP-OF` of the earlier stage, with the same saved weights and each stock's 2019 scalers and Δt, are scored again on the 2019 test days and on later days from 2020-01-30 to 2026-06-12. `LSTM-OF` and the ridge reference were not scored again.
- **Main run, 14 days.** Two network architectures from the catalogue in `configs/models.toml` (after Kolm, Turiel and Westray 2023, Tables 1 and 2): `LSTM-MLP-OF-alldays` and `CNN-LSTM-OF-alldays`, both trained on the 14 training days. A pooled ridge regression on the same inputs (`Ridge-OF-alldays`) is the linear reference.
- **Training of the main run.** Adam with learning rate 1e-5 for the LSTM-MLP, and Adam in the Keras form with learning rate 3e-5 for the CNN-LSTM; epsilon 1e-7 and batch size 256 for both. The validation loss is read every 317,214 steps. A fit stops after 5 checks without a new lowest loss, or after 50 checks, and keeps the weights of its best check. The ridge λ is chosen from 1e-3, 1e-2, 1e-1, 1 and 10 on the validation days.
- **Split of the main run.** Training: the 14 days 2019-01-30, 2019-03-27, 2019-07-30, 2019-08-30, 2019-10-18, 2019-10-30, 2019-12-30, 2020-01-30, 2021-07-13, 2021-08-13, 2025-11-28, 2025-12-08, 2025-12-09 and 2025-12-10. Validation: 2025-12-11 and 2025-12-12. Test: 2026-05-15, 2026-05-18 and 2026-06-12.
- **Order of work of the main run.** The split, the models, the settings and the metrics were written down before any test-day array of this run was loaded ([protocols/preregistration.md](protocols/preregistration.md)). Each model scored the test days once, after every model of the run had finished training; the export of the stored records checks that order.
- **Secondary analysis.** It keeps the test stock-days whose median mid price is inside 0.6 to 1.6 times that of the last training day.
- **R².** R² is out of sample and in percent: per stock, day and horizon against the mean label of that stock-day. A stock's value is the mean of its cells, a table value is the mean over stocks, and s.e. is the standard error across stocks.

## Data and split

The corporate-action file (`configs/protocol/corporate_actions_20251210_20260612.csv`) leaves out 4 test stock-days: BKNG on 2026-05-15, 2026-05-18 and 2026-06-12 (`forward_split`, effective 2026-04-06) and KLAC on 2026-06-12 (`forward_split`, effective 2026-06-12).

Before scoring, the pre-registered rules also leave out reused tickers (GLIBA and FB) and stocks whose shortest horizon would be at or below the 10 ms latency buffer; see [protocols/preregistration.md](protocols/preregistration.md), Revision 1. No market data is in this repository; `results/` holds only statistics computed from it (see [DATA_NOTICE.md](DATA_NOTICE.md)).

The 2019 numbers come from `results/reports/earlier.md`, the decay numbers from `results/reports/decay.md`, the 2026 numbers from `results/reports/alldays.md` and the adjacent-days numbers from `results/reports/adjacent.md`. Each report is rebuilt from the tables in `results/`.

## Limitations and disclosures

- **The 2019 test days (2019-10-18, 2019-10-30 and 2019-12-30) of `LSTM-OF`, `LSTM-MLP-OF`, `CNN-LSTM-OF-lr3e-5-eps1e-7` and `Ridge-OF`.**
  - These three test days were read many times while settings were tried: the earlier stage scored them for linear models on 20 tickers under two sign rules, one model per stock, a pooled model on 20 tickers, learning-rate and epsilon variants of the CNN-LSTM, and the models above. The numbers describe the models; they are not a first-look test.
  - The days were fixed beforehand; the models and settings were not. No protocol document fixed the earlier stage's models, settings and order of work before the test days were read, and the test-day reads of its 20-stock runs sit in records that are not in this repository's tables, so the order of work cannot be audited.
  - The CNN-LSTM's 3e-5/1e-7 setting was one of three picked on the validation day by the 20-stock learning-rate map (the lowest validation loss among the cells with epsilon 1e-7). All three 20-stock fits, this setting included, were scored on these test days. On 115 stocks, the setting with learning rate 0.3 and epsilon 1 was fitted first and scored 0.609 % on these test days (115 stocks; k = 1..10); its weight steps were about three times larger than on 20 stocks, so the 3e-5/1e-7 fit was added after that score had been read. The 0.609 % fit is not in these tables.
- Every day of the decay measurement is a second look for this feature family, and the stock set changes from row to row. The decay tables describe; they do not test.
- 2026-06-12 is not a first look for this feature family: an earlier experiment had scored per-stock ridge models on it. 2026-05-15 and 2026-05-18 are first looks for the main run; the decay and adjacent-days runs scored them afterwards.
- The learning rate 3e-5 and epsilon 1e-7 of `CNN-LSTM-OF-alldays` come from an earlier stage, where they were chosen after that stage's test days had been scored. The choice used none of the 2026 test days.
- Sample selection uses information from the future: a row is a sample only if its label window holds no trading halt, no crossed book and no session end. This is the replication convention; a live trader cannot know that a halt is coming.
- R² is measured against the stock-day's own mean label, which a trader does not know during the day.
- The order-flow sign rule of every run is that of Cont, Kukanov and Stoikov (2014): a bid price drop contributes minus the previous size at that level. Kolm, Turiel and Westray (2023, eqs. 2–3) print minus the current size. The library offers both rules (`current_size`, `cks`); no run in this repository's tables uses the paper's.
- Stocks move together, so the standard errors are too small.
- On the 2026 test days, Δt and the scalers are those of the last training group, five to six months old.
- GPU training with cuDNN's algorithm search is not bit-reproducible, so the tables are rebuilt from the stored score cells, not from new fits.
- R² measures forecasts only; no trading costs or profit and loss are evaluated.
- For context: the paper's LSTM, LSTM-MLP and CNN-LSTM on order flow reach about 1.2–1.3 % (mean over the ten horizons, read from Figure 2 of Kolm, Turiel and Westray 2023). The paper fitted one model per stock in rolling windows (1 week validation, 4 weeks training, 1 week test) on the same 115 tickers over 2019-01 to 2020-01. This study fits one pooled model for all stocks.
- This README states the author's own views. It is not investment advice. The market data are licensed from their providers; see [DATA_NOTICE.md](DATA_NOTICE.md).

## The code

Python 3.11. No market data is included; see [DATA_NOTICE.md](DATA_NOTICE.md).

### Layout

```
src/dof/lob/
  itch/         ITCH 5.0 framing and decoding for a set of tracked stocks (spec, framing, decode, filenames)
  book.py       the order book: A/F/E/C/X/D/U/P/Q/B/H semantics, 10-level snapshots, integrity flags
  tables/       one ITCH day -> per-stock event and state tables (Parquet) and a build log; load_states
  features/     cleaning and invalid intervals, the order-flow series (flow_series), labels, sessions and half
                days, scalers, the one-minute OFI and its first principal component (minute_ofi)
  stats/        sufficient statistics, ridge, out-of-sample R², block t-test
  validation/   reconciliation of the book against the public LOBSTER sample files
  testing/      synthetic ITCH builders, hand-checked scenarios, a seeded random day
  provenance.py code identity of an artifact (package version, source hashes, git state, library versions)
  cli.py        python -m dof.lob build-day ...
src/dof/deep/
  config.py     run configurations from TOML (frozen dataclasses), machine settings, result identities
  cache/        the feature cache of a split: probes, scaling groups, record files, evaluation arrays, summary
  data/         the training stream and its pass order, the evaluation days, the optional compact backend
  models/       the model registry: mlp.py, lstm.py (LSTM and LSTM-MLP), cnn_lstm.py, transformer.py, and
                init.py (the Keras initialisation)
  optim.py      Adam, and Adam in the Keras form (KerasAdam)
  train/        early stopping, checkpoints, one training step, the fit record, the fit
  evaluate/     predictions, validation losses, score cells, the test-day gate, test scoring
  linear/       the pooled ridge references
  runtime/      atomic writers, memory and disk guards, device settings, provenance
  cli.py        python -m dof.deep build | summary | fit | ridge | score | measure
src/dof/experiments/
  spec.py       ExperimentSpec: fits, seeds and rounds, parts, record layouts, protocol hashes, report sections
  store.py, cells.py, export.py   stored records into the result tables: record ids, protocol pins, totals
  gate.py       the round gate before the test scoring, and the audit of the order of work
  analyses/     pure functions of the result tables: score rows, seeds, adjacent days, decay, fits and checks
  report.py, render.py            the report of a spec built from its tables; the renderer only formats
  ledger.py, runner.py            run(spec): fit, gate, score once, the seed rule, collect, report
  specs/        the five stored experiments: alldays, seeds, adjacent, decay, earlier
  cli.py        python -m dof.experiments export | totals | report | run
results/        the result tables (Parquet) and MANIFEST.json; reports/ holds the five reports built from them
protocols/      edited copies of the four pre-registration documents, with the sha256 of each original
configs/        models.toml (the architecture catalogue), smoke.toml and alldays.toml (splits), the universe,
                protocol/ (the corporate-action file and a test fixture), a local-file template
examples/       synthetic_end_to_end.py: from ITCH bytes to R² without any market data
tests/lob/      unit tests of the library (synthetic data only)
tests/deep/     unit tests of the training package (a synthetic run built in the test session)
tests/experiments/  unit tests of the experiment framework, the published tables, this README and the protocols
tests/goldens/  golden tests: outputs pinned bit for bit to the reference implementation (see its README)
```

### Install and test

```
pip install -e ".[test]"                # the library
pip install -e ".[test,deep]"           # with torch: the training package
pip install -e ".[test,deep,compact]"   # with zstandard: the compact backend
pytest
```

`requirements-lock.txt` lists the versions the results and the goldens were made with. Without data, the tests that
need it are skipped: those marked `local` read paths from `DOF_*` environment variables, and those marked `golden`
run only with `DOF_GOLDENS=1` (see `tests/goldens/README.md`). Without torch, the test files of the training package
that need it are skipped as well.

### Build one day

```
python -m dof.lob build-day --itch 01302019.NASDAQ_ITCH50.gz --symbols-file tickers.csv --out book \
    --manifest MANIFEST.csv
```

`--manifest` checks the download against its manifest (size and sha256); `--no-manifest-check` skips that check. A
day is `complete` only when the whole file was read up to the end-of-messages event; a stock-day is used only when
its book never hit an integrity error.

### Example

```
python examples/synthetic_end_to_end.py --out /tmp/dof-example
```

It writes three seeded random days, builds them, fits a ridge regression of the ten horizons on 100 rows of
20-column order flow on two days, and prints the out-of-sample R² per stock and horizon on the third day. The data
are synthetic, so the numbers only show that the chain runs end to end.

### Train

A run is described by two files. The run configuration (`configs/smoke.toml`, `configs/alldays.toml`) holds
everything that defines a result: the split, the half days, the universe, the corporate-action file, the features,
the training and ridge settings, the model catalogue (`configs/models.toml`, included), the fits and the linear
references. The local file (a copy of `configs/local.example.toml`, outside the repository) holds the machine
settings: the order-book folder, the cache folders, the device, threads and loader processes, the memory and disk
guards. A machine setting never enters the identity of a result.

```
python -m dof.deep build --config configs/smoke.toml --local local.toml      # the feature cache of the split
python -m dof.deep fit   --config configs/smoke.toml --local local.toml --fit LSTM-MLP-OF-alldays --seed 0 \
    --out runs/smoke/LSTM-MLP-OF-alldays
python -m dof.deep ridge --config configs/smoke.toml --local local.toml --out runs/smoke
python -m dof.deep score --config configs/smoke.toml --local local.toml --results runs/smoke   # the test days, once
```

- A fit reads the validation loss every `check_steps` training steps and stops after `patience` checks without a
  new lowest loss, or after `max_checks` checks. The best check's weights are saved before any scoring, then the
  record. After every check the whole state is saved, and a stopped fit continues after its last check with the
  batches, and the random-number state, of an uninterrupted fit.
- The test days are read only through the test gate: every scored model must have a finished record, its saved
  weights must be the ones the record names, and the corporate-action file must be the pinned one. A test day is
  scored once. The corporate-action file that `configs/alldays.toml` pins is in `configs/protocol/`. The protocol
  document it pins, under its original name and sha256, is not in this repository (`protocols/` holds an edited
  copy with another hash), so the gate refuses the test days of that split.
- A new model family is one file in `src/dof/deep/models/` with `@register("<family>")`, and one `[models.<NAME>]`
  entry in the catalogue.
- The compact backend (`compact_store = true` in the local file) holds the training records block-compressed in
  memory instead of reading them from disk. It is Linux only: it reads `/proc/meminfo` and the cgroup files to check
  that the machine keeps enough free memory, and on another system loading the store refuses with an error. With
  `compact_store = false`, the default, the record files are read on every system.
- Run outputs go to `runs/`, which git ignores.

### Experiments and result tables

An experiment is a spec (`dof.experiments.ExperimentSpec`): its fits, seeds and rounds, its parts (the held-out
parts declared by day and time of day), where its stored records are, the sha256 of its protocol documents and of
the corporate-action file, and the tables of its report. The five experiments are specs in
`src/dof/experiments/specs/`: `alldays`, `seeds`, `adjacent`, `decay` and `earlier`; their reports are in
`results/reports/`.

```
python -m dof.experiments export --results-root <stored results> --root earlier=<earlier stage results> \
    --out results                                                                         # once
python -m dof.experiments totals --results results --expect tests/goldens/data/results_totals.json
python -m dof.experiments report --results results --all --out-dir results/reports
python -m dof.experiments run --spec alldays --config configs/alldays.toml --local local.toml --out runs/alldays
```

- `export` turns the stored records into the result tables. It refuses a record that names another protocol or
  corporate-action hash than its spec, and a held-out record that was scored before every fit of its round had
  ended.
- `results/cells.parquet`: one row per score cell (model, seed, part, day, stock, horizon: R², SSE, SST).
- `results/stock_days.parquet`: the price ratio of every scored held-out stock-day, and the stock-days left out
  (corporate actions, decay exclusions with their reason).
- `results/records.parquet`: one row per stored record, with its cell count, exact sums and source hash.
- `results/fits.parquet`, `checks.parquet`, `check_losses.parquet`: the fits, the validation loss by check, and by
  check and stock.
- `results/MANIFEST.json`: the sha256 and row count of every table, the totals, the audit of the order of work.
- The analyses (`dof.experiments.analyses`) are pure functions of these tables; the reports are rebuilt from the
  tables and pinned to the stored reports by the golden tests.
- `run` fits, gates, scores once and reports the main-run kind and the seeds kind. The adjacent and decay
  experiments are reported from their stored results; the package does not run their data steps.

### Conventions

- Times are nanoseconds since midnight, US Eastern (ITCH time). Prices are integers in 1e-4 USD.
- The trimmed session drops the first and the last ten minutes of regular trading; Δt (Kolm, Turiel and Westray
  2023, eq. 16) is estimated on training days only, each day counted with its own regular trading time.
- Order-flow sign convention: the library default `current_size` follows Cont, Cucuringu and Zhang (2023, §2.1) and
  Kolm, Turiel and Westray (2023, eqs. 2-3): a bid price drop contributes -q_n, the current size. `cks` follows
  Cont, Kukanov and Stoikov (2014) and uses -q_{n-1}, the previous size. The runs in `configs/` use `cks`.
- R² is reported against the mean of the evaluation set (`r2_os`) and against zero (`r2_zero`).

## References

- Cont, R., Cucuringu, M. and Zhang, C. (2023). Cross-impact of order flow imbalance in equity markets. *Quantitative Finance* 23(10), 1373-1393. https://doi.org/10.1080/14697688.2023.2236159
- Cont, R., Kukanov, A. and Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics* 12(1), 47-88. https://doi.org/10.1093/jjfinec/nbt003
- Kolm, P. N., Turiel, J. and Westray, N. (2023). Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book. *Mathematical Finance* 33(4), 1044-1081. https://doi.org/10.1111/mafi.12413

## Licence

MIT; see [LICENSE](LICENSE).
