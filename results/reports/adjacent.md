# Five adjacent training days: validation on a morning, test on the afternoon

All values are out-of-sample R² in percent, per stock, day part and horizon, against the mean label of the scored rows of that stock on that part of the day. A stock's value is the mean of its cells; a table value is the mean over stocks.

## Result

One half day: the result cannot be carried to other days, and every s.e. across stocks is too small because stocks move together.

- Test, afternoon of 2025-12-12 (13:00 on; one half day; a second look), LSTM-MLP-OF, 5 days: seed 3089492870 0.226, seed 771810125 0.228, seed 18531247 0.312; mean 0.255 %, s.d. across seeds 0.049, s.e. across stocks 0.206 (98 stocks; k = 1..10).
- The same weights on 2026-05-15, 2026-05-18 and 2026-06-12 (days scored before; primary analysis): mean -0.002 %, s.d. across seeds 0.051, s.e. across stocks 0.069 (96 stocks; k = 1..10).
- Paired difference D, afternoon minus the 2026 days, over 96 stocks scored on both parts (k = 1..10; 2026 days: primary analysis): D = 0.250 percentage points; U = 0.217 (s.e. across stocks 0.210, s.d. across seeds 0.091); D/U = 1.15; |D| is at most 2U.

## Method

Training days: 2025-11-28 (a half day) and 2025-12-08 to 2025-12-11. Validation: the samples of 2025-12-12 whose row and ten labels end before 13:00. Test: the samples of 2025-12-12 whose 100 rows of history are all at or after 13:00. A sample that is neither is used nowhere.

One network (LSTM-MLP-OF) with three seeds and one pooled ridge reference. The same weights are also scored on 2026-05-15, 2026-05-18 and 2026-06-12, with the Δt and scalers of the five training days.

Protocol document sha256 babfaf66c72cd5627f6b28cf271c12c8837078722216f9c91570bafd157607e2; experiment protocol sha256 1f1417e75640153875e0a9eebada72be1243a496932f590520092d34a5ca20a3; corporate-action file sha256 df7ef9adb2c52d4af1d112e5ae8a326d2408956ebe43e1d953e1b034f9f6ea4a.

## Results by model

### LSTM-MLP-OF, 5 days (2025-11-28 to 2025-12-11)

#### Validation: morning of 2025-12-12 (rows and labels before 13:00; the fits stopped on it)

| rows scored | horizon | stocks | seed 3089492870 | seed 771810125 | seed 18531247 | mean | s.d. across seeds | s.e. across stocks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| morning of 2025-12-12 | k = 1..10 | 98 | 0.433 | 0.434 | 0.486 | 0.451 | 0.030 | 0.104 |
| morning of 2025-12-12 | k = 10 | 98 | 0.607 | 0.615 | 0.694 | 0.639 | 0.048 | 0.116 |

#### Test: afternoon of 2025-12-12 (window, row and labels at or after 13:00)

| rows scored | horizon | stocks | seed 3089492870 | seed 771810125 | seed 18531247 | mean | s.d. across seeds | s.e. across stocks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| afternoon of 2025-12-12 | k = 1..10 | 98 | 0.226 | 0.228 | 0.312 | 0.255 | 0.049 | 0.206 |
| afternoon of 2025-12-12 | k = 10 | 98 | 0.455 | 0.380 | 0.497 | 0.444 | 0.059 | 0.243 |

#### Test: the same weights on 2026-05-15, 2026-05-18, 2026-06-12

| rows scored | horizon | stocks | seed 3089492870 | seed 771810125 | seed 18531247 | mean | s.d. across seeds | s.e. across stocks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12, primary analysis | k = 1..10 | 96 | -0.050 | 0.051 | -0.007 | -0.002 | 0.051 | 0.069 |
| 2026-05-15, 2026-05-18, 2026-06-12, primary analysis | k = 10 | 96 | 0.124 | 0.231 | 0.169 | 0.175 | 0.054 | 0.087 |
| 2026-05-15, primary analysis | k = 1..10 | 96 | -0.187 | -0.218 | -0.194 | -0.200 | 0.016 | 0.104 |
| 2026-05-18, primary analysis | k = 1..10 | 96 | 0.050 | 0.242 | 0.123 | 0.138 | 0.097 | 0.095 |
| 2026-06-12, primary analysis | k = 1..10 | 95 | -0.022 | 0.121 | 0.042 | 0.047 | 0.072 | 0.102 |
| 2026-05-15, 2026-05-18, 2026-06-12, secondary analysis (price ratio inside the range) | k = 1..10 | 86 | -0.059 | 0.048 | -0.019 | -0.010 | 0.054 | 0.075 |

#### The fits

| model | seed | status | checks | best check | stopped by | steps | passes | best validation loss | interruptions | sha256 of the saved weights |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
| LSTM-MLP-OF-recent | 3089492870 | finished | 9 | 4 | patience | 2,854,926 | 6.51 | 1.251693 | 0 | 41cc7062f9b935ae… |
| LSTM-MLP-OF-recent | 771810125 | finished | 10 | 5 | patience | 3,172,140 | 7.23 | 1.250928 | 0 | 281a3dd1289b6317… |
| LSTM-MLP-OF-recent | 18531247 | finished | 9 | 4 | patience | 2,854,926 | 6.51 | 1.250668 | 0 | e8b2a6ce01c73e22… |

#### Validation loss by check, seed 3089492870

| check | steps | passes | training loss since the last check | validation loss | new best | median weight step |
|---|---:|---:|---:|---:|---|---:|
| 1 | 317,214 | 0.72 | 0.994619 | 1.253335 | yes | 1.855e-06 |
| 2 | 634,428 | 1.45 | 0.992630 | 1.251897 | yes | 1.255e-06 |
| 3 | 951,642 | 2.17 | 0.991259 | 1.253467 |  | 1.505e-06 |
| 4 | 1,268,856 | 2.89 | 0.990084 | 1.251693 | yes | 2.541e-06 |
| 5 | 1,586,070 | 3.61 | 0.989032 | 1.252357 |  | 1.976e-06 |
| 6 | 1,903,284 | 4.34 | 0.987678 | 1.253447 |  | 1.349e-06 |
| 7 | 2,220,498 | 5.06 | 0.985640 | 1.255418 |  | 1.196e-06 |
| 8 | 2,537,712 | 5.78 | 0.985258 | 1.256150 |  | 1.271e-06 |
| 9 | 2,854,926 | 6.51 | 0.982862 | 1.255215 |  | 1.922e-06 |

#### Validation loss by check, seed 771810125

| check | steps | passes | training loss since the last check | validation loss | new best | median weight step |
|---|---:|---:|---:|---:|---|---:|
| 1 | 317,214 | 0.72 | 0.994436 | 1.252874 | yes | 1.378e-06 |
| 2 | 634,428 | 1.45 | 0.992825 | 1.251618 | yes | 2.190e-06 |
| 3 | 951,642 | 2.17 | 0.991027 | 1.251542 | yes | 1.416e-06 |
| 4 | 1,268,856 | 2.89 | 0.990352 | 1.254652 |  | 1.308e-06 |
| 5 | 1,586,070 | 3.61 | 0.988742 | 1.250928 | yes | 4.157e-06 |
| 6 | 1,903,284 | 4.34 | 0.987949 | 1.252289 |  | 9.453e-07 |
| 7 | 2,220,498 | 5.06 | 0.986145 | 1.252831 |  | 1.233e-06 |
| 8 | 2,537,712 | 5.78 | 0.985039 | 1.252732 |  | 1.296e-06 |
| 9 | 2,854,926 | 6.51 | 0.983957 | 1.254584 |  | 1.121e-06 |
| 10 | 3,172,140 | 7.23 | 0.982660 | 1.258071 |  | 1.168e-06 |

#### Validation loss by check, seed 18531247

| check | steps | passes | training loss since the last check | validation loss | new best | median weight step |
|---|---:|---:|---:|---:|---|---:|
| 1 | 317,214 | 0.72 | 0.994893 | 1.252988 | yes | 1.732e-06 |
| 2 | 634,428 | 1.45 | 0.993054 | 1.252264 | yes | 1.221e-06 |
| 3 | 951,642 | 2.17 | 0.990409 | 1.251150 | yes | 1.363e-06 |
| 4 | 1,268,856 | 2.89 | 0.990165 | 1.250668 | yes | 1.790e-06 |
| 5 | 1,586,070 | 3.61 | 0.989199 | 1.252221 |  | 9.835e-07 |
| 6 | 1,903,284 | 4.34 | 0.987380 | 1.251524 |  | 1.259e-06 |
| 7 | 2,220,498 | 5.06 | 0.987381 | 1.253095 |  | 2.015e-06 |
| 8 | 2,537,712 | 5.78 | 0.984772 | 1.254729 |  | 1.550e-06 |
| 9 | 2,854,926 | 6.51 | 0.983493 | 1.257226 |  | 1.363e-06 |

### Ridge-OF, 5 days (2025-11-28 to 2025-12-11)

#### Validation: morning of 2025-12-12 (λ was chosen on it)

| rows scored | horizon | stocks | seed 3089492870 | seed 771810125 | seed 18531247 | mean | s.d. across seeds | s.e. across stocks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| morning of 2025-12-12 | k = 1..10 | 98 |  |  |  | -0.061 |  | 0.063 |
| morning of 2025-12-12 | k = 10 | 98 |  |  |  | 0.004 |  | 0.075 |

#### Test: afternoon of 2025-12-12 (window, row and labels at or after 13:00)

| rows scored | horizon | stocks | seed 3089492870 | seed 771810125 | seed 18531247 | mean | s.d. across seeds | s.e. across stocks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| afternoon of 2025-12-12 | k = 1..10 | 98 |  |  |  | -0.205 |  | 0.162 |
| afternoon of 2025-12-12 | k = 10 | 98 |  |  |  | -0.116 |  | 0.221 |

#### Test: the same weights on 2026-05-15, 2026-05-18, 2026-06-12

| rows scored | horizon | stocks | seed 3089492870 | seed 771810125 | seed 18531247 | mean | s.d. across seeds | s.e. across stocks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12, primary analysis | k = 1..10 | 96 |  |  |  | -0.191 |  | 0.057 |
| 2026-05-15, 2026-05-18, 2026-06-12, primary analysis | k = 10 | 96 |  |  |  | -0.111 |  | 0.074 |
| 2026-05-15, primary analysis | k = 1..10 | 96 |  |  |  | -0.227 |  | 0.062 |
| 2026-05-18, primary analysis | k = 1..10 | 96 |  |  |  | -0.217 |  | 0.079 |
| 2026-06-12, primary analysis | k = 1..10 | 95 |  |  |  | -0.133 |  | 0.082 |
| 2026-05-15, 2026-05-18, 2026-06-12, secondary analysis (price ratio inside the range) | k = 1..10 | 86 |  |  |  | -0.200 |  | 0.063 |

#### The ridge reference

| model | λ | training samples | stocks |
|---|---:|---:|---:|
| Ridge-OF-recent | 0.01 | 112,267,810 | 98 |

## The afternoon against the 2026 days

A seed's difference is the mean, over the stocks scored on both parts, of the stock's value on the afternoon minus its value on the 2026 days. D is the mean of the seed differences. U is the root of the sum of two squares: the s.e. across stocks of the per-stock difference averaged over the seeds, and the s.d. across seeds of the seed differences divided by the root of the number of seeds. U holds no day effect and no correlation across stocks, so it is too small; D is a description, not a test.

#### Paired difference

| quantity | value | relation |
|---|---:|---|
| stocks scored on both parts (the pair) | 96 |  |
| difference of seed 3089492870 | 0.286 |  |
| difference of seed 771810125 | 0.147 |  |
| difference of seed 18531247 | 0.317 |  |
| D, the mean over the seeds | 0.250 |  |
| s.e. across stocks of the per-stock difference averaged over the seeds | 0.210 |  |
| s.d. across seeds of the seed differences | 0.091 |  |
| U | 0.217 |  |
| D / U | 1.15 |  |
| afternoon mean (for the tables only) | 0.255 |  |
| afternoon, uncertainty of a part (for the tables only) | 0.208 |  |
| 2026 mean, primary analysis (for the tables only) | -0.002 |  |
| 2026, uncertainty of a part (for the tables only) | 0.075 |  |
| scored on the afternoon only | 2 (BKNG, HOLX) |  |
| scored on the 2026 days only | 0 |  |
| D against 2U |  | \|D\| is at most 2U |

## Selection-bias reference: the main run's seed-0 fits on the same afternoon

The main run's seed-0 fits, with their own Δt and scalers, on the same afternoon. This afternoon was inside their validation days. No number of this table enters D.

#### Main-run fits on the afternoon of 2025-12-12

| fit | trained on | horizon | stocks | R² % | s.e. |
|---|---|---|---:|---:|---:|
| LSTM-MLP-OF-recent | 4 days (2025-11-28 to 2025-12-10) | k = 1..10 | 98 | 0.262 | 0.170 |
| LSTM-MLP-OF-recent | 4 days (2025-11-28 to 2025-12-10) | k = 10 | 98 | 0.438 | 0.215 |
| LSTM-MLP-OF-alldays | 14 days (2019-01-30 to 2025-12-10) | k = 1..10 | 98 | 0.445 | 0.197 |
| LSTM-MLP-OF-alldays | 14 days (2019-01-30 to 2025-12-10) | k = 10 | 98 | 0.781 | 0.238 |
| CNN-LSTM-OF-alldays | 14 days (2019-01-30 to 2025-12-10) | k = 1..10 | 98 | 0.449 | 0.203 |
| CNN-LSTM-OF-alldays | 14 days (2019-01-30 to 2025-12-10) | k = 10 | 98 | 0.764 | 0.245 |

## Status of the days and validity

2025-12-12 was a validation day of the main run and of the seed extension, so its afternoon is a second look for this feature family. The three 2026 days had been scored before for other models. No day of this experiment is a first look.

The comparison cannot separate the distance in time from the difference between one December afternoon and three days of May and June, nor from the ageing of Δt and the scalers on the 2026 days.

The afternoon is scored against the mean label of its own rows and the 2026 days against the mean of the whole day; this pushes D down.

## Reproducibility

Every number above is computed from the published tables in results/ (cells.parquet, stock_days.parquet, fits.parquet, checks.parquet) by the functions of dof.experiments.analyses.

#### Stored records behind the tables

| record | cells | weights sha256 |
|---|---:|---|
| adjacent/far/LSTM-MLP-OF-recent/seed=18531247/test | 2870 | e8b2a6ce01c73e22 |
| adjacent/far/LSTM-MLP-OF-recent/seed=3089492870/test | 2870 | 41cc7062f9b935ae |
| adjacent/far/LSTM-MLP-OF-recent/seed=771810125/test | 2870 | 281a3dd1289b6317 |
| adjacent/far/Ridge-OF-recent/seed=0/test | 2870 | 7d900bbc2c3f2aa5 |
| adjacent/fit_valid/LSTM-MLP-OF-recent/seed=18531247/valid | 980 | e8b2a6ce01c73e22 |
| adjacent/fit_valid/LSTM-MLP-OF-recent/seed=3089492870/valid | 980 | 41cc7062f9b935ae |
| adjacent/fit_valid/LSTM-MLP-OF-recent/seed=771810125/valid | 980 | 281a3dd1289b6317 |
| adjacent/fit_valid/Ridge-OF-recent/seed=0/valid | 980 | 7d900bbc2c3f2aa5 |
| adjacent/reference/CNN-LSTM-OF-alldays/seed=0/test | 980 | 4ec421d309e5374f |
| adjacent/reference/LSTM-MLP-OF-alldays/seed=0/test | 980 | 0950686a0eef1f14 |
| adjacent/reference/LSTM-MLP-OF-recent/seed=0/test | 980 | d6bb40ed4652fd9e |
| adjacent/test/LSTM-MLP-OF-recent/seed=18531247/test | 980 | e8b2a6ce01c73e22 |
| adjacent/test/LSTM-MLP-OF-recent/seed=3089492870/test | 980 | 41cc7062f9b935ae |
| adjacent/test/LSTM-MLP-OF-recent/seed=771810125/test | 980 | 281a3dd1289b6317 |
| adjacent/test/Ridge-OF-recent/seed=0/test | 980 | 7d900bbc2c3f2aa5 |
