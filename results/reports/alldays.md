# One pooled model on all held trading days

All values are out-of-sample R² in percent, per stock, day and horizon, against the mean label of that stock-day. A stock's value is the mean of its cells; a table value is the mean over stocks; s.e. is the standard error across stocks. Stocks move together, so the standard errors are too small.

## Result

Test days 2026-05-15, 2026-05-18 and 2026-06-12, primary analysis: every test stock-day of the cache that no corporate action removes. Each value is the mean of the ten horizons k = 1..10; k = 10 alone is given beside it. Validation-day numbers chose the stopping check and λ; they are not results and are given in the model sections.

- `CNN-LSTM-OF-alldays` (14 days, 2019-01-30 to 2025-12-10), test days 2026-05-15, 2026-05-18 and 2026-06-12, primary analysis: R² 0.268 % (s.e. 0.059; 96 stocks; k = 1..10); k = 10: 0.428 % (s.e. 0.080).
- `LSTM-MLP-OF-alldays` (14 days, 2019-01-30 to 2025-12-10), test days 2026-05-15, 2026-05-18 and 2026-06-12, primary analysis: R² 0.196 % (s.e. 0.059; 96 stocks; k = 1..10); k = 10: 0.359 % (s.e. 0.087).
- `LSTM-MLP-OF-recent` (4 days, 2025-11-28 to 2025-12-10), test days 2026-05-15, 2026-05-18 and 2026-06-12, primary analysis: R² -0.000 % (s.e. 0.060; 96 stocks; k = 1..10); k = 10: 0.153 % (s.e. 0.075).
- Does the older data help, test days 2026-05-15, 2026-05-18 and 2026-06-12, primary analysis: `LSTM-MLP-OF-alldays` minus `LSTM-MLP-OF-recent` 0.196 percentage points (s.e. 0.031; 96 stocks in both; k = 1..10).

### References

- `Ridge-OF-alldays` (14 days, 2019-01-30 to 2025-12-10), test days 2026-05-15, 2026-05-18 and 2026-06-12, primary analysis: R² -0.194 % (s.e. 0.047; 96 stocks; k = 1..10); k = 10: -0.159 % (s.e. 0.064).
- `Ridge-OF-recent` (4 days, 2025-11-28 to 2025-12-10), test days 2026-05-15, 2026-05-18 and 2026-06-12, primary analysis: R² -0.190 % (s.e. 0.057; 96 stocks; k = 1..10); k = 10: -0.106 % (s.e. 0.073).
- Does the older data help, test days 2026-05-15, 2026-05-18 and 2026-06-12, primary analysis: `Ridge-OF-alldays` minus `Ridge-OF-recent` -0.005 percentage points (s.e. 0.023; 96 stocks in both; k = 1..10).

## Method

Training days: the 14 days from 2019-01-30 to 2025-12-10 ('all') or the 4 days from 2025-11-28 to 2025-12-10 ('recent'). Validation days: 2025-12-11 and 2025-12-12. Test days: 2026-05-15, 2026-05-18 and 2026-06-12, scored once, after every model had ended.

Three networks and two pooled ridge references, one seed (0). Nothing was tuned in this stage.

Protocol document sha256 babfaf66c72cd5627f6b28cf271c12c8837078722216f9c91570bafd157607e2; corporate-action file sha256 df7ef9adb2c52d4af1d112e5ae8a326d2408956ebe43e1d953e1b034f9f6ea4a. The secondary analysis keeps the stock-days whose median mid price is inside [0.6, 1.6] times that of the last training day; nothing is scored twice.

## Results by model

### CNN-LSTM-OF-alldays

#### Test days, primary analysis

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 96 | 0.268 | 0.059 |
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 96 | 0.428 | 0.080 |
| 2026-05-15 | k = 1..10 | 96 | 0.058 | 0.095 |
| 2026-05-18 | k = 1..10 | 96 | 0.372 | 0.089 |
| 2026-06-12 | k = 1..10 | 95 | 0.375 | 0.067 |

#### Test days, secondary analysis (price ratio inside [0.6, 1.6])

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 87 | 0.288 | 0.066 |
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 87 | 0.449 | 0.091 |
| 2026-05-15 | k = 1..10 | 84 | 0.027 | 0.108 |
| 2026-05-18 | k = 1..10 | 87 | 0.393 | 0.097 |
| 2026-06-12 | k = 1..10 | 82 | 0.401 | 0.076 |

#### Validation days (they chose the stopping check: not out-of-sample for that choice)

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2025-12-11, 2025-12-12 | k = 1..10 | 98 | 0.673 | 0.096 |
| 2025-12-11, 2025-12-12 | k = 10 | 98 | 0.860 | 0.094 |
| 2025-12-11 | k = 1..10 | 98 | 0.528 | 0.109 |
| 2025-12-12 | k = 1..10 | 98 | 0.819 | 0.102 |

#### The fit

| model | seed | status | checks | best check | stopped by | steps | passes | best validation loss | interruptions | sha256 of the saved weights |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
| CNN-LSTM-OF-alldays | 0 | finished | 12 | 7 | patience | 3,806,568 | 3.03 | 1.082123 | 0 | 4ec421d309e5374f… |

#### Validation loss by check

| check | steps | passes | training loss since the last check | validation loss | new best | median weight step |
|---|---:|---:|---:|---:|---|---:|
| 1 | 317,214 | 0.25 | 0.987780 | 1.084315 | yes | 3.934e-06 |
| 2 | 634,428 | 0.51 | 0.984216 | 1.084102 | yes | 3.383e-06 |
| 3 | 951,642 | 0.76 | 0.983349 | 1.083018 | yes | 3.882e-06 |
| 4 | 1,268,856 | 1.01 | 0.981980 | 1.082282 | yes | 3.016e-06 |
| 5 | 1,586,070 | 1.26 | 0.980791 | 1.082796 |  | 4.590e-06 |
| 6 | 1,903,284 | 1.52 | 0.979676 | 1.082654 |  | 3.092e-06 |
| 7 | 2,220,498 | 1.77 | 0.978697 | 1.082123 | yes | 3.179e-06 |
| 8 | 2,537,712 | 2.02 | 0.976927 | 1.082822 |  | 3.778e-06 |
| 9 | 2,854,926 | 2.27 | 0.975957 | 1.082805 |  | 2.585e-06 |
| 10 | 3,172,140 | 2.53 | 0.974225 | 1.083897 |  | 2.488e-06 |
| 11 | 3,489,354 | 2.78 | 0.972227 | 1.085946 |  | 3.695e-06 |
| 12 | 3,806,568 | 3.03 | 0.970783 | 1.091934 |  | 3.636e-06 |

### LSTM-MLP-OF-alldays

#### Test days, primary analysis

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 96 | 0.196 | 0.059 |
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 96 | 0.359 | 0.087 |
| 2026-05-15 | k = 1..10 | 96 | 0.020 | 0.094 |
| 2026-05-18 | k = 1..10 | 96 | 0.241 | 0.093 |
| 2026-06-12 | k = 1..10 | 95 | 0.326 | 0.082 |

#### Test days, secondary analysis (price ratio inside [0.6, 1.6])

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 87 | 0.206 | 0.065 |
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 87 | 0.369 | 0.096 |
| 2026-05-15 | k = 1..10 | 84 | -0.006 | 0.106 |
| 2026-05-18 | k = 1..10 | 87 | 0.254 | 0.102 |
| 2026-06-12 | k = 1..10 | 82 | 0.328 | 0.094 |

#### Validation days (they chose the stopping check: not out-of-sample for that choice)

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2025-12-11, 2025-12-12 | k = 1..10 | 98 | 0.533 | 0.090 |
| 2025-12-11, 2025-12-12 | k = 10 | 98 | 0.736 | 0.100 |
| 2025-12-11 | k = 1..10 | 98 | 0.391 | 0.096 |
| 2025-12-12 | k = 1..10 | 98 | 0.676 | 0.104 |

#### The fit

| model | seed | status | checks | best check | stopped by | steps | passes | best validation loss | interruptions | sha256 of the saved weights |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
| LSTM-MLP-OF-alldays | 0 | finished | 13 | 8 | patience | 4,123,782 | 3.28 | 1.083579 | 1 | 0950686a0eef1f14… |

#### Validation loss by check

| check | steps | passes | training loss since the last check | validation loss | new best | median weight step |
|---|---:|---:|---:|---:|---|---:|
| 1 | 317,214 | 0.25 | 0.989541 | 1.086119 | yes | 1.468e-06 |
| 2 | 634,428 | 0.51 | 0.986621 | 1.085232 | yes | 1.326e-06 |
| 3 | 951,642 | 0.76 | 0.986144 | 1.084633 | yes | 1.337e-06 |
| 4 | 1,268,856 | 1.01 | 0.985201 | 1.084245 | yes | 1.028e-06 |
| 5 | 1,586,070 | 1.26 | 0.984475 | 1.083856 | yes | 1.863e-06 |
| 6 | 1,903,284 | 1.52 | 0.983891 | 1.083866 |  | 1.237e-06 |
| 7 | 2,220,498 | 1.77 | 0.983534 | 1.083804 | yes | 1.758e-06 |
| 8 | 2,537,712 | 2.02 | 0.982601 | 1.083579 | yes | 1.268e-06 |
| 9 | 2,854,926 | 2.27 | 0.982610 | 1.083826 |  | 1.337e-06 |
| 10 | 3,172,140 | 2.53 | 0.982019 | 1.083691 |  | 1.170e-06 |
| 11 | 3,489,354 | 2.78 | 0.981364 | 1.083778 |  | 1.170e-06 |
| 12 | 3,806,568 | 3.03 | 0.981398 | 1.083954 |  | 1.356e-06 |
| 13 | 4,123,782 | 3.28 | 0.980937 | 1.084107 |  | 1.531e-06 |

### LSTM-MLP-OF-recent

#### Test days, primary analysis

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 96 | -0.000 | 0.060 |
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 96 | 0.153 | 0.075 |
| 2026-05-15 | k = 1..10 | 96 | -0.084 | 0.082 |
| 2026-05-18 | k = 1..10 | 96 | 0.052 | 0.091 |
| 2026-06-12 | k = 1..10 | 95 | 0.025 | 0.084 |

#### Test days, secondary analysis (price ratio inside [0.6, 1.6])

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 87 | 0.003 | 0.066 |
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 87 | 0.157 | 0.083 |
| 2026-05-15 | k = 1..10 | 84 | -0.107 | 0.092 |
| 2026-05-18 | k = 1..10 | 87 | 0.061 | 0.100 |
| 2026-06-12 | k = 1..10 | 82 | -0.001 | 0.095 |

#### Validation days (they chose the stopping check: not out-of-sample for that choice)

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2025-12-11, 2025-12-12 | k = 1..10 | 98 | 0.369 | 0.080 |
| 2025-12-11, 2025-12-12 | k = 10 | 98 | 0.522 | 0.079 |
| 2025-12-11 | k = 1..10 | 98 | 0.298 | 0.088 |
| 2025-12-12 | k = 1..10 | 98 | 0.441 | 0.091 |

#### The fit

| model | seed | status | checks | best check | stopped by | steps | passes | best validation loss | interruptions | sha256 of the saved weights |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
| LSTM-MLP-OF-recent | 0 | finished | 7 | 2 | patience | 2,220,498 | 6.65 | 1.085639 | 0 | d6bb40ed4652fd9e… |

#### Validation loss by check

| check | steps | passes | training loss since the last check | validation loss | new best | median weight step |
|---|---:|---:|---:|---:|---|---:|
| 1 | 317,214 | 0.95 | 0.994452 | 1.086156 | yes | 1.289e-06 |
| 2 | 634,428 | 1.90 | 0.992540 | 1.085639 | yes | 1.609e-06 |
| 3 | 951,642 | 2.85 | 0.990261 | 1.087051 |  | 1.308e-06 |
| 4 | 1,268,856 | 3.80 | 0.989625 | 1.085711 |  | 1.259e-06 |
| 5 | 1,586,070 | 4.75 | 0.988177 | 1.086536 |  | 1.110e-06 |
| 6 | 1,903,284 | 5.70 | 0.986933 | 1.088060 |  | 9.770e-07 |
| 7 | 2,220,498 | 6.65 | 0.984103 | 1.089131 |  | 1.393e-06 |

### Ridge-OF-alldays

#### Test days, primary analysis

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 96 | -0.194 | 0.047 |
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 96 | -0.159 | 0.064 |
| 2026-05-15 | k = 1..10 | 96 | -0.201 | 0.057 |
| 2026-05-18 | k = 1..10 | 96 | -0.196 | 0.061 |
| 2026-06-12 | k = 1..10 | 95 | -0.187 | 0.068 |

#### Test days, secondary analysis (price ratio inside [0.6, 1.6])

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 87 | -0.196 | 0.051 |
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 87 | -0.164 | 0.070 |
| 2026-05-15 | k = 1..10 | 84 | -0.221 | 0.064 |
| 2026-05-18 | k = 1..10 | 87 | -0.197 | 0.067 |
| 2026-06-12 | k = 1..10 | 82 | -0.194 | 0.078 |

#### Validation days (λ was chosen on them: not out-of-sample for that choice)

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2025-12-11, 2025-12-12 | k = 1..10 | 98 | -0.057 | 0.037 |
| 2025-12-11, 2025-12-12 | k = 10 | 98 | 0.004 | 0.040 |
| 2025-12-11 | k = 1..10 | 98 | -0.070 | 0.043 |
| 2025-12-12 | k = 1..10 | 98 | -0.044 | 0.046 |

#### The ridge reference

| model | λ | training samples | stocks |
|---|---:|---:|---:|
| Ridge-OF-alldays | 1 | 321,309,274 | 115 |

### Ridge-OF-recent

#### Test days, primary analysis

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 96 | -0.190 | 0.057 |
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 96 | -0.106 | 0.073 |
| 2026-05-15 | k = 1..10 | 96 | -0.235 | 0.067 |
| 2026-05-18 | k = 1..10 | 96 | -0.204 | 0.076 |
| 2026-06-12 | k = 1..10 | 95 | -0.132 | 0.078 |

#### Test days, secondary analysis (price ratio inside [0.6, 1.6])

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 87 | -0.192 | 0.062 |
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 87 | -0.110 | 0.080 |
| 2026-05-15 | k = 1..10 | 84 | -0.258 | 0.075 |
| 2026-05-18 | k = 1..10 | 87 | -0.200 | 0.083 |
| 2026-06-12 | k = 1..10 | 82 | -0.154 | 0.088 |

#### Validation days (λ was chosen on them: not out-of-sample for that choice)

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2025-12-11, 2025-12-12 | k = 1..10 | 98 | 0.003 | 0.055 |
| 2025-12-11, 2025-12-12 | k = 10 | 98 | 0.089 | 0.056 |
| 2025-12-11 | k = 1..10 | 98 | -0.014 | 0.062 |
| 2025-12-12 | k = 1..10 | 98 | 0.021 | 0.062 |

#### The ridge reference

| model | λ | training samples | stocks |
|---|---:|---:|---:|
| Ridge-OF-recent | 0.1 | 85,449,237 | 98 |

## Does the older data help

#### Test days, primary analysis

| comparison | days | horizon | stocks in both | difference in R², percentage points | s.e. | stocks where the first is higher |
|---|---|---|---:|---:|---:|---:|
| LSTM-MLP-OF-alldays minus LSTM-MLP-OF-recent | 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 96 | 0.196 | 0.031 | 79 |
| LSTM-MLP-OF-alldays minus LSTM-MLP-OF-recent | 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 96 | 0.207 | 0.034 | 80 |
| Ridge-OF-alldays minus Ridge-OF-recent | 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 96 | -0.005 | 0.023 | 39 |
| Ridge-OF-alldays minus Ridge-OF-recent | 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 96 | -0.053 | 0.018 | 32 |

#### Test days, secondary analysis (price ratio inside [0.6, 1.6])

| comparison | days | horizon | stocks in both | difference in R², percentage points | s.e. | stocks where the first is higher |
|---|---|---|---:|---:|---:|---:|
| LSTM-MLP-OF-alldays minus LSTM-MLP-OF-recent | 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 87 | 0.203 | 0.033 | 72 |
| LSTM-MLP-OF-alldays minus LSTM-MLP-OF-recent | 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 87 | 0.212 | 0.038 | 73 |
| Ridge-OF-alldays minus Ridge-OF-recent | 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 87 | -0.004 | 0.024 | 36 |
| Ridge-OF-alldays minus Ridge-OF-recent | 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 87 | -0.054 | 0.019 | 30 |

#### Validation days (not out-of-sample for the stopping check and λ)

| comparison | days | horizon | stocks in both | difference in R², percentage points | s.e. | stocks where the first is higher |
|---|---|---|---:|---:|---:|---:|
| LSTM-MLP-OF-alldays minus LSTM-MLP-OF-recent | 2025-12-11, 2025-12-12 | k = 1..10 | 98 | 0.164 | 0.040 | 67 |
| LSTM-MLP-OF-alldays minus LSTM-MLP-OF-recent | 2025-12-11, 2025-12-12 | k = 10 | 98 | 0.214 | 0.042 | 76 |
| Ridge-OF-alldays minus Ridge-OF-recent | 2025-12-11, 2025-12-12 | k = 1..10 | 98 | -0.060 | 0.034 | 44 |
| Ridge-OF-alldays minus Ridge-OF-recent | 2025-12-11, 2025-12-12 | k = 10 | 98 | -0.085 | 0.024 | 32 |

## Data and split

The corporate-action file leaves out the test stock-days below; every other test stock-day of the cache is scored. A cell without an R² has labels of zero variance and is left out of the means.

#### Test stock-days left out by a corporate action

| symbol | date | event | effective date |
|---|---|---|---|
| BKNG | 2026-05-15 | forward_split | 2026-04-06 |
| BKNG | 2026-05-18 | forward_split | 2026-04-06 |
| BKNG | 2026-06-12 | forward_split | 2026-04-06 |
| KLAC | 2026-06-12 | forward_split | 2026-06-12 |

#### Cells without an R² (labels of zero variance), left out of the means: validation days

| model | cells |
|---|---:|
| LSTM-MLP-OF-recent | 0 |
| LSTM-MLP-OF-alldays | 0 |
| CNN-LSTM-OF-alldays | 0 |
| Ridge-OF-alldays | 0 |
| Ridge-OF-recent | 0 |

#### Cells without an R² (labels of zero variance), left out of the means: test days, primary analysis

| model | cells |
|---|---:|
| LSTM-MLP-OF-recent | 0 |
| LSTM-MLP-OF-alldays | 0 |
| CNN-LSTM-OF-alldays | 0 |
| Ridge-OF-alldays | 0 |
| Ridge-OF-recent | 0 |

#### Cells without an R² (labels of zero variance), left out of the means: test days, secondary analysis

| model | cells |
|---|---:|
| LSTM-MLP-OF-recent | 0 |
| LSTM-MLP-OF-alldays | 0 |
| CNN-LSTM-OF-alldays | 0 |
| Ridge-OF-alldays | 0 |
| Ridge-OF-recent | 0 |

## Validity

2026-06-12 is not a first look for this feature family: an earlier experiment scored per-stock ridge models on it. 2026-05-15 and 2026-05-18 are first looks.

The learning rate and epsilon of CNN-LSTM-OF-alldays come from an earlier stage whose choice had seen that stage's test days.

The out-of-sample R² compares the forecast with the stock-day's own mean label; a trader does not know that mean during the day. On the test days Δt and the scalers are those of the last training group, five to six months old.

One seed. The cuDNN algorithm search was on, so a repeated fit is not bit-identical; these tables are rebuilt from the stored score cells, not from new fits.

## Reproducibility

Every number above is computed from the published tables in results/ (cells.parquet, stock_days.parquet, fits.parquet, checks.parquet) by the functions of dof.experiments.analyses.

#### Stored records behind the tables

| record | cells | weights sha256 |
|---|---:|---|
| alldays/fit_valid/CNN-LSTM-OF-alldays/seed=0/valid | 1960 | 4ec421d309e5374f |
| alldays/fit_valid/LSTM-MLP-OF-alldays/seed=0/valid | 1960 | 0950686a0eef1f14 |
| alldays/fit_valid/LSTM-MLP-OF-recent/seed=0/valid | 1960 | d6bb40ed4652fd9e |
| alldays/fit_valid/Ridge-OF-alldays/seed=0/valid | 1960 | 64a403767b3706cd |
| alldays/fit_valid/Ridge-OF-recent/seed=0/valid | 1960 | 4c2881b224e37e03 |
| alldays/test/CNN-LSTM-OF-alldays/seed=0/test | 2870 | 4ec421d309e5374f |
| alldays/test/LSTM-MLP-OF-alldays/seed=0/test | 2870 | 0950686a0eef1f14 |
| alldays/test/LSTM-MLP-OF-recent/seed=0/test | 2870 | d6bb40ed4652fd9e |
| alldays/test/Ridge-OF-alldays/seed=0/test | 2870 | 64a403767b3706cd |
| alldays/test/Ridge-OF-recent/seed=0/test | 2870 | 4c2881b224e37e03 |
