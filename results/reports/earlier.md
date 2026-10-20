# The earlier stage: one pooled model on 115 stocks, trained on three days of 2019

All values are out-of-sample R² in percent, per stock, day and horizon, against the mean label of that stock-day. A stock's value is the mean of its cells; a table value is the mean over stocks; s.e. is the standard error across stocks. Stocks move together, so the standard errors are too small.

## Result

Test days 2019-10-18, 2019-10-30 and 2019-12-30, 115 stocks, seed 0. These test days were read many times while this stage tried its models and settings, so the numbers describe; they are not a first-look test (see Validity).

- `LSTM-OF`, test days 2019-10-18, 2019-10-30 and 2019-12-30: R² 1.044 % (s.e. 0.106; 115 stocks; k = 1..10); k = 10: 1.280 % (s.e. 0.110).
- `LSTM-MLP-OF`, test days 2019-10-18, 2019-10-30 and 2019-12-30: R² 1.114 % (s.e. 0.105; 115 stocks; k = 1..10); k = 10: 1.327 % (s.e. 0.108).
- `CNN-LSTM-OF-lr3e-5-eps1e-7`, test days 2019-10-18, 2019-10-30 and 2019-12-30: R² 1.136 % (s.e. 0.108; 115 stocks; k = 1..10); k = 10: 1.372 % (s.e. 0.103).
- `Ridge-OF`, test days 2019-10-18, 2019-10-30 and 2019-12-30: R² 0.241 % (s.e. 0.049; 115 stocks; k = 1..10); k = 10: 0.330 % (s.e. 0.047).

## Method

Training days: 2019-01-30, 2019-03-27 and 2019-07-30. Validation day: 2019-08-30. Test days: 2019-10-18, 2019-10-30 and 2019-12-30. Each network is one model trained on the training rows of all 115 stocks; the pooled Ridge-OF is one closed-form ridge fit on the same rows, with λ chosen on the validation day.

LSTM-OF and LSTM-MLP-OF were trained with Adam (learning rate 1e-5, epsilon 1e-7), CNN-LSTM-OF-lr3e-5-eps1e-7 with Adam in the Keras form (learning rate 3e-5, epsilon 1e-7). The validation loss was read after each pass over the training rows; a fit stopped after 5 passes without a new lowest loss and kept the weights of its best pass.

No protocol document was written before this stage, so its records name no protocol hash and the spec pins none. Each stored file holds the validation day and the three test days of one model; the export splits it into a validation record and a test record. The pooled ridge record keeps only R², so its cells have no sums of squares.

## Results by model

### LSTM-OF

#### Test days

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2019-10-18, 2019-10-30, 2019-12-30 | k = 1..10 | 115 | 1.044 | 0.106 |
| 2019-10-18, 2019-10-30, 2019-12-30 | k = 10 | 115 | 1.280 | 0.110 |
| 2019-10-18 | k = 1..10 | 115 | 1.407 | 0.147 |
| 2019-10-30 | k = 1..10 | 115 | 0.888 | 0.092 |
| 2019-12-30 | k = 1..10 | 115 | 0.836 | 0.148 |

#### Validation day (it chose the best pass: not out-of-sample for that choice)

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2019-08-30 | k = 1..10 | 115 | 1.229 | 0.141 |
| 2019-08-30 | k = 10 | 115 | 1.491 | 0.161 |

### LSTM-MLP-OF

#### Test days

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2019-10-18, 2019-10-30, 2019-12-30 | k = 1..10 | 115 | 1.114 | 0.105 |
| 2019-10-18, 2019-10-30, 2019-12-30 | k = 10 | 115 | 1.327 | 0.108 |
| 2019-10-18 | k = 1..10 | 115 | 1.449 | 0.143 |
| 2019-10-30 | k = 1..10 | 115 | 0.959 | 0.091 |
| 2019-12-30 | k = 1..10 | 115 | 0.932 | 0.152 |

#### Validation day (it chose the best pass: not out-of-sample for that choice)

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2019-08-30 | k = 1..10 | 115 | 1.245 | 0.146 |
| 2019-08-30 | k = 10 | 115 | 1.512 | 0.167 |

### CNN-LSTM-OF-lr3e-5-eps1e-7

#### Test days

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2019-10-18, 2019-10-30, 2019-12-30 | k = 1..10 | 115 | 1.136 | 0.108 |
| 2019-10-18, 2019-10-30, 2019-12-30 | k = 10 | 115 | 1.372 | 0.103 |
| 2019-10-18 | k = 1..10 | 115 | 1.427 | 0.141 |
| 2019-10-30 | k = 1..10 | 115 | 0.928 | 0.087 |
| 2019-12-30 | k = 1..10 | 115 | 1.051 | 0.150 |

#### Validation day (it chose the best pass: not out-of-sample for that choice)

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2019-08-30 | k = 1..10 | 115 | 1.242 | 0.134 |
| 2019-08-30 | k = 10 | 115 | 1.504 | 0.149 |

### Ridge-OF

#### Test days

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2019-10-18, 2019-10-30, 2019-12-30 | k = 1..10 | 115 | 0.241 | 0.049 |
| 2019-10-18, 2019-10-30, 2019-12-30 | k = 10 | 115 | 0.330 | 0.047 |
| 2019-10-18 | k = 1..10 | 115 | 0.371 | 0.069 |
| 2019-10-30 | k = 1..10 | 115 | 0.102 | 0.046 |
| 2019-12-30 | k = 1..10 | 115 | 0.250 | 0.067 |

#### Validation day (λ was chosen on it: not out-of-sample for that choice)

| days scored | horizon | stocks | R² % | s.e. |
|---|---|---:|---:|---:|
| 2019-08-30 | k = 1..10 | 115 | 0.181 | 0.061 |
| 2019-08-30 | k = 10 | 115 | 0.225 | 0.060 |

## The same weights scored again by the decay measurement

The decay measurement scored LSTM-MLP-OF and CNN-LSTM-OF-lr3e-5-eps1e-7 again on these three days, with the same saved weights, scalers and Δt (its frozen basis); LSTM-OF and the ridge reference were not scored again. The table puts both scorings side by side. cuDNN's algorithm search was on in this stage and off in the decay measurement.

#### Stored scores and the second scoring, k = 1..10

| model | day | stocks | stored R² % | stocks, scored again | scored again, R² % | again minus stored, percentage points | cells on both | cells with the same R² |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM-MLP-OF | 2019-10-18 | 115 | 1.449 | 115 | 1.449 | 0.000e+00 | 1150 | 1150 |
| LSTM-MLP-OF | 2019-10-30 | 115 | 0.959 | 115 | 0.959 | 0.000e+00 | 1150 | 1150 |
| LSTM-MLP-OF | 2019-12-30 | 115 | 0.932 | 115 | 0.932 | 0.000e+00 | 1150 | 1150 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | 2019-10-18 | 115 | 1.427 | 115 | 1.428 | 1.305e-04 | 1150 | 0 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | 2019-10-30 | 115 | 0.928 | 115 | 0.928 | 1.401e-04 | 1150 | 0 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | 2019-12-30 | 115 | 1.051 | 115 | 1.052 | 2.875e-04 | 1150 | 0 |

## Validity

No day of this stage is a first look. The three test days were read many times while settings were tried: they were scored for many models of this stage (one model per stock on 20 stocks, a pooled model on 20 stocks, learning-rate and epsilon variants of the CNN-LSTM, and the pooled models of this report).

The days were fixed beforehand; the models and settings were not. No protocol document fixed this stage's models, settings and order of work before the test days were read, and the test-day reads of its 20-stock runs sit in other folders that are not exported, so the order of work cannot be audited from these records: the manifest's audit of this spec is empty.

The 3e-5/1e-7 setting was one of three picked on the validation day by the 20-stock learning-rate map. All three 20-stock fits, this setting included, were scored on these test days. The 115-stock fit of this setting was added after the 115-stock fit with learning rate 0.3 and epsilon 1 had been scored on these test days (0.609 %). That fit is not in these tables.

One seed.

## Reproducibility

Every number above is computed from the published tables in results/ (cells.parquet) by the functions of dof.experiments.analyses; the second scoring comes from the decay measurement's cells.

#### Stored records behind the tables

| record | cells | weights sha256 |
|---|---:|---|
| earlier/test/CNN-LSTM-OF-lr3e-5-eps1e-7/seed=0/test | 3450 |  |
| earlier/test/LSTM-MLP-OF/seed=0/test | 3450 |  |
| earlier/test/LSTM-OF/seed=0/test | 3450 |  |
| earlier/test/Ridge-OF/seed=0/test | 3450 |  |
| earlier/valid/CNN-LSTM-OF-lr3e-5-eps1e-7/seed=0/valid | 1150 |  |
| earlier/valid/LSTM-MLP-OF/seed=0/valid | 1150 |  |
| earlier/valid/LSTM-OF/seed=0/valid | 1150 |  |
| earlier/valid/Ridge-OF/seed=0/valid | 1150 |  |
