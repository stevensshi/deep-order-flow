# Seeds of the three fits, and the ensembles of the first three seeds

All values are out-of-sample R² in percent: per stock, day and horizon against the mean label of that stock-day; a stock's value is the mean of its cells; a table value is the mean over stocks. s.d. is the sample standard deviation across seeds (n - 1); with few seeds it is a rough number. The ensemble's forecast is the mean of the first three seeds' forecasts in dollars, sample by sample.

## Result

Test days 2026-05-15, 2026-05-18 and 2026-06-12, primary analysis, mean of the ten horizons. No seed is selected: every seed is in every table.

- `CNN-LSTM-OF-alldays`, test days 2026-05-15, 2026-05-18 and 2026-06-12, primary analysis, k = 1..10, 96 stocks: mean 0.269 % over 3 seeds (s.d. 0.020; min 0.250, max 0.289); ensemble of the first three seeds 0.364 % (96 stocks).
- `LSTM-MLP-OF-alldays`, test days 2026-05-15, 2026-05-18 and 2026-06-12, primary analysis, k = 1..10, 96 stocks: mean 0.205 % over 3 seeds (s.d. 0.008; min 0.196, max 0.211); ensemble of the first three seeds 0.270 % (96 stocks).
- `LSTM-MLP-OF-recent`, test days 2026-05-15, 2026-05-18 and 2026-06-12, primary analysis, k = 1..10, 96 stocks: mean 0.010 % over 3 seeds (s.d. 0.016; min -0.000, max 0.029); ensemble of the first three seeds 0.076 % (96 stocks).
- Does the older data help, test days 2026-05-15, 2026-05-18 and 2026-06-12, primary analysis, k = 1..10: mean 0.195 percentage points over 3 seeds (s.d. 0.012; 96 stocks in both); the two ensembles 0.194.

## Method

Seed 0 is the main run and was chosen by hand; the other seeds were drawn with secrets.randbits(32) and recorded before their fits started. A seed sets the initial weights and the order of the samples and batches; data, scalers, settings, stopping rule, validation set and test set are the same for every seed.

Order of work, as pre-registered: the six fits of the two drawn seeds, then their test scoring, then the ensembles, then the rule for two more seeds, which reads validation numbers only; further seeds are fitted and scored after that. The test days of the first round were therefore scored before the rule was applied; the rule could not see them.

Protocol document sha256 babfaf66c72cd5627f6b28cf271c12c8837078722216f9c91570bafd157607e2; seed protocol sha256 33f9cef8db47c6ebda964c38489f503c446f3bbe94368e9bc81e76a81e0e5e39; corporate-action file sha256 df7ef9adb2c52d4af1d112e5ae8a326d2408956ebe43e1d953e1b034f9f6ea4a.

## Results by fit

### CNN-LSTM-OF-alldays

#### Test days, primary analysis

| days scored | horizon | stocks | seed 0 | seed 3089492870 | seed 771810125 | mean | s.d. | min | max | seeds | ensemble of the first three | its stocks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 96 | 0.268 | 0.289 | 0.250 | 0.269 | 0.020 | 0.250 | 0.289 | 3 | 0.364 | 96 |
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 96 | 0.428 | 0.456 | 0.430 | 0.438 | 0.016 | 0.428 | 0.456 | 3 | 0.533 | 96 |
| 2026-05-15 | k = 1..10 | 96 | 0.058 | 0.132 | 0.073 | 0.088 | 0.039 | 0.058 | 0.132 | 3 | 0.189 | 96 |
| 2026-05-18 | k = 1..10 | 96 | 0.372 | 0.325 | 0.293 | 0.330 | 0.040 | 0.293 | 0.372 | 3 | 0.419 | 96 |
| 2026-06-12 | k = 1..10 | 95 | 0.375 | 0.411 | 0.381 | 0.389 | 0.019 | 0.375 | 0.411 | 3 | 0.485 | 95 |

#### Test days, secondary analysis (price ratio inside [0.6, 1.6])

| days scored | horizon | stocks | seed 0 | seed 3089492870 | seed 771810125 | mean | s.d. | min | max | seeds | ensemble of the first three | its stocks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 87 | 0.288 | 0.306 | 0.262 | 0.285 | 0.022 | 0.262 | 0.306 | 3 | 0.386 | 87 |
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 87 | 0.449 | 0.474 | 0.444 | 0.456 | 0.016 | 0.444 | 0.474 | 3 | 0.556 | 87 |
| 2026-05-15 | k = 1..10 | 84 | 0.027 | 0.117 | 0.053 | 0.066 | 0.046 | 0.027 | 0.117 | 3 | 0.175 | 84 |
| 2026-05-18 | k = 1..10 | 87 | 0.393 | 0.345 | 0.306 | 0.348 | 0.044 | 0.306 | 0.393 | 3 | 0.444 | 87 |
| 2026-06-12 | k = 1..10 | 82 | 0.401 | 0.423 | 0.387 | 0.404 | 0.018 | 0.387 | 0.423 | 3 | 0.504 | 82 |

#### Validation days (they chose the stopping check: not out-of-sample for that choice)

| days scored | horizon | stocks | seed 0 | seed 3089492870 | seed 771810125 | mean | s.d. | min | max | seeds | ensemble of the first three | its stocks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-12-11, 2025-12-12 | k = 1..10 | 98 | 0.673 | 0.672 | 0.647 | 0.664 | 0.014 | 0.647 | 0.673 | 3 | 0.787 | 98 |
| 2025-12-11, 2025-12-12 | k = 10 | 98 | 0.860 | 0.846 | 0.803 | 0.836 | 0.030 | 0.803 | 0.860 | 3 | 0.954 | 98 |
| 2025-12-11 | k = 1..10 | 98 | 0.528 | 0.532 | 0.540 | 0.533 | 0.006 | 0.528 | 0.540 | 3 | 0.655 | 98 |
| 2025-12-12 | k = 1..10 | 98 | 0.819 | 0.811 | 0.755 | 0.795 | 0.035 | 0.755 | 0.819 | 3 | 0.918 | 98 |

#### The fits

| model | seed | status | checks | best check | stopped by | steps | passes | best validation loss | interruptions | sha256 of the saved weights |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
| CNN-LSTM-OF-alldays | 0 | finished | 12 | 7 | patience | 3,806,568 | 3.03 | 1.082123 | 0 | 4ec421d309e5374f… |
| CNN-LSTM-OF-alldays | 3089492870 | finished | 10 | 5 | patience | 3,172,140 | 2.53 | 1.082130 | 0 | 86952263262922e3… |
| CNN-LSTM-OF-alldays | 771810125 | finished | 9 | 4 | patience | 2,854,926 | 2.27 | 1.082212 | 0 | 65854c2bea4a20a0… |

#### Validation loss by check, seed 0

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

#### Validation loss by check, seed 3089492870

| check | steps | passes | training loss since the last check | validation loss | new best | median weight step |
|---|---:|---:|---:|---:|---|---:|
| 1 | 317,214 | 0.25 | 0.988616 | 1.085319 | yes | 4.943e-06 |
| 2 | 634,428 | 0.51 | 0.984693 | 1.083792 | yes | 4.359e-06 |
| 3 | 951,642 | 0.76 | 0.982315 | 1.082817 | yes | 4.917e-06 |
| 4 | 1,268,856 | 1.01 | 0.981729 | 1.082182 | yes | 4.756e-06 |
| 5 | 1,586,070 | 1.26 | 0.981365 | 1.082130 | yes | 4.649e-06 |
| 6 | 1,903,284 | 1.52 | 0.980283 | 1.082649 |  | 3.323e-06 |
| 7 | 2,220,498 | 1.77 | 0.977908 | 1.083905 |  | 3.472e-06 |
| 8 | 2,537,712 | 2.02 | 0.977894 | 1.083288 |  | 2.131e-06 |
| 9 | 2,854,926 | 2.27 | 0.976289 | 1.084294 |  | 3.338e-06 |
| 10 | 3,172,140 | 2.53 | 0.974420 | 1.085973 |  | 5.007e-06 |

#### Validation loss by check, seed 771810125

| check | steps | passes | training loss since the last check | validation loss | new best | median weight step |
|---|---:|---:|---:|---:|---|---:|
| 1 | 317,214 | 0.25 | 0.987774 | 1.084906 | yes | 3.736e-06 |
| 2 | 634,428 | 0.51 | 0.986027 | 1.082726 | yes | 3.874e-06 |
| 3 | 951,642 | 0.76 | 0.982364 | 1.083094 |  | 4.701e-06 |
| 4 | 1,268,856 | 1.01 | 0.981771 | 1.082212 | yes | 3.912e-06 |
| 5 | 1,586,070 | 1.26 | 0.980578 | 1.083607 |  | 7.227e-06 |
| 6 | 1,903,284 | 1.52 | 0.979959 | 1.083137 |  | 4.958e-06 |
| 7 | 2,220,498 | 1.77 | 0.978888 | 1.082321 |  | 5.156e-06 |
| 8 | 2,537,712 | 2.02 | 0.976779 | 1.082951 |  | 3.852e-06 |
| 9 | 2,854,926 | 2.27 | 0.975910 | 1.083463 |  | 4.776e-06 |

### LSTM-MLP-OF-alldays

#### Test days, primary analysis

| days scored | horizon | stocks | seed 0 | seed 3089492870 | seed 771810125 | mean | s.d. | min | max | seeds | ensemble of the first three | its stocks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 96 | 0.196 | 0.209 | 0.211 | 0.205 | 0.008 | 0.196 | 0.211 | 3 | 0.270 | 96 |
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 96 | 0.359 | 0.369 | 0.389 | 0.372 | 0.015 | 0.359 | 0.389 | 3 | 0.435 | 96 |
| 2026-05-15 | k = 1..10 | 96 | 0.020 | 0.061 | 0.009 | 0.030 | 0.027 | 0.009 | 0.061 | 3 | 0.100 | 96 |
| 2026-05-18 | k = 1..10 | 96 | 0.241 | 0.255 | 0.265 | 0.253 | 0.012 | 0.241 | 0.265 | 3 | 0.314 | 96 |
| 2026-06-12 | k = 1..10 | 95 | 0.326 | 0.308 | 0.356 | 0.330 | 0.024 | 0.308 | 0.356 | 3 | 0.395 | 95 |

#### Test days, secondary analysis (price ratio inside [0.6, 1.6])

| days scored | horizon | stocks | seed 0 | seed 3089492870 | seed 771810125 | mean | s.d. | min | max | seeds | ensemble of the first three | its stocks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 87 | 0.206 | 0.218 | 0.220 | 0.215 | 0.008 | 0.206 | 0.220 | 3 | 0.283 | 87 |
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 87 | 0.369 | 0.380 | 0.398 | 0.382 | 0.014 | 0.369 | 0.398 | 3 | 0.448 | 87 |
| 2026-05-15 | k = 1..10 | 84 | -0.006 | 0.031 | -0.019 | 0.002 | 0.026 | -0.019 | 0.031 | 3 | 0.077 | 84 |
| 2026-05-18 | k = 1..10 | 87 | 0.254 | 0.266 | 0.282 | 0.268 | 0.014 | 0.254 | 0.282 | 3 | 0.333 | 87 |
| 2026-06-12 | k = 1..10 | 82 | 0.328 | 0.320 | 0.351 | 0.333 | 0.016 | 0.320 | 0.351 | 3 | 0.401 | 82 |

#### Validation days (they chose the stopping check: not out-of-sample for that choice)

| days scored | horizon | stocks | seed 0 | seed 3089492870 | seed 771810125 | mean | s.d. | min | max | seeds | ensemble of the first three | its stocks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-12-11, 2025-12-12 | k = 1..10 | 98 | 0.533 | 0.576 | 0.558 | 0.556 | 0.021 | 0.533 | 0.576 | 3 | 0.640 | 98 |
| 2025-12-11, 2025-12-12 | k = 10 | 98 | 0.736 | 0.755 | 0.728 | 0.740 | 0.014 | 0.728 | 0.755 | 3 | 0.819 | 98 |
| 2025-12-11 | k = 1..10 | 98 | 0.391 | 0.436 | 0.406 | 0.411 | 0.023 | 0.391 | 0.436 | 3 | 0.492 | 98 |
| 2025-12-12 | k = 1..10 | 98 | 0.676 | 0.715 | 0.710 | 0.701 | 0.021 | 0.676 | 0.715 | 3 | 0.788 | 98 |

#### The fits

| model | seed | status | checks | best check | stopped by | steps | passes | best validation loss | interruptions | sha256 of the saved weights |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
| LSTM-MLP-OF-alldays | 0 | finished | 13 | 8 | patience | 4,123,782 | 3.28 | 1.083579 | 1 | 0950686a0eef1f14… |
| LSTM-MLP-OF-alldays | 3089492870 | finished | 15 | 10 | patience | 4,758,210 | 3.79 | 1.083185 | 0 | 3be32bd58c365b14… |
| LSTM-MLP-OF-alldays | 771810125 | finished | 12 | 7 | patience | 3,806,568 | 3.03 | 1.083290 | 0 | 1486a77116897702… |

#### Validation loss by check, seed 0

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

#### Validation loss by check, seed 3089492870

| check | steps | passes | training loss since the last check | validation loss | new best | median weight step |
|---|---:|---:|---:|---:|---|---:|
| 1 | 317,214 | 0.25 | 0.990381 | 1.086118 | yes | 1.386e-06 |
| 2 | 634,428 | 0.51 | 0.986890 | 1.084805 | yes | 1.748e-06 |
| 3 | 951,642 | 0.76 | 0.985017 | 1.084361 | yes | 1.930e-06 |
| 4 | 1,268,856 | 1.01 | 0.984776 | 1.083719 | yes | 1.371e-06 |
| 5 | 1,586,070 | 1.26 | 0.984791 | 1.083492 | yes | 1.796e-06 |
| 6 | 1,903,284 | 1.52 | 0.984109 | 1.083356 | yes | 1.283e-06 |
| 7 | 2,220,498 | 1.77 | 0.982296 | 1.083240 | yes | 1.527e-06 |
| 8 | 2,537,712 | 2.02 | 0.982986 | 1.083275 |  | 1.356e-06 |
| 9 | 2,854,926 | 2.27 | 0.982343 | 1.083409 |  | 1.255e-06 |
| 10 | 3,172,140 | 2.53 | 0.981686 | 1.083185 | yes | 1.829e-06 |
| 11 | 3,489,354 | 2.78 | 0.981906 | 1.083506 |  | 1.553e-06 |
| 12 | 3,806,568 | 3.03 | 0.980668 | 1.083605 |  | 1.341e-06 |
| 13 | 4,123,782 | 3.28 | 0.981155 | 1.083574 |  | 9.388e-07 |
| 14 | 4,440,996 | 3.54 | 0.980071 | 1.083997 |  | 1.416e-06 |
| 15 | 4,758,210 | 3.79 | 0.979810 | 1.084754 |  | 1.408e-06 |

#### Validation loss by check, seed 771810125

| check | steps | passes | training loss since the last check | validation loss | new best | median weight step |
|---|---:|---:|---:|---:|---|---:|
| 1 | 317,214 | 0.25 | 0.989352 | 1.086286 | yes | 1.416e-06 |
| 2 | 634,428 | 0.51 | 0.988210 | 1.085029 | yes | 1.416e-06 |
| 3 | 951,642 | 0.76 | 0.985079 | 1.084486 | yes | 1.259e-06 |
| 4 | 1,268,856 | 1.01 | 0.984922 | 1.083855 | yes | 1.557e-06 |
| 5 | 1,586,070 | 1.26 | 0.984146 | 1.083667 | yes | 1.945e-06 |
| 6 | 1,903,284 | 1.52 | 0.984032 | 1.083887 |  | 1.580e-06 |
| 7 | 2,220,498 | 1.77 | 0.983526 | 1.083290 | yes | 1.900e-06 |
| 8 | 2,537,712 | 2.02 | 0.982126 | 1.083380 |  | 1.483e-06 |
| 9 | 2,854,926 | 2.27 | 0.982186 | 1.083693 |  | 1.364e-06 |
| 10 | 3,172,140 | 2.53 | 0.982647 | 1.083487 |  | 1.475e-06 |
| 11 | 3,489,354 | 2.78 | 0.981748 | 1.083405 |  | 1.237e-06 |
| 12 | 3,806,568 | 3.03 | 0.981198 | 1.084108 |  | 1.609e-06 |

### LSTM-MLP-OF-recent

#### Test days, primary analysis

| days scored | horizon | stocks | seed 0 | seed 3089492870 | seed 771810125 | mean | s.d. | min | max | seeds | ensemble of the first three | its stocks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 96 | -0.000 | 0.002 | 0.029 | 0.010 | 0.016 | -0.000 | 0.029 | 3 | 0.076 | 96 |
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 96 | 0.153 | 0.158 | 0.180 | 0.163 | 0.014 | 0.153 | 0.180 | 3 | 0.233 | 96 |
| 2026-05-15 | k = 1..10 | 96 | -0.084 | -0.116 | -0.081 | -0.094 | 0.019 | -0.116 | -0.081 | 3 | -0.018 | 96 |
| 2026-05-18 | k = 1..10 | 96 | 0.052 | 0.067 | 0.104 | 0.074 | 0.027 | 0.052 | 0.104 | 3 | 0.135 | 96 |
| 2026-06-12 | k = 1..10 | 95 | 0.025 | 0.048 | 0.058 | 0.044 | 0.017 | 0.025 | 0.058 | 3 | 0.105 | 95 |

#### Test days, secondary analysis (price ratio inside [0.6, 1.6])

| days scored | horizon | stocks | seed 0 | seed 3089492870 | seed 771810125 | mean | s.d. | min | max | seeds | ensemble of the first three | its stocks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 87 | 0.003 | 0.002 | 0.027 | 0.011 | 0.014 | 0.002 | 0.027 | 3 | 0.080 | 87 |
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 87 | 0.157 | 0.158 | 0.180 | 0.165 | 0.013 | 0.157 | 0.180 | 3 | 0.239 | 87 |
| 2026-05-15 | k = 1..10 | 84 | -0.107 | -0.140 | -0.110 | -0.119 | 0.018 | -0.140 | -0.107 | 3 | -0.037 | 84 |
| 2026-05-18 | k = 1..10 | 87 | 0.061 | 0.077 | 0.110 | 0.082 | 0.025 | 0.061 | 0.110 | 3 | 0.149 | 87 |
| 2026-06-12 | k = 1..10 | 82 | -0.001 | 0.015 | 0.020 | 0.011 | 0.011 | -0.001 | 0.020 | 3 | 0.075 | 82 |

#### Validation days (they chose the stopping check: not out-of-sample for that choice)

| days scored | horizon | stocks | seed 0 | seed 3089492870 | seed 771810125 | mean | s.d. | min | max | seeds | ensemble of the first three | its stocks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-12-11, 2025-12-12 | k = 1..10 | 98 | 0.369 | 0.374 | 0.388 | 0.377 | 0.010 | 0.369 | 0.388 | 3 | 0.457 | 98 |
| 2025-12-11, 2025-12-12 | k = 10 | 98 | 0.522 | 0.524 | 0.550 | 0.532 | 0.016 | 0.522 | 0.550 | 3 | 0.615 | 98 |
| 2025-12-11 | k = 1..10 | 98 | 0.298 | 0.298 | 0.289 | 0.295 | 0.005 | 0.289 | 0.298 | 3 | 0.371 | 98 |
| 2025-12-12 | k = 1..10 | 98 | 0.441 | 0.449 | 0.487 | 0.459 | 0.024 | 0.441 | 0.487 | 3 | 0.543 | 98 |

#### The fits

| model | seed | status | checks | best check | stopped by | steps | passes | best validation loss | interruptions | sha256 of the saved weights |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
| LSTM-MLP-OF-recent | 0 | finished | 7 | 2 | patience | 2,220,498 | 6.65 | 1.085639 | 0 | d6bb40ed4652fd9e… |
| LSTM-MLP-OF-recent | 3089492870 | finished | 7 | 2 | patience | 2,220,498 | 6.65 | 1.085474 | 0 | 35e7c249d9aec89f… |
| LSTM-MLP-OF-recent | 771810125 | finished | 8 | 3 | patience | 2,537,712 | 7.60 | 1.085249 | 0 | 0fa81e07e03ec4d6… |

#### Validation loss by check, seed 0

| check | steps | passes | training loss since the last check | validation loss | new best | median weight step |
|---|---:|---:|---:|---:|---|---:|
| 1 | 317,214 | 0.95 | 0.994452 | 1.086156 | yes | 1.289e-06 |
| 2 | 634,428 | 1.90 | 0.992540 | 1.085639 | yes | 1.609e-06 |
| 3 | 951,642 | 2.85 | 0.990261 | 1.087051 |  | 1.308e-06 |
| 4 | 1,268,856 | 3.80 | 0.989625 | 1.085711 |  | 1.259e-06 |
| 5 | 1,586,070 | 4.75 | 0.988177 | 1.086536 |  | 1.110e-06 |
| 6 | 1,903,284 | 5.70 | 0.986933 | 1.088060 |  | 9.770e-07 |
| 7 | 2,220,498 | 6.65 | 0.984103 | 1.089131 |  | 1.393e-06 |

#### Validation loss by check, seed 3089492870

| check | steps | passes | training loss since the last check | validation loss | new best | median weight step |
|---|---:|---:|---:|---:|---|---:|
| 1 | 317,214 | 0.95 | 0.994890 | 1.085999 | yes | 1.631e-06 |
| 2 | 634,428 | 1.90 | 0.991675 | 1.085474 | yes | 1.632e-06 |
| 3 | 951,642 | 2.85 | 0.991265 | 1.085753 |  | 1.080e-06 |
| 4 | 1,268,856 | 3.80 | 0.989106 | 1.087140 |  | 1.214e-06 |
| 5 | 1,586,070 | 4.75 | 0.987351 | 1.087568 |  | 1.109e-06 |
| 6 | 1,903,284 | 5.70 | 0.986666 | 1.088155 |  | 1.375e-06 |
| 7 | 2,220,498 | 6.65 | 0.983663 | 1.089095 |  | 9.537e-07 |

#### Validation loss by check, seed 771810125

| check | steps | passes | training loss since the last check | validation loss | new best | median weight step |
|---|---:|---:|---:|---:|---|---:|
| 1 | 317,214 | 0.95 | 0.994653 | 1.085870 | yes | 1.376e-06 |
| 2 | 634,428 | 1.90 | 0.992385 | 1.085372 | yes | 1.624e-06 |
| 3 | 951,642 | 2.85 | 0.990494 | 1.085249 | yes | 1.131e-06 |
| 4 | 1,268,856 | 3.80 | 0.988854 | 1.087618 |  | 1.377e-06 |
| 5 | 1,586,070 | 4.75 | 0.987545 | 1.086627 |  | 1.490e-06 |
| 6 | 1,903,284 | 5.70 | 0.985473 | 1.088537 |  | 1.330e-06 |
| 7 | 2,220,498 | 6.65 | 0.983933 | 1.089654 |  | 1.736e-06 |
| 8 | 2,537,712 | 7.60 | 0.981922 | 1.089378 |  | 1.343e-06 |

## Does the older data help

#### Test days, primary analysis

| days scored | horizon | stocks in both | seed 0 | seed 3089492870 | seed 771810125 | mean | s.d. | min | max | seeds | the two ensembles |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 96 | 0.196 | 0.206 | 0.182 | 0.195 | 0.012 | 0.182 | 0.206 | 3 | 0.194 |
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 96 | 0.207 | 0.211 | 0.209 | 0.209 | 0.002 | 0.207 | 0.211 | 3 | 0.202 |
| 2026-05-15 | k = 1..10 | 96 | 0.105 | 0.177 | 0.090 | 0.124 | 0.047 | 0.090 | 0.177 | 3 | 0.118 |
| 2026-05-18 | k = 1..10 | 96 | 0.189 | 0.188 | 0.161 | 0.179 | 0.016 | 0.161 | 0.189 | 3 | 0.179 |
| 2026-06-12 | k = 1..10 | 95 | 0.301 | 0.259 | 0.298 | 0.286 | 0.023 | 0.259 | 0.301 | 3 | 0.290 |

#### Test days, secondary analysis (price ratio inside [0.6, 1.6])

| days scored | horizon | stocks in both | seed 0 | seed 3089492870 | seed 771810125 | mean | s.d. | min | max | seeds | the two ensembles |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 1..10 | 87 | 0.203 | 0.217 | 0.193 | 0.204 | 0.012 | 0.193 | 0.217 | 3 | 0.203 |
| 2026-05-15, 2026-05-18, 2026-06-12 | k = 10 | 87 | 0.212 | 0.222 | 0.217 | 0.217 | 0.005 | 0.212 | 0.222 | 3 | 0.210 |
| 2026-05-15 | k = 1..10 | 84 | 0.102 | 0.171 | 0.090 | 0.121 | 0.044 | 0.090 | 0.171 | 3 | 0.114 |
| 2026-05-18 | k = 1..10 | 87 | 0.193 | 0.190 | 0.172 | 0.185 | 0.011 | 0.172 | 0.193 | 3 | 0.184 |
| 2026-06-12 | k = 1..10 | 82 | 0.329 | 0.304 | 0.331 | 0.322 | 0.015 | 0.304 | 0.331 | 3 | 0.326 |

#### Validation days (not out-of-sample for the stopping check)

| days scored | horizon | stocks in both | seed 0 | seed 3089492870 | seed 771810125 | mean | s.d. | min | max | seeds | the two ensembles |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-12-11, 2025-12-12 | k = 1..10 | 98 | 0.164 | 0.202 | 0.170 | 0.179 | 0.020 | 0.164 | 0.202 | 3 | 0.183 |
| 2025-12-11, 2025-12-12 | k = 10 | 98 | 0.214 | 0.232 | 0.178 | 0.208 | 0.028 | 0.178 | 0.232 | 3 | 0.204 |
| 2025-12-11 | k = 1..10 | 98 | 0.092 | 0.138 | 0.116 | 0.116 | 0.023 | 0.092 | 0.138 | 3 | 0.121 |
| 2025-12-12 | k = 1..10 | 98 | 0.236 | 0.266 | 0.223 | 0.242 | 0.022 | 0.223 | 0.266 | 3 | 0.245 |

## The rule for two more seeds

Two more seeds when the sample standard deviation of the three validation differences (all days minus recent, k = 1..10) is more than half of the absolute value of their mean, or when a difference is missing.

#### The rule, on validation numbers only

| quantity | value |
|---|---:|
| difference seed=0 | 0.164 |
| stocks seed=0 | 98 |
| difference seed=3089492870 | 0.202 |
| stocks seed=3089492870 | 98 |
| difference seed=771810125 | 0.170 |
| stocks seed=771810125 | 98 |
| mean | 0.179 |
| sd | 0.020 |
| half_abs_mean | 0.089 |
| more | no |
| drawn | none |

## Data

#### Test stock-days left out by a corporate action

| symbol | date | event | effective date |
|---|---|---|---|
| BKNG | 2026-05-15 | forward_split | 2026-04-06 |
| BKNG | 2026-05-18 | forward_split | 2026-04-06 |
| BKNG | 2026-06-12 | forward_split | 2026-04-06 |
| KLAC | 2026-06-12 | forward_split | 2026-06-12 |

#### Cells without an R²

| fit | score files | cells | finite | label variance zero | not finite, label variance above zero | other |
|---|---:|---:|---:|---:|---:|---:|
| LSTM-MLP-OF-recent | 8 | 19,320 | 19,320 | 0 | 0 | 0 |
| LSTM-MLP-OF-alldays | 8 | 19,320 | 19,320 | 0 | 0 | 0 |
| CNN-LSTM-OF-alldays | 8 | 19,320 | 19,320 | 0 | 0 | 0 |

## Validity

Every seed after seed 0 reads the same test days as seed 0; they are first looks for seed 0 only (2026-06-12 is not a first look for this feature family).

The cuDNN algorithm search was on, so a repeated fit is not bit-identical; these tables are rebuilt from the stored score cells.

## Reproducibility

Every number above is computed from the published tables in results/ (cells.parquet, stock_days.parquet, fits.parquet, checks.parquet) by the functions of dof.experiments.analyses.

#### Stored records behind the tables

| record | cells | weights sha256 |
|---|---:|---|
| alldays/fit_valid/CNN-LSTM-OF-alldays/seed=0/valid | 1960 | 4ec421d309e5374f |
| alldays/fit_valid/LSTM-MLP-OF-alldays/seed=0/valid | 1960 | 0950686a0eef1f14 |
| alldays/fit_valid/LSTM-MLP-OF-recent/seed=0/valid | 1960 | d6bb40ed4652fd9e |
| alldays/test/CNN-LSTM-OF-alldays/seed=0/test | 2870 | 4ec421d309e5374f |
| alldays/test/LSTM-MLP-OF-alldays/seed=0/test | 2870 | 0950686a0eef1f14 |
| alldays/test/LSTM-MLP-OF-recent/seed=0/test | 2870 | d6bb40ed4652fd9e |
| seeds/ensemble/CNN-LSTM-OF-alldays/members=0+3089492870+771810125/test | 2870 |  |
| seeds/ensemble/CNN-LSTM-OF-alldays/members=0+3089492870+771810125/valid | 1960 |  |
| seeds/ensemble/LSTM-MLP-OF-alldays/members=0+3089492870+771810125/test | 2870 |  |
| seeds/ensemble/LSTM-MLP-OF-alldays/members=0+3089492870+771810125/valid | 1960 |  |
| seeds/ensemble/LSTM-MLP-OF-recent/members=0+3089492870+771810125/test | 2870 |  |
| seeds/ensemble/LSTM-MLP-OF-recent/members=0+3089492870+771810125/valid | 1960 |  |
| seeds/fit_valid/CNN-LSTM-OF-alldays/seed=3089492870/valid | 1960 | 86952263262922e3 |
| seeds/fit_valid/CNN-LSTM-OF-alldays/seed=771810125/valid | 1960 | 65854c2bea4a20a0 |
| seeds/fit_valid/LSTM-MLP-OF-alldays/seed=3089492870/valid | 1960 | 3be32bd58c365b14 |
| seeds/fit_valid/LSTM-MLP-OF-alldays/seed=771810125/valid | 1960 | 1486a77116897702 |
| seeds/fit_valid/LSTM-MLP-OF-recent/seed=3089492870/valid | 1960 | 35e7c249d9aec89f |
| seeds/fit_valid/LSTM-MLP-OF-recent/seed=771810125/valid | 1960 | 0fa81e07e03ec4d6 |
| seeds/test/CNN-LSTM-OF-alldays/seed=3089492870/test | 2870 | 86952263262922e3 |
| seeds/test/CNN-LSTM-OF-alldays/seed=771810125/test | 2870 | 65854c2bea4a20a0 |
| seeds/test/LSTM-MLP-OF-alldays/seed=3089492870/test | 2870 | 3be32bd58c365b14 |
| seeds/test/LSTM-MLP-OF-alldays/seed=771810125/test | 2870 | 1486a77116897702 |
| seeds/test/LSTM-MLP-OF-recent/seed=3089492870/test | 2870 | 35e7c249d9aec89f |
| seeds/test/LSTM-MLP-OF-recent/seed=771810125/test | 2870 | 0fa81e07e03ec4d6 |
