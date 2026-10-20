# Decay: two models trained on three days of 2019, scored on later days

All values are out-of-sample R² in percent, per stock, day and horizon, against the mean label of that stock-day. A stock's value is the mean of its cells; a table value is the mean over stocks. Two models, seed 0 each.

## Result

The two models were trained on 2019-01-30, 2019-03-27 and 2019-07-30.

Further from training, and on the frozen basis, the values fall; the tables by model give every day.

- Frozen 2019 basis on the 2019 test days (2.6 to 5.0 months after training), row 'all', k = 1..10: `CNN-LSTM-OF-lr3e-5-eps1e-7` 1.428, 0.928, 1.052 % (115, 115, 115 stocks); `LSTM-MLP-OF` 1.449, 0.959, 0.932 % (115, 115, 115 stocks).
- Refreshed basis (scalers and Δt of 2025-11-28 to 2025-12-10), 2025-12-12, row 'all', k = 1..10: `CNN-LSTM-OF-lr3e-5-eps1e-7` 0.278 % (98 stocks); `LSTM-MLP-OF` 0.102 % (98 stocks).

## Method

Basis (i), frozen: each stock's 2019 scalers and Δt. Basis (ii), refreshed: scalers and Δt fitted on recent days the model was not trained on (2021-07-13 for 2021-08-13; the main run's group of 2025-11-28 to 2025-12-10 for the later days); its label horizons follow the refreshed Δt, so the labels of the two bases differ.

Row 'all': every held stock-day. Row 'inside': the stock-days whose median mid price is inside [0.6, 1.6] times that of the basis's reference day (2019-07-30 for (i), the refreshed group's last day for (ii)). (ii) − (i): the stocks scored on both bases on that day, in that row.

Experiment protocol sha256 ad9015c9c6fca945ad22fe8fd2e276d9f7d9bf05000e3a2d14b783267152a97b, which extends the protocol document sha256 babfaf66c72cd5627f6b28cf271c12c8837078722216f9c91570bafd157607e2.

## Results by model

### CNN-LSTM-OF-lr3e-5-eps1e-7

#### k = 1..10

| day | row | days since 2019-07-30 | months | (i) stock-days | (i) stocks | (i) R² % | (i) s.e. | (ii) stock-days | (ii) stocks | (ii) R² % | (ii) s.e. | (ii) − (i): stocks on both | difference, pp | s.e. | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2019-10-18 | all | 80 | 2.6 | 115 | 115 | 1.428 | 0.141 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-10-18 | inside [0.6, 1.6] | 80 | 2.6 | 115 | 115 | 1.428 | 0.141 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-10-30 | all | 92 | 3.0 | 115 | 115 | 0.928 | 0.087 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-10-30 | inside [0.6, 1.6] | 92 | 3.0 | 115 | 115 | 0.928 | 0.087 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-12-30 | all | 153 | 5.0 | 115 | 115 | 1.052 | 0.150 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-12-30 | inside [0.6, 1.6] | 153 | 5.0 | 113 | 113 | 1.030 | 0.150 |  |  |  |  |  |  |  | stored result, scored again |
| 2020-01-30 | all | 184 | 6.0 | 115 | 115 | 1.030 | 0.070 |  |  |  |  |  |  |  |  |
| 2020-01-30 | inside [0.6, 1.6] | 184 | 6.0 | 114 | 114 | 1.038 | 0.070 |  |  |  |  |  |  |  |  |
| 2021-07-13 | all | 714 | 23.5 | 113 | 113 | -0.328 | 1.009 |  |  |  |  |  |  |  |  |
| 2021-07-13 | inside [0.6, 1.6] | 714 | 23.5 | 64 | 64 | 0.884 | 0.145 |  |  |  |  |  |  |  |  |
| 2021-08-13 | all | 745 | 24.5 | 112 | 112 | -1.003 | 1.132 | 112 | 112 | -0.144 | 0.285 | 112 | 0.859 | 1.164 | refreshed on one day |
| 2021-08-13 | inside [0.6, 1.6] | 745 | 24.5 | 68 | 68 | 0.332 | 0.157 | 111 | 111 | 0.090 | 0.163 | 67 | 0.013 | 0.116 | refreshed on one day |
| 2025-11-28 | all | 2313 | 76.0 | 99 | 99 | -3.435 | 1.373 |  |  |  |  |  |  |  | half day |
| 2025-11-28 | inside [0.6, 1.6] | 2313 | 76.0 | 44 | 44 | -0.624 | 0.184 |  |  |  |  |  |  |  | half day |
| 2025-12-08 | all | 2323 | 76.3 | 99 | 99 | -3.244 | 1.492 |  |  |  |  |  |  |  |  |
| 2025-12-08 | inside [0.6, 1.6] | 2323 | 76.3 | 44 | 44 | -0.821 | 0.433 |  |  |  |  |  |  |  |  |
| 2025-12-09 | all | 2324 | 76.3 | 99 | 99 | -3.113 | 1.323 |  |  |  |  |  |  |  |  |
| 2025-12-09 | inside [0.6, 1.6] | 2324 | 76.3 | 44 | 44 | -0.233 | 0.219 |  |  |  |  |  |  |  |  |
| 2025-12-10 | all | 2325 | 76.4 | 99 | 99 | -1.867 | 0.801 |  |  |  |  |  |  |  |  |
| 2025-12-10 | inside [0.6, 1.6] | 2325 | 76.4 | 44 | 44 | -0.051 | 0.143 |  |  |  |  |  |  |  |  |
| 2025-12-11 | all | 2326 | 76.4 | 99 | 99 | -2.508 | 1.004 | 98 | 98 | -0.223 | 0.177 | 98 | 2.260 | 0.989 |  |
| 2025-12-11 | inside [0.6, 1.6] | 2326 | 76.4 | 43 | 43 | -1.017 | 0.744 | 98 | 98 | -0.223 | 0.177 | 43 | 0.541 | 0.453 |  |
| 2025-12-12 | all | 2327 | 76.4 | 99 | 99 | -2.633 | 1.344 | 98 | 98 | 0.278 | 0.104 | 98 | 2.833 | 1.351 |  |
| 2025-12-12 | inside [0.6, 1.6] | 2327 | 76.4 | 44 | 44 | -0.262 | 0.263 | 98 | 98 | 0.278 | 0.104 | 44 | 0.408 | 0.254 |  |
| 2026-05-15 | all | 2481 | 81.5 | 98 | 98 | -4.035 | 2.255 | 97 | 97 | -34.703 | 34.363 | 97 | -30.674 | 33.671 |  |
| 2026-05-15 | inside [0.6, 1.6] | 2481 | 81.5 | 43 | 43 | -0.234 | 0.145 | 84 | 84 | -0.398 | 0.130 | 40 | -0.271 | 0.149 |  |
| 2026-05-18 | all | 2484 | 81.6 | 98 | 98 | -2.208 | 1.156 | 97 | 97 | -19.193 | 19.276 | 97 | -16.989 | 18.786 |  |
| 2026-05-18 | inside [0.6, 1.6] | 2484 | 81.6 | 44 | 44 | -0.132 | 0.134 | 87 | 87 | 0.087 | 0.104 | 42 | 0.299 | 0.136 |  |
| 2026-06-12 | all | 2509 | 82.4 | 98 | 98 | -1.949 | 1.236 | 97 | 97 | -8.798 | 8.936 | 97 | -6.871 | 8.755 |  |
| 2026-06-12 | inside [0.6, 1.6] | 2509 | 82.4 | 42 | 42 | 0.226 | 0.135 | 82 | 82 | 0.155 | 0.084 | 39 | 0.018 | 0.212 |  |

#### k = 10

| day | row | days since 2019-07-30 | months | (i) stock-days | (i) stocks | (i) R² % | (i) s.e. | (ii) stock-days | (ii) stocks | (ii) R² % | (ii) s.e. | (ii) − (i): stocks on both | difference, pp | s.e. | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2019-10-18 | all | 80 | 2.6 | 115 | 115 | 1.753 | 0.144 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-10-18 | inside [0.6, 1.6] | 80 | 2.6 | 115 | 115 | 1.753 | 0.144 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-10-30 | all | 92 | 3.0 | 115 | 115 | 1.178 | 0.098 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-10-30 | inside [0.6, 1.6] | 92 | 3.0 | 115 | 115 | 1.178 | 0.098 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-12-30 | all | 153 | 5.0 | 115 | 115 | 1.185 | 0.151 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-12-30 | inside [0.6, 1.6] | 153 | 5.0 | 113 | 113 | 1.164 | 0.152 |  |  |  |  |  |  |  | stored result, scored again |
| 2020-01-30 | all | 184 | 6.0 | 115 | 115 | 1.256 | 0.080 |  |  |  |  |  |  |  |  |
| 2020-01-30 | inside [0.6, 1.6] | 184 | 6.0 | 114 | 114 | 1.265 | 0.080 |  |  |  |  |  |  |  |  |
| 2021-07-13 | all | 714 | 23.5 | 113 | 113 | -0.168 | 1.058 |  |  |  |  |  |  |  |  |
| 2021-07-13 | inside [0.6, 1.6] | 714 | 23.5 | 64 | 64 | 1.055 | 0.159 |  |  |  |  |  |  |  |  |
| 2021-08-13 | all | 745 | 24.5 | 112 | 112 | -0.779 | 1.028 | 112 | 112 | -0.045 | 0.263 | 112 | 0.734 | 1.055 | refreshed on one day |
| 2021-08-13 | inside [0.6, 1.6] | 745 | 24.5 | 68 | 68 | 0.424 | 0.180 | 111 | 111 | 0.156 | 0.170 | 67 | -0.014 | 0.148 | refreshed on one day |
| 2025-11-28 | all | 2313 | 76.0 | 99 | 99 | -4.063 | 1.711 |  |  |  |  |  |  |  | half day |
| 2025-11-28 | inside [0.6, 1.6] | 2313 | 76.0 | 44 | 44 | -0.661 | 0.176 |  |  |  |  |  |  |  | half day |
| 2025-12-08 | all | 2323 | 76.3 | 99 | 99 | -3.710 | 1.863 |  |  |  |  |  |  |  |  |
| 2025-12-08 | inside [0.6, 1.6] | 2323 | 76.3 | 44 | 44 | -0.796 | 0.574 |  |  |  |  |  |  |  |  |
| 2025-12-09 | all | 2324 | 76.3 | 99 | 99 | -3.425 | 1.527 |  |  |  |  |  |  |  |  |
| 2025-12-09 | inside [0.6, 1.6] | 2324 | 76.3 | 44 | 44 | -0.101 | 0.288 |  |  |  |  |  |  |  |  |
| 2025-12-10 | all | 2325 | 76.4 | 99 | 99 | -2.058 | 0.993 |  |  |  |  |  |  |  |  |
| 2025-12-10 | inside [0.6, 1.6] | 2325 | 76.4 | 44 | 44 | 0.207 | 0.139 |  |  |  |  |  |  |  |  |
| 2025-12-11 | all | 2326 | 76.4 | 99 | 99 | -2.633 | 1.172 | 98 | 98 | -0.043 | 0.159 | 98 | 2.567 | 1.168 |  |
| 2025-12-11 | inside [0.6, 1.6] | 2326 | 76.4 | 43 | 43 | -0.794 | 0.603 | 98 | 98 | -0.043 | 0.159 | 43 | 0.505 | 0.430 |  |
| 2025-12-12 | all | 2327 | 76.4 | 99 | 99 | -2.886 | 1.584 | 98 | 98 | 0.462 | 0.132 | 98 | 3.277 | 1.595 |  |
| 2025-12-12 | inside [0.6, 1.6] | 2327 | 76.4 | 44 | 44 | -0.143 | 0.277 | 98 | 98 | 0.462 | 0.132 | 44 | 0.368 | 0.281 |  |
| 2026-05-15 | all | 2481 | 81.5 | 98 | 98 | -4.612 | 2.503 | 97 | 97 | -32.137 | 31.878 | 97 | -27.531 | 30.838 |  |
| 2026-05-15 | inside [0.6, 1.6] | 2481 | 81.5 | 43 | 43 | -0.260 | 0.198 | 84 | 84 | -0.333 | 0.170 | 40 | -0.282 | 0.208 |  |
| 2026-05-18 | all | 2484 | 81.6 | 98 | 98 | -2.354 | 1.233 | 97 | 97 | -14.475 | 14.809 | 97 | -12.123 | 14.118 |  |
| 2026-05-18 | inside [0.6, 1.6] | 2484 | 81.6 | 44 | 44 | 0.036 | 0.156 | 87 | 87 | 0.337 | 0.136 | 42 | 0.405 | 0.203 |  |
| 2026-06-12 | all | 2509 | 82.4 | 98 | 98 | -2.134 | 1.362 | 97 | 97 | -6.165 | 6.430 | 97 | -4.046 | 6.158 |  |
| 2026-06-12 | inside [0.6, 1.6] | 2509 | 82.4 | 42 | 42 | 0.401 | 0.126 | 82 | 82 | 0.268 | 0.130 | 39 | 0.030 | 0.289 |  |

### LSTM-MLP-OF

#### k = 1..10

| day | row | days since 2019-07-30 | months | (i) stock-days | (i) stocks | (i) R² % | (i) s.e. | (ii) stock-days | (ii) stocks | (ii) R² % | (ii) s.e. | (ii) − (i): stocks on both | difference, pp | s.e. | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2019-10-18 | all | 80 | 2.6 | 115 | 115 | 1.449 | 0.143 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-10-18 | inside [0.6, 1.6] | 80 | 2.6 | 115 | 115 | 1.449 | 0.143 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-10-30 | all | 92 | 3.0 | 115 | 115 | 0.959 | 0.091 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-10-30 | inside [0.6, 1.6] | 92 | 3.0 | 115 | 115 | 0.959 | 0.091 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-12-30 | all | 153 | 5.0 | 115 | 115 | 0.932 | 0.152 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-12-30 | inside [0.6, 1.6] | 153 | 5.0 | 113 | 113 | 0.916 | 0.153 |  |  |  |  |  |  |  | stored result, scored again |
| 2020-01-30 | all | 184 | 6.0 | 115 | 115 | 1.021 | 0.099 |  |  |  |  |  |  |  |  |
| 2020-01-30 | inside [0.6, 1.6] | 184 | 6.0 | 114 | 114 | 1.029 | 0.099 |  |  |  |  |  |  |  |  |
| 2021-07-13 | all | 714 | 23.5 | 113 | 113 | -0.538 | 1.258 |  |  |  |  |  |  |  |  |
| 2021-07-13 | inside [0.6, 1.6] | 714 | 23.5 | 64 | 64 | 0.955 | 0.160 |  |  |  |  |  |  |  |  |
| 2021-08-13 | all | 745 | 24.5 | 112 | 112 | -1.684 | 1.709 | 112 | 112 | -0.348 | 0.347 | 112 | 1.336 | 1.739 | refreshed on one day |
| 2021-08-13 | inside [0.6, 1.6] | 745 | 24.5 | 68 | 68 | 0.317 | 0.188 | 111 | 111 | -0.057 | 0.190 | 67 | -0.114 | 0.171 | refreshed on one day |
| 2025-11-28 | all | 2313 | 76.0 | 99 | 99 | -4.754 | 2.011 |  |  |  |  |  |  |  | half day |
| 2025-11-28 | inside [0.6, 1.6] | 2313 | 76.0 | 44 | 44 | -0.675 | 0.213 |  |  |  |  |  |  |  | half day |
| 2025-12-08 | all | 2323 | 76.3 | 99 | 99 | -4.578 | 2.202 |  |  |  |  |  |  |  |  |
| 2025-12-08 | inside [0.6, 1.6] | 2323 | 76.3 | 44 | 44 | -0.760 | 0.427 |  |  |  |  |  |  |  |  |
| 2025-12-09 | all | 2324 | 76.3 | 99 | 99 | -4.446 | 1.986 |  |  |  |  |  |  |  |  |
| 2025-12-09 | inside [0.6, 1.6] | 2324 | 76.3 | 44 | 44 | -0.126 | 0.127 |  |  |  |  |  |  |  |  |
| 2025-12-10 | all | 2325 | 76.4 | 99 | 99 | -2.731 | 1.233 |  |  |  |  |  |  |  |  |
| 2025-12-10 | inside [0.6, 1.6] | 2325 | 76.4 | 44 | 44 | -0.016 | 0.149 |  |  |  |  |  |  |  |  |
| 2025-12-11 | all | 2326 | 76.4 | 99 | 99 | -3.460 | 1.669 | 98 | 98 | -0.314 | 0.144 | 98 | 3.110 | 1.693 |  |
| 2025-12-11 | inside [0.6, 1.6] | 2326 | 76.4 | 43 | 43 | -0.525 | 0.360 | 98 | 98 | -0.314 | 0.144 | 43 | 0.085 | 0.351 |  |
| 2025-12-12 | all | 2327 | 76.4 | 99 | 99 | -3.944 | 2.025 | 98 | 98 | 0.102 | 0.144 | 98 | 3.940 | 2.046 |  |
| 2025-12-12 | inside [0.6, 1.6] | 2327 | 76.4 | 44 | 44 | -0.251 | 0.268 | 98 | 98 | 0.102 | 0.144 | 44 | 0.222 | 0.264 |  |
| 2026-05-15 | all | 2481 | 81.5 | 98 | 98 | -5.504 | 3.180 | 97 | 97 | -50.279 | 49.600 | 97 | -44.783 | 48.943 |  |
| 2026-05-15 | inside [0.6, 1.6] | 2481 | 81.5 | 43 | 43 | -0.286 | 0.143 | 84 | 84 | -0.760 | 0.163 | 40 | -0.683 | 0.209 |  |
| 2026-05-18 | all | 2484 | 81.6 | 98 | 98 | -2.953 | 1.532 | 97 | 97 | -27.451 | 27.264 | 97 | -24.500 | 26.752 |  |
| 2026-05-18 | inside [0.6, 1.6] | 2484 | 81.6 | 44 | 44 | -0.162 | 0.146 | 87 | 87 | -0.207 | 0.151 | 42 | -0.194 | 0.217 |  |
| 2026-06-12 | all | 2509 | 82.4 | 98 | 98 | -2.704 | 1.690 | 97 | 97 | -11.967 | 11.791 | 97 | -9.282 | 11.645 |  |
| 2026-06-12 | inside [0.6, 1.6] | 2509 | 82.4 | 42 | 42 | 0.078 | 0.121 | 82 | 82 | -0.132 | 0.133 | 39 | -0.267 | 0.219 |  |

#### k = 10

| day | row | days since 2019-07-30 | months | (i) stock-days | (i) stocks | (i) R² % | (i) s.e. | (ii) stock-days | (ii) stocks | (ii) R² % | (ii) s.e. | (ii) − (i): stocks on both | difference, pp | s.e. | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2019-10-18 | all | 80 | 2.6 | 115 | 115 | 1.754 | 0.151 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-10-18 | inside [0.6, 1.6] | 80 | 2.6 | 115 | 115 | 1.754 | 0.151 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-10-30 | all | 92 | 3.0 | 115 | 115 | 1.222 | 0.116 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-10-30 | inside [0.6, 1.6] | 92 | 3.0 | 115 | 115 | 1.222 | 0.116 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-12-30 | all | 153 | 5.0 | 115 | 115 | 1.006 | 0.164 |  |  |  |  |  |  |  | stored result, scored again |
| 2019-12-30 | inside [0.6, 1.6] | 153 | 5.0 | 113 | 113 | 0.990 | 0.166 |  |  |  |  |  |  |  | stored result, scored again |
| 2020-01-30 | all | 184 | 6.0 | 115 | 115 | 1.219 | 0.110 |  |  |  |  |  |  |  |  |
| 2020-01-30 | inside [0.6, 1.6] | 184 | 6.0 | 114 | 114 | 1.228 | 0.110 |  |  |  |  |  |  |  |  |
| 2021-07-13 | all | 714 | 23.5 | 113 | 113 | -0.478 | 1.385 |  |  |  |  |  |  |  |  |
| 2021-07-13 | inside [0.6, 1.6] | 714 | 23.5 | 64 | 64 | 1.067 | 0.177 |  |  |  |  |  |  |  |  |
| 2021-08-13 | all | 745 | 24.5 | 112 | 112 | -1.631 | 1.685 | 112 | 112 | -0.311 | 0.332 | 112 | 1.320 | 1.712 | refreshed on one day |
| 2021-08-13 | inside [0.6, 1.6] | 745 | 24.5 | 68 | 68 | 0.266 | 0.215 | 111 | 111 | -0.045 | 0.200 | 67 | -0.061 | 0.184 | refreshed on one day |
| 2025-11-28 | all | 2313 | 76.0 | 99 | 99 | -5.745 | 2.493 |  |  |  |  |  |  |  | half day |
| 2025-11-28 | inside [0.6, 1.6] | 2313 | 76.0 | 44 | 44 | -0.838 | 0.225 |  |  |  |  |  |  |  | half day |
| 2025-12-08 | all | 2323 | 76.3 | 99 | 99 | -5.450 | 2.773 |  |  |  |  |  |  |  |  |
| 2025-12-08 | inside [0.6, 1.6] | 2323 | 76.3 | 44 | 44 | -0.829 | 0.546 |  |  |  |  |  |  |  |  |
| 2025-12-09 | all | 2324 | 76.3 | 99 | 99 | -5.162 | 2.353 |  |  |  |  |  |  |  |  |
| 2025-12-09 | inside [0.6, 1.6] | 2324 | 76.3 | 44 | 44 | -0.085 | 0.184 |  |  |  |  |  |  |  |  |
| 2025-12-10 | all | 2325 | 76.4 | 99 | 99 | -3.208 | 1.525 |  |  |  |  |  |  |  |  |
| 2025-12-10 | inside [0.6, 1.6] | 2325 | 76.4 | 44 | 44 | 0.083 | 0.166 |  |  |  |  |  |  |  |  |
| 2025-12-11 | all | 2326 | 76.4 | 99 | 99 | -4.039 | 2.076 | 98 | 98 | -0.203 | 0.170 | 98 | 3.800 | 2.096 |  |
| 2025-12-11 | inside [0.6, 1.6] | 2326 | 76.4 | 43 | 43 | -0.597 | 0.417 | 98 | 98 | -0.203 | 0.170 | 43 | 0.150 | 0.428 |  |
| 2025-12-12 | all | 2327 | 76.4 | 99 | 99 | -4.564 | 2.375 | 98 | 98 | 0.256 | 0.171 | 98 | 4.718 | 2.400 |  |
| 2025-12-12 | inside [0.6, 1.6] | 2327 | 76.4 | 44 | 44 | -0.336 | 0.334 | 98 | 98 | 0.256 | 0.171 | 44 | 0.357 | 0.336 |  |
| 2026-05-15 | all | 2481 | 81.5 | 98 | 98 | -6.129 | 3.390 | 97 | 97 | -51.440 | 50.790 | 97 | -45.326 | 49.823 |  |
| 2026-05-15 | inside [0.6, 1.6] | 2481 | 81.5 | 43 | 43 | -0.311 | 0.213 | 84 | 84 | -0.758 | 0.214 | 40 | -0.764 | 0.271 |  |
| 2026-05-18 | all | 2484 | 81.6 | 98 | 98 | -3.114 | 1.573 | 97 | 97 | -25.094 | 25.176 | 97 | -21.980 | 24.474 |  |
| 2026-05-18 | inside [0.6, 1.6] | 2484 | 81.6 | 44 | 44 | -0.008 | 0.163 | 87 | 87 | 0.066 | 0.179 | 42 | -0.055 | 0.264 |  |
| 2026-06-12 | all | 2509 | 82.4 | 98 | 98 | -3.050 | 1.910 | 97 | 97 | -10.979 | 10.826 | 97 | -7.945 | 10.608 |  |
| 2026-06-12 | inside [0.6, 1.6] | 2509 | 82.4 | 42 | 42 | 0.161 | 0.148 | 82 | 82 | -0.127 | 0.195 | 39 | -0.327 | 0.305 |  |

## The days

A stock-day is left out for a reused ticker, for no usable scaling group, or for no sample; every other held stock-day is scored, whatever its price ratio.

#### Stock-days per day and basis

| day | basis | role in the other experiments | held | scored ('all') | inside | below 0.6 | above 1.6 | left out: reused | no usable group | no sample | stock-days with no R² (CNN-LSTM-OF-lr3e-5-eps1e-7) | stock-days with no R² (LSTM-MLP-OF) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019-10-18 | (i) | the test days of both 2019 models. | 115 | 115 | 115 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2019-10-30 | (i) | the test days of both 2019 models. | 115 | 115 | 115 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2019-12-30 | (i) | the test days of both 2019 models. | 115 | 115 | 113 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 2020-01-30 | (i) | training days of the main run's models. | 115 | 115 | 114 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 2021-07-13 | (i) | training days of the main run's models. | 113 | 113 | 64 | 4 | 45 | 0 | 0 | 0 | 0 | 0 |
| 2021-08-13 | (i) | training days of the main run's models. | 112 | 112 | 68 | 4 | 40 | 0 | 0 | 0 | 0 | 0 |
| 2021-08-13 | (ii) | training days of the main run's models. | 112 | 112 | 111 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2025-11-28 | (i) | training days of the main run's models. | 101 | 99 | 44 | 19 | 36 | 2 | 0 | 0 | 0 | 0 |
| 2025-12-08 | (i) | training days of the main run's models. | 101 | 99 | 44 | 17 | 38 | 2 | 0 | 0 | 0 | 0 |
| 2025-12-09 | (i) | training days of the main run's models. | 101 | 99 | 44 | 17 | 38 | 2 | 0 | 0 | 0 | 0 |
| 2025-12-10 | (i) | training days of the main run's models. | 101 | 99 | 44 | 17 | 38 | 2 | 0 | 0 | 0 | 0 |
| 2025-12-11 | (i) | a validation day of the main run and of the seed extension; a training day of the adjacent-days experiment. | 101 | 99 | 43 | 17 | 39 | 2 | 0 | 0 | 0 | 0 |
| 2025-12-11 | (ii) | a validation day of the main run and of the seed extension; a training day of the adjacent-days experiment. | 101 | 98 | 98 | 0 | 0 | 2 | 1 | 0 | 0 | 0 |
| 2025-12-12 | (i) | a validation day of the main run and of the seed extension; its morning and afternoon are the validation and test parts of the adjacent-days experiment. | 101 | 99 | 44 | 17 | 38 | 2 | 0 | 0 | 0 | 0 |
| 2025-12-12 | (ii) | a validation day of the main run and of the seed extension; its morning and afternoon are the validation and test parts of the adjacent-days experiment. | 101 | 98 | 98 | 0 | 0 | 2 | 1 | 0 | 0 | 0 |
| 2026-05-15 | (i) | the test days of the main run. | 100 | 98 | 43 | 20 | 35 | 2 | 0 | 0 | 0 | 0 |
| 2026-05-15 | (ii) | the test days of the main run. | 100 | 97 | 84 | 5 | 8 | 2 | 1 | 0 | 0 | 0 |
| 2026-05-18 | (i) | the test days of the main run. | 100 | 98 | 44 | 19 | 35 | 2 | 0 | 0 | 0 | 0 |
| 2026-05-18 | (ii) | the test days of the main run. | 100 | 97 | 87 | 3 | 7 | 2 | 1 | 0 | 0 | 0 |
| 2026-06-12 | (i) | the test days of the main run. | 100 | 98 | 42 | 19 | 37 | 2 | 0 | 0 | 0 | 0 |
| 2026-06-12 | (ii) | the test days of the main run. | 100 | 97 | 82 | 6 | 9 | 2 | 1 | 0 | 0 | 0 |

#### Stock-days with the price ratio outside [0.6, 1.6]

```
frozen 2019-12-30 QRTEA 0.592
frozen 2019-12-30 TSLA 1.731
frozen 2020-01-30 TSLA 2.687
frozen 2021-07-13 CSGP 0.138
frozen 2021-07-13 CSX 0.454
frozen 2021-07-13 NTES 0.480
frozen 2021-07-13 UAL 0.526
frozen 2021-07-13 ADBE 1.991
frozen 2021-07-13 ADSK 1.833
frozen 2021-07-13 ALGN 3.175
frozen 2021-07-13 ALXN 1.624
frozen 2021-07-13 AMAT 2.684
frozen 2021-07-13 AMD 2.713
frozen 2021-07-13 AMZN 1.967
frozen 2021-07-13 ANSS 1.692
frozen 2021-07-13 ASML 3.101
frozen 2021-07-13 ATVI 1.949
frozen 2021-07-13 AVGO 1.617
frozen 2021-07-13 BIDU 1.647
frozen 2021-07-13 CDNS 1.826
frozen 2021-07-13 CHTR 1.804
frozen 2021-07-13 CPRT 1.770
frozen 2021-07-13 EA 1.637
frozen 2021-07-13 EBAY 1.664
frozen 2021-07-13 FAST 1.682
frozen 2021-07-13 FB 1.789
frozen 2021-07-13 GOOG 2.138
frozen 2021-07-13 GOOGL 2.073
frozen 2021-07-13 IDXX 2.320
frozen 2021-07-13 ILMN 1.622
frozen 2021-07-13 INTU 1.811
frozen 2021-07-13 ISRG 1.794
frozen 2021-07-13 JBHT 1.632
frozen 2021-07-13 JD 2.504
frozen 2021-07-13 KLAC 2.232
frozen 2021-07-13 LRCX 2.950
frozen 2021-07-13 LULU 1.932
frozen 2021-07-13 MELI 2.432
frozen 2021-07-13 MSFT 1.997
frozen 2021-07-13 MU 1.667
frozen 2021-07-13 NFLX 1.664
frozen 2021-07-13 NVDA 4.647
frozen 2021-07-13 NXPI 1.946
frozen 2021-07-13 PYPL 2.716
frozen 2021-07-13 QCOM 1.892
frozen 2021-07-13 REGN 1.873
frozen 2021-07-13 SGEN 1.918
frozen 2021-07-13 SNPS 2.052
frozen 2021-07-13 STX 1.872
frozen 2021-07-13 SWKS 2.174
frozen 2021-07-13 TMUS 1.819
frozen 2021-07-13 TSLA 2.828
frozen 2021-08-13 CSGP 0.136
frozen 2021-08-13 CSX 0.484
frozen 2021-08-13 NTES 0.384
frozen 2021-08-13 UAL 0.502
frozen 2021-08-13 ADBE 2.092
frozen 2021-08-13 ADSK 2.059
frozen 2021-08-13 ALGN 3.419
frozen 2021-08-13 AMAT 2.547
frozen 2021-08-13 AMD 3.302
frozen 2021-08-13 AMZN 1.733
frozen 2021-08-13 ANSS 1.740
frozen 2021-08-13 ASML 3.431
frozen 2021-08-13 ATVI 1.753
frozen 2021-08-13 AVGO 1.624
frozen 2021-08-13 CDNS 2.021
frozen 2021-08-13 CDW 1.744
frozen 2021-08-13 CHTR 1.949
frozen 2021-08-13 CPRT 1.810
frozen 2021-08-13 EBAY 1.759
frozen 2021-08-13 FAST 1.751
frozen 2021-08-13 FB 1.840
frozen 2021-08-13 GOOG 2.251
frozen 2021-08-13 GOOGL 2.235
frozen 2021-08-13 IDXX 2.349
frozen 2021-08-13 ILMN 1.726
frozen 2021-08-13 INTU 1.924
frozen 2021-08-13 ISRG 1.872
frozen 2021-08-13 JBHT 1.716
frozen 2021-08-13 JD 2.277
frozen 2021-08-13 KLAC 2.338
frozen 2021-08-13 LRCX 2.745
frozen 2021-08-13 LULU 2.116
frozen 2021-08-13 MELI 2.900
frozen 2021-08-13 MSFT 2.073
frozen 2021-08-13 NXPI 2.052
frozen 2021-08-13 PYPL 2.463
frozen 2021-08-13 QCOM 1.986
frozen 2021-08-13 REGN 2.015
frozen 2021-08-13 SGEN 2.085
frozen 2021-08-13 SNPS 2.163
frozen 2021-08-13 STX 1.917
frozen 2021-08-13 SWKS 2.097
frozen 2021-08-13 TMUS 1.778
frozen 2021-08-13 TSLA 2.999
refreshed 2021-08-13 NVDA 0.247
frozen 2025-11-28 AAL 0.458
frozen 2025-11-28 AMZN 0.122
frozen 2025-11-28 CHTR 0.507
frozen 2025-11-28 CPRT 0.501
frozen 2025-11-28 CSGP 0.111
frozen 2025-11-28 CSX 0.504
frozen 2025-11-28 FISV 0.595
frozen 2025-11-28 GOOG 0.259
frozen 2025-11-28 GOOGL 0.259
frozen 2025-11-28 ILMN 0.437
frozen 2025-11-28 LBTYA 0.421
frozen 2025-11-28 LBTYK 0.432
frozen 2025-11-28 LILAK 0.522
frozen 2025-11-28 MCHP 0.553
frozen 2025-11-28 NFLX 0.331
frozen 2025-11-28 NTES 0.590
frozen 2025-11-28 ORLY 0.264
frozen 2025-11-28 PYPL 0.562
frozen 2025-11-28 XRAY 0.200
frozen 2025-11-28 ADI 2.185
frozen 2025-11-28 ADSK 1.878
frozen 2025-11-28 AMAT 4.925
frozen 2025-11-28 AMD 6.487
frozen 2025-11-28 AMGN 1.951
frozen 2025-11-28 ASML 4.578
frozen 2025-11-28 BKNG 2.573
frozen 2025-11-28 CDNS 4.106
frozen 2025-11-28 CHKP 1.677
frozen 2025-11-28 COST 3.256
frozen 2025-11-28 EA 2.293
frozen 2025-11-28 EBAY 2.004
frozen 2025-11-28 EXPE 1.898
frozen 2025-11-28 FOXA 1.732
frozen 2025-11-28 GILD 1.867
frozen 2025-11-28 IDXX 2.638
frozen 2025-11-28 INTU 2.257
frozen 2025-11-28 JBHT 1.712
frozen 2025-11-28 KLAC 8.320
frozen 2025-11-28 MAR 2.164
frozen 2025-11-28 MELI 3.262
frozen 2025-11-28 MSFT 3.492
frozen 2025-11-28 MU 5.011
frozen 2025-11-28 NTAP 1.878
frozen 2025-11-28 NXPI 1.854
frozen 2025-11-28 QCOM 2.234
frozen 2025-11-28 REGN 2.533
frozen 2025-11-28 ROST 1.667
frozen 2025-11-28 SIRI 3.399
frozen 2025-11-28 SNPS 3.074
frozen 2025-11-28 STX 5.777
frozen 2025-11-28 TMUS 2.556
frozen 2025-11-28 TSLA 1.791
frozen 2025-11-28 TTWO 2.001
frozen 2025-11-28 VRTX 2.531
frozen 2025-11-28 WDC 2.882
frozen 2025-12-08 AAL 0.483
frozen 2025-12-08 AMZN 0.120
frozen 2025-12-08 CHTR 0.514
frozen 2025-12-08 CPRT 0.491
frozen 2025-12-08 CSGP 0.107
frozen 2025-12-08 CSX 0.523
frozen 2025-12-08 GOOG 0.257
frozen 2025-12-08 GOOGL 0.256
frozen 2025-12-08 ILMN 0.429
frozen 2025-12-08 LBTYA 0.418
frozen 2025-12-08 LBTYK 0.424
frozen 2025-12-08 LILAK 0.524
frozen 2025-12-08 NFLX 0.296
frozen 2025-12-08 NTES 0.586
frozen 2025-12-08 ORLY 0.257
frozen 2025-12-08 PYPL 0.550
frozen 2025-12-08 XRAY 0.198
frozen 2025-12-08 ADI 2.308
frozen 2025-12-08 ADSK 1.864
frozen 2025-12-08 AMAT 5.278
frozen 2025-12-08 AMD 6.600
frozen 2025-12-08 AMGN 1.837
frozen 2025-12-08 ASML 4.925
frozen 2025-12-08 BKNG 2.698
frozen 2025-12-08 CDNS 4.489
frozen 2025-12-08 CHKP 1.709
frozen 2025-12-08 COST 3.167
frozen 2025-12-08 EA 2.315
frozen 2025-12-08 EBAY 2.026
frozen 2025-12-08 EXPE 1.956
frozen 2025-12-08 FOX 1.622
frozen 2025-12-08 FOXA 1.831
frozen 2025-12-08 GILD 1.807
frozen 2025-12-08 IDXX 2.479
frozen 2025-12-08 INTU 2.347
frozen 2025-12-08 JBHT 1.867
frozen 2025-12-08 KLAC 8.784
frozen 2025-12-08 MAR 2.049
frozen 2025-12-08 MELI 3.282
frozen 2025-12-08 MSFT 3.484
frozen 2025-12-08 MU 5.179
frozen 2025-12-08 NTAP 1.976
frozen 2025-12-08 NXPI 2.187
frozen 2025-12-08 QCOM 2.332
frozen 2025-12-08 REGN 2.294
frozen 2025-12-08 ROST 1.679
frozen 2025-12-08 SIRI 3.497
frozen 2025-12-08 SNPS 3.421
frozen 2025-12-08 STX 5.991
frozen 2025-12-08 TMUS 2.512
frozen 2025-12-08 TSLA 1.830
frozen 2025-12-08 TTWO 2.021
frozen 2025-12-08 ULTA 1.698
frozen 2025-12-08 VRTX 2.602
frozen 2025-12-08 WDC 3.019
frozen 2025-12-09 AAL 0.490
frozen 2025-12-09 AMZN 0.120
frozen 2025-12-09 CHTR 0.516
frozen 2025-12-09 CPRT 0.491
frozen 2025-12-09 CSGP 0.109
frozen 2025-12-09 CSX 0.520
frozen 2025-12-09 GOOG 0.258
frozen 2025-12-09 GOOGL 0.256
frozen 2025-12-09 ILMN 0.440
frozen 2025-12-09 LBTYA 0.411
frozen 2025-12-09 LBTYK 0.414
frozen 2025-12-09 LILAK 0.515
frozen 2025-12-09 NFLX 0.297
frozen 2025-12-09 NTES 0.581
frozen 2025-12-09 ORLY 0.245
frozen 2025-12-09 PYPL 0.546
frozen 2025-12-09 XRAY 0.198
frozen 2025-12-09 ADI 2.291
frozen 2025-12-09 ADSK 1.871
frozen 2025-12-09 AMAT 5.264
frozen 2025-12-09 AMD 6.651
frozen 2025-12-09 AMGN 1.802
frozen 2025-12-09 ASML 4.870
frozen 2025-12-09 BKNG 2.692
frozen 2025-12-09 CDNS 4.442
frozen 2025-12-09 CHKP 1.726
frozen 2025-12-09 COST 3.163
frozen 2025-12-09 EA 2.314
frozen 2025-12-09 EBAY 2.033
frozen 2025-12-09 EXPE 1.957
frozen 2025-12-09 FOX 1.631
frozen 2025-12-09 FOXA 1.845
frozen 2025-12-09 GILD 1.782
frozen 2025-12-09 IDXX 2.469
frozen 2025-12-09 INTU 2.342
frozen 2025-12-09 JBHT 1.857
frozen 2025-12-09 KLAC 8.729
frozen 2025-12-09 MAR 2.026
frozen 2025-12-09 MELI 3.238
frozen 2025-12-09 MSFT 3.488
frozen 2025-12-09 MU 5.353
frozen 2025-12-09 NTAP 1.976
frozen 2025-12-09 NXPI 2.179
frozen 2025-12-09 QCOM 2.346
frozen 2025-12-09 REGN 2.264
frozen 2025-12-09 ROST 1.668
frozen 2025-12-09 SIRI 3.541
frozen 2025-12-09 SNPS 3.430
frozen 2025-12-09 STX 5.999
frozen 2025-12-09 TMUS 2.464
frozen 2025-12-09 TSLA 1.868
frozen 2025-12-09 TTWO 2.027
frozen 2025-12-09 ULTA 1.727
frozen 2025-12-09 VRTX 2.585
frozen 2025-12-09 WDC 3.029
frozen 2025-12-10 AAL 0.491
frozen 2025-12-10 AMZN 0.122
frozen 2025-12-10 CHTR 0.533
frozen 2025-12-10 CPRT 0.494
frozen 2025-12-10 CSGP 0.110
frozen 2025-12-10 CSX 0.525
frozen 2025-12-10 GOOG 0.259
frozen 2025-12-10 GOOGL 0.258
frozen 2025-12-10 ILMN 0.452
frozen 2025-12-10 LBTYA 0.410
frozen 2025-12-10 LBTYK 0.413
frozen 2025-12-10 LILAK 0.508
frozen 2025-12-10 NFLX 0.290
frozen 2025-12-10 NTES 0.582
frozen 2025-12-10 ORLY 0.243
frozen 2025-12-10 PYPL 0.543
frozen 2025-12-10 XRAY 0.202
frozen 2025-12-10 ADI 2.312
frozen 2025-12-10 ADSK 1.861
frozen 2025-12-10 AMAT 5.334
frozen 2025-12-10 AMD 6.603
frozen 2025-12-10 AMGN 1.796
frozen 2025-12-10 ASML 4.835
frozen 2025-12-10 BKNG 2.749
frozen 2025-12-10 CDNS 4.472
frozen 2025-12-10 CHKP 1.761
frozen 2025-12-10 COST 3.133
frozen 2025-12-10 EA 2.312
frozen 2025-12-10 EBAY 2.005
frozen 2025-12-10 EXPE 2.016
frozen 2025-12-10 FOX 1.664
frozen 2025-12-10 FOXA 1.885
frozen 2025-12-10 GILD 1.788
frozen 2025-12-10 IDXX 2.447
frozen 2025-12-10 INTU 2.353
frozen 2025-12-10 JBHT 1.901
frozen 2025-12-10 KLAC 8.800
frozen 2025-12-10 MAR 2.013
frozen 2025-12-10 MELI 3.122
frozen 2025-12-10 MSFT 3.404
frozen 2025-12-10 MU 5.490
frozen 2025-12-10 NTAP 1.997
frozen 2025-12-10 NXPI 2.184
frozen 2025-12-10 QCOM 2.410
frozen 2025-12-10 REGN 2.313
frozen 2025-12-10 ROST 1.700
frozen 2025-12-10 SIRI 3.584
frozen 2025-12-10 SNPS 3.454
frozen 2025-12-10 STX 6.179
frozen 2025-12-10 TMUS 2.412
frozen 2025-12-10 TSLA 1.865
frozen 2025-12-10 TTWO 2.019
frozen 2025-12-10 ULTA 1.716
frozen 2025-12-10 VRTX 2.572
frozen 2025-12-10 WDC 3.146
frozen 2025-12-11 AAL 0.488
frozen 2025-12-11 AMZN 0.121
frozen 2025-12-11 CHTR 0.532
frozen 2025-12-11 CPRT 0.497
frozen 2025-12-11 CSGP 0.109
frozen 2025-12-11 CSX 0.530
frozen 2025-12-11 GOOG 0.257
frozen 2025-12-11 GOOGL 0.256
frozen 2025-12-11 ILMN 0.455
frozen 2025-12-11 LBTYA 0.416
frozen 2025-12-11 LBTYK 0.420
frozen 2025-12-11 LILAK 0.522
frozen 2025-12-11 NFLX 0.290
frozen 2025-12-11 NTES 0.582
frozen 2025-12-11 ORLY 0.245
frozen 2025-12-11 PYPL 0.549
frozen 2025-12-11 XRAY 0.206
frozen 2025-12-11 ADI 2.328
frozen 2025-12-11 ADSK 1.874
frozen 2025-12-11 AMAT 5.267
frozen 2025-12-11 AMD 6.431
frozen 2025-12-11 AMGN 1.798
frozen 2025-12-11 ASML 4.846
frozen 2025-12-11 BKNG 2.759
frozen 2025-12-11 CDNS 4.406
frozen 2025-12-11 CHKP 1.775
frozen 2025-12-11 COST 3.143
frozen 2025-12-11 EA 2.313
frozen 2025-12-11 EBAY 2.038
frozen 2025-12-11 EXPE 2.080
frozen 2025-12-11 FOX 1.690
frozen 2025-12-11 FOXA 1.916
frozen 2025-12-11 GILD 1.834
frozen 2025-12-11 IDXX 2.477
frozen 2025-12-11 INTU 2.410
frozen 2025-12-11 JBHT 1.988
frozen 2025-12-11 KLAC 8.676
frozen 2025-12-11 MAR 2.098
frozen 2025-12-11 MELI 3.177
frozen 2025-12-11 MSFT 3.419
frozen 2025-12-11 MU 5.473
frozen 2025-12-11 NTAP 1.985
frozen 2025-12-11 NXPI 2.187
frozen 2025-12-11 PCAR 1.601
frozen 2025-12-11 QCOM 2.394
frozen 2025-12-11 REGN 2.401
frozen 2025-12-11 ROST 1.722
frozen 2025-12-11 SIRI 3.585
frozen 2025-12-11 SNPS 3.479
frozen 2025-12-11 STX 6.367
frozen 2025-12-11 TMUS 2.395
frozen 2025-12-11 TSLA 1.855
frozen 2025-12-11 TTWO 1.995
frozen 2025-12-11 ULTA 1.705
frozen 2025-12-11 VRTX 2.615
frozen 2025-12-11 WDC 3.277
frozen 2025-12-12 AAL 0.493
frozen 2025-12-12 AMZN 0.119
frozen 2025-12-12 CHTR 0.524
frozen 2025-12-12 CPRT 0.496
frozen 2025-12-12 CSGP 0.110
frozen 2025-12-12 CSX 0.532
frozen 2025-12-12 GOOG 0.253
frozen 2025-12-12 GOOGL 0.251
frozen 2025-12-12 ILMN 0.453
frozen 2025-12-12 LBTYA 0.416
frozen 2025-12-12 LBTYK 0.419
frozen 2025-12-12 LILAK 0.518
frozen 2025-12-12 NFLX 0.294
frozen 2025-12-12 NTES 0.594
frozen 2025-12-12 ORLY 0.245
frozen 2025-12-12 PYPL 0.552
frozen 2025-12-12 XRAY 0.205
frozen 2025-12-12 ADI 2.322
frozen 2025-12-12 ADSK 1.849
frozen 2025-12-12 AMAT 5.148
frozen 2025-12-12 AMD 6.360
frozen 2025-12-12 AMGN 1.801
frozen 2025-12-12 ASML 4.771
frozen 2025-12-12 BKNG 2.780
frozen 2025-12-12 CDNS 4.307
frozen 2025-12-12 CHKP 1.746
frozen 2025-12-12 COST 3.126
frozen 2025-12-12 EA 2.313
frozen 2025-12-12 EBAY 2.058
frozen 2025-12-12 EXPE 2.051
frozen 2025-12-12 FOX 1.678
frozen 2025-12-12 FOXA 1.899
frozen 2025-12-12 GILD 1.805
frozen 2025-12-12 IDXX 2.458
frozen 2025-12-12 INTU 2.397
frozen 2025-12-12 JBHT 1.957
frozen 2025-12-12 KLAC 8.571
frozen 2025-12-12 MAR 2.127
frozen 2025-12-12 MELI 3.179
frozen 2025-12-12 MSFT 3.402
frozen 2025-12-12 MU 5.182
frozen 2025-12-12 NTAP 1.965
frozen 2025-12-12 NXPI 2.190
frozen 2025-12-12 QCOM 2.403
frozen 2025-12-12 REGN 2.394
frozen 2025-12-12 ROST 1.725
frozen 2025-12-12 SIRI 3.512
frozen 2025-12-12 SNPS 3.387
frozen 2025-12-12 STX 6.117
frozen 2025-12-12 TMUS 2.394
frozen 2025-12-12 TSLA 1.886
frozen 2025-12-12 TTWO 1.976
frozen 2025-12-12 ULTA 1.702
frozen 2025-12-12 VRTX 2.650
frozen 2025-12-12 WDC 3.133
frozen 2026-05-15 AAL 0.407
frozen 2026-05-15 AMZN 0.139
frozen 2026-05-15 BKNG 0.081
frozen 2026-05-15 CHTR 0.353
frozen 2026-05-15 CMCSA 0.569
frozen 2026-05-15 CPRT 0.418
frozen 2026-05-15 CSGP 0.053
frozen 2026-05-15 FISV 0.530
frozen 2026-05-15 GOOG 0.320
frozen 2026-05-15 GOOGL 0.323
frozen 2026-05-15 ILMN 0.471
frozen 2026-05-15 LBTYA 0.426
frozen 2026-05-15 LBTYK 0.427
frozen 2026-05-15 LILAK 0.450
frozen 2026-05-15 NFLX 0.268
frozen 2026-05-15 NTES 0.485
frozen 2026-05-15 ORLY 0.232
frozen 2026-05-15 PYPL 0.402
frozen 2026-05-15 WDAY 0.592
frozen 2026-05-15 XRAY 0.177
frozen 2026-05-15 ADI 3.463
frozen 2026-05-15 AMAT 8.611
frozen 2026-05-15 AMD 12.950
frozen 2026-05-15 AMGN 1.859
frozen 2026-05-15 ASML 6.619
frozen 2026-05-15 CDNS 4.625
frozen 2026-05-15 COST 3.742
frozen 2026-05-15 CSCO 2.067
frozen 2026-05-15 EA 2.276
frozen 2026-05-15 EBAY 2.791
frozen 2026-05-15 EXPE 1.619
frozen 2026-05-15 FOXA 1.708
frozen 2026-05-15 GILD 1.929
frozen 2026-05-15 IDXX 1.860
frozen 2026-05-15 INTC 2.090
frozen 2026-05-15 JBHT 2.562
frozen 2026-05-15 KLAC 13.024
frozen 2026-05-15 MAR 2.509
frozen 2026-05-15 MELI 2.445
frozen 2026-05-15 MSFT 3.014
frozen 2026-05-15 MU 15.618
frozen 2026-05-15 NTAP 2.023
frozen 2026-05-15 NXPI 2.783
frozen 2026-05-15 QCOM 2.728
frozen 2026-05-15 REGN 2.262
frozen 2026-05-15 ROST 2.009
frozen 2026-05-15 SIRI 4.134
frozen 2026-05-15 SNPS 3.691
frozen 2026-05-15 STX 16.651
frozen 2026-05-15 TMUS 2.289
frozen 2026-05-15 TSLA 1.777
frozen 2026-05-15 TTWO 1.991
frozen 2026-05-15 TXN 2.383
frozen 2026-05-15 VRTX 2.578
frozen 2026-05-15 WDC 8.540
refreshed 2026-05-15 BKNG 0.029
refreshed 2026-05-15 CSGP 0.479
refreshed 2026-05-15 CTSH 0.575
refreshed 2026-05-15 INTU 0.597
refreshed 2026-05-15 WDAY 0.556
refreshed 2026-05-15 AMAT 1.614
refreshed 2026-05-15 AMD 1.961
refreshed 2026-05-15 INTC 2.702
refreshed 2026-05-15 LRCX 1.718
refreshed 2026-05-15 MU 2.845
refreshed 2026-05-15 STX 2.695
refreshed 2026-05-15 TXN 1.694
refreshed 2026-05-15 WDC 2.715
frozen 2026-05-18 AAL 0.405
frozen 2026-05-18 AMZN 0.140
frozen 2026-05-18 BKNG 0.081
frozen 2026-05-18 CHTR 0.365
frozen 2026-05-18 CMCSA 0.571
frozen 2026-05-18 CPRT 0.428
frozen 2026-05-18 CSGP 0.055
frozen 2026-05-18 FISV 0.549
frozen 2026-05-18 GOOG 0.324
frozen 2026-05-18 GOOGL 0.327
frozen 2026-05-18 ILMN 0.475
frozen 2026-05-18 LBTYA 0.440
frozen 2026-05-18 LBTYK 0.441
frozen 2026-05-18 LILAK 0.447
frozen 2026-05-18 NFLX 0.275
frozen 2026-05-18 NTES 0.485
frozen 2026-05-18 ORLY 0.236
frozen 2026-05-18 PYPL 0.400
frozen 2026-05-18 XRAY 0.176
frozen 2026-05-18 ADI 3.472
frozen 2026-05-18 AMAT 8.140
frozen 2026-05-18 AMD 12.515
frozen 2026-05-18 AMGN 1.838
frozen 2026-05-18 ASML 6.469
frozen 2026-05-18 CDNS 4.561
frozen 2026-05-18 COST 3.807
frozen 2026-05-18 CSCO 2.066
frozen 2026-05-18 EA 2.284
frozen 2026-05-18 EBAY 2.773
frozen 2026-05-18 EXPE 1.604
frozen 2026-05-18 FOXA 1.734
frozen 2026-05-18 GILD 1.936
frozen 2026-05-18 IDXX 1.910
frozen 2026-05-18 INTC 2.058
frozen 2026-05-18 JBHT 2.571
frozen 2026-05-18 KLAC 12.514
frozen 2026-05-18 MAR 2.566
frozen 2026-05-18 MELI 2.471
frozen 2026-05-18 MSFT 2.998
frozen 2026-05-18 MU 14.717
frozen 2026-05-18 NTAP 2.031
frozen 2026-05-18 NXPI 2.770
frozen 2026-05-18 QCOM 2.662
frozen 2026-05-18 REGN 2.028
frozen 2026-05-18 ROST 1.987
frozen 2026-05-18 SIRI 4.208
frozen 2026-05-18 SNPS 3.640
frozen 2026-05-18 STX 15.521
frozen 2026-05-18 TMUS 2.330
frozen 2026-05-18 TSLA 1.713
frozen 2026-05-18 TTWO 1.976
frozen 2026-05-18 TXN 2.343
frozen 2026-05-18 VRTX 2.557
frozen 2026-05-18 WDC 8.088
refreshed 2026-05-18 BKNG 0.029
refreshed 2026-05-18 CSGP 0.498
refreshed 2026-05-18 WDAY 0.579
refreshed 2026-05-18 AMD 1.896
refreshed 2026-05-18 INTC 2.660
refreshed 2026-05-18 LRCX 1.671
refreshed 2026-05-18 MU 2.681
refreshed 2026-05-18 STX 2.512
refreshed 2026-05-18 TXN 1.666
refreshed 2026-05-18 WDC 2.571
frozen 2026-06-12 AAL 0.486
frozen 2026-06-12 AMZN 0.125
frozen 2026-06-12 BKNG 0.086
frozen 2026-06-12 CHTR 0.358
frozen 2026-06-12 CMCSA 0.553
frozen 2026-06-12 CPRT 0.393
frozen 2026-06-12 CSGP 0.053
frozen 2026-06-12 FISV 0.515
frozen 2026-06-12 GOOG 0.293
frozen 2026-06-12 GOOGL 0.294
frozen 2026-06-12 ILMN 0.541
frozen 2026-06-12 LBTYA 0.449
frozen 2026-06-12 LBTYK 0.443
frozen 2026-06-12 LILAK 0.489
frozen 2026-06-12 NFLX 0.247
frozen 2026-06-12 NTES 0.540
frozen 2026-06-12 ORLY 0.233
frozen 2026-06-12 PYPL 0.369
frozen 2026-06-12 XRAY 0.180
frozen 2026-06-12 ADI 3.451
frozen 2026-06-12 AMAT 11.123
frozen 2026-06-12 AMD 15.383
frozen 2026-06-12 AMGN 2.011
frozen 2026-06-12 ASML 8.159
frozen 2026-06-12 CDNS 5.088
frozen 2026-06-12 COST 3.487
frozen 2026-06-12 CSCO 2.144
frozen 2026-06-12 EA 2.303
frozen 2026-06-12 EBAY 2.615
frozen 2026-06-12 EXPE 1.665
frozen 2026-06-12 FOXA 1.780
frozen 2026-06-12 GILD 1.860
frozen 2026-06-12 IDXX 1.956
frozen 2026-06-12 INTC 2.369
frozen 2026-06-12 JBHT 2.834
frozen 2026-06-12 KLAC 1.786
frozen 2026-06-12 LRCX 1.749
frozen 2026-06-12 MAR 2.844
frozen 2026-06-12 MELI 2.489
frozen 2026-06-12 MSFT 2.753
frozen 2026-06-12 MU 21.074
frozen 2026-06-12 NTAP 2.727
frozen 2026-06-12 NXPI 2.904
frozen 2026-06-12 PCAR 1.678
frozen 2026-06-12 QCOM 2.834
frozen 2026-06-12 REGN 1.989
frozen 2026-06-12 ROST 2.271
frozen 2026-06-12 SIRI 4.381
frozen 2026-06-12 SNPS 3.349
frozen 2026-06-12 STX 19.611
frozen 2026-06-12 TMUS 2.309
frozen 2026-06-12 TSLA 1.658
frozen 2026-06-12 TTWO 1.726
frozen 2026-06-12 TXN 2.344
frozen 2026-06-12 VRTX 2.613
frozen 2026-06-12 WDC 10.059
refreshed 2026-06-12 ADBE 0.590
refreshed 2026-06-12 BKNG 0.031
refreshed 2026-06-12 CSGP 0.484
refreshed 2026-06-12 INTU 0.412
refreshed 2026-06-12 KLAC 0.203
refreshed 2026-06-12 WDAY 0.579
refreshed 2026-06-12 AMAT 2.085
refreshed 2026-06-12 AMD 2.330
refreshed 2026-06-12 ASML 1.688
refreshed 2026-06-12 INTC 3.062
refreshed 2026-06-12 LRCX 2.224
refreshed 2026-06-12 MU 3.839
refreshed 2026-06-12 STX 3.174
refreshed 2026-06-12 TXN 1.666
refreshed 2026-06-12 WDC 3.197
```

## Validity

Every day is a second look for this feature family. The stock set changes from row to row; the counts are in each cell. The tables are a description; no threshold decides anything.

Added after the results were seen: on 2026-05-15, 2026-05-18 and 2026-06-12 the 'all' row of basis (ii) holds stocks with a corporate action between the refreshed group and the day (the main run lists them among its corporate-action left-outs); read the 'inside' row of (ii) for those days. No stock-day is left out after the fact.

cuDNN's algorithm search was off here and on in the 2019 run.

## Reproducibility

Every number above is computed from the published tables in results/ (cells.parquet, stock_days.parquet, fits.parquet, checks.parquet) by the functions of dof.experiments.analyses.

#### Stored records behind the tables

| model | basis | date | records | scored | left out | cells |
|---|---|---|---:|---:|---:|---:|
| CNN-LSTM-OF-lr3e-5-eps1e-7 | frozen | 2019-10-18 | 115 | 115 | 0 | 1150 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | frozen | 2019-10-30 | 115 | 115 | 0 | 1150 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | frozen | 2019-12-30 | 115 | 115 | 0 | 1150 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | frozen | 2020-01-30 | 115 | 115 | 0 | 1150 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | frozen | 2021-07-13 | 113 | 113 | 0 | 1130 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | frozen | 2021-08-13 | 112 | 112 | 0 | 1120 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | frozen | 2025-11-28 | 101 | 99 | 2 | 990 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | frozen | 2025-12-08 | 101 | 99 | 2 | 990 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | frozen | 2025-12-09 | 101 | 99 | 2 | 990 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | frozen | 2025-12-10 | 101 | 99 | 2 | 990 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | frozen | 2025-12-11 | 101 | 99 | 2 | 990 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | frozen | 2025-12-12 | 101 | 99 | 2 | 990 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | frozen | 2026-05-15 | 100 | 98 | 2 | 980 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | frozen | 2026-05-18 | 100 | 98 | 2 | 980 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | frozen | 2026-06-12 | 100 | 98 | 2 | 980 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | refreshed | 2021-08-13 | 112 | 112 | 0 | 1120 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | refreshed | 2025-12-11 | 101 | 98 | 3 | 980 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | refreshed | 2025-12-12 | 101 | 98 | 3 | 980 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | refreshed | 2026-05-15 | 100 | 97 | 3 | 970 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | refreshed | 2026-05-18 | 100 | 97 | 3 | 970 |
| CNN-LSTM-OF-lr3e-5-eps1e-7 | refreshed | 2026-06-12 | 100 | 97 | 3 | 970 |
| LSTM-MLP-OF | frozen | 2019-10-18 | 115 | 115 | 0 | 1150 |
| LSTM-MLP-OF | frozen | 2019-10-30 | 115 | 115 | 0 | 1150 |
| LSTM-MLP-OF | frozen | 2019-12-30 | 115 | 115 | 0 | 1150 |
| LSTM-MLP-OF | frozen | 2020-01-30 | 115 | 115 | 0 | 1150 |
| LSTM-MLP-OF | frozen | 2021-07-13 | 113 | 113 | 0 | 1130 |
| LSTM-MLP-OF | frozen | 2021-08-13 | 112 | 112 | 0 | 1120 |
| LSTM-MLP-OF | frozen | 2025-11-28 | 101 | 99 | 2 | 990 |
| LSTM-MLP-OF | frozen | 2025-12-08 | 101 | 99 | 2 | 990 |
| LSTM-MLP-OF | frozen | 2025-12-09 | 101 | 99 | 2 | 990 |
| LSTM-MLP-OF | frozen | 2025-12-10 | 101 | 99 | 2 | 990 |
| LSTM-MLP-OF | frozen | 2025-12-11 | 101 | 99 | 2 | 990 |
| LSTM-MLP-OF | frozen | 2025-12-12 | 101 | 99 | 2 | 990 |
| LSTM-MLP-OF | frozen | 2026-05-15 | 100 | 98 | 2 | 980 |
| LSTM-MLP-OF | frozen | 2026-05-18 | 100 | 98 | 2 | 980 |
| LSTM-MLP-OF | frozen | 2026-06-12 | 100 | 98 | 2 | 980 |
| LSTM-MLP-OF | refreshed | 2021-08-13 | 112 | 112 | 0 | 1120 |
| LSTM-MLP-OF | refreshed | 2025-12-11 | 101 | 98 | 3 | 980 |
| LSTM-MLP-OF | refreshed | 2025-12-12 | 101 | 98 | 3 | 980 |
| LSTM-MLP-OF | refreshed | 2026-05-15 | 100 | 97 | 3 | 970 |
| LSTM-MLP-OF | refreshed | 2026-05-18 | 100 | 97 | 3 | 970 |
| LSTM-MLP-OF | refreshed | 2026-06-12 | 100 | 97 | 3 | 970 |
