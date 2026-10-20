> **Edited copy.** This is a copy of a pre-registration document, written before the experiment it describes.
> The copy replaces internal file paths, internal names and process wording with neutral words, and calls the
> document's later changes "revisions"; no rule, split, model, setting, date or number is changed.
>
> - sha256 of the original: `ad9015c9c6fca945ad22fe8fd2e276d9f7d9bf05000e3a2d14b783267152a97b`
> - sha256 of the text below the marker line: `215c87e9e0097482f0c102aeac10ec288a642cb378f92c8c8b9789fcda0b4a28`

<!-- document text below -->
# Pre-registration of the decay measurement: two models of 2019 scored on later days

This file was written on 2026-10-14, after validation results and test results of this stage existed and had been read. It selects no setting and fits no model. The two models, their saved weights, the days, the two bases and the rules of section 7 come from the author's question, from the approved design note (approved 2026-10-14 00:45, with its revisions 2 and 3) and from the design decisions of 2026-10-14. No score chose any of them.

This file extends `preregistration.md`. It changes no pre-registration file. Where it is silent, `preregistration.md` with its Revisions 1 and 2 holds.

The sha256 of this file is recorded when the measurement first starts (`decay/begin.json` in the lab folder) and in the file of every scored stock-day. After that, a change of this file stops every step.

## 1. What existed when this text was fixed (2026-10-14 06:16, +11:00)

- `REPORT_ALLDAYS.md`: the main run with seed 0. Known numbers, R² in percent, k = 1..10: validation days 2025-12-11 and 2025-12-12, 98 stocks: 0.369, 0.533, 0.673 for the three networks; test days 2026-05-15, 2026-05-18, 2026-06-12, 96 stocks, primary analysis: -0.000, 0.196, 0.268.
- The stored scores of the two models that are measured here, 115 stocks, k = 1..10, seed 0:
  - `CNN-LSTM-OF-lr3e-5-eps1e-7` (its `pooled_seed0.json`, sha256 `5266cfc38aa38b57db630f697c388b3a42898291c488eab3da1a0862ca079d9f`): validation day 2019-08-30: 1.242; test days 2019-10-18, 2019-10-30, 2019-12-30: 1.136 (by day 1.427, 0.928, 1.051);
  - `LSTM-MLP-OF` (its `pooled_seed0.json`, sha256 `293f8ad6ead5b80f69e166a064a41045ddb25928d60f286ca2f12131033f0551`): validation day 2019-08-30: 1.245; the same test days: 1.114 (by day 1.449, 0.959, 0.932).
  These three test days are decay days here. Their scores were known.
- Experiment two (`preregistration_adjacent.md`) was running. Its fit `LSTM-MLP-OF-recent` with seed 3089492870 had passed 4 checks. Their validation losses were known: 1.253335, 1.251897, 1.253467, 1.251693. `REPORT_ADJ.md` did not exist: experiment two had no test score.
- The seed extension (`preregistration_seeds.md`) was not running; its log ended with `SEEDS_FAILED`. Its fit `LSTM-MLP-OF-recent` with seed 3089492870 had ended after 7 checks (best check 2); its validation scores were in the log. The validation losses of its checks were known: 1.085999, 1.085474, 1.085753, 1.087140, 1.087568, 1.088155, 1.089095.
- Counts made before this file, from median mid prices and with no model: the number of stocks whose price ratio of section 7 against 2019-07-30 is inside [0.6, 1.6] is 115, 115, 113 and 114 of 115 on the four days to 2020-01-30; 64 of 113 and 68 of 112 on the two days of 2021; 44 of 99 on the four days 2025-11-28 to 2025-12-10; 43 and 44 of the 98 stocks with a stored median on 2025-12-11 and 2025-12-12; 43, 44 and 42 of the 97 stocks with a stored median on the three days of 2026.
- No score of either model on a decay day other than its three 2019 test days existed. The folder `decay/` of the lab folder held no file of a scored stock-day.
- sha256 of `preregistration.md`: `babfaf66c72cd5627f6b28cf271c12c8837078722216f9c91570bafd157607e2`. sha256 of `preregistration_seeds.md`: `33f9cef8db47c6ebda964c38489f503c446f3bbe94368e9bc81e76a81e0e5e39`. sha256 of `preregistration_adjacent.md`: `1f1417e75640153875e0a9eebada72be1243a496932f590520092d34a5ca20a3`.

## 2. Status of the days: every day is a second look for this feature family

- 2019-10-18, 2019-10-30, 2019-12-30: the test days of both models. Scored for them in 2026-10. They are also training days of the main run's 14-day models.
- 2020-01-30, 2021-07-13, 2021-08-13, 2025-11-28, 2025-12-08, 2025-12-09, 2025-12-10: training days of the main run's models. The two models were not trained on them.
- 2025-12-11: a validation day of the main run and of the seed extension; a training day of experiment two.
- 2025-12-12: a validation day of the main run and of the seed extension; in experiment two its morning is the validation set and its afternoon the test set.
- 2026-05-15, 2026-05-18, 2026-06-12: the test days of the stage, scored on 2026-10-13 for five models. `preregistration.md` section 9 says that no other model is scored on them. This measurement scores two more models there. It is outside that list, it was decided after those scores were read, and every table says so. `preregistration.md` section 9 binds the models of that stage. This measurement adds no model to that stage's comparison, and no number of it enters `REPORT_ALLDAYS.md` or `REPORT_SEEDS.md`.

## 3. Question, and how the tables are read

The author's question: take a model trained in 2019 and score it up to 2025 and 2026; how much does it decay?

- The answer is one table per model: the days in rows, ordered by the time since the models' last training day, 2019-07-30. Two columns of R², one per basis of section 6.
- The difference of the two bases on one day, over the stocks scored on both, is the part of the loss that stale scalers and a stale Δt explain.
- The tables are a description. No threshold decides anything, and no setting is chosen from them.

Each day has two rows per basis: 'all' (every held stock-day) and 'inside' (the stock-days whose price ratio of section 7 is inside [0.6, 1.6]). The 'all' row of basis (i) is the frozen dollar model on everything it would meet in use, splits included; the 'inside' row removes the nominal-price effect and is a selected set. On basis (ii) the refreshed label scaler removes the scale problem, so the 'all' row of (ii) is the cleanest measure of the weights alone.

What it cannot show:

- It has two models and one seed each. The pooled `LSTM-OF` of the same run also has saved weights (`checkpoints/pooled115/`); it is not measured here, because no later stage uses that network.
- A row is one day. Days differ by about 0.3 percentage points for one model (`REPORT_ALLDAYS.md`: 0.058, 0.372, 0.375). The tables have between one and six days per period.
- The stock set changes from row to row (sections 5 and 7). Each cell gives its number of stocks.
- It cannot separate the distance in time from the difference between the periods.
- Each model was trained on three days of 2019. Its own test level on 115 stocks, 1.136 for the CNN-LSTM and 1.114 for the LSTM-MLP, is the zero-distance reference of its table; the paper's level is not.

## 4. The two models

Both are pooled models of the earlier stage, seed 0, trained on 2019-01-30, 2019-03-27 and 2019-07-30 on 115 stocks and validated on 2019-08-30.

- `CNN-LSTM-OF-lr3e-5-eps1e-7`: the CNN-LSTM-OF network; Adam in the Keras form, learning rate 3e-5, epsilon 1e-7; its best epoch was the first of six. Saved weights: `results/pooled115/CNN-LSTM-OF-lr3e-5-eps1e-7/best_weights_seed0.pt`, sha256 `14ee1f217e32613f28c8129584452496a32e210fdd48042711d6e2fe6e1753b7`.
- `LSTM-MLP-OF`: the LSTM-MLP-OF network (120 units, 128 hidden), the network of experiment two and of fits 1 and 2 of the main run; PyTorch Adam, learning rate 1e-5, epsilon 1e-7; its best epoch was the fourth of nine. Saved weights: `checkpoints/pooled115/LSTM-MLP-OF_seed0.pt`, sha256 `035ef6b645f382a6d0fa2e0895c5b87f408dd14304f6304d2e0e19e1ba7f0055`, written by the 2019 run after it had restored the best epoch and scored the model.
- The code refuses another file for either model.
- Nothing is fitted and nothing is tuned. The measurement is forward passes of these weights. It has no random draw.
- One build of a stock-day serves both models.

## 5. The days

| day | days since 2019-07-30 | months | bases | behind the gate of section 9 |
|---|---:|---:|---|---|
| 2019-10-18 | 80 | 2.6 | (i) | no |
| 2019-10-30 | 92 | 3.0 | (i) | no |
| 2019-12-30 | 153 | 5.0 | (i) | no |
| 2020-01-30 | 184 | 6.0 | (i) | no |
| 2021-07-13 | 714 | 23.5 | (i) | no |
| 2021-08-13 | 745 | 24.5 | (i), (ii) | no |
| 2025-11-28 (half day) | 2313 | 76.0 | (i) | no |
| 2025-12-08 | 2323 | 76.3 | (i) | no |
| 2025-12-09 | 2324 | 76.3 | (i) | no |
| 2025-12-10 | 2325 | 76.4 | (i) | no |
| 2025-12-11 | 2326 | 76.4 | (i), (ii) | no |
| 2025-12-12 | 2327 | 76.4 | (i), (ii) | yes |
| 2026-05-15 | 2481 | 81.5 | (i), (ii) | yes |
| 2026-05-18 | 2484 | 81.6 | (i), (ii) | yes |
| 2026-06-12 | 2509 | 82.4 | (i), (ii) | yes |

A month is 30.44 days. A stock is scored on a day when the day's book holds it as valid and no rule of section 7 leaves it out.

## 6. The two bases

Features, labels and samples of a day are made by the present code, as `preregistration.md` section 3 and Revision 1 say. A basis is the Δt and the two scalers that are used for a stock.

- (i) Frozen. The stock's Δt and scalers of 2019, fitted on the models' three training days and stored then. This is the model as it would run without any update.
- (ii) Refreshed. Δt and scalers fitted on days shortly before the scored day that the models were not trained on, for the same stock:
  - for 2021-08-13: fitted on 2021-07-13 alone, by the scaling-group rule of `preregistration.md` section 3;
  - for 2025-12-11, 2025-12-12 and the three days of 2026: those of the main run's group of 2025-11-28 to 2025-12-10, as they are stored.
- The other days have basis (i) only.
- On basis (ii) the label horizons are those of the refreshed Δt. So the labels of the two bases differ on the same day. The tables say so.
- (ii) minus (i), over the stocks scored on both bases on that day, is the part that stale scalers and a stale Δt explain.

The stored inputs are pinned by sha256 in `decay/begin.json`: the 345 files of the 2019 scalers and metas, the main run's plan and the 295 files of its 2025 groups, and the files of the 2021-07-13 fit.

## 7. Exclusions

Fixed here. The report gives, per day and basis, the number of stock-days each rule leaves out.

1. Reused tickers. GLIBA and FB are left out on every day from 2025-01-01 on.
2. Price range, recorded, not excluding. For every stock-day the ratio of its median mid price to the median mid price of the basis's reference day is recorded: 2019-07-30 for basis (i), the last day of the refreshed group for basis (ii). Every stock-day is scored. The tables give two rows per day and basis: all stock-days, and those with the ratio inside [0.6, 1.6]. The counts of both are given.
3. No usable Δt. On basis (ii) a stock is left out when it has no refreshed group, when that group is left out by rule 3 of `preregistration.md` (Δt at or below 50 ms), or when the group's last day is more than 300 days before the scored day.
4. No sample. A stock-day without a sample row is left out.

What rule 2 does and does not do. It compares nominal prices. A model that works in dollars has no meaning across a share split, and the rule is meant for that. Over six years the 'inside' row also leaves out stocks whose price only moved, and it can keep a stock that split and whose price came back into the range. An example from the held data: NVDA's median mid price falls to 0.247 of its value between 2021-07-13 and 2021-08-13, and its ratio of 2025-12-10 to 2019-07-30 is 1.05. A second example: AAPL split 4-for-1 in 2020; its ratio to 2019-07-30 is 0.70 on 2021-07-13 and 1.34 on 2025-12-10, inside the range both times. So the 'inside' row is a selected set with split artefacts inside it, and the 'all' row carries the nominal-price effect in full. A list of splits from official sources (2019 to 2025) does not exist in this project today. If it is made later, a third row, 'no split since 2019-07-30', may be added from the stored per-stock scores without scoring anything again; that addition is dated and labelled as made after these results.

## 8. Metrics

- Cell, stock value and standard error across stocks: `preregistration.md` section 8. R² is in dollars against the mean label of the stock-day; predictions go back to dollars with the basis's label scaler.
- Table value: the mean over stocks of the stock values for k = 1..10 on that day, basis and row. Horizon k = 10 alone is also given.
- (ii) minus (i): for each stock scored on both bases on that day, the stock's value on (ii) minus its value on (i); the mean and its standard error across those stocks.
- (ii) minus (i) is given for both rows. The stocks of a pair are those scored on both bases on that day, in that row.
- A cell whose labels have zero variance has no R². It is left out of the means and counted.
- Stocks move together, so the standard errors are too small.

## 9. The gate

2025-12-12 and the three days of 2026 are behind the gate of experiment two (`preregistration_adjacent.md` section 7). The code of this measurement opens no file of these days (book, build log or array) until the three fits of experiment two and its Ridge reference have ended and their saved weights have the sha256 their records name. The counts of section 1 for these four days come from the main run's stored build metas, read on 2026-10-14. The code calls experiment two's own gate. The other eleven days may be scored before that.

## 10. Checks

- Before any decay day is scored, each model is scored on the 2019 validation day, 2019-08-30, for all 115 stocks, with the code of this measurement. Every cell must equal that model's stored cell within 1e-4 of R² as a fraction (0.01 percentage points), with the same number of samples. Otherwise nothing is scored.
- After the three 2019 test days are scored, their 'all' rows must hold the same 115 stocks as the stored scores, and every cell must agree in the same way. This holds for each model.
- The whole code path runs first on a tiny split (UAL and FAST on 2019-03-27, 2019-07-30 and 2019-08-30). It reads no decay day.
- A stock-day is scored once per model. Its file is never written again.

## 11. When it runs

The measurement runs only while no training fit of this project holds the card (design note, revision 3). Its runner does not start, and it stops itself within a minute, when a training runner of this stage is alive. It takes no training lock. The other project's job may be on the card: the measurement stays under an allocator cap of 3 GiB with cuDNN's algorithm search off, and it starts a stock-day only when memory is free and the card has room. It can be stopped and started again; a scored stock-day is kept.

## 12. The report

`REPORT_DECAY.md`: one table per model on both bases, with the two rows of section 3; the counts of section 7; the section "Existing evidence" with the wording of the design note's section 0; the existing numbers 1.136, 0.673 and 0.268 with their differences of basis listed apart; the checks of section 10; the measured card memory and time. The report gives, per day and basis, the number of stock-days in each row and the number of stock-days with no R² (zero label variance). Every number carries its day, its row, its number of stocks, its horizon, its basis, its model, "seed 0".

## 13. Local choices

These are not in `preregistration.md` or in the design note. They were fixed before any decay day was scored.

- The three 2019 test days are scored again here, so that every row comes from one code path.
- The reference day of the price ratio for each basis (section 7), and the 300 days of rule 3. Scoring every stock-day and showing two rows, rather than excluding by the ratio (the design note said the outside count is listed; here it is also scored).
- The whole day 2025-12-12 waits behind the gate, not only its afternoon.
- The tolerance of section 10: 1e-4 of R² as a fraction.
- cuDNN's algorithm search is off; it was on when the models were trained and scored in 2019's run.
- A month of 30.44 days.
- Numbers that change no result: the allocator cap, the memory and card rules of section 11.
