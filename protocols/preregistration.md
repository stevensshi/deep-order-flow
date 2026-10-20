> **Edited copy.** This is a copy of a pre-registration document, written before the experiment it describes.
> The copy replaces internal file paths, internal names and process wording with neutral words, and calls the
> document's later changes "revisions"; no rule, split, model, setting, date or number is changed.
>
> - sha256 of the original: `babfaf66c72cd5627f6b28cf271c12c8837078722216f9c91570bafd157607e2`
> - sha256 of the text below the marker line: `7aebf97ab9bfda54aa85051d4ea234872faa0ec461dacbc195d785d77305c7de`

<!-- document text below -->
# Pre-registration: one pooled model on all held trading days

Written 2026-10-11, before any test-day data was read. No file of a test-day folder had been opened when this file was written; only the folder names and file counts were listed.

The sha256 of this file is recorded in every result file. A result made under another sha256 is refused by the scoring step.

Revised twice, on 2026-10-11: see "Revision 1" and "Revision 2" at the end. Where a revision and sections 1 to 9 differ, the revision holds. Revision 2 corrects the first sentence above for one test day.

## 1. Question

The pooled models of an earlier stage were trained on 3 days and overfit. This stage trains on every training day we hold and keeps only validation and test apart (decision of 2026-10-11).

Two questions are answered:

1. What does a pooled model give on the 3 test days when it is trained on all 14 training days?
2. Do the older days (2019 to 2021) help? Fit 2 (14 days) is compared with fit 1 (the four 2025 days).

## 2. The split

| part | days |
|---|---|
| training, 14 days | 2019-01-30, 2019-03-27, 2019-07-30, 2019-08-30, 2019-10-18, 2019-10-30, 2019-12-30, 2020-01-30, 2021-07-13, 2021-08-13, 2025-11-28, 2025-12-08, 2025-12-09, 2025-12-10 |
| validation, 2 days | 2025-12-11, 2025-12-12 |
| test, 3 days | 2026-05-15, 2026-05-18, 2026-06-12 |

- The split is chronological. Every training day is before every validation day. Every validation day is before every test day.
- Every stock held on a day is used on that day, unless a rule of section 4 leaves it out.
- The stock sets differ by year. The model is pooled and has no stock identity input.
- 2025-11-28 is a half trading day. It is processed like every other day: session 09:40 to 15:50, and the same horizon formula.
- The universe is the 115 tickers of the universe file of an earlier replication (its ticker column is `configs/universe_115.csv` in this repository). A stock is held on a day when the day's `build_log.json` gives it the status `valid` and it has at least one cleaned row in the session.

## 3. Features, labels and scaling

Features and labels come from the present code, unchanged: the feature module of the reference implementation and the functions of its baseline module that the earlier stage's data module calls.

- Convention `cks`, same-timestamp rows deduplicated, session 09:40 to 15:50.
- Input: the last 100 cleaned rows of the 20 order-flow columns.
- Label for horizon k = 1..10: mid(t + h_k) − mid(t + 10 ms) in dollars, with h_k = k/5 × Δt.
- A sample is a row with a full 100-row history and a valid label on all ten horizons.

### Scaling groups

Share counts and price levels change across years. One scaler per stock over all training days would mix them. So the scalers and Δt are fitted per scaling group.

- A scaling group is a run of consecutive held training days of one stock.
- A new group starts at a training day when one of these holds against the stock's previous held training day:
  - the gap is longer than 300 calendar days;
  - the ratio of the two days' median mid prices is outside [0.6, 1.6].
- The median mid price of a day is the median of the mid price over the cleaned session rows of that day.
- Per (stock, group), from that group's training days only:
  - Δt = 2.34e7 ms / (mean over the group's days of the day's count of mid-price changes between 09:30 and 16:00), as `features.estimate_delta_t`;
  - the feature scaler, fitted on the group's sample rows: per column, clip at the 0.5 % and 99.5 % quantiles, then z-score, as `features.fit_scaler`;
  - the label scaler: the same, per horizon.
- A validation or test day of a stock uses the Δt and the scalers of the stock's latest training group. The latest group is the group that holds the stock's last held training day.
- Nothing of a validation or test day enters a scaler, a Δt or a training array.

## 4. Exclusion rules

All rules are fixed here. The report lists every stock-day that a rule leaves out, with the rule.

1. Reused tickers. GLIBA and FB are left out on every 2026 day (an earlier decision: other companies reused the tickers).
2. Price discontinuity. A validation or test stock-day is left out when the ratio of its median mid price to the median mid price of the last day of the stock's latest training group is outside [0.6, 1.6].
3. Group not usable for training. A (stock, group) is left out of training when one of these holds:
   - the stock has no mid-price change on the group's days (Δt is not defined);
   - a horizon is at or below the 10 ms latency buffer (Δt ≤ 50 ms). The networks output all ten horizons jointly. This is the rule of the earlier stage's data module;
   - the group has no sample row.
4. No usable group. A validation or test stock-day is left out when the stock has no training group, or when its latest training group is left out by rule 3. There is no fallback to an older group.
5. No samples. A validation or test stock-day without a sample row is left out.

## 5. The three fits

Models and optimizers are those of the earlier stage. Nothing is tuned in this stage.

| fit | name | network | training days | optimizer |
|---|---|---|---|---|
| 1 | `LSTM-MLP-OF-recent` | LSTM-MLP-OF (120 units, 128 hidden) | the four 2025 training days | PyTorch Adam, learning rate 1e-5, epsilon 1e-7, betas (0.9, 0.999) |
| 2 | `LSTM-MLP-OF-alldays` | LSTM-MLP-OF | all 14 training days | as fit 1 |
| 3 | `CNN-LSTM-OF-alldays` | CNN-LSTM-OF | all 14 training days | Adam in the Keras form, learning rate 3e-5, epsilon 1e-7, betas (0.9, 0.999) |

Common settings:

- One seed: 0. The cuDNN algorithm search is on, as before, so a repeated run is not bit-identical.
- Loss: mean squared error on the z-scored labels of all ten horizons.
- A batch is 256 samples of one stock-day. The samples of a stock-day are in random order. The batches of all stock-days are in random order. The order is drawn again when a pass over the training set ends.
- Fit 1 uses the same scaling groups and the same validation set as fits 2 and 3.
- The fits run in the order 1, 2, 3.

## 6. The stopping rule

- A check is taken every 317,214 training steps. This is one pass over the old 3-day training set.
- At a check, the validation loss is read: the equal-weight mean over stocks of each stock's mean squared error on z-scored labels, over the sample rows of both validation days.
- Training stops after 5 checks without a new lowest validation loss. It stops after 50 checks at the latest.
- The weights of the check with the lowest validation loss are restored and saved before any scoring.
- The batch stream does not restart at a check. It continues through the pass.
- A fit whose training loss is not finite is recorded as diverged. It is not scored on the test days.

## 7. The linear references

Two pooled ridge regressions on the CPU, as the earlier stage's pooled ridge command does it.

| name | training days |
|---|---|
| `Ridge-OF-alldays` | all 14 training days |
| `Ridge-OF-recent` | the four 2025 training days |

- Input: the same 100 × 20 window, flattened. Target: the same z-scored labels. The scaling groups are the same.
- λ is chosen from (0.001, 0.01, 0.1, 1, 10) by the lowest validation loss of section 6.

## 8. The metrics

- Cell: the out-of-sample R² of one stock, one day and one horizon. R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)², with ȳ the mean label of that stock-day. Predictions go back to dollars with the group's label scaler. This is the scoring function of the earlier stage.
- Stock value: the mean of the cells of one stock over its used days and over the horizons.
- Headline: the mean of the stock values over the stocks, for the 3 test days and the 10 horizons.
- Standard error: the sample standard deviation of the stock values, divided by the square root of the number of stocks.
- Also reported, each with its standard error across stocks:
  - horizon k = 10 alone;
  - each test day alone.
- The same numbers are computed on the 2 validation days. Validation numbers and test numbers are never put in one table.
- A cell whose labels have zero variance has no R². It is left out of the means, counted and listed.
- Does the older data help: for each stock scored under fit 1 and fit 2, the difference of the two stock values (fit 2 minus fit 1). The mean difference and its standard error across stocks are reported, on validation first and on test second. No threshold decides anything. Stocks move together, so the standard errors are too small.
- Every R² in the report carries its days, its number of stocks and its horizon.

## 9. The test days

- The test days are scored once, at the very end, after all three fits and both linear references have ended and their weights are saved.
- They are scored for exactly these five models: `LSTM-MLP-OF-recent`, `LSTM-MLP-OF-alldays`, `CNN-LSTM-OF-alldays`, `Ridge-OF-alldays`, `Ridge-OF-recent`.
- No other model is scored on them. Nothing is changed after they are scored.
- The code refuses to load a test-day array unless the fit it scores has ended and its saved weights match the hash in its fit record.
- The feature caches of the test days are built after this file exists. Building them computes features and labels with the frozen Δt and scalers of the training groups. It scores nothing.
- The smoke test and the placebo have their own small split made of training days of this split. They never read a validation or test day of this split.

## Revision 1 (2026-10-11), before any test-day array was loaded

### What was known when the changes were decided

- No fit had been run. No model existed. No validation score and no test score existed.
- The caches of the first version had been built, the test caches too. The build computes features and labels. No model and no scoring step had loaded a test-day array: the test-day gate refused every attempt.
- Of the test days these facts were known: per day the number of stocks and of samples; per stock the ratio of the test day's median mid price to that of 2025-12-10; the list of the 38 test stock-days that the price rule of the first version left out.
- Change 4 was decided with that list in view. Changes 1 to 3 rest on facts of the training days only.
- The first version of this file has the sha256 `2c4f2a119cfe43de1676c12849514747541297035c5527df3884a747c303a6a5`.

### The changes, each with its reason

1. **Half trading day.** 2025-11-28 closed at 13:00. It is the only half day among the 19 held days.
   - On a half day the session is 09:40 to 12:50. A sample needs its row and the end of all ten labels inside that session, as on a full day.
   - Δt of a group is the total regular trading time of the group's training days divided by the total number of mid-price changes of those days in regular trading time. A full day enters with 6.5 hours (09:30 to 16:00), a half day with 3.5 hours (09:30 to 13:00). For a group of full days this is the formula of section 3.
   - The day stays a training day.
   - Reason: under the first version the rows after the 13:00 close were samples, and the day's lower count of mid-price changes made Δt of every 2025 group longer than its full days give (by a factor of 1.13 at the median).
2. **Groups at or below 50 ms.** Rule 3 is not changed and has no exception. It is applied after change 1. A group that it leaves out is out on its training days, and the stock is then out on the validation days and on the test days (rule 4).
   - Reason: under the first version one group passed the 50 ms limit only because the half day lengthened its Δt.
3. **FB and GLIBA.** Both tickers are left out on every day from 2025-01-01 on: training, validation and test. They stay in on the 2019 to 2021 days.
   - Reason: in 2025 and 2026 the two tickers belong to other issuers. They are not stocks of this universe.
4. **Exclusions on the test days.** The price-ratio rule was meant to catch share splits. On the test days it also removed stocks whose price had moved much since 2025-12-10. That biases the test set.
   - The price-ratio rule no longer leaves out a test stock-day. It still forms the scaling groups inside the training days, and it still leaves out validation stock-days (rule 2).
   - Primary analysis. A stock is left out on a test day only by rule 1, rule 4, rule 5 or the new rule 6. Rule 6: the corporate-action file `corporate_actions_20251210_20260612.csv` has an event for the ticker with the status `confirmed`, a type of `forward_split`, `reverse_split`, `ticker_change` or `ticker_reused`, and an effective date on or before that test day.
   - The corporate-action file is written by another party from official sources. Its effective date is the first trading day on the new scale. It is read when the test days are scored. Its sha256 is recorded in every test result. Without the file nothing is scored.
   - The file was delivered on 2026-10-11, after this revision was drafted and before any fit was run. Its sha256 is `df7ef9adb2c52d4af1d112e5ae8a326d2408956ebe43e1d953e1b034f9f6ea4a`. The scoring step refuses a file with another sha256.
   - A stock with a large price move and no such event stays in the primary analysis. It is scored with the Δt and the scalers fitted on the training days up to 2025-12-10. These are five to six months old on the test days.
   - Secondary analysis. The scores of the primary analysis, restricted to the stock-days whose ratio of median mid price against the last day of the stock's latest training group is inside [0.6, 1.6]. This is the test set of the first version, less the stock-days that the primary analysis leaves out.
   - Both analyses are reported, the primary one first. The headline of section 8 is the primary analysis. The comparison of fit 2 with fit 1 is made in both.
   - The test caches are built for every held test stock-day that rules 1, 4 and 5 do not leave out. The exclusions of rule 6 are applied when the test days are scored.
5. **Restart.** An interrupted fit continues from its last check. After each check the weights, the optimizer state, the best weights so far, the per-check record, the counters of the stopping rule, and the state and position of the batch stream are saved. The batch stream continues at the same batch. No rule of section 6 changes.
   - Reason: one fit can run for days.

### The exclusion rules after Revision 1 (they replace section 4)

1. Reused tickers. GLIBA and FB are left out on every day from 2025-01-01 on.
2. Price discontinuity, validation days only. A validation stock-day is left out when the ratio of its median mid price to the median mid price of the last day of the stock's latest training group is outside [0.6, 1.6].
3. Group not usable for training. A (stock, group) is left out of training when the stock has no mid-price change on the group's days, when Δt is at or below 50 ms, or when the group has no sample row.
4. No usable group. A validation or test stock-day is left out when the stock has no training group, or when its latest training group is left out by rule 3. There is no fallback to an older group.
5. No samples. A validation or test stock-day without a sample row is left out.
6. Corporate action, test days only, applied at scoring: see change 4.

Everything else in sections 1 to 9 stands: the split, the features and labels, the scaling-group rule, the three fits, the stopping rule, the linear references, the metrics, and the rule that the test days are scored once, for the five models, after all of them have ended.

## Revision 2 (2026-10-11), before the run started: one test day is not a first look

An independent review of this stage, made before any fit of the real stage started, found that the first sentence of this file is true for this stage's code and not true for the project as a whole.

- 2026-06-12 was used before. On 2026-10-03 an earlier experiment (a forward validation of an earlier replication) scored 40 per-stock Ridge-OF models on that day (20 stocks, two sign conventions, trained on 2025-12-08 to 2025-12-10), with the same input, the same label and the same R². The author saw those scores before this stage was designed.
- 2026-05-15 and 2026-05-18 had not been used by any model before this stage.
- Consequence: the 2026-06-12 column of every table is a second look for this feature family. The two May days are first looks. The report gives each test day alone and states this in its limits.
- No rule, split, model or setting of this stage changes with this revision.
- Also recorded: the caches were built before the last edits of the stage's configuration module and of this file; the build summary names an earlier sha256 of this file. The review re-derived 7 stocks (100 stock-days) from the book data with separate code and found the cached arrays bit-identical. No array was rebuilt after that.
