> **Edited copy.** This is a copy of a pre-registration document, written before the experiment it describes.
> The copy replaces internal file paths, internal names and process wording with neutral words, and calls the
> document's later changes "revisions"; no rule, split, model, setting, date or number is changed.
>
> - sha256 of the original: `1f1417e75640153875e0a9eebada72be1243a496932f590520092d34a5ca20a3`
> - sha256 of the text below the marker line: `73375918b57d4b9c3765ed70570f132ba7cca4ff0cf359e4397df19d825160d0`

<!-- document text below -->
# Pre-registration of experiment two: five adjacent training days, validation on a morning, test on the afternoon

This file was written on 2026-10-14, after validation results and test results of this stage existed and had been read. It selects no setting. Every setting of the fit is the one fixed in `preregistration.md` sections 5 and 6 on 2026-10-11, before any result of the stage. The split, the three seed values and the rule at the cut come from the author's question and from the approved design note (approved 2026-10-14 00:45). No score chose any of them.

This file extends `preregistration.md`. It changes neither `preregistration.md` nor `preregistration_seeds.md`. Where this file is silent, `preregistration.md` with its Revisions 1 and 2 holds.

The sha256 of this file is recorded when the runner of the experiment first starts (`adj/results/experiment.json` in the lab folder) and in every score file of the experiment, next to the sha256 of `preregistration.md`. After that, a change of either file stops every step.

## 1. What existed when this text was fixed (2026-10-14 01:32, +11:00)

- `REPORT_ALLDAYS.md`: the main run with seed 0, ended 2026-10-13 20:56. Known numbers, R² in percent, k = 1..10:
  - validation days 2025-12-11 and 2025-12-12, 98 stocks: `LSTM-MLP-OF-recent` 0.369, `LSTM-MLP-OF-alldays` 0.533, `CNN-LSTM-OF-alldays` 0.673, `Ridge-OF-alldays` -0.057, `Ridge-OF-recent` 0.003;
  - test days 2026-05-15, 2026-05-18, 2026-06-12, 96 stocks, primary analysis: -0.000, 0.196, 0.268, -0.194, -0.190, in the same order.
- The seed extension (`preregistration_seeds.md`) was running. Its fit `LSTM-MLP-OF-recent` with seed 3089492870 had passed 5 checks. Their validation losses were known: 1.085999, 1.085474, 1.085753, 1.087140, 1.087568.
- Sample counts of the main run's build: 2025-12-12 has 35,450,865 samples of 98 stocks; 11,568,582 of them are at or after 13:00.
- No fit record, weights file, checkpoint or score file of this experiment existed. The folder `adj/` of the lab folder did not exist.
- sha256 of `preregistration.md`: `babfaf66c72cd5627f6b28cf271c12c8837078722216f9c91570bafd157607e2`. sha256 of `preregistration_seeds.md`: `33f9cef8db47c6ebda964c38489f503c446f3bbe94368e9bc81e76a81e0e5e39`.

The file was put on disk at that time, before the first fit of the experiment.

## 2. Status of the days: no day of this experiment is a first look

- 2025-12-12 was a validation day of the three seed-0 fits (early stopping, together with 2025-12-11) and of the λ of both Ridge references. It is a validation day of every fit of the seed extension.
- So the afternoon of 2025-12-12 is a second look for this feature family. It is not a first observation, and no table presents it as one.
- 2025-12-11 was a validation day in the main run. Here it is a training day.
- The three 2026 days were scored on 2026-10-13 for five models. 2026-06-12 had also been scored on 2026-10-03 (`preregistration.md` Revision 2).

## 3. Question, and how the result is read

The author's expectation: a model must be trained, validated and tested on adjacent days to forecast well; a model tested five to six months later has decayed.

The experiment gives one comparison: the same saved weights on the afternoon of 2025-12-12 (one day after the last training day) and on the three 2026 days (five to six months after it).

- For each seed and each stock scored on both parts, take the stock's value on the afternoon minus its value on the 2026 days (primary analysis, k = 1..10). The mean over these stocks is the seed's difference. The mean over the three seeds is the difference D. The number of stocks in the pair is given.
- The uncertainty U of D is the root of the sum of two squares: the standard error across stocks of the per-stock difference averaged over the seeds; and the sample standard deviation across seeds of the seed differences, divided by the root of the number of seeds.
- If D is above 2U, the result agrees with the expectation. If the absolute value of D is at most 2U, the experiment does not confirm the expectation. If D is below −2U, the expectation does not hold in this experiment.
- The headline of each part is reported beside D. The verdict reads D.

What the experiment cannot show:

- It has one half day. The uncertainty holds no day effect. The three 2026 days alone differ by about 0.3 percentage points for one model (`REPORT_ALLDAYS.md`: 0.058, 0.372, 0.375). The result cannot be carried to other days.
- It is not a first look on any day (section 2).
- It cannot separate the distance in time from the difference between one December afternoon and three days of May and June (day and period effects), nor from the ageing of Δt and the scalers on the 2026 days, which is part of the deployment case. A positive D is consistent with the expectation; it does not prove it.
- The 2026 part is a different stock set (the primary analysis of Revision 1). The pair is the intersection.

## 4. The split

| part | days | rows |
|---|---|---|
| training, 5 days | 2025-11-28 (half day), 2025-12-08, 2025-12-09, 2025-12-10, 2025-12-11 | every sample |
| validation | 2025-12-12 | the samples before 13:00, by the rule below |
| test | 2025-12-12 | the samples at or after 13:00, by the rule below |
| the same weights, days already seen | 2026-05-15, 2026-05-18, 2026-06-12 | every sample |

The day 2025-12-12 is processed once, as a whole day, exactly as `preregistration.md` section 3 and Revision 1 say: session 09:40 to 15:50, the same features, labels and sample definition. Then the cut at 13:00:00 is applied to its samples.

- A validation sample is a sample whose row is before 13:00 and whose ten labels all end before 13:00. In this code the label of horizon k ends at t + h_k. The longest is h_10 = 2 Δt.
- A test sample is a sample whose 100 rows of history are all at or after 13:00. Its row and its ten labels are then at or after 13:00 too.
- A sample that is neither is used nowhere. These are the morning samples whose longest label ends at or after 13:00, and the first afternoon samples whose history reaches back before 13:00.
- A row stamped exactly 13:00:00 belongs to the afternoon. A label that ends exactly at 13:00:00 is not before 13:00.
- The order flow of the first afternoon row is the change of the book at that row's time against the book before it. It is an afternoon event.
- The validation arrays hold no row at or after 13:00. The test arrays hold no row before 13:00. They are separate files.

Scaling:

- Δt and both scalers of a stock are fitted on its five training days and on nothing else, by the scaling-group rule of `preregistration.md` section 3 with Revision 1 (the half day enters with 3.5 hours).
- No row of 2025-12-12 and no row of a 2026 day enters a scaler, a Δt or a training array.
- The validation rows, the test rows and the 2026 days use the Δt and the scalers of the stock's five-day group. On the 2026 days these are five to six months old.

Exclusion rules: rules 1 to 6 of `preregistration.md` Revision 1, with these points for the cut day.

- Rule 2 (price ratio, validation). The median mid price of 2025-12-12 is taken over its cleaned rows before 13:00. A stock that the rule leaves out is out in the morning and in the afternoon.
- Rule 5 (no sample) is applied to the morning and to the afternoon separately.
- Rule 6 (corporate actions, file `corporate_actions_20251210_20260612.csv`, sha256 `df7ef9adb2c52d4af1d112e5ae8a326d2408956ebe43e1d953e1b034f9f6ea4a`) is applied when the afternoon and the 2026 days are scored.
- On the 2026 days the primary and the secondary analysis are those of Revision 1, change 4. The price ratio is taken against 2025-12-11.

The number of stocks follows from these rules. It is given in every table.

## 5. The fit

One model: `LSTM-MLP-OF` (120 units, 128 hidden), the network of fit 1 of `preregistration.md` section 5.

- Optimizer: PyTorch Adam, learning rate 1e-5, epsilon 1e-7, betas (0.9, 0.999).
- Loss: mean squared error on the z-scored labels of all ten horizons.
- Batch: 256 samples of one stock-day; random order inside a stock-day and across batches; drawn again when a pass ends.
- Stopping rule of `preregistration.md` section 6: a check every 317,214 steps; stop after 5 checks without a new lowest validation loss; at most 50 checks; the weights of the best check are saved before any scoring.
- Validation loss: the equal-weight mean over stocks of each stock's mean squared error on z-scored labels, over its validation samples (the morning of 2025-12-12).
- The validation set is the morning of one day, about 24 million samples of 98 or 99 stocks. It is smaller than the main run's two whole days. The stopping rule is otherwise the same.
- An interrupted fit continues from its last check (Revision 1, change 5).
- The training records are read from the compact store (its design note). It changes no batch.
- In the code the fit has the stage's name `LSTM-MLP-OF-recent`. Its training days are the five days above.

Nothing is tuned. No setting differs from `preregistration.md` except the training days and the validation set.

Seeds:

- Three seeds, in this order: **3089492870**, **771810125**, **18531247**. Seed 0 is not used.
- The first two were drawn on 2026-10-12 with `secrets.randbits(32)` (`preregistration_seeds.md` section 3; `proofs/seeds/seed_draw_20261012.txt` in the lab folder).
- The third was drawn on 2026-10-13 at 21:36:52 in the same way (`seed_third_draw_20261013.txt` in the lab folder).
- All three were fixed before any fit of this experiment.
- The first two are also seeds of the seed extension. A fit here starts from the same initial weights as the extension's fit of this network with the same seed. Its training set and its validation set differ.
- No rule adds a seed. The standard deviation across the three seeds is reported as it is.
- A fit whose loss is not finite is recorded as diverged. It is not scored. The statistics are then over the seeds that ended, and the count is given.

## 6. The linear reference

One pooled ridge regression on the CPU, `Ridge-OF` on the five training days, by the method of `preregistration.md` section 7. λ is chosen from (0.001, 0.01, 0.1, 1, 10) by the validation loss of section 5 on the morning of 2025-12-12. It has no seed.

## 7. The gate and the scoring

- The arrays of the afternoon and of the 2026 days are built before they are needed. Building computes features and labels with the frozen Δt and scalers. It fits nothing and scores nothing.
- No afternoon array of 2025-12-12 and no array of a 2026 day is read by a model or by a scoring step before all three fits and the Ridge reference have ended and their saved weights have the sha256 their records name. The code refuses: it checks the fit records, the two pre-registration hashes and the weights of all of them, and then asks the stage's own test-day gate for each model.
- Then, once each:
  1. the three fits and the Ridge reference on the afternoon of 2025-12-12;
  2. the selection-bias reference (below);
  3. the same weights on 2026-05-15, 2026-05-18 and 2026-06-12.
- A score file that exists is not written again. A score file that does not match the present files stops the run.
- Nothing is changed after the scoring.

Selection-bias reference:

- The three seed-0 fits of the main run (`LSTM-MLP-OF-recent`, `LSTM-MLP-OF-alldays`, `CNN-LSTM-OF-alldays`) are scored once on the afternoon of 2025-12-12, by the same rule at the cut.
- They use their own arrays: the Δt and scalers of the four days to 2025-12-10. The cleaned rows are the same as in this experiment; the label horizons differ slightly.
- These fits stopped on 2025-12-11 and 2025-12-12 as whole days, so this afternoon was inside their validation set.
- Their numbers are in a table of their own. No decision uses them. No number of this table enters D.

## 8. Metrics and uncertainty

- Cell, stock value, headline and standard error across stocks: `preregistration.md` section 8. The cell of a half day is computed over the scored rows of that half; its benchmark is the mean label of those rows.
- Headline of a part: R² with k = 1..10 over the days of the part. For the 2026 days: the primary analysis, all three days.
- Across seeds: the mean of the three headlines and their sample standard deviation (3 - 1 in the denominator). With three seeds it is a rough number.
- Uncertainty of a part, in the tables: the root of the sum of two squares. The first is the standard error across stocks of the stock values averaged over the seeds. The second is the standard deviation across seeds divided by the root of the number of seeds.
- The verdict uses the paired difference D and its uncertainty U of section 3, over the stocks scored on both parts. Their count is given.
- Stocks move together, so the standard error across stocks is too small. The uncertainty holds no day effect.
- Also reported: horizon k = 10 alone; each 2026 day alone; the secondary analysis of the 2026 days.

## 9. The report

`REPORT_ADJ.md`. Every value carries its part of the day, its days, its number of stocks, its horizon and its seeds. The validation table and the test tables are apart. The selection-bias reference has its own table. The report holds the log lines of the gate's refusals, the per-check record of each fit, and the sha256 of this file, of `preregistration.md` and of the corporate-action file.

## 10. Checks before the first fit

- The whole code path runs on a tiny split made of training days of `preregistration.md` (UAL and FAST; 2019-01-30, 2019-03-27, 2019-07-30; cut day 2019-08-30; far day 2019-10-18), with the three seeds, on the CPU and then on the GPU. It reads no day of this experiment.
- Proofs on the real split, on the CPU, without reading an afternoon array:
  - the training cache names only the five days;
  - every Δt equals the total regular trading time over the total count of the five days;
  - a book root without 2025-12-12 and without the 2026 days gives the same scalers, Δt and training arrays;
  - the validation arrays equal a build from the book cut at 13:00;
  - the counts at the cut add up for every stock.
- The gate's refusal is logged twice in the real run: before the first fit, and after the first fit while two fits are missing.

## 11. Deviation from `preregistration_seeds.md` section 5

Decision of 2026-10-14: the runner of the seed extension is stopped when this experiment is ready, and started again after it has ended. The fit that is interrupted continues from its last check. The order of the six fits of the extension does not change; their times do. No pre-registration file is edited. The times of the stop and of the new start are in `seeds.log`.

## 12. Local choices

These are not in `preregistration.md`. They were fixed before any fit of the experiment.

- The cut at 13:00:00, and the rule at the cut (section 4). The approved design note fixed both.
- The median mid price before 13:00 for rule 2 on the cut day.
- The combination of the two uncertainty terms as a root of a sum of squares, and the factor two of section 3 (from the approved design note).
- The paired difference across stocks for the verdict (section 3), as `preregistration.md` section 8 pairs fit 2 with fit 1.
- One Ridge reference in place of two.
- Numbers that change no result: worker processes, memory budgets, the pace of the compact conversion.
