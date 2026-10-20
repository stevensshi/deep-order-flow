> **Edited copy.** This is a copy of a pre-registration document, written before the experiment it describes.
> The copy replaces internal file paths, internal names and process wording with neutral words, and calls the
> document's later changes "revisions"; no rule, split, model, setting, date or number is changed.
>
> - sha256 of the original: `33f9cef8db47c6ebda964c38489f503c446f3bbe94368e9bc81e76a81e0e5e39`
> - sha256 of the text below the marker line: `829d9c253d8f9c071f1080e9b61818b3c44d329ba8438c32b4a29117341c7b92`

<!-- document text below -->
# Pre-registration of the seed extension: more seeds for the three fits

Decision of 2026-10-12: one seed is not enough. This file extends `preregistration.md`. It does not change it. Where this file is silent, `preregistration.md` with its Revisions 1 and 2 holds.

The sha256 of this file is recorded in every result file of the extension, next to the sha256 of `preregistration.md`.

## 1. When this file was written, and what existed

First written on 2026-10-12 at 00:10 (+11:00) with the seed values 1 and 2. Rewritten on 2026-10-12 at 00:17 after two decisions recorded on 2026-10-12: the seed values are drawn (section 3), and a rule decides about two more seeds (section 7). Between the two versions no fit of the extension was started and its runner was not started. The first version had the sha256 `7cff946840103b04d47d198528b727645255fab14bee8c3233f6aafd216e421c`.

At both times the main run (`run_alldays.sh`, started 2026-10-11 14:09) was alive. Fit 1 of seed 0 had ended. Fit 2 of seed 0 (`LSTM-MLP-OF-alldays`) was training; at 00:16 it was in the validation reading of its fourth check (step 1,268,856). No test-day score of this stage existed.

Corrected on 2026-10-12 at 00:36, before the runner of the extension was first started. Two things changed: the time of the rewrite above (it read 00:20; the file was written at 00:17), and this paragraph was added. No rule, no seed value and no other sentence changed. The version of 00:17 had the sha256 `972f14c1596298996d28ebc6052bf06d49c90fb3f8bee76db293b615946a1c63`. What happened in between: the main run was stopped by a signal at 00:24:23, right after its fourth check, and was started again at 00:24:35. The extension did not do this; its runner had not been started. The main run went on with fit 2 from its checkpoint (`alldays.log`: "CONTINUES after check 4"). At 00:35:27 the lists below were the same, except that the checkpoint was the one of check 4 (2026-10-12 00:24): no test result file, one fit record, one weights file, no result folder of the extension.

Proof, from the stage's run folder at 2026-10-12T00:16:22 (the smoke folder left aside):

- Test result files (`test_seed*.json`): none. Files with "test" in their name: none.
- Fit records: one, `results/LSTM-MLP-OF-recent/fit_seed0.json` (2026-10-11 19:16; status finished; `test_scored` false).
- Saved weights: one, `results/LSTM-MLP-OF-recent/best_weights_seed0.pt` (2026-10-11 18:58).
- Checkpoints: one, `results/LSTM-MLP-OF-alldays/checkpoint_seed0.pt` (2026-10-11 23:04).
- No result folder of the extension existed (`results_seeds`), and no process of the extension.
- `preregistration.md` had the sha256 `babfaf66c72cd5627f6b28cf271c12c8837078722216f9c91570bafd157607e2`.

Known when this file was written: the validation numbers of seed 0 in `alldays.log`, for fit 1 (7 checks, best check 2) and for the first 3 checks of fit 2. No test number.

## 2. What the extension adds

Two more seeds for the three fits of `preregistration.md` section 5: `LSTM-MLP-OF-recent`, `LSTM-MLP-OF-alldays`, `CNN-LSTM-OF-alldays`. Six new fits. A rule fixed here can add two further seeds, six further fits (section 7).

Nothing else changes:

- the data, the caches, the split, the scaling groups, Δt and the scalers;
- the networks and the optimizer settings;
- the batch rule and the stopping rule (a check every 317,214 steps, patience 5, at most 50 checks, the weights of the best check);
- the exclusion rules 1 to 6 and the corporate-action file, with the same sha256 `df7ef9adb2c52d4af1d112e5ae8a326d2408956ebe43e1d953e1b034f9f6ea4a`;
- the metrics of `preregistration.md` section 8, with the primary and the secondary analysis of Revision 1.

The linear references have no seed: they are closed-form solutions. They are not repeated.

The first seed is the main run. Its fits, its scores and its report stand as they are.

## 3. The seed values

- The first seed is 0. It was chosen by hand on 2026-10-11, before the rule below existed. It stays.
- The two new seed values are **3089492870** and **771810125**.
- They were drawn once, on 2026-10-12 at 00:16:13 (+11:00), with this command:

  `python3 -c "import secrets; a, b = secrets.randbits(32), secrets.randbits(32); print(a, b)"`

  Its output was `3089492870 771810125`. The rule of the draw: two draws; a redraw only if a value is 0 or the two are equal. Neither was the case, so there was one draw. The command and its output are kept in the stage's run folder (`proofs/seeds/seed_draw_20261012.txt`).
- Reason (decision recorded on 2026-10-12): NumPy's documentation advises against small seed values chosen by hand, and random seeds were wanted. Each value is fixed and recorded before its fit starts, so every run can be repeated.
- The two values were fixed in this file before any of their fits started. No fit record, weights file or checkpoint of these seeds existed.

## 4. What the seed changes in this code, and what it does not

The seed is the value `SEED` of the stage's configuration module. The fit (the stage's training function) uses it in two places and nowhere else:

1. `torch.manual_seed(seed)` before the network is built. This sets the initial weights (`models.build`: Glorot-uniform kernels, orthogonal recurrent kernels).
2. `torch.Generator().manual_seed(seed)` for the batch stream (the stage's pass-drawing function). This sets, for every pass, the order of the samples inside each stock-day and the order of the batches across all stock-days.

One side effect: at the start of every pass the data loader takes one number from torch's global generator (the one of point 1) to seed its worker processes. The workers draw no random number, and nothing else reads the global generator after the network is built. So this changes no batch and no weight.

The seed does not change:

- which samples exist, their features and labels, the scalers, Δt;
- which samples are in a batch pool: a batch is still 256 samples of one stock-day;
- the validation set, its order and the validation loss formula;
- the learning rate, epsilon, betas, the stopping rule;
- the test set and the exclusions.

The arithmetic of the GPU is not tied to the seed. The cuDNN algorithm search is on. In the restart test of 2026-10-11 (smoke split, 400 steps) two runs of the same seed agreed to the last digit for the LSTM-MLP and to about 1e-6 of the validation loss for the CNN-LSTM. Over a long fit the CNN-LSTM difference can grow.

## 5. The six fits and their order

1. `LSTM-MLP-OF-recent`, seed 3089492870
2. `LSTM-MLP-OF-alldays`, seed 3089492870
3. `LSTM-MLP-OF-recent`, seed 771810125
4. `LSTM-MLP-OF-alldays`, seed 771810125
5. `CNN-LSTM-OF-alldays`, seed 3089492870
6. `CNN-LSTM-OF-alldays`, seed 771810125

- The extension starts by itself when the main run has ended with `ALLDAYS_DONE`. If the main run ends with `ALLDAYS_FAILED`, the extension does not start.
- So the six fits start after the test scores of seed 0 exist. Nothing of the extension depends on those scores: the fits, their settings, their stopping rule, their scoring and the rule of section 7 are fixed here.
- An interrupted fit continues from its last check (Revision 1, change 5).
- A fit whose loss is not finite is recorded as diverged. It is not scored on the test days.

## 6. Scoring of the six fits

- Each new fit is scored once on the three test days, with the rules of `preregistration.md` Revision 1: the primary analysis and, from the same scores, the secondary analysis.
- This is done after all six fits have ended and their weights are saved. The code refuses to load a test-day array for one of the six before that.
- The validation scores of a new fit are those of its saved weights, as for seed 0.

## 7. The rule for two more seeds

Fixed now. It uses validation numbers only. The runner applies it by itself, after the six fits and their scoring.

- For each of the three seeds (0, 3089492870, 771810125) take the difference `LSTM-MLP-OF-alldays` minus `LSTM-MLP-OF-recent` of the headline metric on the validation days: R² with k = 1..10 over both validation days, as the mean over stocks of the difference of the two stock values (`preregistration.md` section 8).
- Take the mean of the three differences and their sample standard deviation (3 − 1 in the denominator).
- If the standard deviation is more than half of the absolute value of the mean, two more seed values are drawn and all three fits are run with them: five seeds in all. Otherwise the extension ends with three seeds.
- Reason for five (decision recorded on 2026-10-12): it is the number of seeds Henderson et al. (2018) use; they call fewer than five potentially misleading.
- If one of the three differences does not exist because a fit has diverged, the two more seeds are run.
- The two further values are drawn in the same way, by the runner, at the time of the decision: `secrets.randbits(32)`; a redraw only if a value is 0 or equal to a seed already used. They are written with the time into `results_seeds/decision.json` and into `seeds.log` before the first of their fits starts. They are drawn once: a restarted runner reads them from that file.
- The six further fits run in the order of section 5 with the two further values. They are scored once on the test days after all six of them have ended, as in section 6.
- The decision, the three differences, their mean and their standard deviation go into the log and into the report, whichever way the rule falls.

## 8. The seed ensemble

For each of the three fits there is one ensemble: of the first three seeds, 0, 3089492870 and 771810125.

- The forecast of the ensemble is the mean over the three seeds of the three models' forecasts in dollars, sample by sample and horizon by horizon. Each model's forecast goes back to dollars with the same label scaler, as in `train.score`.
- The ensemble is scored once on the validation days and once on the test days, with the metrics and the two analyses of section 6.
- It is scored after the six fits have ended and have been scored, and not before.
- This needs one more forward pass of the saved weights over the validation days and over the test days: of the seed-0 weights, which the main run has already scored there once, and of the weights of the two new seeds. No weights are chosen or changed by it. The single-seed scores of section 6 and of the main run stand; the extra pass is used for the ensemble only.
- If one of the three seeds of a fit has diverged, no ensemble is formed for that fit. The report says so.
- This file defines no ensemble of five seeds. If two more seeds are run, the ensemble stays the one of the first three seeds.

## 9. What is reported

`REPORT_SEEDS.md`. It is built after the decision of section 7. If two more seeds are run, it is built again when they have been scored. Validation tables and test tables are apart. On the test days the primary analysis comes first, the secondary analysis second.

For each of these measures: R² with k = 1..10 over all days of the part (the headline), k = 10 alone, and each day alone:

- per fit: the value of each seed, their mean, their standard deviation, their minimum and their maximum;
- for LSTM-MLP: the difference `LSTM-MLP-OF-alldays` minus `LSTM-MLP-OF-recent` of each seed, its mean and its standard deviation. The difference of a seed is the mean over stocks of the difference of the two stock values (`preregistration.md` section 8);
- per fit: the value of the ensemble beside the mean of the single seeds; and the same difference of the two LSTM-MLP ensembles.

Definitions:

- The standard deviation across seeds is the sample standard deviation, with the number of seeds minus 1 in the denominator. With three seeds it is a rough number.
- The statistics are over all seeds that were run: three, or five when the rule of section 7 adds two.
- If a seed has diverged, the statistics are over the seeds that ended, and the count is given.
- Every value carries its days, its number of stocks and its horizon, as in `REPORT_ALLDAYS.md`.

No seed is selected. All seeds are reported, in every table. Seed 0 has no special place except that it ran first.

Also in the report: the decision of section 7 with its numbers; per new fit the number of checks, the best check, the interruptions and the per-check record; the sha256 of this file, of `preregistration.md` and of the corporate-action file.

## 10. Checks before the six fits

Right after the main run has ended, the extension first runs its whole code path on the smoke split of `preregistration.md` section 9 (UAL and FAST, training days of this split, a few hundred steps, the three seeds, scoring, the rule of section 7, two further seeds whatever the rule says, ensembles and report). If that fails, nothing of the extension is run.

The smoke split holds no validation day and no test day of this split.
