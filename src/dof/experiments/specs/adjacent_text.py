"""Fixed prose of the adjacent-days report."""

from dof.experiments.specs.alldays_text import REPRO as ALLDAYS_REPRO

TITLE = "Five adjacent training days: validation on a morning, test on the afternoon"
INTRO = ("All values are out-of-sample R² in percent, per stock, day part and horizon, against the mean label of the "
         "scored rows of that stock on that part of the day. A stock's value is the mean of its cells; a table value "
         "is the mean over stocks.",)
RESULT = ("One half day: the result cannot be carried to other days, and every s.e. across stocks is too small "
          "because stocks move together.",)
METHOD = ("Training days: 2025-11-28 (a half day) and 2025-12-08 to 2025-12-11. Validation: the samples of "
          "2025-12-12 whose row and ten labels end before 13:00. Test: the samples of 2025-12-12 whose 100 rows of "
          "history are all at or after 13:00. A sample that is neither is used nowhere.",
          "One network (LSTM-MLP-OF) with three seeds and one pooled ridge reference. The same weights are also "
          "scored on 2026-05-15, 2026-05-18 and 2026-06-12, with the Δt and scalers of the five training days.",
          "Protocol document sha256 {main}; experiment protocol sha256 {extension}; corporate-action file sha256 {ca}.")
PAIR = ("A seed's difference is the mean, over the stocks scored on both parts, of the stock's value on the "
        "afternoon minus its value on the 2026 days. D is the mean of the seed differences. U is the root of the sum "
        "of two squares: the s.e. across stocks of the per-stock difference averaged over the seeds, and the s.d. "
        "across seeds of the seed differences divided by the root of the number of seeds. U holds no day effect and "
        "no correlation across stocks, so it is too small; D is a description, not a test.",)
REFERENCE = ("The main run's seed-0 fits, with their own Δt and scalers, on the same afternoon. This afternoon was "
             "inside their validation days. No number of this table enters D.",)
VALIDITY = ("2025-12-12 was a validation day of the main run and of the seed extension, so its afternoon is a second "
            "look for this feature family. The three 2026 days had been scored before for other models. No day of "
            "this experiment is a first look.",
            "The comparison cannot separate the distance in time from the difference between one December afternoon "
            "and three days of May and June, nor from the ageing of Δt and the scalers on the 2026 days.",
            "The afternoon is scored against the mean label of its own rows and the 2026 days against the mean of "
            "the whole day; this pushes D down.")
REPRO = ALLDAYS_REPRO
