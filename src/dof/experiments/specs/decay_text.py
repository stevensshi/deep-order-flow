"""Fixed prose of the decay report."""

from dof.experiments.specs.alldays_text import REPRO as ALLDAYS_REPRO

TITLE = "Decay: two models trained on three days of 2019, scored on later days"
INTRO = ("All values are out-of-sample R² in percent, per stock, day and horizon, against the mean label of that "
         "stock-day. A stock's value is the mean of its cells; a table value is the mean over stocks. Two models, "
         "seed 0 each.",)
RESULT = ("The two models were trained on 2019-01-30, 2019-03-27 and 2019-07-30.",)
LATER = ("Further from training, and on the frozen basis, the values fall; the tables by model give every day.",)
METHOD = ("Basis (i), frozen: each stock's 2019 scalers and Δt. Basis (ii), refreshed: scalers and Δt fitted on "
          "recent days the model was not trained on (2021-07-13 for 2021-08-13; the main run's group of 2025-11-28 "
          "to 2025-12-10 for the later days); its label horizons follow the refreshed Δt, so the labels of the two "
          "bases differ.",
          "Row 'all': every held stock-day. Row 'inside': the stock-days whose median mid price is inside [0.6, 1.6] "
          "times that of the basis's reference day (2019-07-30 for (i), the refreshed group's last day for (ii)). "
          "(ii) − (i): the stocks scored on both bases on that day, in that row.",
          "Experiment protocol sha256 {extension}, which extends the protocol document sha256 {main}.")
DAYS = ("A stock-day is left out for a reused ticker, for no usable scaling group, or for no sample; every other "
        "held stock-day is scored, whatever its price ratio.",)
VALIDITY = ("Every day is a second look for this feature family. The stock set changes from row to row; the counts "
            "are in each cell. The tables are a description; no threshold decides anything.",
            "Added after the results were seen: on 2026-05-15, 2026-05-18 and 2026-06-12 the 'all' row of basis (ii) "
            "holds stocks with a corporate action between the refreshed group and the day (the main run lists them "
            "among its corporate-action left-outs); read the 'inside' row of (ii) for those days. No stock-day is left "
            "out after the fact.",
            "cuDNN's algorithm search was off here and on in the 2019 run.")
REPRO = ALLDAYS_REPRO

TEST_2019 = "the test days of both 2019 models."
MAIN_TRAINING = "training days of the main run's models."
TEST_2026 = "the test days of the main run."
ROLES = {
    "2019-10-18": TEST_2019, "2019-10-30": TEST_2019, "2019-12-30": TEST_2019,
    "2020-01-30": MAIN_TRAINING, "2021-07-13": MAIN_TRAINING, "2021-08-13": MAIN_TRAINING,
    "2025-11-28": MAIN_TRAINING, "2025-12-08": MAIN_TRAINING, "2025-12-09": MAIN_TRAINING, "2025-12-10": MAIN_TRAINING,
    "2025-12-11": "a validation day of the main run and of the seed extension; a training day of the adjacent-days "
                  "experiment.",
    "2025-12-12": "a validation day of the main run and of the seed extension; its morning and afternoon are the "
                  "validation and test parts of the adjacent-days experiment.",
    "2026-05-15": TEST_2026, "2026-05-18": TEST_2026, "2026-06-12": TEST_2026,
}
NOTES = {"2019-10-18": "stored result, scored again", "2019-10-30": "stored result, scored again",
         "2019-12-30": "stored result, scored again", "2025-11-28": "half day", "2021-08-13": "refreshed on one day"}
