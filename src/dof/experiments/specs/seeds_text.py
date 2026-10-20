"""Fixed prose of the seeds report."""

from dof.experiments.specs.alldays_text import REPRO as ALLDAYS_REPRO

TITLE = "Seeds of the three fits, and the ensembles of the first three seeds"
INTRO = ("All values are out-of-sample R² in percent: per stock, day and horizon against the mean label of that "
         "stock-day; a stock's value is the mean of its cells; a table value is the mean over stocks. s.d. is the "
         "sample standard deviation across seeds (n - 1); with few seeds it is a rough number. The ensemble's "
         "forecast is the mean of the first three seeds' forecasts in dollars, sample by sample.",)
RESULT = ("Test days 2026-05-15, 2026-05-18 and 2026-06-12, primary analysis, mean of the ten horizons. No seed is "
          "selected: every seed is in every table.",)
METHOD = ("Seed 0 is the main run and was chosen by hand; the other seeds were drawn with secrets.randbits(32) and "
          "recorded before their fits started. A seed sets the initial weights and the order of the samples and "
          "batches; data, scalers, settings, stopping rule, validation set and test set are the same for every seed.",
          "Order of work, as pre-registered: the six fits of the two drawn seeds, then their test scoring, then the "
          "ensembles, then the rule for two more seeds, which reads validation numbers only; further seeds are fitted "
          "and scored after that. The test days of the first round were therefore scored before the rule was "
          "applied; the rule could not see them.",
          "Protocol document sha256 {main}; seed protocol sha256 {extension}; corporate-action file sha256 {ca}.")
RULE = ("Two more seeds when the sample standard deviation of the three validation differences (all days minus "
        "recent, k = 1..10) is more than half of the absolute value of their mean, or when a difference is missing.",)
VALIDITY = ("Every seed after seed 0 reads the same test days as seed 0; they are first looks for seed 0 only "
            "(2026-06-12 is not a first look for this feature family).",
            "The cuDNN algorithm search was on, so a repeated fit is not bit-identical; these tables are rebuilt from "
            "the stored score cells.")
REPRO = ALLDAYS_REPRO
