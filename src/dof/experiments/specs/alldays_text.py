"""Fixed prose of the main-run report."""

TITLE = "One pooled model on all held trading days"
INTRO = ("All values are out-of-sample R² in percent, per stock, day and horizon, against the mean label of that "
         "stock-day. A stock's value is the mean of its cells; a table value is the mean over stocks; s.e. is the "
         "standard error across stocks. Stocks move together, so the standard errors are too small.",)
RESULT = ("Test days 2026-05-15, 2026-05-18 and 2026-06-12, primary analysis: every test stock-day of the cache that "
          "no corporate action removes. Each value is the mean of the ten horizons k = 1..10; k = 10 alone is given "
          "beside it. Validation-day numbers chose the stopping check and λ; they are not results and are given in "
          "the model sections.",)
METHOD = ("Training days: the 14 days from 2019-01-30 to 2025-12-10 ('all') or the 4 days from 2025-11-28 to "
          "2025-12-10 ('recent'). Validation days: 2025-12-11 and 2025-12-12. Test days: 2026-05-15, 2026-05-18 and "
          "2026-06-12, scored once, after every model had ended.",
          "Three networks and two pooled ridge references, one seed (0). Nothing was tuned in this stage.",
          "Protocol document sha256 {main}; corporate-action file sha256 {ca}. The secondary analysis keeps the "
          "stock-days whose median mid price is inside [0.6, 1.6] times that of the last training day; nothing is "
          "scored twice.")
DATA = ("The corporate-action file leaves out the test stock-days below; every other test stock-day of the cache "
        "is scored. A cell without an R² has labels of zero variance and is left out of the means.",)
VALIDITY = ("2026-06-12 is not a first look for this feature family: an earlier experiment scored per-stock ridge "
            "models on it. 2026-05-15 and 2026-05-18 are first looks.",
            "The learning rate and epsilon of CNN-LSTM-OF-alldays come from an earlier stage whose choice had seen "
            "that stage's test days.",
            "The out-of-sample R² compares the forecast with the stock-day's own mean label; a trader does not know "
            "that mean during the day. On the test days Δt and the scalers are those of the last training group, "
            "five to six months old.",
            "One seed. The cuDNN algorithm search was on, so a repeated fit is not bit-identical; these tables are "
            "rebuilt from the stored score cells, not from new fits.")
REPRO = ("Every number above is computed from the published tables in results/ (cells.parquet, stock_days.parquet, "
         "fits.parquet, checks.parquet) by the functions of dof.experiments.analyses.",)
