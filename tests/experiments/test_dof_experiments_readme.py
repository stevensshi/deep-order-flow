"""Every result number of README.md comes from the published tables in results/.

Each README line that holds a result number is built here from the report tables (``dof.experiments.report.build``),
formatted as the reports format it, and must appear in README.md in this order. Outside those lines the README may
hold no number of three decimals, no count of stocks or stock-days, no percent value, no "percentage points" and no
decimal or scientific number other than the settings in ``ALLOWED_DECIMALS``. One context number is allowed, exactly
once and only in the words of ``CONTEXT``: the paper's level, read from its figure, not a result of this repository.
``LINES``, ``CONTEXT`` and ``ALLOWED_DECIMALS`` are written together, from the README text.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
README = ROOT / "README.md"

if not (RESULTS / "cells.parquet").exists():
    pytest.skip("the README numbers are checked against the published tables", allow_module_level=True)

from dof.experiments.cells import read_tables  # noqa: E402
from dof.experiments.render import fmt_cell  # noqa: E402
from dof.experiments.report import build  # noqa: E402
from dof.experiments.specs import SPECS, TEST_DAYS_LABEL  # noqa: E402
from dof.experiments.specs.decay import DAYS as DECAY_DAYS  # noqa: E402
from dof.experiments.specs.earlier import TEST_2019_LABEL  # noqa: E402

NUM3 = re.compile(r"(?<![\d.])-?\d+\.\d{3}(?!\d)")
COUNT = re.compile(r"\b\d+ (?:stocks|stock-days)\b")
PERCENT_OFF = re.compile(r"(?<![\d.])-?\d+\.(?:\d{1,2}|\d{4,}) ?%")
DECIMAL = re.compile(r"(?<![\w.])-?\d+\.\d+(?![\w.])")
SCI = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?e[-+]?\d+")
PERCENT = re.compile(r"\d+ ?%")
URL = re.compile(r"https?://\S+")
# Settings, not results: ITCH 5.0, Python 3.11, the price range 0.6 and 1.6, a paper's section 2.1, and the
# learning rates, epsilons and the price unit in scientific notation.
ALLOWED_DECIMALS = {"5.0", "3.11", "0.6", "1.6", "2.1", "1e-5", "3e-5", "1e-7", "1e-3", "1e-2", "1e-1", "1e-4"}
# The one context number of the README: the means over the ten horizons of the order-flow curves of
# Figure 2 of Kolm, Turiel and Westray (2023), read from the figure (1.207, 1.223, 1.286).
CONTEXT = ("the paper's LSTM, LSTM-MLP and CNN-LSTM on order flow reach about 1.2–1.3 % (mean over the ten horizons, "
           "read from Figure 2 of Kolm, Turiel and Westray 2023)")
K = "k=1..10"
VALID = "2025-12-11,2025-12-12"
REPORT_SPECS = ("earlier", "decay", "alldays", "adjacent")   # the README uses no number of the seeds report
EARLIER = ("LSTM-OF", "LSTM-MLP-OF", "CNN-LSTM-OF-lr3e-5-eps1e-7", "Ridge-OF")
# The README compares the three models trained on all 14 days; the two fits on the 4 most recent days are reported
# only in results/ and its reports.
ORDER = ("CNN-LSTM-OF-alldays", "LSTM-MLP-OF-alldays", "Ridge-OF-alldays")
CNN_D, LSTM_D = "CNN-LSTM-OF-lr3e-5-eps1e-7", "LSTM-MLP-OF"
ADJ = "LSTM-MLP-OF-recent"


def cell(spec: str, table: str, row: tuple, col: str) -> tuple:
    return ("cell", spec, table, tuple(row), col)


def rows(spec: str, table: str) -> tuple:
    return ("rows", spec, table)


def _p(m: str, col: str, day: str = TEST_DAYS_LABEL, k: str = K) -> tuple:
    return cell("alldays", f"test_primary:{m}", (m, day, k), col)


def _e(m: str, col: str, k: str = K) -> tuple:
    return cell("earlier", f"test:{m}", (m, TEST_2019_LABEL, k), col)


def _decay(m: str, day: str, row: str, col: str) -> tuple:
    return cell("decay", f"decay:{m}:k=1..10", (m, "k=1..10", day, row), col)


def _r(m: str, day: str, col: str) -> tuple:
    return cell("earlier", "rescore", (m, day), col)


def _model_rows(m: str) -> list[tuple[str, dict]]:
    valid = ("validation (chose the stopping check; not a result)" if m.startswith(("CNN", "LSTM"))
             else "validation (chose λ; not a result)")
    out = [("| 2026-05-15, 2026-05-18, 2026-06-12 | test, primary | {n} | {r2} | {se} |",
            {"n": _p(m, "stocks"), "r2": _p(m, "r2"), "se": _p(m, "se")})]
    for day in ("2026-05-15", "2026-05-18", "2026-06-12"):
        out.append((f"| {day} | test, primary | {{n}} | {{r2}} | {{se}} |",
                    {"n": _p(m, "stocks", day), "r2": _p(m, "r2", day), "se": _p(m, "se", day)}))
    sec = ("alldays", f"test_secondary:{m}", (m, TEST_DAYS_LABEL, K))
    out.append(("| 2026-05-15, 2026-05-18, 2026-06-12 | test, secondary (price ratio inside the range) | {n} | {r2} | "
                "{se} |", {"n": cell(*sec, "stocks"), "r2": cell(*sec, "r2"), "se": cell(*sec, "se")}))
    val = ("alldays", f"valid:{m}", (m, VALID, K))
    out.append((f"| 2025-12-11, 2025-12-12 | {valid} | {{n}} | {{r2}} | {{se}} |",
                {"n": cell(*val, "stocks"), "r2": cell(*val, "r2"), "se": cell(*val, "se")}))
    return out


def _decay_row(day: str) -> tuple[str, dict]:
    refs = {"mo": _decay(CNN_D, day, "all", "months")}
    for pre, m in (("c", CNN_D), ("l", LSTM_D)):
        for suf, row in (("", "all"), ("i", "inside")):
            refs[pre + suf] = _decay(m, day, row, "i_r2")
            refs[pre + suf + "n"] = _decay(m, day, row, "i_stocks")
    return f"| {day} | {{mo}} | {{c}} | {{cn}} | {{ci}} | {{cin}} | {{l}} | {{ln}} | {{li}} | {{lin}} |", refs


EARLIER_NAMES = {"LSTM-OF": "`LSTM-OF`", "LSTM-MLP-OF": "`LSTM-MLP-OF`",
                 "CNN-LSTM-OF-lr3e-5-eps1e-7": "`CNN-LSTM-OF-lr3e-5-eps1e-7`", "Ridge-OF": "`Ridge-OF` (pooled ridge)"}
CNN = "CNN-LSTM-OF-alldays"
DAYS_0515 = ("decay", "days", ("2026-05-15", "frozen"))
AFTERNOON = ("adjacent", f"part_afternoon:{ADJ}", (ADJ, "all", K))
FAR = ("adjacent", f"part_far:{ADJ}", (ADJ, "all", K))
CA = ("alldays", "ca_left_out")

# Helpers for the README's tables, whose columns are ordered as below.
#
# A range in the text ("between 0.916 % and 1.449 %") binds its two ends to the two cells that are the lowest and the
# highest of the rows the sentence covers. A stock count that a table column shares between the two decay networks
# binds to the CNN-LSTM's cell; the LSTM-MLP's count is equal on every row (decay.md, both model tables, column
# "(i) stocks").

def _sec(m: str, col: str) -> tuple:
    return cell("alldays", f"test_secondary:{m}", (m, TEST_DAYS_LABEL, K), col)


def _decay_row_c(day: str, label: str) -> tuple[str, dict]:
    """One row of the README's decay table: day | months | 'all' stocks | CNN | LSTM-MLP | 'inside' stocks | CNN | LSTM-MLP."""
    return (f"| {label} | {{mo}} | {{n}} | {{c}} | {{l}} | {{ni}} | {{ci}} | {{li}} |",
            {"mo": _decay(CNN_D, day, "all", "months"), "n": _decay(CNN_D, day, "all", "i_stocks"),
             "c": _decay(CNN_D, day, "all", "i_r2"), "l": _decay(LSTM_D, day, "all", "i_r2"),
             "ni": _decay(CNN_D, day, "inside", "i_stocks"), "ci": _decay(CNN_D, day, "inside", "i_r2"),
             "li": _decay(LSTM_D, day, "inside", "i_r2")})


# Statement (c), in the Summary and (indented) in the disclosures. 0.609 is literal text, not a placeholder: its
# source is results/reports/earlier.md (Validity, rendered from src/dof/experiments/specs/earlier_text.py), not a
# table cell; the fit it describes is not in results/. It is the one number of the README that this test does not
# check. {n} is the earlier stage's stock count: the 0.609 % fit used the same 115 stocks as the three networks of
# Table 1.
_STATEMENT_C = (
    "The CNN-LSTM's 3e-5/1e-7 setting was one of three picked on the validation day by the 20-stock learning-rate map "
    "(the lowest validation loss among the cells with epsilon 1e-7). All three 20-stock fits, this setting included, "
    "were scored on these test days. On {n} stocks, the setting with learning rate 0.3 and epsilon 1 was fitted first "
    "and scored 0.609 % on these test days ({n} stocks; k = 1..10); its weight steps were about three times larger "
    "than on 20 stocks, so the 3e-5/1e-7 fit was added after that score had been read. The 0.609 % fit is not in "
    "these tables.")
_STATEMENT_C_REFS = {"n": _e(CNN_D, "stocks")}

DECAY_LABELS = (("2019-10-18", "2019-10-18"), ("2019-10-30", "2019-10-30"), ("2019-12-30", "2019-12-30"),
                ("2020-01-30", "2020-01-30"), ("2021-07-13", "2021-07-13"), ("2021-08-13", "2021-08-13"),
                ("2025-11-28", "2025-11-28 (half day)"), ("2025-12-08", "2025-12-08"), ("2025-12-09", "2025-12-09"),
                ("2025-12-10", "2025-12-10"), ("2025-12-11", "2025-12-11"), ("2025-12-12", "2025-12-12"),
                ("2026-05-15", "2026-05-15"), ("2026-05-18", "2026-05-18"), ("2026-06-12", "2026-06-12"))
EARLIER_RANKED = ("CNN-LSTM-OF-lr3e-5-eps1e-7", "LSTM-MLP-OF", "LSTM-OF", "Ridge-OF")
T19 = "2019-10-18, 2019-10-30, 2019-12-30"
T26 = "2026-05-15, 2026-05-18, 2026-06-12"
RIDGE = "Ridge-OF-alldays"
LSTM_A = "LSTM-MLP-OF-alldays"

LINES: list[tuple[str, dict]] = [
    # Summary
    ("**Networks forecast better than a linear reference on the 2019 test days.** The earlier stage of this work "
     "trained one pooled model per network for the 115 tickers on three days of 2019. On 2019-10-18, 2019-10-30 and "
     "2019-12-30, pooled networks reach an out-of-sample R² of {c} % (CNN-LSTM; s.e. {cse}), {m} % (LSTM-MLP; s.e. "
     "{mse}) and {l} % (LSTM; s.e. {lse}) on {n} stocks, mean of the horizons k = 1..10. A pooled ridge regression on "
     "the same inputs reaches {r} % (s.e. {rse}; {rn} stocks; k = 1..10).",
     {"c": _e(CNN_D, "r2"), "cse": _e(CNN_D, "se"), "m": _e("LSTM-MLP-OF", "r2"), "mse": _e("LSTM-MLP-OF", "se"),
      "l": _e("LSTM-OF", "r2"), "lse": _e("LSTM-OF", "se"), "n": _e(CNN_D, "stocks"),
      "r": _e("Ridge-OF", "r2"), "rse": _e("Ridge-OF", "se"), "rn": _e("Ridge-OF", "stocks")}),
    ("- " + _STATEMENT_C, _STATEMENT_C_REFS),
    ("**Forecast power holds for months.** On the four days scored from {mo1} to {mo4} months after training, the "
     "same CNN-LSTM and LSTM-MLP stay between {lo} % and {hi} % ({nlo} to {nhi} stocks; k = 1..10). Only 2020-01-30 "
     "had not been scored as a held-out day before the decay run. The values do not fall in a straight line.",
     {"mo1": _decay(CNN_D, "2019-10-18", "all", "months"), "mo4": _decay(CNN_D, "2020-01-30", "all", "months"),
      "lo": _decay(LSTM_D, "2019-12-30", "inside", "i_r2"), "hi": _decay(LSTM_D, "2019-10-18", "all", "i_r2"),
      "nlo": _decay(CNN_D, "2019-12-30", "inside", "i_stocks"), "nhi": _decay(CNN_D, "2019-10-18", "all", "i_stocks")}),
    ("**Forecast power falls over years, for a cause not tested.** From 2021-07-13 on, both networks are below zero "
     "when every stock that the decay run could score is included: from {hi} % to {lo} % ({nlo} to {nhi} stocks; "
     "k = 1..10). The fall is not separated from the change in price levels since 2019.",
     {"hi": _decay(CNN_D, "2021-07-13", "all", "i_r2"), "lo": _decay(LSTM_D, "2026-05-15", "all", "i_r2"),
      "nlo": _decay(CNN_D, "2026-05-15", "all", "i_stocks"), "nhi": _decay(CNN_D, "2021-07-13", "all", "i_stocks")}),
    ("**Retrained networks forecast weakly in 2026.** The main run, the pre-registered stage of this work, trained the "
     "networks again on 14 days from 2019-01-30 to 2025-12-10. On 2026-05-15, 2026-05-18 and 2026-06-12, five to six "
     "months after its last training day, the CNN-LSTM reaches {c} % (s.e. {cse}; {cn} stocks; k = 1..10). The pooled "
     "ridge regression is at {r} % (s.e. {rse}; {rn} stocks; k = 1..10).",
     {"c": _p(CNN, "r2"), "cse": _p(CNN, "se"), "cn": _p(CNN, "stocks"),
      "r": _p(RIDGE, "r2"), "rse": _p(RIDGE, "se"), "rn": _p(RIDGE, "stocks")}),
    # Table 1 and its text
    *[(f"| {i} | {EARLIER_NAMES[m]} | {T19} | {{n}} | {{r2}} | {{se}} | {{r2_10}} | {{se_10}} |",
       {"n": _e(m, "stocks"), "r2": _e(m, "r2"), "se": _e(m, "se"), "r2_10": _e(m, "r2", "k=10"),
        "se_10": _e(m, "se", "k=10")}) for i, m in enumerate(EARLIER_RANKED, 1)],
    ("The CNN-LSTM ranks first at {c} % (s.e. {cse}; {cn} stocks; k = 1..10), and all three networks are above the "
     "pooled ridge regression at {r} % (s.e. {rse}; {rn} stocks; k = 1..10). The three networks lie within one "
     "standard error of each other, from {lo} % to {hi} % ({n} stocks; k = 1..10), and the order rests on test days "
     "that were read many times; this study gives no reason why one architecture leads. A per-stock ridge regression "
     "of the earlier stage (one model per stock) scored higher than the pooled ridge regression on the same stocks and "
     "test days; it is not in these tables. Stocks move together, so these standard errors are too small.",
     {"c": _e(CNN_D, "r2"), "cse": _e(CNN_D, "se"), "cn": _e(CNN_D, "stocks"),
      "r": _e("Ridge-OF", "r2"), "rse": _e("Ridge-OF", "se"), "rn": _e("Ridge-OF", "stocks"),
      "lo": _e("LSTM-OF", "r2"), "hi": _e(CNN_D, "r2"), "n": _e(CNN_D, "stocks")}),
    # Table 2 and its text
    *[_decay_row_c(day, label) for day, label in DECAY_LABELS],
    ("The first three rows are the test days of Table 1, scored again with the same saved weights; "
     "`results/reports/earlier.md` compares the two scorings. From {mo1} to {mo4} months, both networks stay between "
     "{lo} % and {hi} % ({nlo} to {nhi} stocks). At {mo4b} months ({c4} %, {l4} %; {n4} stocks) the values are above "
     "those at {mo2} months ({c2} %, {l2} %; {n2} stocks), so the decline is not a straight line. From 2021-07-13 on, "
     "every 'all' row is below zero. The 'inside' rows reach {c5} % and {l5} % on 2021-07-13 ({n5} stocks), {c6} % "
     "and {l6} % on 2021-08-13 ({n6} stocks), and from {lo7} % to {hi7} % in 2025 and 2026 ({nlo7} to {nhi7} stocks).",
     {"mo1": _decay(CNN_D, "2019-10-18", "all", "months"), "mo4": _decay(CNN_D, "2020-01-30", "all", "months"),
      "lo": _decay(LSTM_D, "2019-12-30", "inside", "i_r2"), "hi": _decay(LSTM_D, "2019-10-18", "all", "i_r2"),
      "nlo": _decay(CNN_D, "2019-12-30", "inside", "i_stocks"), "nhi": _decay(CNN_D, "2019-10-18", "all", "i_stocks"),
      "mo4b": _decay(CNN_D, "2020-01-30", "all", "months"),
      "c4": _decay(CNN_D, "2020-01-30", "all", "i_r2"), "l4": _decay(LSTM_D, "2020-01-30", "all", "i_r2"),
      "n4": _decay(CNN_D, "2020-01-30", "all", "i_stocks"), "mo2": _decay(CNN_D, "2019-10-30", "all", "months"),
      "c2": _decay(CNN_D, "2019-10-30", "all", "i_r2"), "l2": _decay(LSTM_D, "2019-10-30", "all", "i_r2"),
      "n2": _decay(CNN_D, "2019-10-30", "all", "i_stocks"),
      "c5": _decay(CNN_D, "2021-07-13", "inside", "i_r2"), "l5": _decay(LSTM_D, "2021-07-13", "inside", "i_r2"),
      "n5": _decay(CNN_D, "2021-07-13", "inside", "i_stocks"),
      "c6": _decay(CNN_D, "2021-08-13", "inside", "i_r2"), "l6": _decay(LSTM_D, "2021-08-13", "inside", "i_r2"),
      "n6": _decay(CNN_D, "2021-08-13", "inside", "i_stocks"),
      "lo7": _decay(CNN_D, "2025-12-11", "inside", "i_r2"), "hi7": _decay(CNN_D, "2026-06-12", "inside", "i_r2"),
      "nlo7": _decay(CNN_D, "2026-06-12", "inside", "i_stocks"), "nhi7": _decay(CNN_D, "2025-11-28", "inside", "i_stocks")}),
    ("With scalers and Δt refitted on 2025-11-28 to 2025-12-10, the two networks reach {c} % ({cn} stocks) and {l} % "
     "({ln} stocks) on 2025-12-12 (k = 1..10); `results/reports/decay.md` gives this refreshed basis for every later "
     "day.",
     {"c": _decay(CNN_D, "2025-12-12", "all", "ii_r2"), "cn": _decay(CNN_D, "2025-12-12", "all", "ii_stocks"),
      "l": _decay(LSTM_D, "2025-12-12", "all", "ii_r2"), "ln": _decay(LSTM_D, "2025-12-12", "all", "ii_stocks")}),
    ("The fall over years is not separated from the change in price levels. On 2026-05-15 the price ratio against "
     "2019-07-30 was below 0.6 on {below} stock-days and above 1.6 on {above} stock-days, out of {scored} stock-days "
     "scored. The 'inside' rows hold fewer stocks, and the stock set changes from day to day. This run does not test "
     "the cause. Every day of Table 2 is a second look for this feature family.",
     {"below": cell(*DAYS_0515, "below"), "above": cell(*DAYS_0515, "above"), "scored": cell(*DAYS_0515, "scored")}),
    # Table 3 and its text
    *[(f"| {i} | `{m}` | {T26} | {{n}} | {{r2}} | {{se}} | {{r2_10}} | {{se_10}} |",
       {"n": _p(m, "stocks"), "r2": _p(m, "r2"), "se": _p(m, "se"), "r2_10": _p(m, "r2", k="k=10"),
        "se_10": _p(m, "se", k="k=10")}) for i, m in enumerate(ORDER, 1)],
    ("The CNN-LSTM ranks first again, at {c} % (s.e. {cse}; {cn} stocks; k = 1..10). Stocks move together, so these "
     "standard errors are too small. 2026-05-15 is the weakest test day for both networks: {d1} % and {d2} % ({dn} "
     "stocks; k = 1..10). The secondary analysis (price ratio inside 0.6 to 1.6 of the last training day) gives {s1} %, "
     "{s2} % and {s3} % ({sn} stocks; same test days; k = 1..10). Validation numbers chose the stopping check of each "
     "network and the λ of the ridge regression; they are not results, and `results/reports/alldays.md` gives them in "
     "its model sections.",
     {"c": _p(CNN, "r2"), "cse": _p(CNN, "se"), "cn": _p(CNN, "stocks"),
      "d1": _p(CNN, "r2", "2026-05-15"), "d2": _p(LSTM_A, "r2", "2026-05-15"), "dn": _p(CNN, "stocks", "2026-05-15"),
      "s1": _sec(CNN, "r2"), "s2": _sec(LSTM_A, "r2"), "s3": _sec(RIDGE, "r2"), "sn": _sec(CNN, "stocks")}),
    # Adjacent training days
    ("**Adjacent training days (one half day).** An `LSTM-MLP-OF` trained on five adjacent days (2025-11-28, a half "
     "day, and 2025-12-08 to 2025-12-11) and stopped on the morning of 2025-12-12 reaches {a} % on the afternoon of "
     "2025-12-12 (s.e. {ase}; {an} stocks; k = 1..10; mean of three training runs). The same weights reach {f} % on "
     "the three 2026 test days (s.e. {fse}; {fn} stocks; k = 1..10; primary analysis). One half day cannot show that "
     "adjacent training days forecast better in general; `results/reports/adjacent.md` gives the paired difference "
     "and its uncertainty.",
     {"a": cell(*AFTERNOON, "mean"), "ase": cell(*AFTERNOON, "se"), "an": cell(*AFTERNOON, "stocks"),
      "f": cell(*FAR, "mean"), "fse": cell(*FAR, "se"), "fn": cell(*FAR, "stocks")}),
    # Data and split (no checked number, kept as in the working LINES so the corporate actions stay pinned)
    ("The corporate-action file (`configs/protocol/corporate_actions_20251210_20260612.csv`) leaves out {n} test "
     "stock-days: {s1} on {d1}, {d2} and {d3} (`{e1}`, effective {f1}) and {s4} on {d4} (`{e4}`, effective {f4}).",
     {"n": rows(*CA),
      "s1": cell(*CA, ("BKNG", "2026-05-15"), "symbol"), "d1": cell(*CA, ("BKNG", "2026-05-15"), "date"),
      "d2": cell(*CA, ("BKNG", "2026-05-18"), "date"), "d3": cell(*CA, ("BKNG", "2026-06-12"), "date"),
      "e1": cell(*CA, ("BKNG", "2026-05-15"), "event"), "f1": cell(*CA, ("BKNG", "2026-05-15"), "effective_date"),
      "s4": cell(*CA, ("KLAC", "2026-06-12"), "symbol"), "d4": cell(*CA, ("KLAC", "2026-06-12"), "date"),
      "e4": cell(*CA, ("KLAC", "2026-06-12"), "event"), "f4": cell(*CA, ("KLAC", "2026-06-12"), "effective_date")}),
    # Limitations and disclosures: statement (c) again, indented under the 2019 lead
    ("  - " + _STATEMENT_C, _STATEMENT_C_REFS),
]


def _walk(sections):
    for s in sections:
        yield from s.tables
        yield from _walk(s.subsections)


@pytest.fixture(scope="module")
def built() -> dict:
    tables = read_tables(RESULTS)
    return {name: {t.id: t for t in _walk(build(tables, SPECS[name], strict=True).sections)} for name in REPORT_SPECS}


def _value(built: dict, ref: tuple) -> str:
    if ref[0] == "rows":
        return str(len(built[ref[1]][ref[2]].rows))
    _, spec, tid, row, col = ref
    t = built[spec][tid]
    hit = [r for r in t.rows if r.key == row]
    assert len(hit) == 1, f"{spec}: table {tid} has {len(hit)} rows {row}"
    c = hit[0].cells[[x.key for x in t.columns].index(col)]
    assert c.value is not None, f"{spec}: table {tid}, row {row}, column {col} has no value"
    return fmt_cell(c)


def expected_lines(built: dict) -> list[str]:
    return [template.format(**{k: _value(built, r) for k, r in refs.items()}) for template, refs in LINES]


def _readme() -> list[str]:
    return README.read_text(encoding="utf-8").splitlines()


def _without_context(line: str) -> str:
    return line.replace(CONTEXT, "")


def test_readme_lines_equal_tables(built) -> None:
    want, i = expected_lines(built), 0
    for line in _readme():
        if i < len(want) and line == want[i]:
            i += 1
    assert i == len(want), f"README.md lacks line {i + 1} of {len(want)} (or has it out of order):\n{want[i]}"


def test_readme_has_no_unchecked_number(built) -> None:
    want = set(expected_lines(built))
    bad = [f"{n}: {line}" for n, line in enumerate(_readme(), 1)
           if (NUM3.search(line) or COUNT.search(line)) and line not in want]
    assert not bad, "README.md lines with a result number that no entry builds:\n" + "\n".join(bad)


def test_readme_has_no_untracked_decimal(built) -> None:
    want, bad = set(expected_lines(built)), []
    for n, line in enumerate(_readme(), 1):
        if line in want:
            continue
        text = URL.sub("", _without_context(line))
        extra = [d for d in DECIMAL.findall(text) + SCI.findall(text) if d not in ALLOWED_DECIMALS]
        if extra or PERCENT.search(text) or "percentage points" in text:
            bad.append(f"{n}: {line}")
    assert not bad, "README.md lines with a number that no entry builds:\n" + "\n".join(bad)


def test_readme_percent_numbers_have_three_decimals() -> None:
    bad = [f"{n}: {line}" for n, line in enumerate(_readme(), 1) if PERCENT_OFF.search(_without_context(line))]
    assert not bad, "README.md percent values not written with three decimals:\n" + "\n".join(bad)


def test_readme_context_number_once() -> None:
    assert README.read_text(encoding="utf-8").count(CONTEXT) == 1
