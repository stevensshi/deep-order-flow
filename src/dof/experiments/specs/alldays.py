"""The main run: three networks and two pooled ridge references, seed 0, on all held trading days."""

from __future__ import annotations

from dof.experiments.spec import ExperimentSpec, HeadlineSpec, Part, Protocol, RecordSource, SectionSpec, ts
from dof.experiments.specs import (
    CORPORATE_ACTIONS,
    FITS,
    LINEAR,
    MAIN_DOCUMENT,
    PROTOCOL_MAIN,
    TEST_DAYS,
    TEST_DAYS_LABEL,
    TEST_DAYS_WORDS,
    VALID_DAYS,
)
from dof.experiments.specs import alldays_text as TEXT

ALL_DAYS, RECENT_DAYS = "14 days, 2019-01-30 to 2025-12-10", "4 days, 2025-11-28 to 2025-12-10"
TRAINED = {"CNN-LSTM-OF-alldays": ALL_DAYS, "LSTM-MLP-OF-alldays": ALL_DAYS, "LSTM-MLP-OF-recent": RECENT_DAYS,
           "Ridge-OF-alldays": ALL_DAYS, "Ridge-OF-recent": RECENT_DAYS}
REPORT_ORDER = ("CNN-LSTM-OF-alldays", "LSTM-MLP-OF-alldays", "LSTM-MLP-OF-recent", "Ridge-OF-alldays", "Ridge-OF-recent")
PAIRS = (("LSTM-MLP-OF-alldays", "LSTM-MLP-OF-recent"), ("Ridge-OF-alldays", "Ridge-OF-recent"))


def model_headline(m: str) -> HeadlineSpec:
    t, all_k, k10 = f"test_primary:{m}", (m, TEST_DAYS_LABEL, "k=1..10"), (m, TEST_DAYS_LABEL, "k=10")
    return HeadlineSpec(
        f"`{m}` ({TRAINED[m]}), test days {TEST_DAYS_WORDS}, primary analysis: R² {{r2}} % (s.e. {{se}}; {{stocks}} "
        "stocks; k = 1..10); k = 10: {r2_10} % (s.e. {se_10}).",
        (("r2", t, all_k, "r2"), ("se", t, all_k, "se"), ("stocks", t, all_k, "stocks"), ("r2_10", t, k10, "r2"),
         ("se_10", t, k10, "se")))


def older_headline(a: str, b: str) -> HeadlineSpec:
    row = (a, b, "k=1..10")
    return HeadlineSpec(
        f"Does the older data help, test days {TEST_DAYS_WORDS}, primary analysis: `{a}` minus `{b}` {{diff}} "
        "percentage points (s.e. {se}; {stocks} stocks in both; k = 1..10).",
        (("diff", "older_test_primary", row, "diff"), ("se", "older_test_primary", row, "se"),
         ("stocks", "older_test_primary", row, "stocks")))


def model_section(m: str) -> SectionSpec:
    tables = [
        ts("score_rows", f"test_primary:{m}", "Test days, primary analysis", record="test", model=m,
           golden="test_primary_scores"),
        ts("score_rows", f"test_secondary:{m}", "Test days, secondary analysis (price ratio inside [0.6, 1.6])",
           record="test", model=m, secondary=True, golden="test_secondary_scores"),
    ]
    if m in FITS:
        tables += [
            ts("score_rows", f"valid:{m}", "Validation days (they chose the stopping check: not out-of-sample for that "
               "choice)", record="fit_valid", model=m, golden="valid_scores"),
            ts("fits_table", f"fit:{m}", "The fit", models=(m,), seeds=(0,)),
            ts("checks_table", f"checks:{m}", "Validation loss by check", model=m, seed=0, golden="checks"),
        ]
    else:
        tables += [
            ts("score_rows", f"valid:{m}", "Validation days (λ was chosen on them: not out-of-sample for that choice)",
               record="fit_valid", model=m, golden="valid_scores"),
            ts("linear_table", f"linear:{m}", "The ridge reference", models=(m,)),
        ]
    return SectionSpec(m, m, tables=tuple(tables))


NO_R2_TITLE = "Cells without an R² (labels of zero variance), left out of the means"

SECTIONS = (
    SectionSpec("result", "Result", paragraphs=TEXT.RESULT,
                headlines=(*(model_headline(m) for m in REPORT_ORDER[:3]), older_headline(*PAIRS[0])),
                subsections=(SectionSpec("references", "References",
                                         headlines=(model_headline("Ridge-OF-alldays"), model_headline("Ridge-OF-recent"),
                                                    older_headline(*PAIRS[1]))),)),
    SectionSpec("method", "Method", paragraphs=TEXT.METHOD),
    SectionSpec("models", "Results by model", subsections=tuple(model_section(m) for m in REPORT_ORDER)),
    SectionSpec("older", "Does the older data help", tables=(
        ts("older_rows", "older_test_primary", "Test days, primary analysis", record="test", golden="test_primary_older"),
        ts("older_rows", "older_test_secondary", "Test days, secondary analysis (price ratio inside [0.6, 1.6])",
           record="test", secondary=True, golden="test_secondary_older"),
        ts("older_rows", "older_valid", "Validation days (not out-of-sample for the stopping check and λ)",
           record="fit_valid", golden="valid_older"),
    )),
    SectionSpec("data", "Data and split", paragraphs=TEXT.DATA, tables=(
        ts("left_out_table", "ca_left_out", "Test stock-days left out by a corporate action", record="test",
           model="LSTM-MLP-OF-recent"),
        ts("no_r2_table", "no_r2_valid", f"{NO_R2_TITLE}: validation days", record="fit_valid", models=FITS + LINEAR,
           golden="no_r2_valid"),
        ts("no_r2_table", "no_r2_test_primary", f"{NO_R2_TITLE}: test days, primary analysis", record="test",
           models=FITS + LINEAR, golden="no_r2_test_primary"),
        ts("no_r2_table", "no_r2_test_secondary", f"{NO_R2_TITLE}: test days, secondary analysis", record="test",
           models=FITS + LINEAR, secondary=True, golden="no_r2_test_secondary"),
    )),
    SectionSpec("validity", "Validity", paragraphs=TEXT.VALIDITY),
    SectionSpec("reproducibility", "Reproducibility", paragraphs=TEXT.REPRO,
                tables=(ts("records_table", "records", "Stored records behind the tables"),)),
)

SPEC = ExperimentSpec(
    name="alldays", title=TEXT.TITLE, intro=TEXT.INTRO, root="results", config="alldays.toml",
    steps=("fit", "score", "collect", "report"), fits=FITS, linear=LINEAR, models=FITS + LINEAR,
    report_order=REPORT_ORDER, seeds=(0,), rounds=(tuple((f, 0) for f in FITS),),
    parts=(Part("valid", VALID_DAYS, ("fit_valid",), False), Part("test", TEST_DAYS, ("test",), True)),
    sources=(RecordSource("fit_valid", "{model}/fit_seed0.json", ("main",), "finished"),
             RecordSource("test", "{model}/test_seed0.json", ("main", "corporate_actions"))),
    exclude=("smoke/",),
    protocol=Protocol(PROTOCOL_MAIN, None, CORPORATE_ACTIONS, MAIN_DOCUMENT, None),
    pairs=PAIRS,
    sections=SECTIONS,
)
