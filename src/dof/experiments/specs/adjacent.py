"""The adjacent-days experiment: one network with three seeds and one ridge reference trained on five adjacent days,
validated on a morning, tested on its afternoon, and the same weights scored on three later days. Reported from its
stored records; this package does not run its data steps."""

from __future__ import annotations

from dof.experiments.spec import (
    AdjacentPlan,
    ExperimentSpec,
    HeadlineSpec,
    Part,
    Protocol,
    RecordSource,
    SectionSpec,
    ts,
)
from dof.experiments.specs import CORPORATE_ACTIONS, FITS, MAIN_DOCUMENT, PROTOCOL_ADJACENT, PROTOCOL_MAIN, TEST_DAYS
from dof.experiments.specs import adjacent_text as TEXT

NET, RIDGE = "LSTM-MLP-OF-recent", "Ridge-OF-recent"
SEEDS = (3089492870, 771810125, 18531247)
PINS = ("main", "extension", "corporate_actions")
ALL_K = (NET, "all", "k=1..10")
SEED_REFS = tuple((f"s{i}", f"seed={s}") for i, s in enumerate(SEEDS, 1))


def part_headline(template: str, table: str) -> HeadlineSpec:
    return HeadlineSpec(template, tuple((ph, table, ALL_K, col) for ph, col in (
        *SEED_REFS, ("mean", "mean"), ("sd", "sd"), ("se", "se"), ("stocks", "stocks"))))


AFTERNOON_HEADLINE = part_headline(
    "Test, afternoon of 2025-12-12 (13:00 on; one half day; a second look), LSTM-MLP-OF, 5 days: seed 3089492870 {s1}, "
    "seed 771810125 {s2}, seed 18531247 {s3}; mean {mean} %, s.d. across seeds {sd}, s.e. across stocks {se} "
    "({stocks} stocks; k = 1..10).", f"part_afternoon:{NET}")
FAR_HEADLINE = part_headline(
    "The same weights on 2026-05-15, 2026-05-18 and 2026-06-12 (days scored before; primary analysis): mean {mean} %, "
    "s.d. across seeds {sd}, s.e. across stocks {se} ({stocks} stocks; k = 1..10).", f"part_far:{NET}")
D_HEADLINE = HeadlineSpec(
    "Paired difference D, afternoon minus the 2026 days, over {n} stocks scored on both parts (k = 1..10; 2026 days: "
    "primary analysis): D = {d} percentage points; U = {u} (s.e. across stocks {se}, s.d. across seeds {sd}); "
    "D/U = {ratio}; {relation}.",
    tuple((ph, "pair", row, col) for ph, row, col in (
        ("n", ("n_pairs",), "value"), ("d", ("d",), "value"), ("u", ("u",), "value"), ("se", ("se_stocks",), "value"),
        ("sd", ("sd_seeds",), "value"), ("ratio", ("d_over_u",), "value"), ("relation", ("relation",), "relation"))))


def part_tables(model: str, valid_title: str) -> tuple:
    return (
        ts("part_table", f"part_valid:{model}", valid_title, part="valid", model=model, golden="valid"),
        ts("part_table", f"part_afternoon:{model}", "Test: afternoon of 2025-12-12 (window, row and labels at or after "
           "13:00)", part="afternoon", model=model, golden="afternoon"),
        ts("part_table", f"part_far:{model}", "Test: the same weights on 2026-05-15, 2026-05-18, 2026-06-12",
           part="far", model=model, golden="far"),
    )


SECTIONS = (
    SectionSpec("result", "Result", paragraphs=TEXT.RESULT, headlines=(AFTERNOON_HEADLINE, FAR_HEADLINE, D_HEADLINE)),
    SectionSpec("method", "Method", paragraphs=TEXT.METHOD),
    SectionSpec("models", "Results by model", subsections=(
        SectionSpec(NET, "LSTM-MLP-OF, 5 days (2025-11-28 to 2025-12-11)", tables=(
            *part_tables(NET, "Validation: morning of 2025-12-12 (rows and labels before 13:00; the fits stopped on it)"),
            ts("fits_table", "fits", "The fits", models=(NET,), seeds=SEEDS, golden="fits"),
            *(ts("checks_table", f"checks:{s}", f"Validation loss by check, seed {s}", model=NET, seed=s, golden="checks")
              for s in SEEDS),
        )),
        SectionSpec(RIDGE, "Ridge-OF, 5 days (2025-11-28 to 2025-12-11)", tables=(
            *part_tables(RIDGE, "Validation: morning of 2025-12-12 (λ was chosen on it)"),
            ts("linear_table", "linear", "The ridge reference", models=(RIDGE,)),
        )),
    )),
    SectionSpec("pair", "The afternoon against the 2026 days", paragraphs=TEXT.PAIR,
                tables=(ts("pair_table", "pair", "Paired difference", golden="pair"),)),
    SectionSpec("reference", "Selection-bias reference: the main run's seed-0 fits on the same afternoon",
                paragraphs=TEXT.REFERENCE,
                tables=(ts("reference_table", "reference", "Main-run fits on the afternoon of 2025-12-12",
                           golden="reference"),)),
    SectionSpec("validity", "Status of the days and validity", paragraphs=TEXT.VALIDITY),
    SectionSpec("reproducibility", "Reproducibility", paragraphs=TEXT.REPRO,
                tables=(ts("records_table", "records", "Stored records behind the tables"),)),
)

SPEC = ExperimentSpec(
    name="adjacent", title=TEXT.TITLE, intro=TEXT.INTRO, root="adj/results", config=None, steps=("collect", "report"),
    fits=(NET,), linear=(RIDGE,), models=(NET, RIDGE), report_order=(NET, RIDGE), seeds=SEEDS,
    rounds=(tuple((NET, s) for s in SEEDS),),
    parts=(Part("valid", ("2025-12-12",), ("fit_valid",), False, None, "13:00"),
           Part("afternoon", ("2025-12-12",), ("test", "reference"), True, "13:00", None),
           Part("far", TEST_DAYS, ("far",), True)),
    sources=(RecordSource("fit_valid", "seed{seed}/{model}/fit_seed{seed}.json", ("main",), "finished", None),
             RecordSource("test", "seed{seed}/{model}/test_seed{seed}.json", PINS, "needs_finished_fit", None),
             RecordSource("far", "seed{seed}/{model}/far_seed{seed}.json", PINS, "needs_finished_fit", None),
             RecordSource("reference", "reference/{model}/test_seed0.json", ("main", "extension")),
             RecordSource("fit_valid", "ridge/{model}/fit_seed0.json", ("main",), "finished"),
             RecordSource("test", "ridge/{model}/test_seed0.json", PINS, "needs_finished_fit"),
             RecordSource("far", "ridge/{model}/far_seed0.json", PINS, "needs_finished_fit")),
    exclude=(),
    protocol=Protocol(PROTOCOL_MAIN, PROTOCOL_ADJACENT, CORPORATE_ACTIONS, MAIN_DOCUMENT,
                      "protocol/preregistration_adjacent.md"),
    adjacent=AdjacentPlan(model=NET, ridge=RIDGE, train_days=("2025-11-28", "2025-12-08", "2025-12-09", "2025-12-10",
                                                              "2025-12-11"),
                          cut_day="2025-12-12", cut="13:00", far_days=TEST_DAYS, reference_fits=FITS),
    sections=SECTIONS,
)
