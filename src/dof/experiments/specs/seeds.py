"""The seeds of the main run's three fits, round by round in the pre-registered order, and the ensembles of the
first three seeds. Seed 0 is the main run's."""

from __future__ import annotations

from dof.experiments.spec import (
    ExperimentSpec,
    HeadlineSpec,
    Part,
    Protocol,
    RecordSource,
    SectionSpec,
    SeedPolicy,
    pair_round,
    ts,
)
from dof.experiments.specs import (
    CORPORATE_ACTIONS,
    FITS,
    MAIN_DOCUMENT,
    PROTOCOL_MAIN,
    PROTOCOL_SEEDS,
    TEST_DAYS,
    TEST_DAYS_LABEL,
    TEST_DAYS_WORDS,
    VALID_DAYS,
)
from dof.experiments.specs import seeds_text as TEXT

FIRST = (0, 3089492870, 771810125)
DRAWN: tuple[int, ...] = ()          # the seeds the rule drew, in drawn order; set from the stored decision
SEEDS = FIRST + DRAWN
ROUNDS = (pair_round(FITS, 3089492870, 771810125),) + ((pair_round(FITS, *DRAWN),) if DRAWN else ())
REPORT_ORDER = ("CNN-LSTM-OF-alldays", "LSTM-MLP-OF-alldays", "LSTM-MLP-OF-recent")
OLDER, RECENT = "LSTM-MLP-OF-alldays", "LSTM-MLP-OF-recent"
ALL_K = (TEST_DAYS_LABEL, "k=1..10")


def fit_headline(f: str) -> HeadlineSpec:
    t, row = f"seed_test_primary:{f}", (f, *ALL_K)
    return HeadlineSpec(
        f"`{f}`, test days {TEST_DAYS_WORDS}, primary analysis, k = 1..10, {{stocks}} stocks: mean {{mean}} % over "
        "{n} seeds (s.d. {sd}; min {min}, max {max}); ensemble of the first three seeds {ens} % ({ens_stocks} stocks).",
        tuple((ph, t, row, col) for ph, col in (("stocks", "stocks"), ("mean", "mean"), ("n", "seeds"), ("sd", "sd"),
                                                ("min", "min"), ("max", "max"), ("ens", "ensemble"),
                                                ("ens_stocks", "ensemble_stocks"))))


OLDER_HEADLINE = HeadlineSpec(
    f"Does the older data help, test days {TEST_DAYS_WORDS}, primary analysis, k = 1..10: mean {{mean}} percentage "
    "points over {n} seeds (s.d. {sd}; {stocks} stocks in both); the two ensembles {ens}.",
    tuple((ph, "seed_older_test_primary", ALL_K, col) for ph, col in (("mean", "mean"), ("n", "seeds"), ("sd", "sd"),
                                                                    ("stocks", "stocks"), ("ens", "ensembles"))))


def fit_section(f: str) -> SectionSpec:
    return SectionSpec(f, f, tables=(
        ts("seed_scores", f"seed_test_primary:{f}", "Test days, primary analysis", record="test", fit=f,
           golden="seed_test_primary"),
        ts("seed_scores", f"seed_test_secondary:{f}", "Test days, secondary analysis (price ratio inside [0.6, 1.6])",
           record="test", fit=f, secondary=True, golden="seed_test_secondary"),
        ts("seed_scores", f"seed_valid:{f}", "Validation days (they chose the stopping check: not out-of-sample for "
           "that choice)", record="fit_valid", fit=f, golden="seed_valid"),
        ts("fits_table", f"fits:{f}", "The fits", models=(f,), seeds=SEEDS, golden="fits"),
        *(ts("checks_table", f"checks:{f}:{s}", f"Validation loss by check, seed {s}", model=f, seed=s, golden="checks")
          for s in SEEDS),
    ))


SECTIONS = (
    SectionSpec("result", "Result", paragraphs=TEXT.RESULT,
                headlines=(*(fit_headline(f) for f in REPORT_ORDER), OLDER_HEADLINE)),
    SectionSpec("method", "Method", paragraphs=TEXT.METHOD),
    SectionSpec("fits", "Results by fit", subsections=tuple(fit_section(f) for f in REPORT_ORDER)),
    SectionSpec("older", "Does the older data help", tables=(
        ts("seed_older", "seed_older_test_primary", "Test days, primary analysis", record="test",
           golden="seed_older_test_primary"),
        ts("seed_older", "seed_older_test_secondary", "Test days, secondary analysis (price ratio inside [0.6, 1.6])",
           record="test", secondary=True, golden="seed_older_test_secondary"),
        ts("seed_older", "seed_older_valid", "Validation days (not out-of-sample for the stopping check)",
           record="fit_valid", golden="seed_older_valid"),
    )),
    SectionSpec("rule", "The rule for two more seeds", paragraphs=TEXT.RULE,
                tables=(ts("seed_rule", "rule", "The rule, on validation numbers only"),)),
    SectionSpec("data", "Data", tables=(
        ts("left_out_table", "ca_left_out", "Test stock-days left out by a corporate action", record="test",
           model="LSTM-MLP-OF-recent", seed=3089492870),
        ts("finite_table", "finite", "Cells without an R²", golden="finite"),
    )),
    SectionSpec("validity", "Validity", paragraphs=TEXT.VALIDITY),
    SectionSpec("reproducibility", "Reproducibility", paragraphs=TEXT.REPRO,
                tables=(ts("records_table", "records", "Stored records behind the tables"),)),
)

SPEC = ExperimentSpec(
    name="seeds", title=TEXT.TITLE, intro=TEXT.INTRO, root="results_seeds", config="alldays.toml",
    steps=("fit", "score", "decide", "collect", "report"), fits=FITS, linear=(), models=FITS,
    report_order=REPORT_ORDER, seeds=SEEDS, rounds=ROUNDS,
    parts=(Part("valid", VALID_DAYS, ("fit_valid", "ensemble"), False),
           Part("test", TEST_DAYS, ("test", "ensemble"), True)),
    sources=(RecordSource("fit_valid", "seed{seed}/{model}/fit_seed{seed}.json", ("main",), "finished_saved", None),
             RecordSource("test", "seed{seed}/{model}/test_seed{seed}.json", ("main", "extension", "corporate_actions"),
                          "none", None),
             RecordSource("ensemble", "ensemble/{model}/{part}_first3seeds.json", ("main", "extension"), "none", None)),
    exclude=(),
    protocol=Protocol(PROTOCOL_MAIN, PROTOCOL_SEEDS, CORPORATE_ACTIONS, MAIN_DOCUMENT,
                      "protocol/preregistration_seeds.md"),
    pairs=((OLDER, RECENT),),
    seed0_from="alldays",
    uses=("alldays",),
    seed_policy=SeedPolicy(first=FIRST, drawn=DRAWN, older=OLDER, recent=RECENT, half=0.5, more=2),
    sections=SECTIONS,
)
