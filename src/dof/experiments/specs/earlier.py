"""The earlier stage's pooled models on 115 stocks: three networks and the pooled ridge reference, seed 0, trained on
three days of 2019. Reported from its stored records; no protocol document exists for this stage."""

from __future__ import annotations

from dof.experiments.spec import ExperimentSpec, HeadlineSpec, Part, Protocol, RecordSource, SectionSpec, ts
from dof.experiments.specs import earlier_text as TEXT

NETWORKS = ("LSTM-OF", "LSTM-MLP-OF", "CNN-LSTM-OF-lr3e-5-eps1e-7")
RIDGE = "Ridge-OF"
MODELS = NETWORKS + (RIDGE,)
VALID_2019 = ("2019-08-30",)
TEST_2019 = ("2019-10-18", "2019-10-30", "2019-12-30")
TEST_2019_LABEL = ",".join(TEST_2019)
TEST_2019_WORDS = "2019-10-18, 2019-10-30 and 2019-12-30"
POOLED_RIDGE = (("training", "pooled"), ("model", RIDGE))


def model_headline(m: str) -> HeadlineSpec:
    t, all_k, k10 = f"test:{m}", (m, TEST_2019_LABEL, "k=1..10"), (m, TEST_2019_LABEL, "k=10")
    return HeadlineSpec(
        f"`{m}`, test days {TEST_2019_WORDS}: R² {{r2}} % (s.e. {{se}}; {{stocks}} stocks; k = 1..10); k = 10: "
        "{r2_10} % (s.e. {se_10}).",
        (("r2", t, all_k, "r2"), ("se", t, all_k, "se"), ("stocks", t, all_k, "stocks"), ("r2_10", t, k10, "r2"),
         ("se_10", t, k10, "se")))


def model_section(m: str) -> SectionSpec:
    choice = "λ was chosen on it" if m == RIDGE else "it chose the best pass"
    return SectionSpec(m, m, tables=(
        ts("score_rows", f"test:{m}", "Test days", record="test", model=m, golden="test_scores"),
        ts("score_rows", f"valid:{m}", f"Validation day ({choice}: not out-of-sample for that choice)", record="valid",
           model=m, golden="valid_scores"),
    ))


SECTIONS = (
    SectionSpec("result", "Result", paragraphs=TEXT.RESULT, headlines=tuple(model_headline(m) for m in MODELS)),
    SectionSpec("method", "Method", paragraphs=TEXT.METHOD),
    SectionSpec("models", "Results by model", subsections=tuple(model_section(m) for m in MODELS)),
    SectionSpec("rescore", "The same weights scored again by the decay measurement", paragraphs=TEXT.RESCORE, tables=(
        ts("rescore_table", "rescore", "Stored scores and the second scoring, k = 1..10", other="decay",
           basis="frozen", models=NETWORKS, golden="rescore"),
    )),
    SectionSpec("validity", "Validity", paragraphs=TEXT.VALIDITY),
    SectionSpec("reproducibility", "Reproducibility", paragraphs=TEXT.REPRO,
                tables=(ts("records_table", "records", "Stored records behind the tables"),)),
)

SPEC = ExperimentSpec(
    name="earlier", title=TEXT.TITLE, intro=TEXT.INTRO, root="pooled115", config=None, steps=("collect", "report"),
    fits=(), linear=(), models=MODELS, report_order=MODELS, seeds=(0,), rounds=(),
    parts=(Part("valid", VALID_2019, ("valid",), False), Part("test", TEST_2019, ("test",), False)),
    sources=(RecordSource("valid", "{model}/pooled_seed0.json", (), part="valid"),
             RecordSource("test", "{model}/pooled_seed0.json", (), part="test"),
             RecordSource("valid", "linear/linear.json", (), part="valid", model=RIDGE, select=POOLED_RIDGE),
             RecordSource("test", "linear/linear.json", (), part="test", model=RIDGE, select=POOLED_RIDGE)),
    exclude=("CNN-LSTM-OF-lr0.3-eps1/",),
    protocol=Protocol(None, None, None),
    uses=("decay",),
    sections=SECTIONS,
)
