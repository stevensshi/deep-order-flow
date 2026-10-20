"""The decay measurement: two models trained on three days of 2019, scored on fifteen later days on a frozen and a
refreshed basis. Reported from its stored records; this package does not run its data steps."""

from __future__ import annotations

from dof.experiments.spec import (
    DecayDay,
    DecayPlan,
    ExperimentSpec,
    HeadlineSpec,
    Part,
    Protocol,
    RecordSource,
    SectionSpec,
    ts,
)
from dof.experiments.specs import MAIN_DOCUMENT, PROTOCOL_DECAY, PROTOCOL_MAIN
from dof.experiments.specs import decay_text as TEXT

CNN, LSTM = "CNN-LSTM-OF-lr3e-5-eps1e-7", "LSTM-MLP-OF"
MODELS = (CNN, LSTM)
FROZEN_ONLY = ("2019-10-18", "2019-10-30", "2019-12-30", "2020-01-30", "2021-07-13", "2025-11-28", "2025-12-08",
               "2025-12-09", "2025-12-10")
BOTH = ("2021-08-13", "2025-12-11", "2025-12-12", "2026-05-15", "2026-05-18", "2026-06-12")
DAYS = tuple(DecayDay(d, ("frozen", "refreshed") if d in BOTH else ("frozen",), TEXT.NOTES.get(d, ""), TEXT.ROLES[d])
             for d in sorted(FROZEN_ONLY + BOTH))
TEST_2019 = ("2019-10-18", "2019-10-30", "2019-12-30")


def _row(model: str, date: str) -> tuple[str, ...]:
    return (model, "k=1..10", date, "all")


FROZEN_HEADLINE = HeadlineSpec(
    "Frozen 2019 basis on the 2019 test days (2.6 to 5.0 months after training), row 'all', k = 1..10: "
    f"`{CNN}` {{c1}}, {{c2}}, {{c3}} % ({{cs1}}, {{cs2}}, {{cs3}} stocks); "
    f"`{LSTM}` {{l1}}, {{l2}}, {{l3}} % ({{ls1}}, {{ls2}}, {{ls3}} stocks).",
    tuple((f"{p}{i}", f"decay:{m}:k=1..10", _row(m, d), col)
          for m, pre in ((CNN, "c"), (LSTM, "l")) for p, col in ((pre, "i_r2"), (pre + "s", "i_stocks"))
          for i, d in enumerate(TEST_2019, 1)))
REFRESHED_HEADLINE = HeadlineSpec(
    "Refreshed basis (scalers and Δt of 2025-11-28 to 2025-12-10), 2025-12-12, row 'all', k = 1..10: "
    f"`{CNN}` {{c}} % ({{cn}} stocks); `{LSTM}` {{l}} % ({{ln}} stocks).",
    tuple((ph, f"decay:{m}:k=1..10", _row(m, "2025-12-12"), col)
          for m, pre in ((CNN, "c"), (LSTM, "l")) for ph, col in ((pre, "ii_r2"), (pre + "n", "ii_stocks"))))

SECTIONS = (
    SectionSpec("result", "Result", paragraphs=TEXT.RESULT + TEXT.LATER, headlines=(FROZEN_HEADLINE, REFRESHED_HEADLINE)),
    SectionSpec("method", "Method", paragraphs=TEXT.METHOD),
    SectionSpec("models", "Results by model", subsections=tuple(
        SectionSpec(m, m, tables=(
            ts("decay_table", f"decay:{m}:k=1..10", "k = 1..10", model=m, ks=None, golden="table"),
            ts("decay_table", f"decay:{m}:k=10", "k = 10", model=m, ks=(10,), golden="table"),
        )) for m in MODELS)),
    SectionSpec("days", "The days", paragraphs=TEXT.DAYS, tables=(
        ts("decay_days", "days", "Stock-days per day and basis", golden="days"),
        ts("decay_outside", "outside", "Stock-days with the price ratio outside [0.6, 1.6]", golden="outside"),
    )),
    SectionSpec("validity", "Validity", paragraphs=TEXT.VALIDITY),
    SectionSpec("reproducibility", "Reproducibility", paragraphs=TEXT.REPRO,
                tables=(ts("records_table", "records", "Stored records behind the tables",
                           by=("model", "basis", "date")),)),
)

SPEC = ExperimentSpec(
    name="decay", title=TEXT.TITLE, intro=TEXT.INTRO, root="decay/scores", config=None, steps=("collect", "report"),
    fits=(), linear=(), models=MODELS, report_order=MODELS, seeds=(0,), rounds=(),
    parts=(Part("decay", tuple(d.date for d in DAYS), ("decay",), False),),
    sources=(RecordSource("decay", "{model}/{basis}/{date}/{symbol}.json", ("extension",)),),
    exclude=(),
    protocol=Protocol(PROTOCOL_MAIN, PROTOCOL_DECAY, None, MAIN_DOCUMENT, "protocol/preregistration_decay.md"),
    decay=DecayPlan(MODELS, DAYS, distance_from="2019-07-30", month_days=30.44, ratio_lo=0.6, ratio_hi=1.6),
    sections=SECTIONS,
)
