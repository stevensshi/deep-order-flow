"""Specs over the synthetic training run of ``tests/deep_synth.py`` (not collected): a main-run kind and a seeds kind
with the seed rule forced to draw."""

from __future__ import annotations

from dof.experiments.runner import RUN_LAYOUT
from dof.experiments.spec import ExperimentSpec, Part, Protocol, SectionSpec, SeedPolicy, pair_round, ts

FITS = ("TinyMLP-all", "TinyLSTM-recent")
LINEAR = ("Ridge-all", "Ridge-recent")


def _parts(cfg) -> tuple[Part, ...]:
    return (Part("valid", cfg.split.valid_days, ("fit_valid",), False), Part("test", cfg.split.test_days, ("test",), True))


def alldays_like(cfg) -> ExperimentSpec:
    tables = tuple(ts("score_rows", f"test:{m}", f"Test day, {m}", record="test", model=m) for m in FITS + LINEAR)
    return ExperimentSpec(
        name="alldays", title="A synthetic main run", intro=("Two tiny networks and two ridge references.",), root=".",
        config=None, steps=("fit", "score", "collect", "report"), fits=FITS, linear=LINEAR, models=FITS + LINEAR,
        report_order=FITS + LINEAR, seeds=(0,), rounds=(tuple((f, 0) for f in FITS),), parts=_parts(cfg),
        sources=RUN_LAYOUT, exclude=(), protocol=Protocol(None, None, cfg.protocol.corporate_actions_sha256),
        pairs=(FITS,),
        sections=(SectionSpec("scores", "Scores", tables=tables),
                  SectionSpec("older", "Older data", tables=(ts("older_rows", "older", "Test day", record="test"),)),
                  SectionSpec("fits", "Fits", tables=(ts("fits_table", "fits", "The fits", models=FITS, seeds=(0,)),))))


def seeds_like(cfg) -> ExperimentSpec:
    tables = tuple(ts("seed_scores", f"seeds:{f}", f"Test day, {f}", record="test", fit=f) for f in FITS)
    return ExperimentSpec(
        name="seeds", title="Synthetic seeds", intro=("Seeds 1 and 2, then the rule.",), root=".", config=None,
        steps=("fit", "score", "decide", "collect", "report"), fits=FITS, linear=(), models=FITS, report_order=FITS,
        seeds=(0, 1, 2), rounds=(pair_round(FITS, 1, 2),), parts=_parts(cfg), sources=RUN_LAYOUT, exclude=(),
        protocol=Protocol(None, None, cfg.protocol.corporate_actions_sha256), seed0_from="alldays", uses=("alldays",),
        seed_policy=SeedPolicy(first=(0, 1, 2), drawn=(), older=FITS[0], recent=FITS[1], half=0.5, more=2,
                               force_more=True),
        sections=(SectionSpec("scores", "Scores", tables=tables),
                  SectionSpec("rule", "Rule", tables=(ts("seed_rule", "rule", "The rule"),))))
