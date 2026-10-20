"""Fit-progress tables: the fits, the validation loss by check, the linear references."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dof.experiments.cells import plain
from dof.experiments.tables import (
    Column,
    Row,
    Table,
    count,
    flag,
    general,
    label,
    loss,
    passes,
    sci,
    seed_label,
    steps,
    text,
)

if TYPE_CHECKING:
    from dof.experiments.cells import ResultTables
    from dof.experiments.spec import ExperimentSpec


def _fit_record(rt: ResultTables, spec: ExperimentSpec, model: str, seed: int) -> str | None:
    return rt.rid(experiment=spec.experiment_of(seed), record="fit_valid", model=model, seed=seed)


def fits_table(rt: ResultTables, spec: ExperimentSpec, *, models: tuple[str, ...], seeds: tuple[int, ...], id: str,
               title: str, golden: str | None = None) -> Table:
    """One row per exported fit record: checks, the best check, why it stopped, steps, passes, the best validation
    loss, interruptions and the first digits of the saved weights' sha256."""
    out = []
    for m in models:
        for s in seeds:
            rid = _fit_record(rt, spec, m, s)
            if rid is None:
                continue
            f = rt.fit(rid)
            sha = f["weights_sha256"]
            out.append(Row((m, seed_label(s)), (
                text(m), text(str(s)), text(f["status"]), count(f["n_checks"]), count(f["best_check"]),
                text(f["stopped_by"]), steps(f["steps"]), passes(f["passes"]), loss(f["best_valid_loss"]),
                count(f["resumes"]), text(None if sha is None else sha[:16] + "…"))))
    columns = (label("model"), label("seed"), Column("status", "status", align="left"), Column("checks", "checks"),
               Column("best_check", "best check"), Column("stopped_by", "stopped by", align="left"),
               Column("steps", "steps"), Column("passes", "passes"), Column("best_valid_loss", "best validation loss"),
               Column("interruptions", "interruptions"), Column("weights", "sha256 of the saved weights", align="left"))
    return Table(id, golden, title, columns, tuple(out), "valid")


def checks_table(rt: ResultTables, spec: ExperimentSpec, *, model: str, seed: int, id: str, title: str,
                 golden: str | None = None) -> Table:
    """The validation loss at each check of one fit, with the training loss since the check before."""
    rid = _fit_record(rt, spec, model, seed)
    out = []
    if rid is not None:
        ck = rt.checks_of(rid)
        cols = ("check", "steps", "passes", "train_loss", "valid_loss", "new_best", "median_abs_dw")
        for n, st, ps, tl, vl, nb, dw in zip(*(ck[c].tolist() for c in cols)):
            n = plain(n)
            out.append(Row((model, seed_label(seed), f"check={n}"), (
                text(str(n)), steps(plain(st)), passes(plain(ps)), loss(plain(tl)), loss(plain(vl)), flag(plain(nb)),
                sci(plain(dw)))))
    columns = (label("check"), Column("steps", "steps"), Column("passes", "passes"),
               Column("train_loss", "training loss since the last check"), Column("valid_loss", "validation loss"),
               Column("new_best", "new best", align="left"), Column("median_dw", "median weight step"))
    return Table(id, golden, title, columns, tuple(out), "valid")


def linear_table(rt: ResultTables, spec: ExperimentSpec, *, models: tuple[str, ...], id: str, title: str) -> Table:
    """The ridge references: λ chosen on the validation loss, training samples and stocks."""
    out = []
    for m in models:
        rid = _fit_record(rt, spec, m, 0)
        if rid is None:
            continue
        f = rt.fit(rid)
        out.append(Row((m,), (text(m), general(f["lambda"]), steps(f["train_samples"]), count(f["train_stocks"]))))
    columns = (label("model"), Column("lambda", "λ", compare=False), Column("samples", "training samples", compare=False),
               Column("stocks", "stocks", compare=False))
    return Table(id, None, title, columns, tuple(out), None)
