"""Score tables of one experiment: one model's rows, the older-data comparison, cells without an R², the stock-days
a corporate action leaves out, and the records behind the tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dof.experiments.analyses.stocks import agg, inside, paired
from dof.experiments.cells import plain
from dof.experiments.tables import Column, Row, Table, count, days_label, horizon_label, label, pct, text

if TYPE_CHECKING:
    from dof.experiments.cells import ResultTables, Rows
    from dof.experiments.spec import ExperimentSpec


def part_of(record: str) -> str:
    return "valid" if record in ("fit_valid", "valid") else "test"


def horizon_text(ks) -> str:
    return "k = 1..10" if ks is None else "k = " + ", ".join(str(k) for k in ks)


def record_rows(rt: ResultTables, spec: ExperimentSpec, record: str, model: str, seed: int = 0,
                secondary: bool = False) -> Rows | None:
    """The cells of one record of the spec (seed 0 from the experiment that holds it); the inside-range cells when
    ``secondary``; None when the record is not exported."""
    rid = rt.rid(experiment=spec.experiment_of(seed), record=record, model=model, seed=seed)
    if rid is None:
        return None
    rows = rt.rows(rid)
    return inside(rows, rt.stock_days_of(rid)) if secondary else rows


def part_days(spec: ExperimentSpec, record: str) -> tuple[str, ...]:
    return next(p.days for p in spec.parts if record in p.records)


SCORE_COLUMNS = (label("days scored", "days"), label("horizon"), Column("stocks", "stocks"), Column("r2", "R² %"),
                 Column("se", "s.e."))


def score_rows(rt: ResultTables, spec: ExperimentSpec, *, record: str, model: str, secondary: bool = False, id: str,
               title: str, golden: str | None = None) -> Table:
    """One model on one part: all days and k = 1..10, all days and k = 10, then each day alone (when the record holds
    more than one day)."""
    rows = record_rows(rt, spec, record, model, 0, secondary)
    out = []
    if rows is not None:
        present = sorted(set(rows.date))
        days = [(None, {d}) for d in present] if len(present) > 1 else []
        for ks, dates in [(None, None), ((10,), None), *days]:
            a = agg(rows, ks, dates)
            out.append(Row((model, days_label(dates or present), horizon_label(ks)),
                           (text(", ".join(sorted(dates)) if dates else ", ".join(present)), text(horizon_text(ks)),
                            count(a["n_stocks"]), pct(a["mean"]), pct(a["se"]))))
    return Table(id, golden, title, SCORE_COLUMNS, tuple(out), part_of(record))


def older_rows(rt: ResultTables, spec: ExperimentSpec, *, record: str, secondary: bool = False, id: str, title: str,
               golden: str | None = None) -> Table:
    """Does the older data help: the first model of each pair minus the second, over the stocks scored under both."""
    days = ", ".join(part_days(spec, record))
    out = []
    for a, b in spec.pairs:
        ra, rb = record_rows(rt, spec, record, a, 0, secondary), record_rows(rt, spec, record, b, 0, secondary)
        if ra is None or rb is None:
            continue
        for ks in (None, (10,)):
            p = paired(ra, rb, ks)
            out.append(Row((a, b, horizon_label(ks)),
                           (text(f"{a} minus {b}"), text(days), text(horizon_text(ks)), count(p["n_stocks"]),
                            pct(p["mean"]), pct(p["se"]), count(p["n_better"]))))
    columns = (label("comparison"), label("days"), label("horizon"), Column("stocks", "stocks in both"),
               Column("diff", "difference in R², percentage points"), Column("se", "s.e."),
               Column("higher", "stocks where the first is higher"))
    return Table(id, golden, title, columns, tuple(out), part_of(record))


def no_r2_table(rt: ResultTables, spec: ExperimentSpec, *, record: str, models: tuple[str, ...], secondary: bool = False,
                id: str, title: str, golden: str | None = None) -> Table:
    """Per model: the cells without an R² (labels of zero variance), which every mean leaves out."""
    out = []
    for m in models:
        rows = record_rows(rt, spec, record, m, 0, secondary)
        if rows is not None:
            out.append(Row((m,), (text(m), count(agg(rows)["n_no_r2"]))))
    return Table(id, golden, title, (label("model"), Column("cells", "cells")), tuple(out), part_of(record))


def left_out_table(rt: ResultTables, spec: ExperimentSpec, *, record: str, model: str, seed: int = 0, id: str,
                   title: str) -> Table:
    """The stock-days that the corporate-action file leaves out of one held-out record."""
    rid = rt.rid(experiment=spec.experiment_of(seed), record=record, model=model, seed=seed)
    out = []
    if rid is not None:
        sd = rt.stock_days_of(rid)
        sd = sd[sd["status"] == "left_out"]
        for sym, day, event, eff in zip(sd["symbol"], sd["date"], sd["event"], sd["effective_date"]):
            if plain(event) is None:
                continue
            out.append(Row((sym, day), (text(sym), text(day), text(event), text(plain(eff)))))
    columns = (label("symbol"), label("date"), label("event"), label("effective date", "effective_date"))
    return Table(id, None, title, columns, tuple(out), None)


def _spec_records(rt: ResultTables, spec: ExperimentSpec) -> list[dict]:
    recs = rt.records
    own = recs[recs["experiment"] == spec.name]
    out = [rt.record(r) for r in own["record_id"]]
    if spec.seed0_from is not None:
        mask = (recs["experiment"] == spec.seed0_from) & (recs["seed"] == 0).fillna(False) & recs["model"].isin(list(spec.fits))
        other = recs[mask.astype(bool)]
        out += [rt.record(r) for r in other["record_id"]]
    return sorted(out, key=lambda r: r["record_id"])


def records_table(rt: ResultTables, spec: ExperimentSpec, *, id: str, title: str, by: tuple[str, ...] | None = None) -> Table:
    """The stored records behind the tables: one row per record, or with ``by`` one row per group of records."""
    recs = _spec_records(rt, spec)
    if by is None:
        rows = tuple(Row((r["record_id"],), (text(r["record_id"]), count(r["cells"]),
                                             text((r["weights_sha256"] or "")[:16])))
                     for r in recs)
        columns = (label("record"), Column("cells", "cells", compare=False), label("weights sha256", "weights"))
        return Table(id, None, title, columns, rows, None)
    groups: dict[tuple, list[dict]] = {}
    for r in recs:
        groups.setdefault(tuple(str(r[f]) for f in by), []).append(r)
    rows = tuple(Row(key, (*(text(v) for v in key), count(len(g)), count(sum(r["status"] == "scored" for r in g)),
                           count(sum(r["status"] == "left_out" for r in g)), count(sum(r["cells"] for r in g))))
                 for key, g in sorted(groups.items()))
    columns = (*(label(f) for f in by), Column("records", "records", compare=False),
               Column("scored", "scored", compare=False), Column("left_out", "left out", compare=False),
               Column("cells", "cells", compare=False))
    return Table(id, None, title, columns, rows, None)
