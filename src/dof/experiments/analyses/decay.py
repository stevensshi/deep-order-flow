"""Tables of the decay experiment: one cell of a day, basis and row; the difference of the two bases; the table of a
model; the stock-days per day and basis; the stock-days with the price ratio outside the range.

A unit is one stored stock-day of one model and basis (scored, or left out with a reason), in the file-name order of
the stored units. Row "all" holds every scored unit, row "inside" those with the price ratio inside the range.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import numpy as np

from dof.experiments.analyses.stocks import agg, paired
from dof.experiments.cells import Rows
from dof.experiments.tables import Cell, Column, Row, Table, count, fixed, horizon_label, label, pct, text

if TYPE_CHECKING:
    from dof.experiments.cells import ResultTables
    from dof.experiments.spec import ExperimentSpec

ROWS = ("all", "inside")
BASES = ("frozen", "refreshed")


def _selected(units: list[dict], row: str) -> list[dict]:
    return [u for u in units if u["status"] == "scored" and (row == "all" or u["ratio_inside"])]


def decay_cell(rt: ResultTables, units: list[dict], row: str, ks=None) -> dict:
    """The mean over stocks of the stock values of the selected units, its s.e., and the counts."""
    sel = _selected(units, row)
    parts = [rt.rows(u["record_id"]) for u in sel]
    a = agg(Rows.concat(parts), ks)
    no_r2 = sum(not any(np.isfinite(r2) for k, r2 in zip(p.k, p.r2) if ks is None or k in ks) for p in parts)
    return {**a, "stock_days": len(sel), "stock_days_no_r2": no_r2}


def decay_diff(rt: ResultTables, units_ii: list[dict], units_i: list[dict], row: str, ks=None) -> dict:
    """Basis (ii) minus basis (i), per stock scored on both bases in that row."""
    def rows(units):
        return Rows.concat(rt.rows(u["record_id"]) for u in _selected(units, row))
    return paired(rows(units_ii), rows(units_i), ks)


def _days_since(spec: ExperimentSpec, date: str) -> int:
    return (datetime.date.fromisoformat(date) - datetime.date.fromisoformat(spec.decay.distance_from)).days


def decay_table(rt: ResultTables, spec: ExperimentSpec, *, model: str, ks: tuple[int, ...] | None, id: str, title: str,
                golden: str | None = None) -> Table:
    """Per day and row: the frozen basis, the refreshed basis, and their difference on the stocks scored on both."""
    plan = spec.decay
    first = plan.models[0]
    out = []
    for day in plan.days:
        n = _days_since(spec, day.date)
        scored = bool(rt.decay_units(first, day.date, "frozen"))       # a day with no frozen unit is not scored yet
        units = {b: rt.decay_units(model, day.date, b) for b in day.bases}
        for row in ROWS:
            cells = [text(day.date), text("all" if row == "all" else f"inside [{plan.ratio_lo}, {plan.ratio_hi}]"),
                     count(n), fixed(n / plan.month_days, 1)]
            for basis in BASES:
                if basis in day.bases and scored:
                    c = decay_cell(rt, units[basis], row, ks)
                    cells += [count(c["stock_days"]), count(c["n_stocks"]), pct(c["mean"]), pct(c["se"])]
                else:
                    cells += [Cell(None)] * 4
            if "refreshed" in day.bases and scored:
                p = decay_diff(rt, units["refreshed"], units["frozen"], row, ks)
                cells += [count(p["n_stocks"]), pct(p["mean"]), pct(p["se"])]
            else:
                cells += [Cell(None)] * 3
            cells.append(text(day.note))
            out.append(Row((model, horizon_label(ks), day.date, row), tuple(cells)))
    columns = (label("day"), label("row"), Column("days_since", f"days since {plan.distance_from}"),
               Column("months", "months"),
               *(Column(f"{pre}_{c}", f"({word}) {t}") for pre, word in (("i", "i"), ("ii", "ii"))
                 for c, t in (("stock_days", "stock-days"), ("stocks", "stocks"), ("r2", "R² %"), ("se", "s.e."))),
               Column("diff_stocks", "(ii) − (i): stocks on both"), Column("diff", "difference, pp"),
               Column("diff_se", "s.e."), label("note"))
    return Table(id, golden, title, columns, tuple(out), None)


def _reason_kind(why: str) -> str:
    return ("reused" if why.startswith("reused ticker") else
            "group" if any(w in why for w in ("rule 4", "rule 3", "no group of the same period", "no frozen 2019 scaler"))
            else "sample" if any(w in why for w in ("rule 5", "no cleaned row")) else "other")


def decay_days(rt: ResultTables, spec: ExperimentSpec, *, id: str, title: str, golden: str | None = None) -> Table:
    """Per day and basis: the held stock-days (the units of the first model), scored, inside, below and above the
    range, left out by reason, and the stock-days without an R² of each model."""
    plan = spec.decay
    lo, hi, first = plan.ratio_lo, plan.ratio_hi, plan.models[0]
    out = []
    for day in plan.days:
        for basis in day.bases:
            us = rt.decay_units(first, day.date, basis)
            head = (text(day.date), text("(i)" if basis == "frozen" else "(ii)"), text(day.role))
            if not us:
                out.append(Row((day.date, basis), head + tuple(Cell(None) for _ in range(8 + len(plan.models)))))
                continue
            sc = [u for u in us if u["status"] == "scored"]
            gone = {"reused": 0, "group": 0, "sample": 0, "other": 0}
            for u in us:
                if u["status"] != "scored":
                    gone[_reason_kind(u.get("reason") or "")] += 1
            no_r2 = [count(decay_cell(rt, rt.decay_units(m, day.date, basis), "all")["stock_days_no_r2"])
                     for m in plan.models]
            out.append(Row((day.date, basis), head + (
                count(len(us)), count(len(sc)), count(sum(bool(u["ratio_inside"]) for u in sc)),
                count(sum(u["ratio"] < lo for u in sc)), count(sum(u["ratio"] > hi for u in sc)),
                count(gone["reused"]), count(gone["group"]), count(gone["sample"]), *no_r2)))
    columns = (label("day"), label("basis"), label("role in the other experiments", "role"), Column("held", "held"),
               Column("scored", "scored ('all')"), Column("inside", "inside"), Column("below", f"below {lo}"),
               Column("above", f"above {hi}"), Column("reused", "left out: reused"), Column("group", "no usable group"),
               Column("sample", "no sample"), *(Column(f"no_r2:{m}", f"stock-days with no R² ({m})") for m in plan.models))
    return Table(id, golden, title, columns, tuple(out), None)


def decay_outside(rt: ResultTables, spec: ExperimentSpec, *, id: str, title: str, golden: str | None = None) -> Table:
    """The scored stock-days with the price ratio outside the range: per day and basis, those below, then those
    above, each in unit order."""
    plan = spec.decay
    out = []
    for day in plan.days:
        for basis in day.bases:
            sc = [u for u in rt.decay_units(plan.models[0], day.date, basis) if u["status"] == "scored"]
            for u in [u for u in sc if u["ratio"] < plan.ratio_lo] + [u for u in sc if u["ratio"] > plan.ratio_hi]:
                out.append(Row((basis, day.date, u["symbol"]),
                               (text(basis), text(day.date), text(u["symbol"]), fixed(u["ratio"], 3))))
    columns = (label("basis"), label("date"), label("symbol"), Column("ratio", "price ratio"))
    return Table(id, golden, title, columns, tuple(out), None)
