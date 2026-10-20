"""The earlier stage's stored scores against the decay measurement's second scoring of the same saved weights.

The decay measurement scored two networks of the earlier stage again on that stage's test days (frozen basis). This
table puts both scorings of a model and day side by side: the stock means, their difference, and how many cells
(stock and horizon) carry the same R² to the last bit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dof.experiments.analyses.scores import part_days
from dof.experiments.analyses.stocks import agg
from dof.experiments.cells import Rows
from dof.experiments.errors import AnalysisError
from dof.experiments.tables import Cell, Column, Row, Table, count, finite, label, pct, text

if TYPE_CHECKING:
    from dof.experiments.cells import ResultTables
    from dof.experiments.spec import ExperimentSpec

RESCORE_COLUMNS = (label("model"), label("day"), Column("stocks", "stocks"), Column("r2", "stored R² %"),
                   Column("again_stocks", "stocks, scored again"), Column("again_r2", "scored again, R² %"),
                   Column("diff", "again minus stored, percentage points"), Column("cells", "cells on both"),
                   Column("equal", "cells with the same R²"))


def _pp_sci(x) -> Cell:
    """A difference of two fractions in percentage points, in scientific notation (it is far below 0.001)."""
    return Cell(100 * x, ".3e", raw=x) if finite(x) else Cell(None)


def rescore_table(rt: ResultTables, spec: ExperimentSpec, *, other: str, basis: str, models: tuple[str, ...], id: str,
                  title: str, golden: str | None = None) -> Table:
    """Per model of ``models`` and test day of the spec: the stored test record against the ``basis`` units of the
    experiment ``other`` (the decay measurement) on the same day, k = 1..10."""
    if other != "decay":
        raise AnalysisError(f"{id}: only the decay measurement holds a second scoring, not {other!r}")
    out = []
    for m in models:
        rid = rt.rid(experiment=spec.name, record="test", model=m, seed=0)
        if rid is None:
            continue
        rows = rt.rows(rid)
        for day in part_days(spec, "test"):
            units = [u for u in rt.decay_units(m, day, basis) if u["status"] == "scored"]
            if not units:
                continue
            again = Rows.concat(rt.rows(u["record_id"]) for u in units)
            a, b = agg(rows, None, {day}), agg(again)
            stored = {(s, k): r2 for s, d, k, r2 in zip(rows.symbol, rows.date, rows.k, rows.r2) if d == day}
            both = [(stored[(s, k)], r2) for s, k, r2 in zip(again.symbol, again.k, again.r2) if (s, k) in stored]
            out.append(Row((m, day), (text(m), text(day), count(a["n_stocks"]), pct(a["mean"]), count(b["n_stocks"]),
                                      pct(b["mean"]), _pp_sci(b["mean"] - a["mean"]), count(len(both)),
                                      count(sum(x == y for x, y in both)))))
    return Table(id, golden, title, RESCORE_COLUMNS, tuple(out), None)      # test records beside decay records
