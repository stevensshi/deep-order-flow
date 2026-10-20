"""Tables of an experiment with several seeds: statistics over seeds, the rule for more seeds, the seed ensembles,
and the kinds of the cells without an R²."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from dof.experiments.analyses.scores import horizon_text, part_of, record_rows
from dof.experiments.analyses.stocks import agg, finite_kind, inside, paired
from dof.experiments.tables import Cell, Column, Row, Table, count, days_label, horizon_label, label, pct, seed_label, text

if TYPE_CHECKING:
    from dof.experiments.cells import ResultTables, Rows
    from dof.experiments.spec import ExperimentSpec

KINDS = ("finite", "zero_variance", "not_finite", "other")


def seed_stats(values: Mapping[int, float]) -> dict:
    """Mean, sample standard deviation (n - 1), minimum and maximum over the seeds that have a finite value."""
    v = np.array([x for x in values.values() if np.isfinite(x)])
    return {"n": len(v), "mean": float(v.mean()) if len(v) else float("nan"),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
            "min": float(v.min()) if len(v) else float("nan"), "max": float(v.max()) if len(v) else float("nan")}


def rule_values(rt: ResultTables, spec: ExperimentSpec) -> dict:
    """The rule for more seeds, on validation numbers only: per first seed the difference older minus recent of the
    headline (k = 1..10, both validation days); more seeds when the sample s.d. of the differences is above
    ``half`` times the absolute value of their mean, or when a difference is missing."""
    pol = spec.seed_policy
    differences, stocks = {}, {}
    for s in pol.first:
        ra = record_rows(rt, spec, "fit_valid", pol.older, s)
        rb = record_rows(rt, spec, "fit_valid", pol.recent, s)
        if ra is not None and rb is not None:
            p = paired(ra, rb)
            differences[s], stocks[s] = p["mean"], p["n_stocks"]
    if len(differences) == len(pol.first):
        v = [differences[s] for s in pol.first]
        mean, sd = float(np.mean(v)), float(np.std(v, ddof=1))
        more = sd > pol.half * abs(mean)
    else:
        mean = sd = None
        more = True
    return {"differences": differences, "stocks": stocks, "mean": mean, "sd": sd,
            "half_abs_mean": None if mean is None else pol.half * abs(mean), "more": more}


def seed_rule(rt: ResultTables, spec: ExperimentSpec, *, id: str, title: str, golden: str | None = None) -> Table:
    pol, v = spec.seed_policy, rule_values(rt, spec)
    rows = []
    for s in pol.first:
        rows.append(Row(("difference", seed_label(s)), (text(f"difference {seed_label(s)}"), pct(v["differences"].get(s)))))
        rows.append(Row(("stocks", seed_label(s)), (text(f"stocks {seed_label(s)}"), count(v["stocks"].get(s)))))
    for q in ("mean", "sd", "half_abs_mean"):
        rows.append(Row((q,), (text(q), pct(v[q]))))
    rows.append(Row(("more",), (text("more"), text("yes" if v["more"] else "no"))))
    rows.append(Row(("drawn",), (text("drawn"), text(", ".join(map(str, pol.drawn)) or "none"))))
    return Table(id, golden, title, (label("quantity"), Column("value", "value")), tuple(rows), "valid")


def _measures(rt, spec, record, secondary):
    """The days of the first present (fit, seed) in protocol order: all days, all days at k = 10, each day."""
    for f in spec.fits:
        for s in spec.seeds:
            some = record_rows(rt, spec, record, f, s, secondary)
            if some is not None:
                days = sorted(set(some.date))
                return days, [(None, None), ((10,), None), *[(None, {d}) for d in days]]
    return [], []


def _ensemble(rt, spec, record, fit, secondary) -> Rows | None:
    members = "+".join(str(s) for s in spec.seed_policy.first)
    rid = rt.rid(experiment=spec.name, record="ensemble", model=fit, members=members, part=part_of(record))
    if rid is None:
        return None
    rows = rt.rows(rid)
    return inside(rows, rt.stock_days_of(rid)) if secondary else rows


def _stats_cells(spec, by_seed: dict, stt: dict) -> list[Cell]:
    return [*(pct(by_seed[s]["mean"]) if s in by_seed else Cell(None) for s in spec.seeds),
            pct(stt["mean"]), pct(stt["sd"]), pct(stt["min"]), pct(stt["max"]), count(stt["n"])]


def _stats_columns(spec) -> list[Column]:
    return [*(Column(seed_label(s), f"seed {s}") for s in spec.seeds), Column("mean", "mean"), Column("sd", "s.d."),
            Column("min", "min"), Column("max", "max"), Column("seeds", "seeds")]


def seed_scores(rt: ResultTables, spec: ExperimentSpec, *, record: str, fit: str, secondary: bool = False, id: str,
                title: str, golden: str | None = None) -> Table:
    """One fit over every seed: per seed the headline, their statistics, and the ensemble of the first seeds."""
    days, measures = _measures(rt, spec, record, secondary)
    by_rows = {s: r for s in spec.seeds if (r := record_rows(rt, spec, record, fit, s, secondary)) is not None}
    ens = _ensemble(rt, spec, record, fit, secondary)
    out = []
    for ks, dates in measures:
        ag = {s: agg(r, ks, dates) for s, r in by_rows.items()}
        if not ag:
            continue
        stt = seed_stats({s: x["mean"] for s, x in ag.items()})
        ea = agg(ens, ks, dates) if ens is not None else None
        out.append(Row((fit, days_label(dates or days), horizon_label(ks)), (
            text(", ".join(sorted(dates)) if dates else ", ".join(days)), text(horizon_text(ks)),
            text("/".join(str(x) for x in sorted({x["n_stocks"] for x in ag.values()}))), *_stats_cells(spec, ag, stt),
            pct(ea["mean"]) if ea else Cell(None), count(ea["n_stocks"]) if ea else Cell(None))))
    columns = (label("days scored", "days"), label("horizon"), Column("stocks", "stocks"), *_stats_columns(spec),
               Column("ensemble", "ensemble of the first three"), Column("ensemble_stocks", "its stocks"))
    return Table(id, golden, title, columns, tuple(out), part_of(record))


def seed_older(rt: ResultTables, spec: ExperimentSpec, *, record: str, secondary: bool = False, id: str, title: str,
               golden: str | None = None) -> Table:
    """Does the older data help, per seed: older minus recent over the stocks scored under both, and the same for the
    two ensembles."""
    pol = spec.seed_policy
    days, measures = _measures(rt, spec, record, secondary)
    older = {s: r for s in spec.seeds if (r := record_rows(rt, spec, record, pol.older, s, secondary)) is not None}
    recent = {s: r for s in spec.seeds if (r := record_rows(rt, spec, record, pol.recent, s, secondary)) is not None}
    e_old, e_rec = _ensemble(rt, spec, record, pol.older, secondary), _ensemble(rt, spec, record, pol.recent, secondary)
    out = []
    for ks, dates in measures:
        pr = {s: paired(older[s], recent[s], ks, dates) for s in spec.seeds if s in older and s in recent}
        if not pr:
            continue
        stt = seed_stats({s: x["mean"] for s, x in pr.items()})
        ep = paired(e_old, e_rec, ks, dates) if e_old is not None and e_rec is not None else None
        out.append(Row((days_label(dates or days), horizon_label(ks)), (
            text(", ".join(sorted(dates)) if dates else ", ".join(days)), text(horizon_text(ks)),
            text("/".join(str(x) for x in sorted({x["n_stocks"] for x in pr.values()}))), *_stats_cells(spec, pr, stt),
            pct(ep["mean"]) if ep else Cell(None))))
    columns = (label("days scored", "days"), label("horizon"), Column("stocks", "stocks in both"), *_stats_columns(spec),
               Column("ensembles", "the two ensembles"))
    return Table(id, golden, title, columns, tuple(out), part_of(record))


def finite_table(rt: ResultTables, spec: ExperimentSpec, *, id: str, title: str, golden: str | None = None) -> Table:
    """Per fit: the score records behind the tables (fit and test records of every seed, then the ensembles) and their
    cells by kind."""
    members = "+".join(str(s) for s in spec.seed_policy.first)
    by_model: dict[str, dict] = {}
    for f in spec.fits:
        rids = [rid for s in spec.seeds for record in ("fit_valid", "test")
                if (rid := rt.rid(experiment=spec.experiment_of(s), record=record, model=f, seed=s)) is not None]
        rids += [rid for part in ("valid", "test")
                 if (rid := rt.rid(experiment=spec.name, record="ensemble", model=f, members=members, part=part)) is not None]
        for rid in rids:
            m = by_model.setdefault(str(rt.record(rid)["model"]), {"files": 0, "cells": 0, **dict.fromkeys(KINDS, 0)})
            m["files"] += 1
            rows = rt.rows(rid)
            for r2, sse, sst in zip(rows.r2, rows.sse, rows.sst):
                m[finite_kind(r2, sse, sst)] += 1
                m["cells"] += 1
    out = tuple(Row((model,), (text(model), count(m["files"]), Cell(m["cells"], ","),
                               *(Cell(m[k], ",") for k in KINDS))) for model, m in by_model.items())
    columns = (label("fit"), Column("files", "score files"), Column("cells", "cells"), Column("finite", "finite"),
               Column("zero_variance", "label variance zero"),
               Column("not_finite", "not finite, label variance above zero"), Column("other", "other"))
    return Table(id, golden, title, columns, out, None)
