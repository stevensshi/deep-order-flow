"""Tables of the adjacent-days experiment: a part over the seeds, the paired difference of the afternoon and the
later days, and the selection-bias reference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from dof.experiments.analyses.scores import horizon_text
from dof.experiments.analyses.seeds import seed_stats
from dof.experiments.analyses.stocks import agg, inside, stock_values
from dof.experiments.tables import Cell, Column, Row, Table, count, fixed, horizon_label, label, pct, seed_label, text

if TYPE_CHECKING:
    from dof.experiments.cells import ResultTables, Rows
    from dof.experiments.spec import ExperimentSpec

RECORD = {"valid": "fit_valid", "afternoon": "test", "far": "far"}


def seed_table(rows_by_seed: Mapping[int, Rows], ks=None, dates=None) -> dict:
    """The headline of one part over the seeds, and the uncertainty of the part (for the tables only)."""
    per = {s: agg(rows, ks, dates) for s, rows in rows_by_seed.items()}
    stt = seed_stats({s: a["mean"] for s, a in per.items()})
    sv = {s: stock_values(rows, ks, dates) for s, rows in rows_by_seed.items()}
    common = sorted(set.intersection(*(set(v) for v in sv.values()))) if sv else []
    vbar = np.array([np.mean([sv[s][i] for s in sv]) for i in common])       # per stock: its mean over the seeds
    se = float(vbar.std(ddof=1) / np.sqrt(len(vbar))) if len(vbar) > 1 else float("nan")
    u = math.sqrt(se**2 + stt["sd"]**2 / stt["n"]) if stt["n"] > 1 and np.isfinite(se) and np.isfinite(stt["sd"]) \
        else float("nan")
    return {"per_seed": {s: a["mean"] for s, a in per.items()}, "mean": stt["mean"], "sd": stt["sd"], "n_seeds": stt["n"],
            "se_stocks": se, "n_stocks": len(common), "u": u}


def paired_difference(aft: Mapping[int, Rows], far: Mapping[int, Rows], ks=None) -> dict:
    """Per seed and per stock scored on both parts: the stock's value on the afternoon minus its value on the later
    days. The pair is the set of stocks with both values under every seed; seeds in numeric order."""
    nan = float("nan")
    seeds = sorted(set(aft) & set(far))
    a = {s: stock_values(aft[s], ks) for s in seeds}
    f = {s: stock_values(far[s], ks) for s in seeds}
    in_a = set().union(*(set(v) for v in a.values()))
    in_f = set().union(*(set(v) for v in f.values()))
    pair = sorted(set.intersection(*(set(a[s]) & set(f[s]) for s in seeds))) if seeds else []
    per_seed = {s: (float(np.mean([a[s][i] - f[s][i] for i in pair])) if pair else nan) for s in seeds}
    dbar = np.array([np.mean([a[s][i] - f[s][i] for s in seeds]) for i in pair])
    stt = seed_stats(per_seed)
    se = float(dbar.std(ddof=1) / np.sqrt(len(dbar))) if len(dbar) > 1 else nan
    u = math.sqrt(se**2 + stt["sd"]**2 / stt["n"]) if stt["n"] > 1 and np.isfinite(se) and np.isfinite(stt["sd"]) else nan
    return {"seeds": seeds, "pair": pair, "n_pairs": len(pair), "per_seed": per_seed, "d": stt["mean"], "se_stocks": se,
            "sd_seeds": stt["sd"], "u": u, "only_afternoon": sorted(in_a - in_f), "only_far": sorted(in_f - in_a),
            "partial": sorted((in_a & in_f) - set(pair))}


def _seed_records(rt, spec, record: str, model: str) -> dict:
    out = {}
    for s in spec.seeds:
        rid = rt.rid(experiment=spec.name, record=record, model=model, seed=s)
        if rid is not None:
            out[s] = rid
    return out


def part_table(rt: ResultTables, spec: ExperimentSpec, *, part: str, model: str, id: str, title: str,
               golden: str | None = None) -> Table:
    """One part of the day (validation morning, afternoon, later days) for the network over its seeds, or for the
    ridge reference."""
    plan, record = spec.adjacent, RECORD[part]
    days = ", ".join(plan.far_days)
    if part == "far":
        scopes = [("all", f"{days}, primary analysis", None, None), ("all", f"{days}, primary analysis", (10,), None),
                  *[(d, f"{d}, primary analysis", None, {d}) for d in plan.far_days],
                  ("secondary", f"{days}, secondary analysis (price ratio inside the range)", None, None)]
    else:
        word = f"{'morning' if part == 'valid' else 'afternoon'} of {plan.cut_day}"
        scopes = [("all", word, None, None), ("all", word, (10,), None)]
    out = []
    if model == plan.model:
        rids = _seed_records(rt, spec, record, model)
        rows = {s: rt.rows(rid) for s, rid in rids.items()}
        for scope, scored, ks, dates in scopes:
            if not rows:
                break
            rbs = {s: inside(r, rt.stock_days_of(rids[s])) for s, r in rows.items()} if scope == "secondary" else rows
            t = seed_table(rbs, ks, dates)
            out.append(Row((model, scope, horizon_label(ks)), (
                text(scored), text(horizon_text(ks)), count(t["n_stocks"]),
                *(pct(t["per_seed"][s]) if s in t["per_seed"] else Cell(None) for s in spec.seeds),
                pct(t["mean"]), pct(t["sd"]), pct(t["se_stocks"]))))
    else:
        rid = rt.rid(experiment=spec.name, record=record, model=model, seed=0)
        for scope, scored, ks, dates in scopes if rid is not None else []:
            rows = rt.rows(rid)
            if scope == "secondary":
                rows = inside(rows, rt.stock_days_of(rid))
            a = agg(rows, ks, dates)
            out.append(Row((model, scope, horizon_label(ks)), (
                text(scored), text(horizon_text(ks)), count(a["n_stocks"]), *(Cell(None) for _ in spec.seeds),
                pct(a["mean"]), Cell(None), pct(a["se"]))))
    columns = (label("rows scored", "scored"), label("horizon"), Column("stocks", "stocks"),
               *(Column(seed_label(s), f"seed {s}") for s in spec.seeds), Column("mean", "mean"),
               Column("sd", "s.d. across seeds"), Column("se", "s.e. across stocks"))
    return Table(id, golden, title, columns, tuple(out), "valid" if part == "valid" else "test")


def _relation(d: float, u: float) -> str:
    if not (np.isfinite(d) and np.isfinite(u)):
        return "U is not defined"
    if d > 2 * u:
        return "D is above 2U"
    if d < -2 * u:
        return "D is below -2U"
    return "|D| is at most 2U"


def _names(v: list[str]) -> str:
    return f"{len(v)}" + (f" ({', '.join(v)})" if v else "")


def pair_table(rt: ResultTables, spec: ExperimentSpec, *, id: str, title: str, golden: str | None = None) -> Table:
    """D, the paired difference of the afternoon and the later days, its uncertainty U, and the facts around it. The
    relation of D to 2U is a fact in its own column (not compared with the stored report, which held a verdict)."""
    net = spec.adjacent.model
    aft = {s: rt.rows(rid) for s, rid in _seed_records(rt, spec, "test", net).items()}
    far = {s: rt.rows(rid) for s, rid in _seed_records(rt, spec, "far", net).items()}
    ta, tf, p = seed_table(aft), seed_table(far), paired_difference(aft, far)

    def num(x) -> Cell:
        return pct(x, missing="not defined")
    rows = [(("n_pairs",), "stocks scored on both parts (the pair)", count(p["n_pairs"]))]
    rows += [(("diff", seed_label(s)), f"difference of seed {s}",
              num(p["per_seed"][s]) if s in p["per_seed"] else text("no score")) for s in spec.seeds]
    ok = np.isfinite(p["d"]) and np.isfinite(p["u"]) and p["u"] > 0
    rows += [(("d",), "D, the mean over the seeds", num(p["d"])),
             (("se_stocks",), "s.e. across stocks of the per-stock difference averaged over the seeds", num(p["se_stocks"])),
             (("sd_seeds",), "s.d. across seeds of the seed differences", num(p["sd_seeds"])),
             (("u",), "U", num(p["u"])),
             (("d_over_u",), "D / U", fixed(p["d"] / p["u"], 2) if ok else text("not defined")),
             (("afternoon", "mean"), "afternoon mean (for the tables only)", num(ta["mean"])),
             (("afternoon", "u"), "afternoon, uncertainty of a part (for the tables only)", num(ta["u"])),
             (("far", "mean"), "2026 mean, primary analysis (for the tables only)", num(tf["mean"])),
             (("far", "u"), "2026, uncertainty of a part (for the tables only)", num(tf["u"])),
             (("only_afternoon",), "scored on the afternoon only", text(_names(p["only_afternoon"]))),
             (("only_far",), "scored on the 2026 days only", text(_names(p["only_far"])))]
    if p["partial"]:
        rows.append((("partial",), "a value missing under a seed", text(_names(p["partial"]))))
    out = [Row(key, (text(q), value, Cell(None))) for key, q, value in rows]
    out.append(Row(("relation",), (text("D against 2U"), Cell(None), text(_relation(p["d"], p["u"])))))
    columns = (label("quantity"), Column("value", "value"), Column("relation", "relation", compare=False, align="left"))
    return Table(id, golden, title, columns, tuple(out), "test")


def reference_table(rt: ResultTables, spec: ExperimentSpec, *, id: str, title: str, golden: str | None = None) -> Table:
    """The main run's seed-0 fits scored on the same afternoon (selection-bias reference)."""
    out = []
    for n in spec.adjacent.reference_fits:
        rid = rt.rid(experiment=spec.name, record="reference", model=n, seed=0)
        if rid is None:
            continue
        main = rt.rid(experiment="alldays", record="fit_valid", model=n, seed=0)
        f = rt.fit(main) if main is not None else {}
        trained = f"{f['train_days']} days ({f['train_first']} to {f['train_last']})" if f.get("train_days") else ""
        rows = rt.rows(rid)
        for ks in (None, (10,)):
            a = agg(rows, ks)
            out.append(Row((n, horizon_label(ks)), (text(n), text(trained), text(horizon_text(ks)), count(a["n_stocks"]),
                                                    pct(a["mean"]), pct(a["se"]))))
    columns = (label("fit"), label("trained on", "trained"), label("horizon"), Column("stocks", "stocks"),
               Column("r2", "R² %"), Column("se", "s.e."))
    return Table(id, golden, title, columns, tuple(out), "test")
