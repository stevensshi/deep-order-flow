"""The summary of a built cache: per day the stocks held and used and the sample rows; every stock-day left out,
with its rule."""

from __future__ import annotations

import json
from collections import Counter

from dof.deep.cache import layout as L
from dof.deep.cache.build import day_log, load_plan
from dof.deep.config import CachePaths, RunConfig, Split
from dof.deep.errors import CacheError
from dof.deep.runtime.io import now


def summarise(cfg: RunConfig, split: Split, cache: CachePaths, parts=("train", "valid", "test")) -> dict:
    plan = load_plan(cache)
    days: dict[str, dict] = {}
    left_out: list[dict] = []
    ratio_outside: list[dict] = []      # stock-days built although their price ratio is outside the range (test days)
    for s, x in plan["stocks"].items():
        for d in x["no_clean_row"]:
            left_out.append({"part": "train", "symbol": s, "date": d, "reason": "no cleaned row in the session"})
        for d in x["reused"]:
            left_out.append({"part": "train", "symbol": s, "date": d, "reason": "reused ticker (rule 1)"})
        held = {p["date"] for p in x["days"]} | set(x["reused"])
        for d in held:
            days.setdefault(d, {"part": "train", "held": 0, "used": 0, "n_rows": 0, "n_samples": 0, "samples_from_1300": 0})["held"] += 1
        for g in x["groups"]:
            gm = g
            if g["status"] == "ok":
                p = L.group_path(cache, s, g["id"])
                if not p.exists():
                    raise CacheError(f"{p} missing: the training phase has not finished")
                gm = json.loads(p.read_text(encoding="utf-8"))
            if gm["status"] != "ok":
                left_out += [{"part": "train", "symbol": s, "date": d, "reason": gm["reason"] + " (rule 3)"} for d in g["days"]]
                continue
            for r in gm["day_rows"]:
                e = days[r["date"]]
                e["used"] += 1
                for k in ("n_rows", "n_samples", "samples_from_1300"):
                    e[k] += r[k]
    for d, bad in plan["not_valid"].items():
        left_out += [{"part": "train", "symbol": s, "date": d, "reason": "book status not valid"} for s in bad]
    for part in parts:
        if part == "train":
            continue
        for d in split.days(part):
            held, bad = day_log(cfg, split, d)
            e = days.setdefault(d, {"part": part, "held": len(held), "used": 0, "n_rows": 0, "n_samples": 0, "samples_from_1300": 0})
            left_out += [{"part": part, "symbol": s, "date": d, "reason": "book status not valid"} for s in bad]
            for s in held:
                p = L.eval_meta_path(cache, part, s, d)
                if not p.exists():
                    raise CacheError(f"{p} missing: the {part} phase has not finished")
                m = json.loads(p.read_text(encoding="utf-8"))
                if m["status"] == "used":
                    e["used"] += 1
                    for k in ("n_rows", "n_samples", "samples_from_1300"):
                        e[k] += m[k]
                    if not m["ratio_inside"]:
                        ratio_outside.append({"part": part, "symbol": s, "date": d, "ratio_to_group_last_day": m["ratio_to_group_last_day"],
                                              "group_last_day": m["group_last_day"], "n_samples": m["n_samples"]})
                else:
                    left_out.append({"part": part, "symbol": s, "date": d, "reason": m["reason"],
                                     **{k: m[k] for k in ("ratio_to_group_last_day", "group_last_day") if k in m}})
    groups = [g for x in plan["stocks"].values() for g in x["groups"]]
    # The protocol document's hash is not kept here: it lives in the fit and score records.
    return {"stage": split.name, "parts": list(parts), "days": dict(sorted(days.items())), "left_out": left_out,
            "ratio_outside_built": ratio_outside, "half_days": {d: c for d, c in cfg.half_days.items() if d in days},
            "n_stocks": len(plan["stocks"]), "n_groups": len(groups),
            "groups_by_first_year": dict(sorted(Counter(g["days"][0][:4] for g in groups).items())),
            "stocks_by_group_count": dict(sorted(Counter(len(x["groups"]) for x in plan["stocks"].values()).items())),
            "bytes_train_cache": sum(f.stat().st_size for f in cache.train.rglob("*") if f.is_file()),
            "bytes_eval_cache": {p: sum(f.stat().st_size for f in (cache.eval / p).rglob("*") if f.is_file())
                                 for p in parts if p != "train" and (cache.eval / p).exists()},
            "written": now()}


def print_summary(doc: dict, log=print) -> None:
    log(f"{'day':<11}{'part':<7}{'held':>5}{'used':>5}{'rows':>14}{'samples':>14}{'from 13:00':>12}")
    for d, e in doc["days"].items():
        log(f"{d:<11}{e['part']:<7}{e['held']:>5}{e['used']:>5}{e['n_rows']:>14,}{e['n_samples']:>14,}{e['samples_from_1300']:>12,}")
    for part in doc["parts"]:
        sel = [e for e in doc["days"].values() if e["part"] == part]
        log(f"{part}: {len(sel)} days, {sum(e['used'] for e in sel)} stock-days, {sum(e['n_samples'] for e in sel):,} samples")
    log(f"stocks {doc['n_stocks']}, scaling groups {doc['n_groups']} (by first year {doc['groups_by_first_year']}; stocks by number "
        f"of groups {doc['stocks_by_group_count']}); left out: {len(doc['left_out'])} stock-days")
    for reason, n in Counter(x["part"] + ": " + x["reason"] for x in doc["left_out"]).most_common():
        log(f"  {n:4d}  {reason}")
    out = doc["ratio_outside_built"]
    log(f"built with a price ratio outside the range against the last training day (test days: the exclusions are "
        f"applied at scoring): {len(out)} stock-days of {len({x['symbol'] for x in out})} stocks")
    log(f"training cache {doc['bytes_train_cache'] / 1e9:.1f} GB; "
        + "; ".join(f"{p} cache {b / 1e9:.1f} GB" for p, b in doc["bytes_eval_cache"].items()))
