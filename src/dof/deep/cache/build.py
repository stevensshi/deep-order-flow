"""The tasks of a cache build: probes of the training days, the plan of scaling groups, one task per (stock,
group) of training days, one task per validation or test stock-day.

Every task returns the file it finds instead of building again, and writes its main JSON file last (under a
temporary name, then renamed), so a file that exists is whole and a finished unit is never rebuilt.
"""

from __future__ import annotations

import datetime
import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dof.deep.cache import layout as L
from dof.deep.cache.day import check_windows, day_frame
from dof.deep.cache.groups import continuous, group_delta_t, scaling_groups
from dof.deep.config import CachePaths, RunConfig, Split
from dof.deep.errors import CacheError, ConfigError, TestDayRefused
from dof.deep.runtime.io import now, save_npy_atomic, write_json_atomic
from dof.deep.runtime.memory import release_memory
from dof.deep.splits import session_of
from dof.deep.universe import reused, universe
from dof.lob.features import (
    ASK_PX,
    BID_PX,
    TrainWindow,
    apply_scaler,
    count_mid_changes,
    fit_scaler_columnwise,
    hms,
    k_valid_from,
    flow_clean,
    mid_at,
    sample_mask,
)
from dof.lob.provenance import package_sources
from dof.lob.tables import load_states

PROBE_COLUMNS = ["ts_ns", "seq", "flags", "msg_type", *BID_PX, *ASK_PX]     # what the cleaning and the mid price read


def _book_root(cfg: RunConfig) -> Path:
    if cfg.machine.book_root is None:
        raise ConfigError("machine.book_root is not set: the cache is built from the order-book tables")
    return cfg.machine.book_root


def day_log(cfg: RunConfig, split: Split, date: str) -> tuple[dict[str, int], list[str]]:
    """(tickers whose book of ``date`` is built and valid -> state rows, tickers present but not valid)."""
    log = json.loads((_book_root(cfg) / date / "build_log.json").read_text(encoding="utf-8"))
    if log.get("status") != "complete":
        raise CacheError(f"{date}: build_log status is {log.get('status')!r}, not complete")
    held, bad = {}, []
    for s in universe(cfg, split):
        e = log["symbols"].get(s)
        if e is None:
            continue
        if e.get("status") == "valid":
            held[s] = int(e["states_rows"])
        else:
            bad.append(s)
    return held, bad


def probe_task(cfg: RunConfig, split: Split, cache: CachePaths, symbol: str, date: str) -> dict:
    """One training stock-day's count of mid-price changes (for Δt), median mid price (for the group rule) and
    number of cleaned rows (for the size estimate). Reads the price columns only."""
    out = L.probe_path(cache, symbol, date)
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    frame = load_states(_book_root(cfg) / date, symbol, columns=PROBE_COLUMNS)
    session = session_of(cfg, date)
    # Regular trading time, deduplicated: what estimate_delta_t counts (09:30 to 16:00; to the close on a half day).
    n_changes = count_mid_changes(frame, session)
    pos, _seq, _seg, _iv, _funnel = flow_clean(frame, session, cfg.features.dedupe)
    mid = mid_at(frame, pos)                     # the mid of the cleaned session rows: FlowFrame.mid
    ts = frame["ts_ns"].to_numpy()[pos]
    doc = {"symbol": symbol, "date": date, "n_raw": len(frame), "n_clean": len(pos), "mid_changes": int(n_changes),
           "median_mid": float(np.median(mid)) if len(pos) else None,
           "clean_rows_from_1300": int((ts >= hms(13, 0)).sum()), "seconds": round(time.perf_counter() - t0, 2)}
    del frame, mid, ts, pos
    write_json_atomic(out, doc, indent=1)
    release_memory()
    return doc


def make_plan(cfg: RunConfig, split: Split, cache: CachePaths) -> dict:
    """Days and scaling groups of every stock, from the probes of the training days."""
    held_by_day, not_valid = {}, {}
    for d in split.train_days:
        held_by_day[d], bad = day_log(cfg, split, d)
        if bad:
            not_valid[d] = bad
    stocks: dict[str, dict] = {}
    for s in universe(cfg, split):
        days_in_book = [d for d in split.train_days if s in held_by_day[d]]
        probes = [json.loads(L.probe_path(cache, s, d).read_text(encoding="utf-8")) for d in days_in_book
                  if not reused(split, s, d)]
        if not days_in_book:
            continue
        held = [p for p in probes if p["n_clean"] > 0]
        groups = []
        for gid, g in enumerate(scaling_groups(cfg, held)):
            per_day, delta_t = group_delta_t(cfg, [p["mid_changes"] for p in g], [p["date"] for p in g])
            groups.append({"id": gid, "days": [p["date"] for p in g], "mid_changes_per_day": per_day, "delta_t_ms": delta_t,
                           "last_day": g[-1]["date"], "last_median_mid": g[-1]["median_mid"],
                           "n_clean": int(sum(p["n_clean"] for p in g)), "max_raw": int(max(p["n_raw"] for p in g)),
                           "status": "ok" if delta_t is not None else "excluded",
                           "reason": None if delta_t is not None else "no mid-price change on the group's days"})
        stocks[s] = {"days": [{k: p[k] for k in ("date", "n_raw", "n_clean", "mid_changes", "median_mid", "clean_rows_from_1300")}
                              for p in probes],
                     "no_clean_row": [p["date"] for p in probes if p["n_clean"] == 0],
                     "reused": [d for d in days_in_book if reused(split, s, d)], "groups": groups}
    f = cfg.features
    return {"stage": split.name, "train_days": list(split.train_days), "gap_days": f.gap_days,
            "ratio_range": [f.ratio_lo, f.ratio_hi],
            "half_days": {d: c for d, c in cfg.half_days.items() if d in split.train_days},
            "reused_from": dict(split.reused_from), "convention": f.convention, "dedupe": f.dedupe,
            "not_valid": not_valid, "stocks": stocks}


def load_plan(cache: CachePaths) -> dict:
    p = L.plan_path(cache)
    if not p.exists():
        raise CacheError(f"{p} missing: build the training part of the cache first")
    return json.loads(p.read_text(encoding="utf-8"))


def latest_group(stock: dict) -> dict | None:
    """The group that holds the stock's last held training day."""
    return stock["groups"][-1] if stock["groups"] else None


def group_task(cfg: RunConfig, split: Split, cache: CachePaths, symbol: str, gid: int) -> dict:
    """The days of one (stock, group): built and held in memory, the two scalers fitted on the group's sample rows
    and nothing else, one record file per stock-day; ``g<k>.json`` last."""
    out = L.group_path(cache, symbol, gid)
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    hist_len, ks = cfg.features.history, list(cfg.features.horizons)
    g = load_plan(cache)["stocks"][symbol]["groups"][gid]
    out.parent.mkdir(parents=True, exist_ok=True)
    dates = [datetime.date.fromisoformat(d) for d in g["days"]]
    tw = TrainWindow(tuple(dates), g["mid_changes_per_day"], g["delta_t_ms"])
    base = {"symbol": symbol, "group": gid, "days": g["days"], "delta_t_ms": tw.delta_t_ms, "mid_changes_per_day": tw.mid_changes_per_day,
            "last_day": g["last_day"], "last_median_mid": g["last_median_mid"], "source_sha256": package_sources()}
    funnel: Counter = Counter()
    held: list[SimpleNamespace] = []
    k_valid, h_ms = None, {}
    for d in dates:
        kf, dd, labels, lab_seq = day_frame(cfg, _book_root(cfg), symbol, d, tw, funnel)
        if k_valid is None:
            k_valid = k_valid_from(labels)
            h_ms = {int(k): float(h) for k, h in zip(labels.k, labels.horizons_ms, strict=True)}
            if k_valid != ks:
                # The networks output all horizons jointly: a stock with a horizon inside the latency buffer is out.
                doc = {**base, "status": "excluded", "k_valid": k_valid, "h_ms": h_ms,
                       "reason": f"a horizon is at or below the 10 ms latency buffer (delta_t {tw.delta_t_ms:.3f} ms)"}
                write_json_atomic(out, doc, indent=1)
                return doc
        mask = sample_mask(labels, k_valid, dd.mask)
        y = labels.y[:, [ks.index(k) for k in k_valid]]
        if not (kf.seq[mask] <= lab_seq[mask]).all():
            raise AssertionError(f"{symbol} {d}: a label starts before its feature row")
        held.append(SimpleNamespace(date=d.isoformat(), mask=mask, of=dd.of, y=y, seq=kf.seq.astype(np.int64), labseq=lab_seq,
                                    samples_from_1300=int((dd.ts[mask] >= hms(13, 0)).sum())))
        del kf, dd, labels, lab_seq, mask, y
        release_memory()
    if sum(int(h.mask.sum()) for h in held) == 0:
        doc = {**base, "status": "excluded", "k_valid": k_valid, "h_ms": h_ms, "reason": "the group has no sample row"}
        write_json_atomic(out, doc, indent=1)
        return doc
    # The two scalers, on the sample rows of THIS group's days and nothing else, one column at a time: column c is
    # the column's values on the sample rows, day after day in the group's order.
    scalers = {"of": fit_scaler_columnwise(lambda c: np.concatenate([h.of[h.mask, c] for h in held]), 20),
               "y": fit_scaler_columnwise(lambda c: np.concatenate([h.y[h.mask, c] for h in held]), len(k_valid))}
    for key, sc in scalers.items():
        L.save_scaler(L.scaler_path(cache, symbol, gid, key), sc)
    days = []
    for h in held:
        rows = np.flatnonzero(h.mask)
        if len(h.mask) >= 2**31:
            raise AssertionError(f"{symbol} {h.date}: too many rows for an int32 row index")
        z = apply_scaler(h.of, scalers["of"]).astype(np.float32)
        check_windows(z, rows, hist_len, f"{symbol} {h.date}")
        if not np.isfinite(h.y[rows]).all():
            raise AssertionError(f"{symbol} {h.date}: NaN label on a sample row")
        rec = np.empty(len(h.mask), dtype=L.REC)
        rec["zof"], rec["y"], rec["seq"], rec["labseq"] = z, h.y.astype(np.float32), h.seq, h.labseq
        save_npy_atomic(L.rec_path(cache, symbol, h.date), rec)
        save_npy_atomic(L.rows_path(cache, symbol, h.date), rows.astype(np.int32))
        days.append({"date": h.date, "n_rows": len(h.mask), "n_samples": len(rows), "samples_from_1300": h.samples_from_1300})
        del z, rec, rows
        h.of = h.y = None
    doc = {**base, "status": "ok", "reason": None, "k_valid": k_valid, "h_ms": h_ms, "day_rows": days,
           "n_samples": int(sum(d["n_samples"] for d in days)),
           "zero_std_cols": {k: [int(c) for c in sc.zero_std_cols] for k, sc in scalers.items()},
           "funnel": dict(sorted(funnel.items())), "seconds": round(time.perf_counter() - t0, 1), "created": now()}
    write_json_atomic(out, doc, indent=1)            # written last: its existence means the group's files are whole
    del held
    release_memory()
    return doc


def eval_task(cfg: RunConfig, split: Split, cache: CachePaths, part: str, symbol: str, date: str) -> dict:
    """One validation or test stock-day: Δt and both scalers are READ from the stock's latest training group."""
    out = L.eval_meta_path(cache, part, symbol, date)
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    p = cfg.protocol
    if part == "test" and p.test_days_need_document and (p.document is None or not p.document.is_file()):
        raise TestDayRefused("the test days are built only after the protocol document exists")
    t0 = time.perf_counter()
    ks = list(cfg.features.horizons)
    base = {"symbol": symbol, "date": date, "part": part, "source_sha256": package_sources()}

    def left_out(reason: str, **more) -> dict:
        doc = {**base, "status": "left_out", "reason": reason, **more}
        write_json_atomic(out, doc, indent=1)
        return doc

    # Rule 1 is decided before the day's data is opened.
    if reused(split, symbol, date):
        return left_out("reused ticker (rule 1)")
    stock = load_plan(cache)["stocks"].get(symbol)
    g = latest_group(stock) if stock else None
    if g is None:
        return left_out("the stock has no training group (rule 4)")
    gm = json.loads(L.group_path(cache, symbol, g["id"]).read_text(encoding="utf-8")) if g["status"] == "ok" else g
    if gm["status"] != "ok":
        return left_out(f"the latest training group is not usable: {gm['reason']} (rule 4)", group=g["id"])
    tw = TrainWindow(tuple(datetime.date.fromisoformat(d) for d in gm["days"]), gm["mid_changes_per_day"], gm["delta_t_ms"])
    scalers = {key: L.load_scaler(L.scaler_path(cache, symbol, g["id"], key)) for key in ("of", "y")}
    funnel: Counter = Counter()
    kf, dd, labels, lab_seq = day_frame(cfg, _book_root(cfg), symbol, datetime.date.fromisoformat(date), tw, funnel)
    info = {"group": g["id"], "group_last_day": gm["last_day"], "delta_t_ms": tw.delta_t_ms, "n_rows": len(kf.ts_ns)}
    if len(kf.mid) == 0:
        return left_out("no cleaned row in the session", **info)
    median_mid = float(np.median(kf.mid))
    inside = continuous(cfg, gm["last_median_mid"], median_mid)
    info.update(median_mid=median_mid, ratio_to_group_last_day=median_mid / gm["last_median_mid"], ratio_inside=inside)
    # The price-ratio rule leaves out the stock-days of the parts it names (the validation days). A test stock-day
    # is built whatever its ratio: the ratio is recorded, and the test exclusions are applied at scoring.
    if part in split.price_rule_parts and not inside:
        return left_out("price discontinuity against the last day of the latest training group (rule 2)", **info)
    k_valid = k_valid_from(labels)
    if k_valid != gm["k_valid"]:
        raise AssertionError(f"{symbol} {date}: horizons {k_valid} differ from the training group's {gm['k_valid']}")
    mask = sample_mask(labels, k_valid, dd.mask)
    rows = np.flatnonzero(mask)
    if len(rows) == 0:
        return left_out("no sample row (rule 5)", **info)
    y = labels.y[:, [ks.index(k) for k in k_valid]]
    if not (kf.seq[mask] <= lab_seq[mask]).all():
        raise AssertionError(f"{symbol} {date}: a label starts before its feature row")
    z = apply_scaler(dd.of, scalers["of"]).astype(np.float32)
    check_windows(z, rows, cfg.features.history, f"{symbol} {date}")
    zy = apply_scaler(y, scalers["y"]).astype(np.float32)
    if not np.isfinite(zy[rows]).all():
        raise AssertionError(f"{symbol} {date}: NaN label on a sample row")
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"zof": z, "zy": zy, "y": y, "seq": kf.seq.astype(np.int64), "labseq": lab_seq, "rows": rows.astype(np.int32)}
    for name, arr in arrays.items():
        save_npy_atomic(L.eval_arr_path(cache, part, symbol, date, name), arr)
    doc = {**base, "status": "used", "reason": None, **info, "n_samples": len(rows), "k_valid": k_valid, "h_ms": gm["h_ms"],
           "samples_from_1300": int((dd.ts[mask] >= hms(13, 0)).sum()), "funnel": dict(sorted(funnel.items())),
           "seconds": round(time.perf_counter() - t0, 1), "created": now()}
    write_json_atomic(out, doc, indent=1)            # written last: its existence means the day's files are whole
    del kf, dd, labels, z, zy, y, arrays
    release_memory()
    return doc
