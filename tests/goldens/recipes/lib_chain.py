"""Library golden recipes: the seeded synthetic days (G1, G2, G2b), the real smoke days (G3) and the stored
build logs of the real days (G4).

Each recipe takes a backend (``recipes.backends.LibBackend``) and returns flat golden values. ``chain`` is the
whole order-book chain for one stock: Δt on the training days, per day the cleaned rows of the flow branch under
both sign conventions and both dedupe settings, the labels and the sample mask, the change events and buckets of
the minute branch; then the scalers, the sufficient statistics and the ridge solutions on the training days, the
scores of the evaluation days and the first principal component of the minute branch.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from . import hashing as H
from . import synth_stream as SS

NS = 1_000_000_000
KS = tuple(range(1, 11))
HISTORY = 100
CHUNK = 16_384
LAMBDAS = (0.0, 1e-3, 1e-1, 1.0, 10.0)
LAM_SCORE = 0.1
BLOCK_NS = 600 * NS
CONVENTIONS = ("cks", "current_size")
DEDUPES = (True, False)
PRIMARY = ("cks", True)
OF_COLS = 20
OFI_COLS = [f"ofi_{m}" for m in range(1, 11)]


def hms(h: int, m: int, s: int = 0) -> int:
    return ((h * 60 + m) * 60 + s) * NS


SYNTH_TRACKED = SS.TRACKED
SYNTH_CHAIN = ("AAA", "BBB", "CCC")
SYNTH_DAYS = ("2019-01-30", "2019-01-31", "2019-02-01")
SYNTH_TRAIN = ("2019-01-30", "2019-01-31")
SYNTH_EVAL = ("2019-02-01",)
SYNTH_SEAM_ROWS = 3
HALF_DAY = "2019-11-29"
HALF_DAYS = ("2019-01-30", HALF_DAY)
HALF_WINDOWS = ((hms(9, 30), SS.REGULAR_CLOSE), (hms(9, 30), SS.CLOSE[HALF_DAY]))
HALF_SESSION = (hms(9, 40), hms(12, 50))
REAL_SYMBOLS = ("UAL", "FAST")
REAL_TRAIN = ("2019-01-30", "2019-03-27", "2019-07-30")
REAL_EVAL = ("2019-08-30", "2019-10-18")
REAL_SEAM_ROWS = 4099
SMOKE_DAYS = REAL_TRAIN + REAL_EVAL


def _dates(days) -> list[datetime.date]:
    return [datetime.date.fromisoformat(d) for d in days]


class LazyFrames:
    """A sequence of day frames loaded on access (``len`` and iteration), so the training days are never all
    held at once."""

    def __init__(self, load, days) -> None:
        self.load, self.days = load, list(days)

    def __len__(self) -> int:
        return len(self.days)

    def __getitem__(self, i: int):
        return self.load(self.days[i])

    def __iter__(self):
        for d in self.days:
            yield self.load(d)


class _Values(dict):
    """Flat golden values with a duplicate-key check."""

    def rec(self, key: str, value) -> None:
        for k, v in H.flatten(value, key).items():
            if k in self:
                raise ValueError(f"duplicate golden key {k}")
            self[k] = v


def _flow_values(kf) -> dict:
    iv, ps = kf.intervals, kf.prices
    return {"ts_ns": kf.ts_ns, "mid": kf.mid, "of": kf.of, "ofi": kf.ofi, "seq": kf.seq, "segment": kf.segment,
            "of_valid": kf.of_valid,
            "intervals": {"start_seq": iv.start_seq, "end_seq": iv.end_seq, "close_seq": iv.close_seq,
                          "start_ts": iv.start_ts, "close_ts": iv.close_ts, "reason": iv.reason},
            "prices_ts": ps.ts_ns, "prices_seq": ps.seq, "prices_mid": ps.mid, "funnel": H.json_value(kf.funnel)}


def _label_values(labels) -> dict:
    return {"y": labels.y, "valid": labels.valid, "horizons_ms": labels.horizons_ms, "k": labels.k,
            "h_le_latency": labels.h_le_latency, "funnel": H.json_value(labels.funnel)}


def _scaler_values(s) -> dict:
    return {"lo": s.lo, "hi": s.hi, "mean": s.mean, "std": s.std, "zero_std_cols": [int(c) for c in s.zero_std_cols]}


def _events_values(ev) -> dict:
    return {"ts_ns": ev.ts_ns, "bid_px": ev.bid_px, "bid_sz": ev.bid_sz, "ask_px": ev.ask_px, "ask_sz": ev.ask_sz,
            "mid": ev.mid, "segment": ev.segment, "has_prev": ev.has_prev, "seq": ev.seq,
            "funnel": H.json_value(ev.funnel)}


def chain(b, load, sym: str, train, eval_, chunk_seam_rows: int, load_full=None) -> dict:
    """The order-book chain of one stock. ``load(date)`` gives the day's frame of the flow columns; ``load_full``
    (optional) the frame of every column. Keys start with ``<sym>/``."""
    out = _Values()

    def rec(key: str, value) -> None:
        out.rec(f"{sym}/{key}", value)

    # 1. Δt from the training days only.
    tw = b.estimate_delta_t(LazyFrames(load, train), _dates(train))
    rec("delta_t/mid_changes_per_day", float(tw.mid_changes_per_day))
    rec("delta_t/delta_t_ms", float(tw.delta_t_ms))
    rec("delta_t/train_dates", [d.isoformat() for d in tw.train_dates])

    # 2. Per day.
    kept: dict[str, dict] = {}
    for d in (*train, *eval_):
        if load_full is not None:
            full = load_full(d)
            rec(f"{d}/frame_full", H.frame_sha256(full))
            del full
        frame = load(d)
        rec(f"{d}/frame_flow", H.frame_sha256(frame))
        rec(f"{d}/mid_changes", int(b.count_mid_changes(frame)))
        primary = None
        for conv in CONVENTIONS:
            for dedupe in DEDUPES:
                kf = b.flow_pipeline(frame, convention=conv, dedupe=dedupe)
                rec(f"{d}/flow/{conv}/d{int(dedupe)}", _flow_values(kf))
                if (conv, dedupe) == PRIMARY:
                    primary = kf
                del kf
        with b.set_chunk_rows(chunk_seam_rows):
            seam = b.flow_pipeline(frame, convention=PRIMARY[0], dedupe=PRIMARY[1])
        rec(f"{d}/flow/chunk_seam_equal",
            all(H.array_sha256(getattr(seam, a)) == H.array_sha256(getattr(primary, a)) for a in ("of", "ofi", "of_valid")))
        del seam
        labels = b.horizon_labels(primary, KS, tw)
        rec(f"{d}/labels", _label_values(labels))
        hist = b.history_valid(primary, HISTORY)
        k_valid = [int(k) for k in b.k_valid_from(labels)]
        mask = b.sample_mask(labels, k_valid, hist)
        rec(f"{d}/hist", hist)
        rec(f"{d}/k_valid", k_valid)
        rec(f"{d}/mask", mask)
        rec(f"{d}/n_samples", int(np.count_nonzero(mask)))
        buckets = {}
        for conv in CONVENTIONS:
            ev = b.change_events(frame, conv)
            rec(f"{d}/minute/{conv}/events", _events_values(ev))
            del ev
            bk = b.minute_buckets(frame, conv)
            rec(f"{d}/minute/{conv}/buckets", H.frame_sha256(bk))
            rec(f"{d}/minute/{conv}/n_valid", int(np.count_nonzero(bk["valid"].to_numpy())))
            buckets[conv] = bk
        ks = [int(k) for k in labels.k]
        kept[d] = {"of": primary.of, "y": labels.y[:, [ks.index(k) for k in k_valid]], "mask": mask,
                   "ts": primary.ts_ns, "k_valid": k_valid, "buckets": buckets}
        del frame, primary, labels, hist

    # 3. Scalers of the training days' sample rows, matrix fit and column-wise fit.
    if len({tuple(x["k_valid"]) for x in kept.values()}) != 1:
        raise ValueError(f"{sym}: the horizons differ between days: {[x['k_valid'] for x in kept.values()]}")
    tr = [kept[d] for d in train]
    ny = len(tr[0]["k_valid"])
    scalers = {}
    for key, ncols in (("of", OF_COLS), ("y", ny)):
        x_all = np.concatenate([x[key][x["mask"]] for x in tr])
        rec(f"scaler/{key}/matrix", _scaler_values(b.fit_scaler(x_all)))
        del x_all
        col = b.fit_scaler_columnwise(lambda c, key=key: np.concatenate([x[key][x["mask"], c] for x in tr]), ncols)
        rec(f"scaler/{key}/columnwise", _scaler_values(col))
        scalers[key] = col

    # 4. Downstream uses the column-wise scalers, as the training cache does.
    for d, x in kept.items():
        x["z_of"] = b.apply_scaler(x.pop("of"), scalers["of"])
        x["zy"] = b.apply_scaler(x["y"], scalers["y"])
        rec(f"{d}/z_of", x["z_of"])
        rec(f"{d}/zy", x["zy"])

    # 5. Sufficient statistics over the training days, in date order and chunks of CHUNK sample rows.
    s = b.suff_stats_init(HISTORY * OF_COLS, ny)
    for d in train:
        x = kept[d]
        rows = np.flatnonzero(x["mask"])
        for a in range(0, len(rows), CHUNK):
            ch = rows[a:a + CHUNK]
            b.suff_stats_update(s, b.lagged_design(x["z_of"], ch, HISTORY), x["zy"][ch])
    rec("ridge/suff", {"n": int(s.n), "sx": s.sx, "sy": s.sy, "sxx": s.sxx, "sxy": s.sxy, "syy": s.syy})

    # 6. Ridge for every λ.
    solved = {}
    for lam in LAMBDAS:
        w, bias, info = b.solve_ridge(s, lam)
        rec(f"ridge/lam={lam!r}", {"W": w, "b": bias, "cond": float(info["cond"]), "fallback": info["fallback"]})
        solved[lam] = (w, bias)
    del s

    # 7. Evaluation days with λ = LAM_SCORE: predictions in dollars, R² against the day's mean and against zero,
    #    and the block t-test of k = 1 with 10-minute blocks.
    w, bias = solved[LAM_SCORE]
    for d in eval_:
        x = kept[d]
        rows = np.flatnonzero(x["mask"])
        rec(f"{d}/eval/n_rows", int(len(rows)))
        if len(rows) == 0:
            continue
        zh = np.empty((len(rows), ny))
        for a in range(0, len(rows), CHUNK):
            ch = rows[a:a + CHUNK]
            zh[a:a + len(ch)] = b.lagged_design(x["z_of"], ch, HISTORY) @ w + bias
        yhat = b.inverse_target(zh, scalers["y"])
        y = x["y"][rows]
        rec(f"{d}/eval/yhat", yhat)
        for j, k in enumerate(x["k_valid"]):
            rec(f"{d}/eval/r2_os/k{k}", float(b.r2_os(y[:, j], yhat[:, j])))
            rec(f"{d}/eval/r2_zero/k{k}", float(b.r2_zero(y[:, j], yhat[:, j])))
        if 1 in x["k_valid"]:
            j = x["k_valid"].index(1)
            loss_diff = (y[:, j] - yhat[:, j]) ** 2 - (y[:, j] - y[:, j].mean()) ** 2
            t, df, n_blocks, p = b.block_t_test(loss_diff, x["ts"][rows] // BLOCK_NS)
            rec(f"{d}/eval/block_t", {"t": float(t), "df": int(df), "B": int(n_blocks), "p": float(p)})

    # 8. Minute branch: first principal component on the training buckets that are valid and inside the analysis window;
    #    the integrated OFI of the evaluation days.
    for conv in CONVENTIONS:
        parts = []
        for d in train:
            bk = kept[d]["buckets"][conv]
            sel = bk["valid"].to_numpy() & bk["in_analysis"].to_numpy()
            parts.append(bk[OFI_COLS].to_numpy(dtype=np.float64)[sel])
        try:
            pw = b.fit_pc1(np.concatenate(parts))
        except ValueError as exc:
            rec(f"minute/{conv}/pc1/error", f"ValueError:{exc}")
            continue
        rec(f"minute/{conv}/pc1", {"w": pw.w, "evr": float(pw.explained_variance_ratio), "n_obs": int(pw.n_obs)})
        for d in eval_:
            bk = kept[d]["buckets"][conv]
            rec(f"{d}/minute/{conv}/integrated_ofi", b.integrated_ofi(bk[OFI_COLS].to_numpy(dtype=np.float64), pw))
    return dict(out)


# -- ITCH layer and builds of the synthetic days -----------------------------------------------------


def _parse_stats(st) -> dict:
    return {"messages_total": st.messages_total, "bytes_decompressed": st.bytes_decompressed,
            "counts_by_type": dict(sorted(st.counts_by_type.items())),
            "tracked_by_type": dict(sorted(st.tracked_by_type.items())),
            "skipped_untracked": st.skipped_untracked,
            "unknown_type_counts": dict(sorted(st.unknown_type_counts.items())),
            "system_events": [list(e) for e in st.system_events], "stopped_by_limit": st.stopped_by_limit,
            "first_ts_ns": st.first_ts_ns, "last_ts_ns": st.last_ts_ns}


def _messages_sha(b, path: Path, symbols, **kw) -> tuple[str, int, object, object]:
    flt, stats = b.LocateFilter(list(symbols)), b.ParseStats()
    h, n = hashlib.sha256(), 0
    for m in b.iter_messages(path, flt, stats=stats, **kw):
        h.update((json.dumps(list(m)) + "\n").encode())
        n += 1
    return h.hexdigest(), n, flt, stats


def _itch_values(b, path: Path) -> dict:
    sha, n, flt, stats = _messages_sha(b, path, SYNTH_TRACKED)
    sha4096, _, _, _ = _messages_sha(b, path, SYNTH_TRACKED, chunk_size=4096)
    half = _messages_sha(b, path, SYNTH_TRACKED, limit_bytes=path.stat().st_size // 2)
    return {"messages_sha256": sha, "n_messages": n, "stats": _parse_stats(stats),
            "locates": sorted(flt.locates), "locate_to_symbol": {str(k): v for k, v in sorted(flt.locate_to_symbol.items())},
            "chunk4096_equal": sha4096 == sha,
            "limit_half": {"n_messages": half[1], "stopped_by_limit": half[3].stopped_by_limit},
            "scan_counts": H.json_value(_parse_stats(b.scan_counts(path)))}


def _build(b, path: Path, symbols, book: Path, date: str) -> dict:
    """Build one day and return its build log."""
    res = b.build_day(path, list(symbols), book, datetime.date.fromisoformat(date))
    return json.loads(Path(res.log_path).read_text(encoding="utf-8"))


def _log_values(log: dict) -> dict:
    v = H.scrub(log, frozenset({"timing", "provenance"}))
    v["input"] = {k: x for k, x in v["input"].items() if k != "path"}
    return v


def _parquet_values(book: Path, date: str, log: dict) -> dict:
    import pyarrow.parquet as pq
    out = {}
    for sym, files in log.get("outputs", {}).items():
        for table in ("events", "states"):
            path = book / date / files[table]
            out[f"{sym}/{table}"] = {"file_sha256": H.sha256_file(path),
                                     "content_sha256": H.table_content_sha256(pq.read_table(path))}
    return out


def _sym_load(b, book: Path, sym: str, full: bool = False):
    if full:
        return lambda d: b.load_states(book / d, sym)
    return lambda d: b.load_states(book / d, sym, columns=b.FLOW_COLUMNS)


def run_synth(b, work: Path) -> tuple[dict, dict]:
    """G1 and G2: the three full synthetic days. ``work`` must not hold a "synth" folder yet.
    Returns (values, build log by date)."""
    inp, book = Path(work) / "synth" / "input", Path(work) / "synth" / "book"
    inp.mkdir(parents=True, exist_ok=False)
    book.mkdir(parents=True, exist_ok=False)
    out, logs = _Values(), {}
    for d in SYNTH_DAYS:
        path = SS.write_day(inp, d)
        out.rec(f"synth/{d}/input_sha256", H.sha256_file(path))
        out.rec(f"synth/{d}/itch", _itch_values(b, path))
        log = _build(b, path, SYNTH_TRACKED, book, d)
        logs[d] = log
        out.rec(f"synth/{d}/log", _log_values(log))
        out.rec(f"synth/{d}/parquet", _parquet_values(book, d, log))
        try:
            b.load_states(book / d, "DDD")
            out.rec(f"DDD/{d}/load_error", None)
        except b.InvalidStockDay as exc:
            out.rec(f"DDD/{d}/load_error", f"InvalidStockDay:{exc.reason}")
    for sym in SYNTH_CHAIN:
        for k, v in chain(b, _sym_load(b, book, sym), sym, SYNTH_TRAIN, SYNTH_EVAL, SYNTH_SEAM_ROWS,
                          _sym_load(b, book, sym, full=True)).items():
            out.rec(k, v)
    return dict(out), logs


def run_halfday(b, work: Path) -> dict:
    """G2b: a full day and a half day (close 13:00). Δt with each day's regular trading time, the half day's
    session 09:40 to 12:50, and how many rows the old full-day session keeps from 12:50 on.
    ``work`` must not hold a "halfday" folder yet."""
    inp, book = Path(work) / "halfday" / "input", Path(work) / "halfday" / "book"
    inp.mkdir(parents=True, exist_ok=False)
    book.mkdir(parents=True, exist_ok=False)
    out = _Values()
    for d in HALF_DAYS:
        path = SS.write_day(inp, d)
        out.rec(f"halfday/{d}/input_sha256", H.sha256_file(path))
        out.rec(f"halfday/{d}/status", _build(b, path, SYNTH_TRACKED, book, d)["status"])
    for sym in SYNTH_CHAIN:
        load = _sym_load(b, book, sym)
        tw = b.estimate_delta_t(LazyFrames(load, HALF_DAYS), _dates(HALF_DAYS), list(HALF_WINDOWS))
        out.rec(f"{sym}/delta_t", {"mid_changes_per_day": float(tw.mid_changes_per_day), "delta_t_ms": float(tw.delta_t_ms)})
        for d, window in zip(HALF_DAYS, HALF_WINDOWS):
            frame = load(d)
            out.rec(f"{sym}/{d}/mid_changes_window", int(b.count_mid_changes(frame, window)))
            del frame
        frame = load(HALF_DAY)
        p = f"{sym}/{HALF_DAY}"
        primary = None
        for conv in CONVENTIONS:
            for dedupe in DEDUPES:
                kf = b.flow_pipeline(frame, convention=conv, dedupe=dedupe, session=HALF_SESSION)
                out.rec(f"{p}/flow/{conv}/d{int(dedupe)}", _flow_values(kf))
                if (conv, dedupe) == PRIMARY:
                    primary = kf
                del kf
        labels = b.horizon_labels(primary, KS, tw)
        out.rec(f"{p}/labels", _label_values(labels))
        hist = b.history_valid(primary, HISTORY)
        k_valid = [int(k) for k in b.k_valid_from(labels)]
        mask = b.sample_mask(labels, k_valid, hist)
        out.rec(f"{p}/hist", hist)
        out.rec(f"{p}/k_valid", k_valid)
        out.rec(f"{p}/mask", mask)
        out.rec(f"{p}/n_samples", int(np.count_nonzero(mask)))
        old = b.flow_pipeline(frame, convention=PRIMARY[0], dedupe=PRIMARY[1])
        out.rec(f"{p}/fullday_default/rows_at_or_after_1250", int(np.count_nonzero(old.ts_ns >= hms(12, 50))))
        del frame, primary, labels, old
    return dict(out)


# -- the real smoke days and the stored build logs --------------------------------------------------


def run_real(b, book_root: Path) -> dict:
    """G3: the chain for UAL and FAST on the five smoke days (stock-level Δt and scalers)."""
    book = Path(book_root)
    out = _Values()
    for sym in REAL_SYMBOLS:
        for k, v in chain(b, _sym_load(b, book, sym), sym, REAL_TRAIN, REAL_EVAL, REAL_SEAM_ROWS,
                          _sym_load(b, book, sym, full=True)).items():
            out.rec(k, v)
    return dict(out)


def build_log_values(log: dict, date: str, symbols=None) -> dict:
    """Hashes and statuses of one stored build log (no counts); ``symbols`` limits the per-stock keys."""
    v = {f"real_build/{date}/input_sha256": log["input"]["sha256"], f"real_build/{date}/status": log["status"]}
    outputs = log.get("outputs", {})
    for sym in sorted(log.get("symbols", {})):
        if symbols is not None and sym not in symbols:
            continue
        o = outputs.get(sym, {})
        v[f"real_build/{date}/{sym}/symbol_status"] = log["symbols"][sym].get("status")
        v[f"real_build/{date}/{sym}/events_sha256"] = o.get("events_sha256")
        v[f"real_build/{date}/{sym}/states_sha256"] = o.get("states_sha256")
    return v


def run_build_logs(book_root: Path) -> tuple[dict, dict]:
    """G4: every stored build log; then the Parquet files of UAL and FAST on the five smoke days hashed on disk
    and compared with their logs. Returns (values, the comparison by file)."""
    book = Path(book_root)
    values: dict = {}
    for day_dir in sorted(p for p in book.iterdir() if (p / "build_log.json").is_file()):
        values.update(build_log_values(json.loads((day_dir / "build_log.json").read_text(encoding="utf-8")), day_dir.name))
    verified = {}
    for d in SMOKE_DAYS:
        log = json.loads((book / d / "build_log.json").read_text(encoding="utf-8"))
        for sym in REAL_SYMBOLS:
            for table in ("events", "states"):
                o = log["outputs"][sym]
                verified[f"real_build/verified/{d}/{sym}/{table}"] = H.sha256_file(book / d / o[table]) == o[f"{table}_sha256"]
    values.update(verified)
    return values, verified
