"""Build of one ITCH day into per-stock tables and its build log."""

from __future__ import annotations

import datetime
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dof.lob import SCHEMA_VERSION
from dof.lob.book import LEVELS, SENTINEL_ASK_PX, SENTINEL_BID_PX
from dof.lob.itch import EVENT_TYPES, LocateFilter, ParseStats, iter_messages, parse_itch_date
from dof.lob.provenance import code_identity, sha256_file
from dof.lob.tables.manifest import read_manifest
from dof.lob.tables.schema import _REFS, EMIT_RULE
from dof.lob.tables.writer import _event_row, _state_row, _SymbolSink, _write_json_atomic
from dof.lob.units import PRICE_SCALE

try:
    import resource
except ImportError:          # the module exists on POSIX systems only
    resource = None


def _max_rss_kb() -> int | None:
    """Peak resident memory of this process in kB; None where the platform does not report it. Linux reports
    ru_maxrss in kB, macOS in bytes."""
    if resource is None:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss // 1024 if sys.platform == "darwin" else rss


@dataclass
class BuildResult:
    day_dir: Path
    date: datetime.date
    status: str                          # "complete" | "partial" (raised errors write "failed" to the log)
    files: dict[str, dict[str, Path]]    # symbol -> {"events": path, "states": path}
    log_path: Path
    invalid_symbols: list[str]
    missing_symbols: list[str]
    error: str | None


def _symbol_log(sink: _SymbolSink, flt: LocateFilter) -> dict[str, object]:
    st = sink.book.stats
    return {
        "locate": sink.book.locate,
        "status": "valid" if sink.book.valid else "invalid",
        "directory": flt.directory.get(sink.symbol, {}),
        "events_rows": sink.events.rows,
        "states_rows": sink.states.rows,
        "unknown_ref": st.unknown_ref,
        "size_underflow": st.size_underflow,
        "duplicate_ref": st.duplicate_ref,
        "first_error_seq": st.first_error_seq,
        "first_error": st.first_error,
        "locked_rows": st.locked_rows,
        "crossed_rows": st.crossed_rows,
        "halted_rows": st.halted_rows,
        "crossed_segment_count": st.crossed_segment_count,
        "crossed_segments": [list(s) for s in st.crossed_segments],
        "trading_state_changes": [list(s) for s in st.trading_state_changes],
        "cross_prints": [list(s) for s in st.cross_prints],
        "trades_printable": st.trades_printable,
        "trade_shares_printable": st.trade_shares_printable,
        "nonprintable_exec": st.nonprintable_exec,
        "broken": st.broken,
        "orders_resting_end": sink.book.resting_orders(),
    }


def _parse_log(stats: ParseStats, flt: LocateFilter) -> dict[str, object]:
    return {
        "messages_total": stats.messages_total,
        "bytes_decompressed": stats.bytes_decompressed,
        "counts_by_type": dict(sorted(stats.counts_by_type.items())),
        "tracked_by_type": dict(sorted(stats.tracked_by_type.items())),
        "unknown_type_counts": dict(sorted(stats.unknown_type_counts.items())),
        "skipped_untracked": stats.skipped_untracked,
        "system_events": [list(e) for e in stats.system_events],
        "directory_size": flt.directory_size,
        "directory_duplicates": flt.directory_duplicates,
        "stopped_by_limit": stats.stopped_by_limit,
        "first_ts_ns": stats.first_ts_ns,
        "last_ts_ns": stats.last_ts_ns,
    }


def build_day(
    itch_path: str | os.PathLike[str],
    symbols: Sequence[str],
    out_dir: str | os.PathLike[str],
    *,
    manifest_csv: Path | None,
    date: datetime.date | None = None,     # None -> parse_itch_date(itch_path)
    limit_bytes: int | None = None,
    overwrite: bool = False,
    flush_rows: int = 32768,
) -> BuildResult:
    itch_path = Path(itch_path)
    # 1. day directory
    date = date or parse_itch_date(itch_path)
    day_dir = Path(out_dir) / date.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    log_path = day_dir / "build_log.json"
    lock = day_dir / ".lock"

    # 2. lock OUTSIDE the try: a held lock belongs to another
    # process, so we must neither write nor delete anything, the lock included.
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"day_dir locked: {lock} exists (another build running, or a stale lock)") from None
    try:
        os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
        return _build_locked(itch_path, list(symbols), day_dir, log_path, date, manifest_csv,
                             limit_bytes, overwrite, flush_rows)
    finally:
        os.close(fd)
        os.remove(lock)   # only the lock this process created


def _reject_or_hash(itch_path: Path, log_path: Path, manifest_csv: Path | None,
                    overwrite: bool) -> tuple[dict[str, str] | None, str, float]:
    """Steps 3-5: the rejections, which raise only (no log written, nothing deleted), then the input hash.
    Returns (manifest row or None, input sha256, seconds spent hashing)."""
    # 3. already built
    if log_path.exists() and not overwrite:
        try:
            old_status = json.loads(log_path.read_text(encoding="utf-8")).get("status")
        except (OSError, ValueError):
            old_status = None
        if old_status == "complete":
            raise RuntimeError(f"already built; pass overwrite=True ({log_path})")
    # 4. manifest
    row = None
    if manifest_csv is not None:
        row = read_manifest(Path(manifest_csv)).get(itch_path.name)
        size = os.stat(itch_path).st_size
        if row is None or row.get("gzip_test") != "ok" or size != int(row["bytes_got"]):
            got = None if row is None else (row.get("gzip_test"), row.get("bytes_got"))
            raise ValueError(f"input not verified complete in MANIFEST.csv: {itch_path.name} "
                             f"(manifest row={got}, size on disk={size})")
    # 5. input hash
    t0 = time.perf_counter()
    input_sha = sha256_file(itch_path)
    sha256_s = time.perf_counter() - t0
    if row is not None and row["sha256"] != input_sha:
        raise ValueError(f"input sha256 differs from MANIFEST.csv: {itch_path.name} "
                         f"manifest={row['sha256']} actual={input_sha}")
    return row, input_sha, sha256_s


def _stream(itch_path: Path, flt: LocateFilter, stats: ParseStats, sinks: dict[int, _SymbolSink], day_dir: Path,
            date: datetime.date, flush_rows: int, limit_bytes: int | None) -> None:
    """Step 6: every message into its stock's book and tables. A stock's sink opens at its first R message."""
    for msg in iter_messages(itch_path, flt, stats=stats, limit_bytes=limit_bytes):
        t = msg.msg_type
        if t == "S":
            continue  # recorded in stats.system_events
        if t == "R":
            # Only the first R of a tracked symbol opens a book (a duplicate
            # R is yielded but its locate is not the tracked one).
            if flt.locate_to_symbol.get(msg.locate) == msg.symbol and msg.locate not in sinks:
                sinks[msg.locate] = _SymbolSink(msg.symbol, msg.locate, day_dir, date, flush_rows)
            continue
        sink = sinks[msg.locate]
        book = sink.book
        target = book.orders.get(msg.order_ref) if t in _REFS else None
        if target is not None:
            target = list(target)  # snapshot before apply mutates it
        snap = book.apply(msg)
        if t in EVENT_TYPES:
            _event_row(sink, msg, target, book.trading_state)
        if snap is not None:
            _state_row(sink, snap)


def _finish(sinks: dict[int, _SymbolSink], finals: list[Path], symbols: list[str], flt: LocateFilter,
            stats: ParseStats, log: dict[str, object], log_path: Path, day_dir: Path, date: datetime.date,
            t0: float, sha256_s: float) -> BuildResult:
    """Steps 7-9: close, rename and hash the tables; the per-stock status; the complete log."""
    # 7. close, rename, hash
    for sink in sinks.values():
        sink.close()
    files: dict[str, dict[str, Path]] = {}
    outputs: dict[str, dict[str, str]] = {}
    for sink in sinks.values():
        for table in ("events", "states"):
            tmp = getattr(sink, table).path
            os.replace(tmp, sink.final[table])
            finals.append(sink.final[table])
        files[sink.symbol] = dict(sink.final)
        outputs[sink.symbol] = {
            "events": sink.final["events"].name, "states": sink.final["states"].name,
            "events_sha256": sha256_file(sink.final["events"]),
            "states_sha256": sha256_file(sink.final["states"]),
        }
    # 8. per-symbol status
    by_symbol = {s.symbol: s for s in sinks.values()}
    missing = [s for s in symbols if s.strip() not in by_symbol]
    invalid = [s.symbol for s in sinks.values() if not s.book.valid]
    # 9. log. The build is complete only when the whole stream was read: no byte limit stopped it and the end
    # of messages (S "C") was seen. Otherwise it is partial, and load_states refuses its tables.
    ended = any(code == "C" for _seq, _ts, code in stats.system_events)
    status = "complete" if ended and not stats.stopped_by_limit else "partial"
    total_s = time.perf_counter() - t0
    write_s = sum(s.write_s for s in sinks.values())
    log.update({
        "status": status,
        "parse": _parse_log(stats, flt),
        "symbols": {s.symbol: _symbol_log(s, flt) for s in sinks.values()},
        "missing_symbols": missing,
        "invalid_symbols": invalid,
        "timing": {"sha256_s": sha256_s, "parse_book_s": total_s - sha256_s - write_s, "write_s": write_s,
                   "total_s": total_s, "max_rss_kb": _max_rss_kb()},
        "outputs": outputs,
    })
    _write_json_atomic(log_path, log)
    return BuildResult(day_dir, date, status, files, log_path, invalid, missing, None)


def _fail(exc: BaseException, sinks: dict[int, _SymbolSink], finals: list[Path], flt: LocateFilter,
          stats: ParseStats, log: dict[str, object], log_path: Path, day_dir: Path, t0: float, sha256_s: float) -> None:
    """Step 10: fail closed. No Parquet file of this run survives; the log says why. The caller re-raises."""
    for sink in sinks.values():
        for w in (sink.events, sink.states):
            try:
                w.writer.close()
            except Exception:  # noqa: BLE001 — cleanup must not mask the original error
                pass
    for p in day_dir.glob("*.parquet.tmp"):
        p.unlink()
    for p in finals:
        if p.exists():
            p.unlink()
    total_s = time.perf_counter() - t0
    log.update({
        "status": "failed",
        "error": repr(exc),
        "parse": _parse_log(stats, flt),
        "symbols": {s.symbol: _symbol_log(s, flt) for s in sinks.values()},
        "missing_symbols": [],
        "invalid_symbols": [s.symbol for s in sinks.values() if not s.book.valid],
        "timing": {"sha256_s": sha256_s, "parse_book_s": None, "write_s": None, "total_s": total_s,
                   "max_rss_kb": _max_rss_kb()},
        "outputs": {},
    })
    _write_json_atomic(log_path, log)


def _build_locked(
    itch_path: Path, symbols: list[str], day_dir: Path, log_path: Path, date: datetime.date,
    manifest_csv: Path | None, limit_bytes: int | None, overwrite: bool, flush_rows: int,
) -> BuildResult:
    row, input_sha, sha256_s = _reject_or_hash(itch_path, log_path, manifest_csv, overwrite)
    t0 = time.perf_counter() - sha256_s          # the build's clock starts at the input hash
    if overwrite:
        # Checks passed: only now remove the previous artifacts (if this
        # build then fails, they are gone and the day must be rebuilt).
        for pattern in ("*.parquet", "*.parquet.tmp"):
            for p in day_dir.glob(pattern):
                p.unlink()
        if log_path.exists():
            log_path.unlink()

    flt = LocateFilter(symbols)
    stats = ParseStats()
    sinks: dict[int, _SymbolSink] = {}
    finals: list[Path] = []
    log: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "error": None,
        "date": date.isoformat(),
        "input": {"path": str(itch_path), "bytes": os.stat(itch_path).st_size, "sha256": input_sha,
                  "manifest_sha256": None if row is None else row["sha256"],
                  "manifest_checked": manifest_csv is not None},
        "provenance": code_identity(),
        "params": {"symbols_requested": symbols, "limit_bytes": limit_bytes, "flush_rows": flush_rows,
                   "levels": LEVELS, "price_scale": PRICE_SCALE, "sentinel_bid_px": SENTINEL_BID_PX,
                   "sentinel_ask_px": SENTINEL_ASK_PX, "emit_rule": EMIT_RULE, "initial_trading_state": "H"},
    }
    try:
        _stream(itch_path, flt, stats, sinks, day_dir, date, flush_rows, limit_bytes)
        return _finish(sinks, finals, symbols, flt, stats, log, log_path, day_dir, date, t0, sha256_s)
    except BaseException as exc:
        _fail(exc, sinks, finals, flt, stats, log, log_path, day_dir, t0, sha256_s)
        raise
