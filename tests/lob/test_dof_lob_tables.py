from __future__ import annotations

import csv
import datetime
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dof.lob import tables as export
from dof.lob.book import Book
from dof.lob.itch import ItchFormatError
from dof.lob.tables import build_day, read_manifest
from dof.lob.testing import synth as ls

DATE = datetime.date(2019, 1, 30)
GZ_NAME = "01302019.NASDAQ_ITCH50.gz"


def _gz(tmp_path: Path, payloads: list[bytes] = ls.SCENARIO_FULL) -> Path:
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    return ls.write_stream(src / GZ_NAME, ls.stream(payloads), gz=True)


def _build(tmp_path: Path, **kw):
    gz = kw.pop("gz", None) or _gz(tmp_path)
    kw.setdefault("manifest_csv", None)
    kw.setdefault("date", DATE)
    return build_day(gz, kw.pop("symbols", ["TEST"]), tmp_path / "out", **kw)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    cols = ["file", "bytes_expected", "bytes_got", "sha256", "gzip_test", "finished_utc"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path


def _ok_row(gz: Path, **over) -> dict[str, object]:
    size = gz.stat().st_size
    row = {"file": gz.name, "bytes_expected": size, "bytes_got": size, "sha256": _sha(gz),
           "gzip_test": "ok", "finished_utc": "2026-10-02T00:00:00Z"}
    row.update(over)
    return row


EVENTS_TYPES = [
    ("date", pa.date32()), ("symbol", pa.dictionary(pa.int32(), pa.string())), ("seq", pa.int64()),
    ("ts_ns", pa.int64()), ("msg_type", pa.string()), ("order_ref", pa.uint64()),
    ("new_order_ref", pa.uint64()), ("side", pa.string()), ("price", pa.int64()),
    ("exec_price", pa.int64()), ("shares", pa.int64()), ("exec_shares", pa.int64()),
    ("printable", pa.string()), ("match_number", pa.uint64()), ("cross_type", pa.string()),
    ("trading_state", pa.string()), ("broken_match", pa.uint64()),
]


def _states_types() -> list[tuple[str, pa.DataType]]:
    out = [("seq", pa.int64()), ("ts_ns", pa.int64())]
    for side in ("bid", "ask"):
        out += [(f"{side}_px_{k}", pa.int64()) for k in range(1, 11)]
        out += [(f"{side}_sz_{k}", pa.int64()) for k in range(1, 11)]
        out += [(f"{side}_ct_{k}", pa.int32()) for k in range(1, 11)]
    return out + [("flags", pa.int8())]


def test_build_day_writes_two_tables(tmp_path: Path) -> None:
    res = _build(tmp_path)
    day = tmp_path / "out" / "2019-01-30"
    assert res.day_dir == day and res.status == "complete"
    assert sorted(p.name for p in day.glob("*.parquet")) == ["TEST.events.parquet", "TEST.states.parquet"]
    assert not list(day.glob("OTHR*")) and not list(day.glob("*.tmp")) and not (day / ".lock").exists()
    for name, want, table in (("TEST.events.parquet", EVENTS_TYPES, "events"),
                              ("TEST.states.parquet", _states_types(), "states")):
        schema = pq.read_schema(day / name)
        assert [(f.name, f.type) for f in schema] == want
        meta = {k.decode(): v.decode() for k, v in schema.metadata.items() if not k.startswith(b"ARROW")}
        assert meta["schema_version"] == "lob-export-v1"
        assert meta["table"] == table
        assert meta["price_scale"] == "10000"
        assert meta["sentinel_bid_px"] == "-9999999999" and meta["sentinel_ask_px"] == "9999999999"
        assert meta["symbol"] == "TEST" and meta["date"] == "2019-01-30"


def test_states_table_equals_expected(tmp_path: Path) -> None:
    res = _build(tmp_path, flush_rows=4)  # several row groups
    t = pq.read_table(res.files["TEST"]["states"])
    assert pq.ParquetFile(res.files["TEST"]["states"]).num_row_groups == 5
    names = [f"{s}_{k}_{i}" for s in ("bid", "ask") for k in ("px", "sz", "ct") for i in range(1, 11)]
    d = t.to_pydict()
    rows = []
    for r in range(t.num_rows):
        vals = [d[n][r] for n in names]
        rows.append((d["seq"][r], d["ts_ns"][r], *(tuple(vals[j:j + 10]) for j in range(0, 60, 10)), d["flags"][r]))
    assert rows == ls.EXPECTED_FULL


def test_states_seq_subset_of_events(tmp_path: Path) -> None:
    res = _build(tmp_path)
    ev = set(pq.read_table(res.files["TEST"]["events"], columns=["seq"]).column("seq").to_pylist())
    st = pq.read_table(res.files["TEST"]["states"], columns=["seq"]).column("seq").to_pylist()
    assert st == sorted(st) and set(st) <= ev


def test_event_table_enrichment(tmp_path: Path) -> None:
    res = _build(tmp_path)
    rows = pq.read_table(res.files["TEST"]["events"]).to_pylist()
    assert len(rows) == 21
    by = {r["seq"]: r for r in rows}
    assert [r["seq"] for r in rows] == sorted(by)
    assert all(r["date"] == DATE and r["symbol"] == "TEST" for r in rows)
    assert (by[18]["side"], by[18]["price"], by[18]["shares"]) == ("S", ls.P100, 10)
    u = by[15]
    assert (u["order_ref"], u["new_order_ref"], u["side"], u["price"], u["shares"]) == (4, 6, "B", ls.P100_05, 80)
    c = by[13]
    assert (c["exec_price"], c["printable"], c["exec_shares"], c["side"], c["price"]) == (ls.P100_05, "N", 60, "B", ls.P100)
    assert (by[20]["side"], by[20]["order_ref"]) == ("B", 0)
    assert (by[21]["cross_type"], by[21]["shares"]) == ("O", 1000)
    assert by[22]["broken_match"] == 103
    assert (by[25]["side"], by[25]["price"]) == ("", 0)
    assert by[23]["trading_state"] == "H"
    assert all(r["trading_state"] == "T" for s, r in by.items() if s >= 24)
    # E takes the resting order's display price as exec price.
    assert (by[12]["exec_price"], by[12]["exec_shares"], by[12]["side"]) == (ls.P100, 40, "B")


def test_log_contents(tmp_path: Path) -> None:
    res = _build(tmp_path)
    log = json.loads(res.log_path.read_text())
    assert log["status"] == "complete" and log["error"] is None
    assert log["schema_version"] == "lob-export-v1" and log["date"] == "2019-01-30"
    assert log["parse"]["unknown_type_counts"] == {"Z": 1}
    assert log["parse"]["skipped_untracked"] == 1
    assert log["parse"]["messages_total"] == 31
    sym = log["symbols"]["TEST"]
    assert sym["status"] == "invalid" and sym["first_error_seq"] == 25
    assert sym["events_rows"] == 21 and sym["states_rows"] == 19
    assert sym["trades_printable"] == 3 and sym["trade_shares_printable"] == 1065
    assert sym["orders_resting_end"] == 3
    assert sym["crossed_segments"] == [[17, 18, ls.T(9, 30, 0, 5), ls.T(9, 30, 0, 6)]]
    assert sym["directory"]["round_lot"] == 100 and sym["locate"] == 1
    assert log["invalid_symbols"] == ["TEST"] and res.invalid_symbols == ["TEST"]
    prov = log["provenance"]
    assert {"git_head", "git_dirty", "source_sha256"} <= set(prov)
    assert "dof/lob/tables/build.py" in prov["source_sha256"]
    assert log["outputs"]["TEST"]["states_sha256"] == _sha(res.files["TEST"]["states"])
    assert log["outputs"]["TEST"]["events_sha256"] == _sha(res.files["TEST"]["events"])
    assert log["input"]["manifest_checked"] is False
    assert set(log["timing"]) == {"sha256_s", "parse_book_s", "write_s", "total_s", "max_rss_kb"}


def test_missing_symbol_logged(tmp_path: Path) -> None:
    res = _build(tmp_path, symbols=["TEST", "NOPE"])
    log = json.loads(res.log_path.read_text())
    assert log["missing_symbols"] == ["NOPE"] and res.missing_symbols == ["NOPE"]
    assert not list(res.day_dir.glob("NOPE*"))


def test_refuses_rebuild_without_overwrite(tmp_path: Path) -> None:
    gz = _gz(tmp_path)
    res = _build(tmp_path, gz=gz)
    before = {p.name: _sha(p) for p in res.day_dir.glob("*.parquet")}
    with pytest.raises(RuntimeError, match="already built"):
        _build(tmp_path, gz=gz)
    assert json.loads(res.log_path.read_text())["status"] == "complete"
    assert {p.name: _sha(p) for p in res.day_dir.glob("*.parquet")} == before
    assert not (res.day_dir / ".lock").exists()
    res2 = _build(tmp_path, gz=gz, overwrite=True)
    assert res2.status == "complete"
    assert json.loads(res2.log_path.read_text())["status"] == "complete"


def test_manifest_refuses_unverified_input(tmp_path: Path) -> None:
    gz = _gz(tmp_path)
    other = _manifest(tmp_path / "MANIFEST.csv", [_ok_row(gz, file="other.gz")])
    with pytest.raises(ValueError, match="input not verified complete in MANIFEST.csv"):
        _build(tmp_path, gz=gz, manifest_csv=other)
    day = tmp_path / "out" / "2019-01-30"
    assert not list(day.glob("*.parquet")) and not (day / "build_log.json").exists()
    assert not (day / ".lock").exists()
    # A failed attempt followed by a good one: the last ok row wins.
    retried = _manifest(tmp_path / "MANIFEST2.csv", [
        _ok_row(gz, bytes_got=gz.stat().st_size - 5, gzip_test="SIZE_MISMATCH"),
        _ok_row(gz),
    ])
    assert read_manifest(retried)[gz.name]["gzip_test"] == "ok"
    res = _build(tmp_path, gz=gz, manifest_csv=retried)
    assert json.loads(res.log_path.read_text())["input"]["manifest_checked"] is True
    # Only a non-ok row -> treated as absent.
    bad_only = _manifest(tmp_path / "MANIFEST3.csv", [_ok_row(gz, gzip_test="SIZE_MISMATCH")])
    assert read_manifest(bad_only) == {}


def test_manifest_size_mismatch_refused(tmp_path: Path) -> None:
    gz = _gz(tmp_path)
    m = _manifest(tmp_path / "MANIFEST.csv", [_ok_row(gz, bytes_got=gz.stat().st_size + 1)])
    with pytest.raises(ValueError, match="input not verified complete"):
        _build(tmp_path, gz=gz, manifest_csv=m)


def test_manifest_sha_mismatch_refused(tmp_path: Path) -> None:
    gz = _gz(tmp_path)
    m = _manifest(tmp_path / "MANIFEST.csv", [_ok_row(gz, sha256="0" * 64)])
    with pytest.raises(ValueError, match="sha256"):
        _build(tmp_path, gz=gz, manifest_csv=m)
    day = tmp_path / "out" / "2019-01-30"
    assert not list(day.glob("*.parquet")) and not (day / "build_log.json").exists()


def test_truncated_input_fails_closed(tmp_path: Path) -> None:
    gz = _gz(tmp_path)
    gz.write_bytes(gz.read_bytes()[:-20])
    with pytest.raises(ItchFormatError):
        _build(tmp_path, gz=gz)
    day = tmp_path / "out" / "2019-01-30"
    assert not list(day.glob("*.parquet")) and not list(day.glob("*.tmp"))
    assert not (day / ".lock").exists()
    log = json.loads((day / "build_log.json").read_text())
    assert log["status"] == "failed" and "ItchFormatError" in log["error"]


def test_lock_blocks_concurrent(tmp_path: Path) -> None:
    day = tmp_path / "out" / "2019-01-30"
    day.mkdir(parents=True)
    (day / ".lock").write_text("pid=1\n")
    with pytest.raises(RuntimeError, match="day_dir locked"):
        _build(tmp_path)
    assert (day / ".lock").read_text() == "pid=1\n"
    assert not (day / "build_log.json").exists() and not list(day.glob("*.parquet"))


def test_export_never_seeds() -> None:
    assert not hasattr(Book, "seed_level") and not hasattr(Book, "reduce_level")
    for p in sorted(Path(export.__file__).parent.glob("*.py")):
        src = p.read_text()
        assert "seed_level" not in src and "reduce_level" not in src, p.name


# -- pins: each fails if the behaviour named in its docstring changes --------------------------------


def _interleaved() -> list[bytes]:
    """Four stocks whose messages alternate; DDD later executes an order it never added (unknown ref)."""
    syms = ["AAA", "BBB", "CCC", "DDD"]
    out = [ls.msg_S(ls.T(3, 0), "O")]
    out += [ls.msg_R(i + 1, ls.T(3, 5, 0, i), s) for i, s in enumerate(syms)]
    out += [ls.msg_H(i + 1, ls.T(9, 28, 0, i), s, "T") for i, s in enumerate(syms)]
    out.append(ls.msg_S(ls.T(9, 30), "Q"))
    ref, ts, live = 1, ls.T(9, 30, 1), {s: [] for s in syms}
    for k in range(40):
        for i, s in enumerate(syms):
            loc, base = i + 1, 1_000_000 * (i + 1)
            for side, sign in (("B", -1), ("S", 1)):
                out.append(ls.msg_A(loc, ts, ref, side, 100 + k, s, base + sign * 100 * (1 + k % 12)))
                live[s].append(ref)
                ref, ts = ref + 1, ts + 1
            if k % 3 == 2:
                out.append(ls.msg_D(loc, ts, live[s].pop(0)))
                ts += 1
            if k % 4 == 3:
                out.append(ls.msg_E(loc, ts, live[s][0], 10, 50_000 + ref))
                ts += 1
            if s == "DDD" and k == 20:
                out.append(ls.msg_E(loc, ts, 999_999, 10, 60_000))
                ts += 1
    out += [ls.msg_S(ls.T(16, 0), "M"), ls.msg_S(ls.T(20, 0), "E"), ls.msg_S(ls.T(20, 5), "C")]
    return out


def test_build_four_symbols_interleaved(tmp_path: Path) -> None:
    """A stock's tables do not depend on the other stocks of the build; DDD is invalid."""
    syms = ["AAA", "BBB", "CCC", "DDD"]
    gz = _gz(tmp_path, _interleaved())
    together = build_day(gz, syms, tmp_path / "all", manifest_csv=None, date=DATE)
    log = json.loads(together.log_path.read_text())
    assert [s for s in syms if log["symbols"][s]["status"] == "invalid"] == ["DDD"]
    for s in syms:
        alone = build_day(gz, [s], tmp_path / f"only_{s}", manifest_csv=None, date=DATE)
        for table in ("events", "states"):
            assert pq.read_table(alone.files[s][table]).equals(pq.read_table(together.files[s][table])), (s, table)


def test_limit_bytes_build_reports_stop(tmp_path: Path) -> None:
    """A build stopped by --limit-bytes says so in its log, and it is not complete."""
    limit = len(ls.stream(ls.SCENARIO_FULL[:20]))
    res = _build(tmp_path, limit_bytes=limit)
    log = json.loads(res.log_path.read_text())
    assert log["parse"]["stopped_by_limit"] is True
    assert res.status == "partial" and log["status"] == "partial"


def test_build_without_end_of_messages_is_partial(tmp_path: Path) -> None:
    res = _build(tmp_path, gz=_gz(tmp_path, ls.SCENARIO_FULL[:29]))      # ends with S "M": no S "E", no S "C"
    log = json.loads(res.log_path.read_text())
    assert log["parse"]["stopped_by_limit"] is False
    assert res.status == "partial" and log["status"] == "partial"
