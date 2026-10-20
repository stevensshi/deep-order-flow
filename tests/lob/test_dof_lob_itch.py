from __future__ import annotations

import datetime
import gzip
import os
from pathlib import Path

import pytest

from dof.lob.itch import (
    MSG_LEN,
    ItchFormatError,
    LocateFilter,
    Message,
    ParseStats,
    iter_messages,
    parse_itch_date,
    scan_counts,
)
from dof.lob.testing import synth as ls

REAL_ITCH_ENV = "DOF_LOB_REAL_ITCH"   # path of the real file 01302019.NASDAQ_ITCH50.gz


def _write(tmp_path: Path, payloads: list[bytes], name: str = "s.itch", gz: bool = False) -> Path:
    return ls.write_stream(tmp_path / name, ls.stream(payloads), gz)


def _parse(path: Path, locates, **kw) -> tuple[list[Message], ParseStats]:
    stats = ParseStats()
    return list(iter_messages(path, locates, stats=stats, **kw)), stats


def test_msg_len_matches_spec() -> None:
    # Hard-coded from the ITCH 5.0 spec (rev. 2023-04-28), independently of the module.
    spec = {
        "S": 12, "R": 39, "H": 25, "Y": 20, "L": 26, "V": 35, "W": 12, "K": 28,
        "J": 35, "h": 21, "A": 36, "F": 40, "E": 31, "C": 36, "X": 23, "D": 19,
        "U": 35, "P": 44, "Q": 40, "B": 19, "I": 50, "N": 20, "O": 48,
    }
    assert len(spec) == 23
    assert MSG_LEN == {ord(k): v for k, v in spec.items()}


def test_each_type_fields_decoded(tmp_path: Path) -> None:
    big_ts = 5 * 2**32 + 7  # > 2^32: a u32 read of the u48 field would lose the high bits
    q_shares = 5 * 2**32    # Q shares are u64 (spec §1.5.2)
    payloads = [
        ls.msg_S(big_ts, "O"),
        ls.msg_R(3, big_ts + 1, "ABC"),
        ls.msg_H(3, big_ts + 2, "ABC", "T", "R1  "),
        ls.msg_A(3, big_ts + 3, 11, "B", 101, "ABC", 1_234_500),
        ls.msg_F(3, big_ts + 4, 12, "S", 102, "ABC", 1_234_600, mpid="MPID"),
        ls.msg_E(3, big_ts + 5, 11, 7, 9001),
        ls.msg_C(3, big_ts + 6, 12, 8, 9002, "N", 1_234_700),
        ls.msg_X(3, big_ts + 7, 13, 9),
        ls.msg_D(3, big_ts + 8, 14),
        ls.msg_U(3, big_ts + 9, 15, 16, 103, 1_234_800),
        ls.msg_P(3, big_ts + 10, 104, "ABC", 1_234_900, 9003, side="B", ref=17),
        ls.msg_Q(3, big_ts + 11, q_shares, "ABC", 1_235_000, 9004, "C"),
        ls.msg_B(3, big_ts + 12, 9005),
    ]
    path = _write(tmp_path, payloads)
    filt = LocateFilter({"ABC"})
    msgs, _ = _parse(path, filt)
    expected = [
        Message(0, big_ts, "S", 0, event_code="O"),
        Message(1, big_ts + 1, "R", 3, symbol="ABC"),
        Message(2, big_ts + 2, "H", 3, symbol="ABC", trading_state="T", reason="R1"),
        Message(3, big_ts + 3, "A", 3, order_ref=11, side="B", shares=101, price=1_234_500, symbol="ABC"),
        Message(4, big_ts + 4, "F", 3, order_ref=12, side="S", shares=102, price=1_234_600, symbol="ABC"),
        Message(5, big_ts + 5, "E", 3, order_ref=11, exec_shares=7, match_number=9001),
        Message(6, big_ts + 6, "C", 3, order_ref=12, exec_shares=8, match_number=9002, printable="N",
                exec_price=1_234_700),
        Message(7, big_ts + 7, "X", 3, order_ref=13, shares=9),
        Message(8, big_ts + 8, "D", 3, order_ref=14),
        Message(9, big_ts + 9, "U", 3, order_ref=15, new_order_ref=16, shares=103, price=1_234_800),
        Message(10, big_ts + 10, "P", 3, order_ref=17, side="B", shares=104, price=1_234_900,
                match_number=9003, symbol="ABC"),
        Message(11, big_ts + 11, "Q", 3, shares=q_shares, symbol="ABC", price=1_235_000,
                match_number=9004, cross_type="C"),
        Message(12, big_ts + 12, "B", 3, broken_match=9005),
    ]
    assert len(msgs) == len(expected)
    for got, want in zip(msgs, expected):  # per message, as dicts, for a readable diff
        assert got._asdict() == want._asdict()
    # R directory fields at the spec offsets (§1.2.1); ETP flag is @33, not @36.
    d = filt.directory["ABC"]
    assert d == {
        "locate": 3, "seq": 1, "ts_ns": big_ts + 1, "market_category": "Q", "financial_status": "\x00",
        "round_lot": 100, "round_lots_only": "\x00", "issue_classification": "\x00",
        "issue_subtype": "\x00\x00", "authenticity": "\x00", "short_sale_threshold": "\x00",
        "ipo_flag": "\x00", "luld_reference": "\x00", "etp_flag": "N", "etp_leverage": 0,
        "inverse_indicator": "\x00",
    }


def test_header_only_types_yielded_when_tracked(tmp_path: Path) -> None:
    # Known types with no Message field (Y, L, V, …) are framed and yielded with their header.
    others = [t for t in MSG_LEN if chr(t) not in "SRHAFECXDUPQB"]
    path = _write(tmp_path, [ls.msg_unknown(chr(t), MSG_LEN[t]) for t in others])
    msgs, stats = _parse(path, None)
    assert [m.msg_type for m in msgs] == [chr(t) for t in others]
    assert all(m == Message(i, 0, chr(t), 0) for i, (m, t) in enumerate(zip(msgs, others)))
    assert stats.unknown_type_counts == {}


def test_untracked_locate_skipped_and_counted(tmp_path: Path) -> None:
    path = _write(tmp_path, ls.SCENARIO_FULL)
    msgs, stats = _parse(path, LocateFilter({"TEST"}))
    seqs = [m.seq for m in msgs]
    assert 7 not in seqs
    assert stats.skipped_untracked == 1
    assert stats.messages_total == 31


def test_unknown_type_skipped_by_length(tmp_path: Path) -> None:
    path = _write(tmp_path, ls.SCENARIO_FULL)
    msgs, stats = _parse(path, LocateFilter({"TEST"}))
    by_seq = {m.seq: m for m in msgs}
    assert 9 not in by_seq
    assert stats.unknown_type_counts == {"Z": 1}
    assert stats.counts_by_type["Z"] == 1
    assert by_seq[10] == Message(10, ls.T(4, 0, 0, 5), "A", 1, order_ref=5, side="B", shares=300,
                                 price=ls.P99_90, symbol="TEST")


def test_known_type_wrong_length_raises(tmp_path: Path) -> None:
    bad = ls.msg_A(1, 1, 1, "B", 100, "TEST", ls.P100)[:35]
    path = _write(tmp_path, [ls.msg_S(0, "O"), bad])
    with pytest.raises(ItchFormatError, match="length mismatch type=A seq=1 got=35 want=36"):
        _parse(path, None)


def test_seq_counts_every_message(tmp_path: Path) -> None:
    path = _write(tmp_path, ls.SCENARIO_FULL)
    msgs, _ = _parse(path, LocateFilter({"TEST"}))
    for m in msgs:
        payload = ls.SCENARIO_FULL[m.seq]
        assert m.msg_type == chr(payload[0])
        assert m.locate == int.from_bytes(payload[1:3], "big")
        assert m.ts_ns == int.from_bytes(payload[5:11], "big")
    # Everything except the untracked R (2), the untracked A (7) and the unknown type (9).
    assert [m.seq for m in msgs] == [i for i in range(31) if i not in (2, 7, 9)]


def test_locate_filter_learns_from_directory(tmp_path: Path) -> None:
    path = _write(tmp_path, ls.SCENARIO_FULL)
    filt = LocateFilter({"TEST"})
    msgs, _ = _parse(path, filt)
    r_syms = [m.symbol for m in msgs if m.msg_type == "R"]
    assert r_syms == ["TEST"]
    assert filt.locates == {1}
    assert filt.locate_to_symbol == {1: "TEST"}
    assert filt.directory["TEST"]["round_lot"] == 100
    assert filt.directory_size == 2
    assert filt.directory_duplicates == 0


def test_directory_duplicate_keeps_first(tmp_path: Path) -> None:
    path = _write(tmp_path, [ls.msg_R(1, 10, "TEST"), ls.msg_R(5, 20, "TEST"), ls.msg_A(5, 30, 1, "B", 1, "TEST", 1)])
    filt = LocateFilter({"TEST"})
    msgs, stats = _parse(path, filt)
    assert filt.directory["TEST"]["locate"] == 1
    assert filt.locates == {1}
    assert filt.directory_duplicates == 1
    assert stats.skipped_untracked == 1  # the A on the duplicate's locate is not tracked


def test_set_and_none_filters(tmp_path: Path) -> None:
    path = _write(tmp_path, ls.SCENARIO_FULL)
    msgs_set, stats_set = _parse(path, {2})
    assert [m.seq for m in msgs_set] == [0, 2, 4, 7, 11, 28, 29, 30]
    assert stats_set.skipped_untracked == 21
    msgs_all, stats_all = _parse(path, None)
    assert [m.seq for m in msgs_all] == [i for i in range(31) if i != 9]
    assert stats_all.skipped_untracked == 0
    assert stats_all.tracked_by_type["A"] == 7  # seq 5, 6, 7, 10, 16, 17, 26
    assert sum(stats_all.tracked_by_type.values()) == 30
    assert stats_all.system_events == [
        (0, ls.T(3, 0), "O"), (4, ls.T(4, 0), "S"), (11, ls.T(9, 30), "Q"),
        (28, ls.T(16, 0), "M"), (29, ls.T(20, 0), "E"), (30, ls.T(20, 5), "C"),
    ]
    assert stats_all.first_ts_ns == ls.T(3, 0)
    assert stats_all.last_ts_ns == ls.T(20, 5)
    assert stats_all.bytes_decompressed == len(ls.stream(ls.SCENARIO_FULL))


def test_gzip_and_plain_identical(tmp_path: Path) -> None:
    plain = _write(tmp_path, ls.SCENARIO_FULL, "s.itch", gz=False)
    gz = _write(tmp_path, ls.SCENARIO_FULL, "s.itch.gz", gz=True)
    assert gz.read_bytes()[:2] == b"\x1f\x8b"
    msgs_plain, stats_plain = _parse(plain, LocateFilter({"TEST"}))
    msgs_gz, stats_gz = _parse(gz, LocateFilter({"TEST"}))
    assert msgs_plain == msgs_gz
    assert stats_plain == stats_gz


def test_chunk_boundary_carry(tmp_path: Path) -> None:
    path = _write(tmp_path, ls.SCENARIO_FULL, "s.itch.gz", gz=True)
    msgs_default, stats_default = _parse(path, LocateFilter({"TEST"}))
    msgs_small, stats_small = _parse(path, LocateFilter({"TEST"}), chunk_size=7)
    assert msgs_small == msgs_default
    assert stats_small == stats_default
    assert len(msgs_default) == 28


def test_truncated_gzip_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, ls.SCENARIO_FULL, "s.itch.gz", gz=True)
    path.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(ItchFormatError, match="truncated gzip"):
        _parse(path, None)


def test_truncated_plain_tail_raises(tmp_path: Path) -> None:
    path = tmp_path / "s.itch"
    path.write_bytes(ls.stream(ls.SCENARIO_FULL) + b"\x00\x13S")
    with pytest.raises(ItchFormatError, match="truncated stream at seq=31"):
        _parse(path, None)
    # Messages before the fragment were yielded before the error.
    got: list[Message] = []
    with pytest.raises(ItchFormatError):
        for m in iter_messages(path, None):
            got.append(m)
    assert got[-1].seq == 30


def test_limit_bytes_stops_on_boundary(tmp_path: Path) -> None:
    data = ls.stream(ls.SCENARIO_FULL)
    path = ls.write_stream(tmp_path / "s.itch.gz", data, gz=True)
    end_of_5 = len(ls.stream(ls.SCENARIO_FULL[:6]))
    limit = end_of_5 + 10  # inside message 6 (an A, 38 framed bytes)
    for chunk in (7, 1 << 20):
        msgs, stats = _parse(path, None, limit_bytes=limit, chunk_size=chunk)
        assert msgs[-1].seq == 5
        assert stats.stopped_by_limit is True
        assert stats.messages_total == 6
        assert stats.bytes_decompressed == end_of_5
    # A limit exactly on a boundary keeps the message that ends there.
    msgs, stats = _parse(path, None, limit_bytes=end_of_5)
    assert msgs[-1].seq == 5 and stats.stopped_by_limit is True
    # A limit past the end is a normal end.
    msgs, stats = _parse(path, None, limit_bytes=len(data) + 100)
    assert msgs[-1].seq == 30 and stats.stopped_by_limit is False


def test_scan_counts_matches_iter(tmp_path: Path) -> None:
    path = _write(tmp_path, ls.SCENARIO_FULL, "s.itch.gz", gz=True)
    _, stats_iter = _parse(path, None)
    stats_scan = scan_counts(path, chunk_size=7)
    assert stats_scan.counts_by_type == stats_iter.counts_by_type
    assert stats_scan.messages_total == stats_iter.messages_total == 31
    assert stats_scan.unknown_type_counts == stats_iter.unknown_type_counts == {"Z": 1}
    assert stats_scan.bytes_decompressed == stats_iter.bytes_decompressed
    assert (stats_scan.first_ts_ns, stats_scan.last_ts_ns) == (ls.T(3, 0), ls.T(20, 5))
    with pytest.raises(ItchFormatError):
        bad = tmp_path / "bad.itch"
        bad.write_bytes(ls.stream([ls.msg_S(0, "O")]) + b"\x00\x23A")
        scan_counts(bad)


def test_parse_itch_date() -> None:
    assert parse_itch_date("01302019.NASDAQ_ITCH50.gz") == datetime.date(2019, 1, 30)
    assert parse_itch_date(Path("/x/S120825-v50.txt.gz")) == datetime.date(2025, 12, 8)
    assert parse_itch_date("S101819-v50.txt") == datetime.date(2019, 10, 18)
    with pytest.raises(ValueError):
        parse_itch_date("foo.gz")


@pytest.mark.local
def test_real_file_head() -> None:
    real = os.environ.get(REAL_ITCH_ENV)
    if not real:
        pytest.skip(f"{REAL_ITCH_ENV} is not set")
    filt = LocateFilter({"AAPL"})
    stats = ParseStats()
    try:
        msgs = list(iter_messages(Path(real), filt, stats=stats, limit_bytes=2_000_000))
    except OSError as exc:
        pytest.skip(f"real ITCH file not readable: {exc}")
    first = msgs[0]
    assert (first.seq, first.msg_type, first.event_code) == (0, "S", "O")
    assert first.ts_ns == 11039687760787  # 03:03:59.687760787
    assert filt.locates == {14}
    assert filt.locate_to_symbol == {14: "AAPL"}
    # A length mismatch would have raised; the limit, not the stream end, stopped the read.
    assert stats.stopped_by_limit is True
    assert stats.unknown_type_counts == {}
    assert 2_000_000 - 50 < stats.bytes_decompressed <= 2_000_000
