from __future__ import annotations

from pathlib import Path

from dof.lob.book import (
    FLAG_CROSSED,
    FLAG_HALTED,
    FLAG_INVALID,
    FLAG_LOCKED,
    LEVELS,
    SENTINEL_ASK_PX,
    SENTINEL_BID_PX,
    Book,
    Snapshot10,
)
from dof.lob.itch import LocateFilter, Message, ParseStats, iter_messages
from dof.lob.testing import synth as ls


def _run(tmp_path: Path, payloads: list[bytes], symbol: str) -> tuple[Book, dict[int, Snapshot10 | None], ParseStats]:
    """Parse with iter_messages and apply every yielded message; return results keyed by seq."""
    path = ls.write_stream(tmp_path / f"{symbol}.itch.gz", ls.stream(payloads), gz=True)
    stats = ParseStats()
    book = Book(symbol, 1)
    out: dict[int, Snapshot10 | None] = {}
    for msg in iter_messages(path, LocateFilter({symbol}), stats=stats):
        out[msg.seq] = book.apply(msg)
    return book, out, stats


def _rows(out: dict[int, Snapshot10 | None]) -> list[Snapshot10]:
    return [s for _, s in sorted(out.items()) if s is not None]


def _full(tmp_path: Path):
    return _run(tmp_path, ls.SCENARIO_FULL, "TEST")


def _book_with(*msgs: Message, state: str = "T") -> Book:
    book = Book("TEST", 1)
    book.trading_state = state
    for m in msgs:
        book.apply(m)
    return book


def _add(seq: int, ref: int, side: str, shares: int, price: int) -> Message:
    return Message(seq, seq, "A", 1, order_ref=ref, side=side, shares=shares, price=price, symbol="TEST")


def test_full_scenario_rows_exact(tmp_path: Path) -> None:
    _, out, _ = _full(tmp_path)
    rows = _rows(out)
    assert [r.seq for r in rows] == [e[0] for e in ls.EXPECTED_FULL]
    for got, want in zip(rows, ls.EXPECTED_FULL):
        assert tuple(got) == want, got.seq
    assert len(rows) == 19


def test_full_scenario_stats(tmp_path: Path) -> None:
    book, _, pstats = _full(tmp_path)
    for name, want in ls.EXPECTED_FULL_STATS.items():
        assert getattr(book.stats, name) == want, name
    assert book.stats.first_error == "unknown_ref E ref=99"
    assert book.stats.rows_emitted == 19
    assert book.resting_orders() == ls.EXPECTED_FULL_RESTING
    assert sorted(book.orders) == [2, 5, 6]
    assert book.valid is False
    for name, want in ls.EXPECTED_FULL_PARSE.items():
        assert getattr(pstats, name) == want, name


def test_depth_beyond_ten_enters_window(tmp_path: Path) -> None:
    book, out, _ = _run(tmp_path, ls.SCENARIO_DEPTH, "DEEP")
    rows = _rows(out)
    assert [tuple(r) for r in rows] == ls.EXPECTED_DEPTH
    assert out[13] is None and out[14] is None and out[16] is None  # i = 10, 11 adds; X on rank 10
    assert book.orders[111] == [True, 1_000_000 - 1_100, 50]


def test_same_price_aggregation(tmp_path: Path) -> None:
    _, out, _ = _full(tmp_path)
    s8 = out[8]
    assert (s8.bid_px[0], s8.bid_sz[0], s8.bid_ct[0]) == (ls.P100, 150, 2)


def test_replace_inherits_side_and_moves_level(tmp_path: Path) -> None:
    book = Book("TEST", 1)
    path = ls.write_stream(tmp_path / "t.itch", ls.stream(ls.SCENARIO_FULL[:16]), gz=False)
    snaps = {m.seq: book.apply(m) for m in iter_messages(path, LocateFilter({"TEST"}))}
    s15 = snaps[15]
    assert ls.P100 not in s15.bid_px
    assert (s15.bid_px[0], s15.bid_sz[0], s15.bid_ct[0]) == (ls.P100_05, 80, 1)
    assert ls.P100_05 not in s15.ask_px
    assert 4 not in book.orders
    assert book.orders[6] == [True, ls.P100_05, 80]


def test_c_nonprintable_reduces_book_not_trades(tmp_path: Path) -> None:
    book = Book("TEST", 1)
    path = ls.write_stream(tmp_path / "t.itch", ls.stream(ls.SCENARIO_FULL), gz=False)
    trades_after: dict[int, tuple[int, int]] = {}
    snaps: dict[int, Snapshot10 | None] = {}
    for m in iter_messages(path, LocateFilter({"TEST"})):
        snaps[m.seq] = book.apply(m)
        trades_after[m.seq] = (book.stats.trades_printable, book.stats.trade_shares_printable)
    assert snaps[13].bid_sz[0] == 50
    assert book.stats.nonprintable_exec == 1
    assert trades_after[12] == (1, 40)
    assert trades_after[13] == trades_after[12]  # the C(N) is not a trade print
    assert trades_after[21][1] - trades_after[20][1] == 1000


def test_p_q_repeat_rows(tmp_path: Path) -> None:
    _, out, _ = _full(tmp_path)
    key19 = out[19].book_key()
    assert out[20] is not None and out[21] is not None
    assert out[20].book_key() == key19
    assert out[21].book_key() == key19


def test_b_no_row(tmp_path: Path) -> None:
    book = Book("TEST", 1)
    path = ls.write_stream(tmp_path / "t.itch", ls.stream(ls.SCENARIO_FULL), gz=False)
    for m in iter_messages(path, LocateFilter({"TEST"})):
        if m.seq == 22:
            before = (book.snapshot(0, 0), {k: list(v) for k, v in book.orders.items()})
            assert book.apply(m) is None
            assert (book.snapshot(0, 0), book.orders) == before
            break
        book.apply(m)
    assert book.stats.broken == 1


def test_initial_state_halted_until_T() -> None:
    book = Book("TEST", 1)
    snap = book.apply(_add(0, 1, "B", 100, ls.P100))
    assert snap is not None and snap.flags & FLAG_HALTED


def test_unknown_ref_marks_invalid(tmp_path: Path) -> None:
    book, out, _ = _full(tmp_path)
    assert out[25] is None
    assert book.valid is False
    assert out[26].flags == FLAG_INVALID
    assert out[24].flags == 0


def test_size_underflow_marks_invalid(tmp_path: Path) -> None:
    book, out, _ = _full(tmp_path)
    assert 9 not in book.orders
    assert book.stats.size_underflow == 1
    assert ls.P99_80 not in out[27].bid_px
    assert all(sz >= 0 for sz in out[27].bid_sz)


def test_duplicate_add_ref_invalid() -> None:
    book = _book_with(_add(0, 1, "B", 100, ls.P100))
    before = book.snapshot(0, 0).book_key()
    assert book.apply(_add(1, 1, "S", 50, ls.P100_10)) is None
    assert book.stats.duplicate_ref == 1
    assert book.valid is False
    assert book.orders == {1: [True, ls.P100, 100]}
    after = book.snapshot(0, 0).book_key()
    assert after[:6] == before[:6]  # levels unchanged; only the INVALID flag differs
    assert after[6] == before[6] | FLAG_INVALID


def test_replace_unknown_old_ref_invalid() -> None:
    book = _book_with(_add(0, 1, "B", 100, ls.P100))
    snap = book.apply(Message(1, 1, "U", 1, order_ref=42, new_order_ref=43, shares=10, price=ls.P100_05))
    assert snap is None
    assert book.stats.unknown_ref == 1
    assert 43 not in book.orders
    assert ls.P100_05 not in book.bids.levels and ls.P100_05 not in book.asks.levels


def test_zero_share_cancel_no_row() -> None:
    book = _book_with(_add(0, 1, "B", 100, ls.P100))
    assert book.apply(Message(1, 1, "X", 1, order_ref=1, shares=0)) is None
    assert book.orders[1] == [True, ls.P100, 100]


def test_deep_add_no_row() -> None:
    book = _book_with(*[_add(i, 100 + i, "B", 100, 1_000_000 - 100 * i) for i in range(10)])
    assert book.apply(_add(10, 200, "B", 100, 1_000_000 - 100 * 10)) is None
    assert book.orders[200] == [True, 999_000, 100]
    assert book.bids.levels[999_000] == [100, 1]


def test_zero_share_add_not_booked() -> None:
    # A 0-share add is counted, not booked, and emits no row.
    book = _book_with(_add(0, 1, "B", 100, ls.P100))
    assert book.apply(_add(1, 2, "S", 0, ls.P100_10)) is None
    assert book.stats.zero_share_add == 1
    assert 2 not in book.orders and book.valid is True


# -- pins: each fails if the behaviour named in its docstring changes --------------------------------


def test_printable_c_counts_as_trade() -> None:
    """A C marked printable "Y" is a trade print; one marked "N" only reduces the book."""
    book = _book_with(_add(0, 1, "B", 100, ls.P100))
    book.apply(Message(1, 1, "C", 1, order_ref=1, exec_shares=30, match_number=7, printable="Y", exec_price=ls.P100))
    assert (book.stats.trades_printable, book.stats.trade_shares_printable, book.stats.nonprintable_exec) == (1, 30, 0)
    book.apply(Message(2, 2, "C", 1, order_ref=1, exec_shares=20, match_number=8, printable="N", exec_price=ls.P100))
    assert (book.stats.trades_printable, book.stats.trade_shares_printable, book.stats.nonprintable_exec) == (1, 30, 1)
    assert book.orders[1] == [True, ls.P100, 50]


def _cross_and_restore(book: Book, seq: int, ref: int) -> int:
    """An ask below the best bid crosses the book; deleting it restores it. Returns the next seq."""
    book.apply(_add(seq, ref, "S", 10, ls.P99_90))
    book.apply(Message(seq + 1, seq + 1, "D", 1, order_ref=ref))
    return seq + 2


def test_crossed_segments_cap(monkeypatch) -> None:
    import dof.lob.book as book_module
    monkeypatch.setattr(book_module, "CROSSED_SEGMENTS_CAP", 2)
    book = _book_with(_add(0, 1, "B", 100, ls.P100), _add(1, 2, "S", 100, ls.P100_10))
    seq = 2
    for i in range(3):
        seq = _cross_and_restore(book, seq, 100 + i)
    assert book.stats.crossed_segment_count == 3
    assert len(book.stats.crossed_segments) == 2


def test_crossed_segment_open_at_end() -> None:
    book = _book_with(_add(0, 1, "B", 100, ls.P100), _add(1, 2, "S", 100, ls.P100_10))
    book.apply(_add(5, 3, "S", 10, ls.P99_90))
    assert book.stats.crossed_segments == [(5, -1, 5, -1)]


class _ReferenceBook:
    """A plain dict-of-orders book: the top 10 levels and flags recomputed from scratch after every message."""

    def __init__(self) -> None:
        self.orders: dict[int, list] = {}
        self.state, self.valid = "H", True

    def apply(self, m: Message) -> None:
        t = m.msg_type
        if t in ("A", "F"):
            if m.order_ref in self.orders:
                self.valid = False
            elif m.shares:
                self.orders[m.order_ref] = [m.side == "B", m.price, m.shares]
        elif t in ("E", "C", "X"):
            o = self.orders.get(m.order_ref)
            if o is None:
                self.valid = False
                return
            n = m.shares if t == "X" else m.exec_shares
            if n > o[2]:
                self.valid, n = False, o[2]
            o[2] -= n
            if o[2] == 0:
                del self.orders[m.order_ref]
        elif t == "D":
            if self.orders.pop(m.order_ref, None) is None:
                self.valid = False
        elif t == "U":
            o = self.orders.pop(m.order_ref, None)
            if o is None:
                self.valid = False
            elif m.new_order_ref in self.orders:
                self.valid = False
            elif m.shares:
                self.orders[m.new_order_ref] = [o[0], m.price, m.shares]
        elif t == "H":
            self.state = m.trading_state

    def snapshot(self) -> tuple:
        levels: dict[bool, dict[int, list[int]]] = {True: {}, False: {}}
        for is_bid, price, shares in self.orders.values():
            agg = levels[is_bid].setdefault(price, [0, 0])
            agg[0] += shares
            agg[1] += 1
        out: list = []
        for is_bid, sentinel in ((True, SENTINEL_BID_PX), (False, SENTINEL_ASK_PX)):
            prices = sorted(levels[is_bid], reverse=is_bid)[:LEVELS]
            pad = LEVELS - len(prices)
            out += [tuple(prices) + (sentinel,) * pad, tuple(levels[is_bid][p][0] for p in prices) + (0,) * pad,
                    tuple(levels[is_bid][p][1] for p in prices) + (0,) * pad]
        flags = 0
        if levels[True] and levels[False]:
            bb, ba = max(levels[True]), min(levels[False])
            flags |= FLAG_LOCKED if bb == ba else (FLAG_CROSSED if bb > ba else 0)
        if self.state != "T":
            flags |= FLAG_HALTED
        if not self.valid:
            flags |= FLAG_INVALID
        return (*out, flags)


def test_book_state_matches_reference_book(tmp_path: Path) -> None:
    """Up to 20,000 messages of one synthetic stock (the seeded day holds about 14,400): after each, the book's
    snapshot equals the reference book's."""
    from recipes import synth_stream
    path = tmp_path / "day.itch"
    path.write_bytes(synth_stream.day_stream("2019-01-30"))
    locate = synth_stream.LOCATE["AAA"]
    book, ref, n = Book("AAA", locate), _ReferenceBook(), 0
    for m in iter_messages(path, LocateFilter({"AAA"})):
        if m.locate != locate:
            continue
        book.apply(m)
        ref.apply(m)
        snap = book.snapshot(m.seq, m.ts_ns)
        assert (snap.bid_px, snap.bid_sz, snap.bid_ct, snap.ask_px, snap.ask_sz, snap.ask_ct, snap.flags) == ref.snapshot(), m
        n += 1
        if n == 20_000:
            break
    assert n >= 10_000
