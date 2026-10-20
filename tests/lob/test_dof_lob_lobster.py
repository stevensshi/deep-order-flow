"""LOBSTER reconciliation tests.

Synthetic cases write tiny LOBSTER CSVs into tmp_path; the two sample tests read the
public LOBSTER sample files from ``DOF_LOB_LOBSTER_DIR`` and skip when it is unset.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dof.lob.itch import Message
from dof.lob.testing import synth as ls
from dof.lob.validation import (
    LOBSTER_SENTINEL,
    SeededBook,
    _apply_lobster,
    read_lobster,
    reconcile_continuous,
    reconcile_transition,
)

SAMPLES_ENV = "DOF_LOB_LOBSTER_DIR"   # the folder that holds the two LOBSTER sample folders
SAMPLES = Path(os.environ[SAMPLES_ENV]) if os.environ.get(SAMPLES_ENV) else None
L10 = SAMPLES / "lobster_AAPL_2012-06-21_level10" if SAMPLES else None
L50 = SAMPLES / "lobster_AAPL_2012-06-21_level50" if SAMPLES else None
L10_STEM = "AAPL_2012-06-21_34200000_57600000"
L50_STEM = "AAPL_2012-06-21_34200000_37800000"


def _row(asks: list[tuple[int, int]], bids: list[tuple[int, int]], levels: int = 2) -> list[int]:
    out: list[int] = []
    for i in range(levels):
        ap, az = asks[i] if i < len(asks) else (LOBSTER_SENTINEL, 0)
        bp, bz = bids[i] if i < len(bids) else (-LOBSTER_SENTINEL, 0)
        out += [ap, az, bp, bz]
    return out


def _write(tmp_path: Path, msgs: list[tuple], rows: list[list[int]]) -> tuple[Path, Path]:
    m, o = tmp_path / "x_message_2.csv", tmp_path / "x_orderbook_2.csv"
    m.write_text("".join(",".join(str(v) for v in msg) + "\n" for msg in msgs))
    o.write_text("".join(",".join(str(v) for v in r) + "\n" for r in rows))
    return m, o


# A synthetic 2-level day. Message 0's order is already in row 0;
# message 1 deletes it and 980000 enters from beyond the window; message 2 acts
# on an order never submitted; message 3 is a hidden execution.
ASK0 = [(1010000, 10), (1020000, 10)]
MSGS = [
    (34200.1, 1, 1, 10, 1000000, 1),
    (34200.2, 3, 1, 10, 1000000, 1),
    (34200.3, 4, 777, 5, 1010000, -1),
    (34200.4, 5, 0, 3, 1005000, 1),
]


def _rows(ask2_size: int = 5) -> list[list[int]]:
    r0 = _row(ASK0, [(1000000, 10), (990000, 10)])
    r1 = _row(ASK0, [(990000, 10), (980000, 5)])
    r2 = _row([(1010000, ask2_size), (1020000, 10)], [(990000, 10), (980000, 5)])
    r3 = _row([(1010000, 5), (1020000, 10)], [(990000, 10), (980000, 5)])
    return [r0, r1, r2, r3]


def _day(tmp_path: Path, msgs=MSGS, rows=None):
    return read_lobster(*_write(tmp_path, msgs, rows if rows is not None else _rows()))


def test_continuous_synthetic(tmp_path: Path) -> None:
    rep = reconcile_continuous(_day(tmp_path))
    assert rep.diffs == 0, rep.first_diffs
    assert rep.entering_units == 3          # rows 1-3: bid level 2 at 980000, beyond row 0's window
    assert rep.entering_side_rows == 3
    assert rep.unseen_order_msgs == 1
    assert rep.unseen_by_type == {"2": 0, "3": 0, "4": 1}
    assert rep.hidden_exec_rows == 1
    assert rep.hidden_exec_book_changed == 0
    assert rep.hidden_exec_file_changed == 0
    assert rep.order_attr_mismatch == 0 and rep.type3_size_mismatch == 0
    assert rep.to_dict()["mode_label"].startswith("continuous replay from orderbook row 0, no filling")


def test_continuous_detects_wrong_size(tmp_path: Path) -> None:
    rep = reconcile_continuous(_day(tmp_path, rows=_rows(ask2_size=6)))
    assert rep.diffs == 1
    assert rep.first_diffs[0]["row"] == 2 and rep.first_diffs[0]["side"] == "ask"
    assert rep.first_diffs[0]["file"] == [1010000, 6] and rep.first_diffs[0]["book"] == [1010000, 5]
    assert rep.hidden_exec_file_changed == 1   # row 3 no longer equals row 2


def test_transition_synthetic(tmp_path: Path) -> None:
    rep = reconcile_transition(_day(tmp_path))
    assert rep.diffs == 0, rep.first_diffs
    assert rep.entering_side_rows == 1        # only k=1: 980000 lies beyond row 0's window
    assert rep.unseen_order_msgs == 0
    assert rep.hidden_exec_book_changed == 0
    assert "not end-to-end equality" in rep.to_dict()["mode_label"]


def test_never_fills_from_file(tmp_path: Path) -> None:
    book = SeededBook("LOBSTER", 0)
    reconcile_continuous(_day(tmp_path), book=book)
    # 98.0000 appears only in file rows 1-3, never in a message: it must not be in the book.
    assert 980000 not in book.bids.levels
    assert sorted(book.bids.levels) == [990000]


def test_type7_and_type6_adapter(tmp_path: Path) -> None:
    r0 = _row(ASK0, [(1000000, 10), (990000, 10)])
    msgs = [
        (34200.1, 1, 1, 10, 1000000, 1),
        (34200.2, 7, 0, 0, -1, -1),          # halt (ReadMe: price -1)
        (34200.3, 6, 0, 500, 1005000, -1),   # cross trade
        (34200.4, 7, 0, 0, 1, -1),           # resume
    ]
    day = _day(tmp_path, msgs=msgs, rows=[r0, r0, r0, r0])
    rep = reconcile_continuous(day)
    assert rep.diffs == 0 and rep.halts == 2
    assert reconcile_transition(day).diffs == 0

    book = SeededBook("LOBSTER", 0)
    book.apply(Message(seq=-1, ts_ns=0, msg_type="H", locate=0, trading_state="T"))
    book.seed_level(True, 1000000, 10)
    before = {p: list(v) for p, v in book.bids.levels.items()}
    _apply_lobster(book, day, 1, set())
    assert book.trading_state == "H"
    assert {p: list(v) for p, v in book.bids.levels.items()} == before
    _apply_lobster(book, day, 2, set())
    assert book.stats.cross_prints[-1][3] == 500
    assert {p: list(v) for p, v in book.bids.levels.items()} == before
    _apply_lobster(book, day, 3, set())
    assert book.trading_state == "T"


def _sample(dir_: Path | None, stem: str, levels: int):
    if dir_ is None:
        pytest.skip(f"{SAMPLES_ENV} is not set")
    m = dir_ / f"{stem}_message_{levels}.csv"
    o = dir_ / f"{stem}_orderbook_{levels}.csv"
    if not (m.exists() and o.exists()):
        pytest.skip(f"LOBSTER sample absent: {dir_}")
    return read_lobster(m, o)


@pytest.mark.local
def test_lobster_level10_sample() -> None:
    day = _sample(L10, L10_STEM, 10)
    cont = reconcile_continuous(day)
    assert cont.diffs == 0, cont.first_diffs
    assert cont.unseen_order_msgs == 8381
    assert cont.compared_units > 0
    assert cont.hidden_exec_book_changed == 0
    tr = reconcile_transition(day)
    assert tr.diffs == 0, tr.first_diffs
    assert tr.entering_side_rows == 104562


@pytest.mark.local
def test_lobster_level50_sample() -> None:
    day = _sample(L50, L50_STEM, 50)
    cont = reconcile_continuous(day)
    assert cont.diffs == 0, cont.first_diffs
    assert cont.compared_units > 1_000_000
    assert reconcile_transition(day).diffs == 0


def _add(seq: int, ref: int, side: str, shares: int, price: int) -> Message:
    return Message(seq, seq, "A", 1, order_ref=ref, side=side, shares=shares, price=price, symbol="TEST")


def test_reduce_level_never_touches_named_orders() -> None:
    p = ls.P100
    book = SeededBook("TEST", 1)
    book.trading_state = "T"
    book.seed_level(True, p, 10)
    book.apply(_add(0, 7, "B", 5, p))
    assert book.bids.levels[p] == [15, 2]
    key = book.snapshot(0, 0).book_key()
    assert book.reduce_level(True, p, 12) is False
    assert book.snapshot(0, 0).book_key() == key
    assert book.reduce_level(True, p, 10) is True
    s = book.snapshot(0, 0)
    assert (s.bid_px[0], s.bid_sz[0], s.bid_ct[0]) == (p, 5, 1)
    assert book.bids.anon == {}
    d = book.apply(Message(1, 1, "D", 1, order_ref=7))
    assert d is not None and d.bid_px[0] == ls.lvl(True, [])[0][0]
    assert book.valid is True and p not in book.bids.levels
