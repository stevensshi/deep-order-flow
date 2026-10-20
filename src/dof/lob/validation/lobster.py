"""LOBSTER sample reader and external reconciliation of ``Book``.

Sources: the samples' own ``LOBSTER_SampleFiles_ReadMe.txt`` (columns, empty
level sentinels -9999999999 / +9999999999, type 7 halt rows repeat the prior
state) and https://php.lobsterdata.com/info/DataStructure.php (type 6 is a cross
trade; the message file only holds events inside the requested price range).

Why there are two modes, reported separately:
- The message file has no submissions for orders resting before 09:30
  (10-level sample: 8,381 messages act on unseen orders) and no events beyond
  the requested levels, so a continuous replay cannot reproduce every file row.
  ``reconcile_continuous`` starts from orderbook row 0, NEVER fills the book
  from later file rows, and compares only units it can still vouch for (the
  "trusted region"). Its 50-level run is the main external check.
- ``reconcile_transition`` rebuilds the prior state from file row k-1, applies
  message k and compares with row k: a message->state transition test of one
  message's price-level semantics. It is not end-to-end equality and must never
  be reported as such; the 104,562 "levels entering from beyond the window"
  count only exists in this mode.

Comparison looks at price and size only; LOBSTER has no order counts or flags.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from dof.lob.book import Book
from dof.lob.itch import Message

LOBSTER_SENTINEL = 9999999999
FIRST_DIFFS_CAP = 20
MODE_LABELS = {
    "continuous": "continuous replay from orderbook row 0, no filling, comparable units only",
    "transition": "message->state transition test, prior state from file row k-1; not end-to-end equality",
}
_HALT_STATE = {-1: "H", 0: "Q", 1: "T"}  # ReadMe: type 7 price -1 halt, 0 quoting, 1 resume


class SeededBook(Book):
    """A ``Book`` that can also hold anonymous shares seeded from an outside file: the LOBSTER reconciliation
    only. The ITCH build uses the plain ``Book``, which has no such methods."""

    def seed_level(self, is_bid: bool, price: int, shares: int) -> None:
        """Seed anonymous shares at a level; all anonymous shares there count as one order."""
        side = self._side(is_bid)
        level = side._ensure(price)
        level[0] += shares
        if side.anon.get(price, 0) == 0:
            level[1] += 1
        side.anon[price] = side.anon.get(price, 0) + shares

    def reduce_level(self, is_bid: bool, price: int, shares: int) -> bool:
        """Take shares off the anonymous part only; False (book unchanged) if not enough.

        Reducing the level total instead would eat named orders and
        delete their level (10-level sample message 5,227; 50-level 41,949).
        """
        side = self._side(is_bid)
        have = side.anon.get(price, 0)
        if have == 0 or have < shares:  # have == 0: nothing anonymous here (also guards shares == 0)
            return False
        level = side.levels[price]
        level[0] -= shares
        have -= shares
        if have == 0:
            del side.anon[price]
            level[1] -= 1
        else:
            side.anon[price] = have
        if level[0] == 0 and level[1] == 0:
            del side.levels[price]
            del side.keys[side.rank(price)]
        return True


@dataclass
class LobsterDay:
    time: np.ndarray        # float64 seconds after midnight
    etype: np.ndarray       # int64 1..7
    oid: np.ndarray         # int64
    size: np.ndarray        # int64
    price: np.ndarray       # int64 (USD * 10000)
    direction: np.ndarray   # int64 (1 buy, -1 sell)
    book: np.ndarray        # int64 (N, 4*levels): AskP1, AskS1, BidP1, BidS1, ...
    levels: int


def read_lobster(message_csv: Path, orderbook_csv: Path) -> LobsterDay:
    msg = pd.read_csv(message_csv, header=None)
    ob = pd.read_csv(orderbook_csv, header=None).to_numpy(dtype=np.int64)
    if len(msg) != len(ob):
        raise ValueError(f"message rows {len(msg)} != orderbook rows {len(ob)}")
    if ob.shape[1] % 4:
        raise ValueError(f"orderbook has {ob.shape[1]} columns, not a multiple of 4")
    return LobsterDay(
        time=msg[0].to_numpy(dtype=np.float64),
        etype=msg[1].to_numpy(dtype=np.int64),
        oid=msg[2].to_numpy(dtype=np.int64),
        size=msg[3].to_numpy(dtype=np.int64),
        price=msg[4].to_numpy(dtype=np.int64),
        direction=msg[5].to_numpy(dtype=np.int64),
        book=ob,
        levels=ob.shape[1] // 4,
    )


@dataclass
class ReconcileReport:
    mode: str                     # "continuous" | "transition"
    levels_in_file: int
    compare_levels: int           # 10
    rows: int
    compared_units: int = 0       # (row, side, rank) units compared on price AND size
    diffs: int = 0
    first_diffs: list[dict] = field(default_factory=list)  # up to 20
    entering_units: int = 0       # file units at rank <= compare_levels that lie outside the trusted region
    entering_side_rows: int = 0   # (row, side) pairs with >= 1 entering unit
    rows_zero_comparable: dict[str, int] = field(default_factory=lambda: {"bid": 0, "ask": 0})
    unseen_order_msgs: int = 0    # type 2/3/4 whose order id was never submitted in the file
    unseen_by_type: dict[str, int] = field(default_factory=lambda: {"2": 0, "3": 0, "4": 0})
    hidden_exec_rows: int = 0     # type 5
    hidden_exec_file_changed: int = 0  # type-5 rows where file row k != row k-1 (expected 0)
    hidden_exec_book_changed: int = 0  # type-5 rows where our book changed (must be 0)
    order_attr_mismatch: int = 0  # known order: message price/direction differ from Book's stored order
    type3_size_mismatch: int = 0  # known order: type-3 size != Book's remaining shares
    halts: int = 0                # type 7 rows

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mode_label"] = MODE_LABELS[self.mode]
        return d


def _start_trading(book: Book) -> None:
    # LOBSTER has no initial trading state; without this every row would carry HALTED.
    book.apply(Message(seq=-1, ts_ns=0, msg_type="H", locate=0, trading_state="T"))


def _new_book() -> SeededBook:
    book = SeededBook("LOBSTER", 0)
    _start_trading(book)
    return book


def _apply_lobster(book: SeededBook, day: LobsterDay, k: int, known: set[int], *,
                   rep: ReconcileReport | None = None, anonymous: bool = False) -> str:
    """Apply LOBSTER message k to ``book``; return "known" / "unseen" / "other".

    ``anonymous=True`` is the transition mode: no order ids, type 1 seeds and
    types 2/3/4 reduce anonymous shares, types 5/6/7 leave the book alone.
    Mismatch and unseen counters go to ``rep`` when given.
    """
    et = int(day.etype[k])
    oid = int(day.oid[k])
    size = int(day.size[k])
    price = int(day.price[k])
    is_bid = int(day.direction[k]) == 1
    if anonymous:
        if et == 1:
            book.seed_level(is_bid, price, size)
        elif et in (2, 3, 4):
            book.reduce_level(is_bid, price, size)
        return "other"
    ts = round(float(day.time[k]) * 1e9)
    if et == 1:
        book.apply(Message(seq=k, ts_ns=ts, msg_type="A", locate=0, order_ref=oid,
                           side="B" if is_bid else "S", shares=size, price=price))
        known.add(oid)
        return "known"
    if et in (2, 3, 4):
        if oid in known:
            o = book.orders.get(oid)
            if rep is not None and o is not None:
                if o[0] != is_bid or o[1] != price:
                    rep.order_attr_mismatch += 1
                if et == 3 and o[2] != size:
                    rep.type3_size_mismatch += 1
            if et == 2:
                book.apply(Message(seq=k, ts_ns=ts, msg_type="X", locate=0, order_ref=oid, shares=size))
            elif et == 3:
                book.apply(Message(seq=k, ts_ns=ts, msg_type="D", locate=0, order_ref=oid))
            else:
                book.apply(Message(seq=k, ts_ns=ts, msg_type="E", locate=0, order_ref=oid, exec_shares=size))
            return "known"
        # An order resting before the file starts: only its level's anonymous
        # shares can absorb it. False leaves the book unchanged on purpose — the
        # comparison then shows the difference instead of hiding it.
        book.reduce_level(is_bid, price, size)
        if rep is not None:
            rep.unseen_order_msgs += 1
            rep.unseen_by_type[str(et)] += 1
        return "unseen"
    if et == 5:
        book.apply(Message(seq=k, ts_ns=ts, msg_type="P", locate=0, shares=size, price=price))
    elif et == 6:
        book.apply(Message(seq=k, ts_ns=ts, msg_type="Q", locate=0, shares=size, price=price, cross_type=""))
    elif et == 7:
        book.apply(Message(seq=k, ts_ns=ts, msg_type="H", locate=0, trading_state=_HALT_STATE[price]))
    return "other"


# -- row helpers ---------------------------------------------------------------

def _sides(row: list[int]) -> tuple[list[int], list[int], list[int], list[int]]:
    """(bid_px, bid_sz, ask_px, ask_sz), best first, sentinels included."""
    return row[2::4], row[3::4], row[0::4], row[1::4]


def _nonempty(px: list[int], is_bid: bool) -> int:
    empty = -LOBSTER_SENTINEL if is_bid else LOBSTER_SENTINEL
    for i, p in enumerate(px):
        if p == empty:
            return i
    return len(px)


def _window_edge(px: list[int], is_bid: bool) -> float:
    """Deepest file price when the side fills every file level, else the whole side is shown."""
    if _nonempty(px, is_bid) == len(px):
        return px[-1]
    return -math.inf if is_bid else math.inf


def _tighten(bound: float, edge: float, is_bid: bool) -> float:
    # A price that once left the file window gets no further messages, so it
    # can never be trusted again: the bound only moves towards the inside.
    return max(bound, edge) if is_bid else min(bound, edge)


def _compare_side(rep: ReconcileReport, day: LobsterDay, k: int, side_name: str, is_bid: bool,
                  file_px: list[int], file_sz: list[int], bound: float, book: Book) -> None:
    cl = rep.compare_levels
    n_file = _nonempty(file_px, is_bid)
    top = min(cl, n_file)
    n_in = 0
    while n_in < top and (file_px[n_in] >= bound if is_bid else file_px[n_in] <= bound):
        n_in += 1
    entering = top - n_in
    if entering:
        rep.entering_units += entering
        rep.entering_side_rows += 1

    side = book.bids if is_bid else book.asks
    keys = side.keys
    levels = side.levels
    book_trusted: list[int] = []
    for key in keys[:cl]:
        p = -key if is_bid else key
        if (p >= bound) if is_bid else (p <= bound):
            book_trusted.append(p)
        else:
            break  # trusted levels are a prefix of the priority order

    compared = n_in
    for i in range(n_in):
        fp, fs = file_px[i], file_sz[i]
        if i < len(book_trusted):
            bp = book_trusted[i]
            bs = levels[bp][0]
            if bp == fp and bs == fs:
                continue
            got = [bp, bs]
        else:
            got = None
        _diff(rep, day, k, side_name, i, [fp, fs], got)
    # The file shows the whole side (fewer than compare_levels non-empty levels,
    # all trusted): any further trusted book level inside the window is a level
    # the file says does not exist.
    if n_in == n_file and n_file < cl:
        for i in range(n_in, len(book_trusted)):
            bp = book_trusted[i]
            compared += 1
            _diff(rep, day, k, side_name, i, None, [bp, levels[bp][0]])
    rep.compared_units += compared
    if compared == 0:
        rep.rows_zero_comparable[side_name] += 1


def _diff(rep: ReconcileReport, day: LobsterDay, k: int, side: str, rank: int, file_unit, book_unit) -> None:
    rep.diffs += 1
    if len(rep.first_diffs) < FIRST_DIFFS_CAP:
        msg = [float(day.time[k]), int(day.etype[k]), int(day.oid[k]),
               int(day.size[k]), int(day.price[k]), int(day.direction[k])]
        rep.first_diffs.append({"row": k, "side": side, "rank": rank, "file": file_unit, "book": book_unit, "msg": msg})


def _seed_row(book: SeededBook, row: list[int], skip: tuple[bool, int, int] | None = None) -> None:
    """Seed every non-empty level of a file row as anonymous shares; ``skip`` subtracts one order."""
    bid_px, bid_sz, ask_px, ask_sz = _sides(row)
    for is_bid, px, sz in ((True, bid_px, bid_sz), (False, ask_px, ask_sz)):
        for i in range(_nonempty(px, is_bid)):
            q = sz[i]
            if skip is not None and skip[0] == is_bid and skip[1] == px[i]:
                q -= skip[2]
            if q > 0:
                book.seed_level(is_bid, px[i], q)


def _book_key(book: Book) -> tuple:
    return book.snapshot(0, 0).book_key()


def _count_hidden_and_halts(rep: ReconcileReport, et: int, row: list[int], prev: list[int], book: Book,
                            before: tuple | None) -> None:
    """Type 5 (hidden execution): the file row must repeat the previous one and the book must not change
    (``before`` is the book key before the message). Type 7 (trading halt): counted."""
    if et == 5:
        rep.hidden_exec_rows += 1
        if row != prev:
            rep.hidden_exec_file_changed += 1
        if _book_key(book) != before:
            rep.hidden_exec_book_changed += 1
    elif et == 7:
        rep.halts += 1


# -- public --------------------------------------------------------------------

def reconcile_continuous(day: LobsterDay, compare_levels: int = 10, *, book: SeededBook | None = None) -> ReconcileReport:
    """Replay every message from orderbook row 0; compare only the trusted region, never fill.

    ``book`` lets a test inspect the replayed book afterwards; it must be a fresh SeededBook.
    """
    n = len(day.etype)
    rep = ReconcileReport(mode="continuous", levels_in_file=day.levels, compare_levels=compare_levels,
                          rows=max(n - 1, 0))
    if n == 0:
        return rep
    if book is None:
        book = _new_book()
    else:
        _start_trading(book)
    known: set[int] = set()
    row0 = day.book[0].tolist()
    # Row 0 already contains message 0. If that is a submission, book it as
    # a real order (so later messages on it count as known) and
    # seed only the rest of its level anonymously.
    skip = None
    if int(day.etype[0]) == 1:
        skip = (int(day.direction[0]) == 1, int(day.price[0]), int(day.size[0]))
    _seed_row(book, row0, skip)
    if skip is not None:
        _apply_lobster(book, day, 0, known)
    bid_px, _, ask_px, _ = _sides(row0)
    bid_bound = _window_edge(bid_px, True)
    ask_bound = _window_edge(ask_px, False)
    prev = row0
    etype = day.etype
    for k in range(1, n):
        et = int(etype[k])
        before = _book_key(book) if et == 5 else None
        _apply_lobster(book, day, k, known, rep=rep)
        row = day.book[k].tolist()
        _count_hidden_and_halts(rep, et, row, prev, book, before)
        bid_px, bid_sz, ask_px, ask_sz = _sides(row)
        bid_bound = _tighten(bid_bound, _window_edge(bid_px, True), True)
        ask_bound = _tighten(ask_bound, _window_edge(ask_px, False), False)
        _compare_side(rep, day, k, "bid", True, bid_px, bid_sz, bid_bound, book)
        _compare_side(rep, day, k, "ask", False, ask_px, ask_sz, ask_bound, book)
        prev = row
    return rep


def reconcile_transition(day: LobsterDay, compare_levels: int = 10) -> ReconcileReport:
    """Message->state transition test: prior state from file row k-1, apply message k, compare with row k.

    Proves one message's price-level semantics only, not the continuous book.
    No order ids are tracked, so unseen/attribute counters stay 0.
    """
    n = len(day.etype)
    rep = ReconcileReport(mode="transition", levels_in_file=day.levels, compare_levels=compare_levels,
                          rows=max(n - 1, 0))
    prev = day.book[0].tolist() if n else []
    for k in range(1, n):
        row = day.book[k].tolist()
        book = _new_book()
        _seed_row(book, prev)
        et = int(day.etype[k])
        before = _book_key(book) if et == 5 else None
        _apply_lobster(book, day, k, set(), anonymous=True)
        _count_hidden_and_halts(rep, et, row, prev, book, before)
        pb_px, _, pa_px, _ = _sides(prev)
        bid_px, bid_sz, ask_px, ask_sz = _sides(row)
        bid_bound = _tighten(_window_edge(pb_px, True), _window_edge(bid_px, True), True)
        ask_bound = _tighten(_window_edge(pa_px, False), _window_edge(ask_px, False), False)
        _compare_side(rep, day, k, "bid", True, bid_px, bid_sz, bid_bound, book)
        _compare_side(rep, day, k, "ask", False, ask_px, ask_sz, ask_bound, book)
        prev = row
    return rep
