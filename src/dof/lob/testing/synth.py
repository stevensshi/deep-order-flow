"""Synthetic ITCH 5.0 byte builders and hand-checked scenarios for the order-book tests.

Every builder packs big-endian fields at the spec offsets and asserts the
payload length equals ``MSG_LEN`` for its type, so a builder that drifts from
the spec fails here instead of silently agreeing with a parser that drifted
the same way. Fields the scenarios do not need (tracking number and the like)
are zero.

``SCENARIO_FULL`` exercises the parser (an untracked locate, an unknown type,
the seq layout) and the book; ``lvl``/``row`` expand the hand-computed state
rows; ``SCENARIO_DEPTH`` moves levels in and out of the 10-level window.
"""

from __future__ import annotations

import gzip
import random
import struct
from pathlib import Path
from typing import Iterable, Sequence

from dof.lob.book import LEVELS, SENTINEL_ASK_PX, SENTINEL_BID_PX
from dof.lob.itch import MSG_LEN


def _check(payload: bytes) -> bytes:
    want = MSG_LEN[payload[0]]
    assert len(payload) == want, (chr(payload[0]), len(payload), want)
    return payload


def ts48(ns: int) -> bytes:
    """6-byte big-endian nanoseconds since midnight."""
    return ns.to_bytes(6, "big")


def sym8(s: str) -> bytes:
    """Stock field: left-justified, space padded to 8 bytes."""
    b = s.encode("ascii")
    assert len(b) <= 8, s
    return b.ljust(8, b" ")


def _head(t: str, locate: int, ts: int) -> bytes:
    # type(1) locate(2) tracking(2, zero) timestamp(6)
    return t.encode("ascii") + struct.pack(">HH", locate, 0) + ts48(ts)


def msg_S(ts: int, code: str) -> bytes:
    return _check(_head("S", 0, ts) + code.encode("ascii"))


def msg_R(locate: int, ts: int, symbol: str) -> bytes:
    # §1.2.1: Market Category "Q" @19, Round Lot 100 @21, ETP Flag "N" @33; every
    # other directory field zero.
    body = (sym8(symbol) + b"Q" + b"\x00" + struct.pack(">I", 100) + b"\x00" * 8
            + b"N" + struct.pack(">I", 0) + b"\x00")
    return _check(_head("R", locate, ts) + body)


def msg_H(locate: int, ts: int, symbol: str, state: str, reason: str = "    ") -> bytes:
    return _check(_head("H", locate, ts) + sym8(symbol) + state.encode("ascii") + b"\x00"
                  + reason.encode("ascii").ljust(4, b" "))


def msg_A(locate: int, ts: int, ref: int, side: str, shares: int, symbol: str, price: int) -> bytes:
    return _check(_head("A", locate, ts) + struct.pack(">QcI", ref, side.encode("ascii"), shares)
                  + sym8(symbol) + struct.pack(">I", price))


def msg_F(locate: int, ts: int, ref: int, side: str, shares: int, symbol: str, price: int,
          mpid: str = "TEST") -> bytes:
    return _check(_head("F", locate, ts) + struct.pack(">QcI", ref, side.encode("ascii"), shares)
                  + sym8(symbol) + struct.pack(">I", price) + mpid.encode("ascii").ljust(4, b" "))


def msg_E(locate: int, ts: int, ref: int, shares: int, match: int) -> bytes:
    return _check(_head("E", locate, ts) + struct.pack(">QIQ", ref, shares, match))


def msg_C(locate: int, ts: int, ref: int, shares: int, match: int, printable: str, price: int) -> bytes:
    return _check(_head("C", locate, ts)
                  + struct.pack(">QIQcI", ref, shares, match, printable.encode("ascii"), price))


def msg_X(locate: int, ts: int, ref: int, shares: int) -> bytes:
    return _check(_head("X", locate, ts) + struct.pack(">QI", ref, shares))


def msg_D(locate: int, ts: int, ref: int) -> bytes:
    return _check(_head("D", locate, ts) + struct.pack(">Q", ref))


def msg_U(locate: int, ts: int, old: int, new: int, shares: int, price: int) -> bytes:
    return _check(_head("U", locate, ts) + struct.pack(">QQII", old, new, shares, price))


def msg_P(locate: int, ts: int, shares: int, symbol: str, price: int, match: int,
          side: str = "B", ref: int = 0) -> bytes:
    return _check(_head("P", locate, ts) + struct.pack(">QcI", ref, side.encode("ascii"), shares)
                  + sym8(symbol) + struct.pack(">IQ", price, match))


def msg_Q(locate: int, ts: int, shares: int, symbol: str, price: int, match: int, cross_type: str) -> bytes:
    return _check(_head("Q", locate, ts) + struct.pack(">Q", shares) + sym8(symbol)
                  + struct.pack(">IQ", price, match) + cross_type.encode("ascii"))


def msg_B(locate: int, ts: int, match: int) -> bytes:
    return _check(_head("B", locate, ts) + struct.pack(">Q", match))


def msg_unknown(type_char: str, payload_len: int) -> bytes:
    """Type byte plus zero padding, ``payload_len`` bytes in total."""
    assert payload_len >= 1
    return type_char.encode("ascii") + b"\x00" * (payload_len - 1)


def frame(payload: bytes) -> bytes:
    return struct.pack(">H", len(payload)) + payload


def stream(payloads: Iterable[bytes]) -> bytes:
    return b"".join(frame(p) for p in payloads)


def write_stream(path: Path, data: bytes, gz: bool) -> Path:
    if gz:
        with gzip.open(path, "wb") as fh:
            fh.write(data)
    else:
        path.write_bytes(data)
    return path


def T(h: int, m: int, s: int = 0, ns: int = 0) -> int:
    """Nanoseconds since midnight."""
    return ((h * 60 + m) * 60 + s) * 1_000_000_000 + ns


def lvl(is_bid: bool, pairs: list[tuple[int, int, int]]) -> tuple[tuple, tuple, tuple]:
    """Expand [(px, sz, ct), ...] best first to 10-tuples (px, sz, ct) padded with the side's sentinel.

    Takes the side because the empty-level
    price sentinel is signed by side: bid -9999999999, ask +9999999999.
    """
    assert len(pairs) <= LEVELS
    pad = LEVELS - len(pairs)
    sentinel = SENTINEL_BID_PX if is_bid else SENTINEL_ASK_PX
    px = tuple(p for p, _, _ in pairs) + (sentinel,) * pad
    sz = tuple(s for _, s, _ in pairs) + (0,) * pad
    ct = tuple(c for _, _, c in pairs) + (0,) * pad
    return px, sz, ct


def row(seq: int, ts: int, bids: list[tuple[int, int, int]], asks: list[tuple[int, int, int]], flags: int) -> tuple:
    """One expected state row, field for field equal to a ``Snapshot10``."""
    return (seq, ts, *lvl(True, bids), *lvl(False, asks), flags)


# Prices in 1e-4 USD (spec p.4).
P100 = 1_000_000
P100_10 = 1_001_000
P99_90 = 999_000
P100_05 = 1_000_500
P99_80 = 998_000
P50 = 500_000

# Main scenario: locate 1 = TEST tracked, locate 2 = OTHR not.
# seq is the list index.
SCENARIO_FULL: list[bytes] = [
    msg_S(T(3, 0), "O"),                                          # 0
    msg_R(1, T(3, 5), "TEST"),                                    # 1
    msg_R(2, T(3, 5, 0, 1), "OTHR"),                              # 2
    msg_H(1, T(3, 6), "TEST", "T"),                               # 3
    msg_S(T(4, 0), "S"),                                          # 4
    msg_A(1, T(4, 0, 0, 1), 1, "B", 100, "TEST", P100),           # 5
    msg_A(1, T(4, 0, 0, 2), 2, "S", 200, "TEST", P100_10),        # 6
    msg_A(2, T(4, 0, 0, 3), 3, "B", 50, "OTHR", P50),             # 7
    msg_F(1, T(4, 0, 0, 4), 4, "B", 50, "TEST", P100),            # 8
    msg_unknown("Z", 5),                                          # 9
    msg_A(1, T(4, 0, 0, 5), 5, "B", 300, "TEST", P99_90),         # 10
    msg_S(T(9, 30), "Q"),                                         # 11
    msg_E(1, T(9, 30, 0, 1), 1, 40, 101),                         # 12
    msg_C(1, T(9, 30, 0, 1), 1, 60, 102, "N", P100_05),           # 13 (same ns as 12)
    msg_X(1, T(9, 30, 0, 2), 5, 100),                             # 14
    msg_U(1, T(9, 30, 0, 3), 4, 6, 80, P100_05),                  # 15
    msg_A(1, T(9, 30, 0, 4), 7, "S", 100, "TEST", P100_05),       # 16
    msg_A(1, T(9, 30, 0, 5), 8, "S", 10, "TEST", P100),           # 17
    msg_D(1, T(9, 30, 0, 6), 8),                                  # 18
    msg_D(1, T(9, 30, 0, 7), 7),                                  # 19
    msg_P(1, T(9, 30, 0, 8), 25, "TEST", 1_000_700, 103, side="B", ref=0),  # 20
    msg_Q(1, T(9, 30, 0, 9), 1000, "TEST", 1_000_600, 104, "O"),  # 21
    msg_B(1, T(9, 30, 0, 10), 103),                               # 22
    msg_H(1, T(10, 0), "TEST", "H"),                              # 23
    msg_H(1, T(10, 5), "TEST", "T"),                              # 24
    msg_E(1, T(10, 5, 0, 1), 99, 10, 105),                        # 25 (unknown ref)
    msg_A(1, T(10, 5, 0, 2), 9, "B", 10, "TEST", P99_80),         # 26
    msg_X(1, T(10, 5, 0, 3), 9, 20),                              # 27 (underflow: only 10)
    msg_S(T(16, 0), "M"),                                         # 28
    msg_S(T(20, 0), "E"),                                         # 29
    msg_S(T(20, 5), "C"),                                         # 30
]

_TS_FULL = [int.from_bytes(p[5:11], "big") for p in SCENARIO_FULL]
_B1 = [(P100_05, 80, 1), (P99_90, 200, 1)]      # bids from seq15 on
_A1 = [(P100_10, 200, 1)]

# Hand-computed state rows (19 rows): seq, ts, bids, asks, flags.
EXPECTED_FULL: list[tuple] = [
    row(s, _TS_FULL[s], b, a, f) for s, b, a, f in [
        (3, [], [], 0),
        (5, [(P100, 100, 1)], [], 0),
        (6, [(P100, 100, 1)], _A1, 0),
        (8, [(P100, 150, 2)], _A1, 0),
        (10, [(P100, 150, 2), (P99_90, 300, 1)], _A1, 0),
        (12, [(P100, 110, 2), (P99_90, 300, 1)], _A1, 0),
        (13, [(P100, 50, 1), (P99_90, 300, 1)], _A1, 0),
        (14, [(P100, 50, 1), (P99_90, 200, 1)], _A1, 0),
        (15, _B1, _A1, 0),
        (16, _B1, [(P100_05, 100, 1), (P100_10, 200, 1)], 1),                    # LOCKED
        (17, _B1, [(P100, 10, 1), (P100_05, 100, 1), (P100_10, 200, 1)], 2),     # CROSSED
        (18, _B1, [(P100_05, 100, 1), (P100_10, 200, 1)], 1),
        (19, _B1, _A1, 0),
        (20, _B1, _A1, 0),                                                       # P repeat row
        (21, _B1, _A1, 0),                                                       # Q repeat row
        (23, _B1, _A1, 4),                                                       # HALTED
        (24, _B1, _A1, 0),
        (26, _B1 + [(P99_80, 10, 1)], _A1, 8),                                   # INVALID
        (27, _B1, _A1, 8),
    ]
]

# Expected BookStats fields after SCENARIO_FULL and resting orders (ref2, ref5, ref6).
EXPECTED_FULL_STATS: dict[str, object] = {
    "unknown_ref": 1,
    "size_underflow": 1,
    "duplicate_ref": 0,
    "first_error_seq": 25,
    "locked_rows": 2,
    "crossed_rows": 1,
    "crossed_segment_count": 1,
    "crossed_segments": [(17, 18, T(9, 30, 0, 5), T(9, 30, 0, 6))],
    "halted_rows": 1,
    "trades_printable": 3,             # E seq12, P seq20, Q seq21 (C/N seq13 excluded)
    "trade_shares_printable": 1065,    # 40 + 25 + 1000
    "nonprintable_exec": 1,
    "broken": 1,
    "cross_prints": [(21, T(9, 30, 0, 9), "O", 1000, 1_000_600)],
}
EXPECTED_FULL_RESTING = 3
EXPECTED_FULL_PARSE: dict[str, object] = {
    "messages_total": 31,
    "unknown_type_counts": {"Z": 1},
    "skipped_untracked": 1,            # seq7
}

# Depth scenario: locate 1 = DEEP; 12 bids one tick (100) apart,
# so i = 10, 11 start outside the 10-level window and enter it as better
# levels are deleted.
_DEPTH_PX = [1_000_000 - 100 * i for i in range(12)]
SCENARIO_DEPTH: list[bytes] = (
    [msg_S(T(3, 0), "O"), msg_R(1, T(3, 5), "DEEP"), msg_H(1, T(3, 6), "DEEP", "T")]
    + [msg_A(1, T(4, 0, 0, 1 + i), 100 + i, "B", 100, "DEEP", _DEPTH_PX[i]) for i in range(12)]
    + [msg_D(1, T(9, 30, 0, 1), 100), msg_X(1, T(9, 30, 0, 2), 111, 50), msg_D(1, T(9, 30, 0, 3), 101)]
)
_TS_DEPTH = [int.from_bytes(p[5:11], "big") for p in SCENARIO_DEPTH]


def _bids(lo: int, hi: int, last_size: int = 100) -> list[tuple[int, int, int]]:
    out = [(_DEPTH_PX[i], 100, 1) for i in range(lo, hi + 1)]
    out[-1] = (out[-1][0], last_size, 1)
    return out


# 13 rows: H; adds i = 0..9 (seq 3..12); D ref100 (seq 15) with i = 10 entering as
# level 10; X ref111 (seq 16, rank 10) no row; D ref101 (seq 17) with i = 11 at 50 shares.
EXPECTED_DEPTH: list[tuple] = (
    [row(2, _TS_DEPTH[2], [], [], 0)]
    + [row(3 + i, _TS_DEPTH[3 + i], _bids(0, i), [], 0) for i in range(10)]
    + [row(15, _TS_DEPTH[15], _bids(1, 10), [], 0),
       row(17, _TS_DEPTH[17], _bids(2, 11, last_size=50), [], 0)]
)
assert len(EXPECTED_FULL) == 19 and len(EXPECTED_DEPTH) == 13
assert EXPECTED_DEPTH[-1][2][9] == 1_000_000 - 1_100 and EXPECTED_DEPTH[-1][3][9] == 50


# -- a seeded random trading day for examples (no golden depends on it) -------------------------------

TICK, REFILL_LEVELS = 100, 11


class _RandomBook:
    """One stock's resting orders as the generator sees them: refs in time order per price level."""

    def __init__(self) -> None:
        self.orders: dict[int, list] = {}                                  # ref -> [is_bid, price, shares]
        self.levels: dict[bool, dict[int, list[int]]] = {True: {}, False: {}}

    def add(self, ref: int, is_bid: bool, price: int, shares: int) -> None:
        self.orders[ref] = [is_bid, price, shares]
        self.levels[is_bid].setdefault(price, []).append(ref)

    def take(self, ref: int, n: int | None = None) -> None:
        """Take ``n`` shares off an order (all of them when None); an order at zero shares is gone."""
        o = self.orders[ref]
        o[2] -= o[2] if n is None else n
        if o[2] == 0:
            del self.orders[ref]
            refs = self.levels[o[0]][o[1]]
            refs.remove(ref)
            if not refs:
                del self.levels[o[0]][o[1]]

    def edge(self, is_bid: bool, best: bool) -> int | None:
        """The best (or the worst) price of a side; None when the side is empty."""
        lv = self.levels[is_bid]
        return (max(lv) if is_bid == best else min(lv)) if lv else None


def random_day(seed: int, symbols: Sequence[str] = ("AAA", "BBB", "CCC"), n_events: int = 60_000) -> bytes:
    """The framed bytes of a seeded random trading day of ``symbols`` (locates 1, 2, ...): books seeded with 12
    levels a side around 1,000,000 x locate; ``n_events`` events (45 % adds, 30 % deletes, 15 % executions at the
    best price, a third of them sweeping the whole level, 10 % partial cancels) spread evenly over 09:30-16:00,
    every side of the stock refilled to 11 price levels after each event; the end-of-messages event at 20:05, so a
    build of the day is complete."""
    rng = random.Random(seed)
    ri = lambda n: rng.getrandbits(32) % n      # noqa: E731
    ids = iter(range(1, 1 << 62))               # one counter for refs and match numbers
    loc = {s: i + 1 for i, s in enumerate(symbols)}
    base = {s: 1_000_000 * loc[s] for s in symbols}
    books = {s: _RandomBook() for s in symbols}
    out = [msg_S(T(3, 0), "O")] + [msg_R(loc[s], T(3, 5) + loc[s], s) for s in symbols]
    out += [msg_H(loc[s], T(9, 28) + loc[s], s, "T") for s in symbols]

    def add(s: str, is_bid: bool, price: int, shares: int, ts: int) -> None:
        ref = next(ids)
        books[s].add(ref, is_bid, price, shares)
        out.append(msg_A(loc[s], ts, ref, "B" if is_bid else "S", shares, s, price))

    def delete(s: str, ref: int, ts: int) -> None:
        books[s].take(ref)
        out.append(msg_D(loc[s], ts, ref))

    ts = T(9, 29)
    for s in symbols:
        for k in range(1, 13):
            for is_bid, sign in ((True, -1), (False, 1)):
                add(s, is_bid, base[s] + sign * TICK * k, 100 * (1 + ri(5)), ts)
                ts += 1
    out.append(msg_S(T(9, 30), "Q"))
    step = 23_400 * 1_000_000_000 // n_events
    for j in range(1, n_events + 1):
        t, s = T(9, 30) + j * step, symbols[ri(len(symbols))]
        book, r = books[s], ri(100)
        bb, ba = book.edge(True, True) or base[s] - TICK, book.edge(False, True) or base[s] + TICK
        if r < 45:
            is_bid, off = ri(2) == 0, ri(8)
            price = min(bb + TICK - off * TICK, ba - TICK) if is_bid else max(ba - TICK + off * TICK, bb + TICK)
            add(s, is_bid, max(price, TICK), 100 * (1 + ri(5)), t)
        elif r < 75:
            if book.orders:
                delete(s, list(book.orders)[ri(len(book.orders))], t)
        elif r < 90:
            # One execution in three sweeps the best level (every order in full, oldest first, one E each), so the
            # level empties and the mid moves; without it the best level almost never empties and the mid moves a
            # few times a day. The others fill part of the oldest order at the best price.
            is_bid, sweep = ri(2) == 0, ri(3) == 0
            price = book.edge(is_bid, True)
            if price is not None:
                queue = book.levels[is_bid][price]
                for ref in (list(queue) if sweep else queue[:1]):
                    n = book.orders[ref][2] if sweep else 1 + ri(book.orders[ref][2])
                    book.take(ref, n)
                    out.append(msg_E(loc[s], t, ref, n, next(ids)))
        elif book.orders:
            ref = list(book.orders)[ri(len(book.orders))]
            sh = book.orders[ref][2]
            if sh <= 1:
                delete(s, ref, t)
            else:
                n = 1 + ri(sh - 1)
                book.take(ref, n)
                out.append(msg_X(loc[s], t, ref, n))
        refill_ts = t + 1
        for is_bid, sign in ((True, -1), (False, 1)):
            while len(book.levels[is_bid]) < REFILL_LEVELS:
                w = book.edge(is_bid, False)
                add(s, is_bid, (base[s] if w is None else w) + sign * TICK, 100 * (1 + ri(5)), refill_ts)
                refill_ts += 1
    out += [msg_S(T(16, 0, 0, 500_000_000), "M"), msg_S(T(20, 0), "E"), msg_S(T(20, 5), "C")]
    return stream(out)


def write_random_day(directory: Path, date: str, seed: int) -> Path:
    """``random_day(seed)`` as a plain file named for ``date`` (ISO): ``MMDDYYYY.NASDAQ_ITCH50``."""
    y, m, d = date.split("-")
    path = Path(directory) / f"{m}{d}{y}.NASDAQ_ITCH50"
    path.write_bytes(random_day(seed))
    return path
