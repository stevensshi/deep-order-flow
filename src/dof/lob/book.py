"""Order-by-order limit order book for one stock-day, emitting 10-level snapshots.

Semantics follow the ITCH 5.0 spec (``itch.SPEC_URL``):
- A/F add, E/C/X reduce (cumulative, the order dies at zero shares, spec §1.4),
  D deletes, U atomically deletes the old ref and adds the new one on the
  same side (spec §1.4.5: side is inherited from the original order).
- P and Q never change the book (spec §1.5.1/§1.5.2) but always emit a row
  repeating the current state, as does H (trading state change).
- Trade statistics count E, C printable "Y", P and Q. A C marked
  non-printable reduces the book but is not a trade print: cross executions
  arrive as C(N) and their volume enters once, via the Q bulk print
  (spec §1.4.2 "included into a later bulk print").
- An integrity error (unknown ref, size underflow, duplicate ref) marks the
  stock-day invalid; processing continues for diagnosis but every later row
  carries ``FLAG_INVALID`` and the feature layer rejects the whole day.

Row rule ("LOBSTER-style"): a book message emits a row only if a touched price
level ranks < 10 on its side (before the operation for removals, after it for
additions) and the resulting snapshot differs from the last emitted one.

``BookStats.crossed_segments`` entries are ``(start_seq, end_seq, start_ts,
end_ts)``; while a crossed segment is still open, ``end_seq`` and ``end_ts``
hold the placeholder ``-1``.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple

from dof.lob.itch import Message

LEVELS = 10
SENTINEL_BID_PX = -9999999999
SENTINEL_ASK_PX = 9999999999
FLAG_LOCKED = 1    # bid1 == ask1, both sides non-empty
FLAG_CROSSED = 2   # bid1 >  ask1, both sides non-empty
FLAG_HALTED = 4    # current trading_state != "T"
FLAG_INVALID = 8   # stock-day already hit an integrity error (unknown ref / underflow / duplicate ref) at or before this row

CROSSED_SEGMENTS_CAP = 1000
_OPEN = -1  # end_seq / end_ts of a crossed segment that has not ended yet


class Snapshot10(NamedTuple):
    seq: int
    ts_ns: int
    bid_px: tuple[int, ...]   # length 10, best first, SENTINEL_BID_PX when empty
    bid_sz: tuple[int, ...]   # 0 when empty
    bid_ct: tuple[int, ...]   # number of orders at the level, 0 when empty
    ask_px: tuple[int, ...]   # SENTINEL_ASK_PX when empty
    ask_sz: tuple[int, ...]
    ask_ct: tuple[int, ...]
    flags: int

    def book_key(self) -> tuple:
        """Everything but seq/ts — used for change detection."""
        return (self.bid_px, self.bid_sz, self.bid_ct, self.ask_px, self.ask_sz, self.ask_ct, self.flags)


@dataclass
class BookStats:
    unknown_ref: int = 0
    size_underflow: int = 0
    duplicate_ref: int = 0
    first_error_seq: int | None = None
    first_error: str | None = None        # e.g. "unknown_ref E ref=99"
    rows_emitted: int = 0
    locked_rows: int = 0
    crossed_rows: int = 0
    halted_rows: int = 0
    crossed_segments: list[tuple[int, int, int, int]] = field(default_factory=list)  # (start_seq, end_seq, start_ts, end_ts); end filled when crossed ends; cap 1000 entries
    crossed_segment_count: int = 0
    trading_state_changes: list[tuple[int, int, str]] = field(default_factory=list)  # (seq, ts_ns, new_state)
    cross_prints: list[tuple[int, int, str, int, int]] = field(default_factory=list) # (seq, ts_ns, cross_type, shares, price)
    trades_printable: int = 0      # E + C(printable Y) + P + Q messages applied (C with printable N excluded; spec §1.4.2/§1.5.2)
    trade_shares_printable: int = 0  # cross volume enters once, via Q
    nonprintable_exec: int = 0     # C with printable N
    broken: int = 0
    zero_share_add: int = 0        # A/F with 0 shares, or U with 0 new shares: new order not booked
    msgs_applied: Counter = field(default_factory=Counter)


class _Side:
    __slots__ = ("is_bid", "keys", "levels", "anon")

    def __init__(self, is_bid: bool) -> None:
        self.is_bid = is_bid
        self.keys: list[int] = []                 # ascending; -price for bid, +price for ask, so keys[0] is best
        self.levels: dict[int, list[int]] = {}    # price -> [shares_total, order_count] (total includes anon)
        self.anon: dict[int, int] = {}            # price -> anonymous shares (reconciliation only; empty in export)

    def _key(self, price: int) -> int:
        return -price if self.is_bid else price

    def _ensure(self, price: int) -> list[int]:
        level = self.levels.get(price)
        if level is None:
            level = self.levels[price] = [0, 0]
            insort(self.keys, self._key(price))
        return level

    def add(self, price: int, shares: int) -> int:
        """Add one order's shares; return the level's 0-based rank after insertion."""
        level = self._ensure(price)
        level[0] += shares
        level[1] += 1
        return self.rank(price)

    def remove(self, price: int, shares: int, orders: int) -> int:
        """Subtract shares and orders; return the level's rank before removal."""
        r = self.rank(price)
        level = self.levels[price]
        level[0] -= shares
        level[1] -= orders
        # Invariant: shares > 0 iff count > 0. Anything else can only follow an
        # underflow, already counted where it happened, so just drop the level.
        if level[0] <= 0 or level[1] <= 0:
            del self.levels[price]
            del self.keys[r]
            self.anon.pop(price, None)
        return r

    def rank(self, price: int) -> int:
        return bisect_left(self.keys, self._key(price))

    def top(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """10 prices, sizes, counts, best first, padded with this side's sentinel."""
        keys = self.keys[:LEVELS]
        prices = [-k for k in keys] if self.is_bid else list(keys)
        levels = self.levels
        sizes = [levels[p][0] for p in prices]
        counts = [levels[p][1] for p in prices]
        pad = LEVELS - len(prices)
        if pad:
            prices += [SENTINEL_BID_PX if self.is_bid else SENTINEL_ASK_PX] * pad
            sizes += [0] * pad
            counts += [0] * pad
        return tuple(prices), tuple(sizes), tuple(counts)

    def best(self) -> int | None:
        if not self.keys:
            return None
        k = self.keys[0]
        return -k if self.is_bid else k


class Book:
    def __init__(self, symbol: str, locate: int) -> None:
        self.symbol, self.locate = symbol, locate
        self.orders: dict[int, list] = {}   # ref -> [is_bid: bool, price: int, shares: int]
        self.bids, self.asks = _Side(True), _Side(False)
        self.trading_state = "H"            # spec §1.2.2: absent from the spin => treated as halted
        self.valid = True
        self.stats = BookStats()
        self._last_key: tuple | None = None
        self._crossed = False

    # -- helpers ---------------------------------------------------------------

    def _side(self, is_bid: bool) -> _Side:
        return self.bids if is_bid else self.asks

    def _error(self, kind: str, msg: str, seq: int) -> None:
        st = self.stats
        setattr(st, kind, getattr(st, kind) + 1)
        if st.first_error_seq is None:
            st.first_error_seq = seq
            st.first_error = f"{kind} {msg}"
        self.valid = False

    def _flags(self) -> int:
        flags = 0
        b, a = self.bids.best(), self.asks.best()
        if b is not None and a is not None:
            if b == a:
                flags |= FLAG_LOCKED
            elif b > a:
                flags |= FLAG_CROSSED
        if self.trading_state != "T":
            flags |= FLAG_HALTED
        if not self.valid:
            flags |= FLAG_INVALID
        return flags

    def snapshot(self, seq: int, ts_ns: int) -> Snapshot10:
        bid_px, bid_sz, bid_ct = self.bids.top()
        ask_px, ask_sz, ask_ct = self.asks.top()
        return Snapshot10(seq, ts_ns, bid_px, bid_sz, bid_ct, ask_px, ask_sz, ask_ct, self._flags())

    def _emit(self, seq: int, ts_ns: int, force: bool) -> Snapshot10 | None:
        snap = self.snapshot(seq, ts_ns)
        key = snap.book_key()
        if not force and key == self._last_key:
            return None
        st = self.stats
        self._last_key = key
        st.rows_emitted += 1
        flags = snap.flags
        if flags & FLAG_LOCKED:
            st.locked_rows += 1
        if flags & FLAG_HALTED:
            st.halted_rows += 1
        crossed = bool(flags & FLAG_CROSSED)
        if crossed:
            st.crossed_rows += 1
            if not self._crossed:
                st.crossed_segment_count += 1
                if len(st.crossed_segments) < CROSSED_SEGMENTS_CAP:
                    st.crossed_segments.append((seq, _OPEN, ts_ns, _OPEN))
        elif self._crossed:
            # Close the open segment, if it was recorded under the cap.
            if st.crossed_segments and st.crossed_segments[-1][1] == _OPEN:
                start_seq, _, start_ts, _ = st.crossed_segments[-1]
                st.crossed_segments[-1] = (start_seq, seq, start_ts, ts_ns)
        self._crossed = crossed
        return snap

    def _reduce(self, msg: Message, shares: int) -> int | None:
        """E/C/X: take ``shares`` off an order; return the level's rank, or None if not applied."""
        o = self.orders.get(msg.order_ref)
        if o is None:
            self._error("unknown_ref", f"{msg.msg_type} ref={msg.order_ref}", msg.seq)
            return None
        is_bid, price, left = o
        if shares > left:
            self._error("size_underflow", f"{msg.msg_type} ref={msg.order_ref} shares={shares} left={left}", msg.seq)
            shares = left
        o[2] = left - shares
        if o[2] == 0:
            del self.orders[msg.order_ref]
            return self._side(is_bid).remove(price, shares, 1)
        return self._side(is_bid).remove(price, shares, 0)

    # -- public ------------------------------------------------------------------

    def apply(self, msg: Message) -> Snapshot10 | None:
        """Apply one message; return the state row it emits, or None. Kept as one function on purpose: it runs
        once per message of a stock (millions per day), and a call per branch would cost more than it gives."""
        t = msg.msg_type
        st = self.stats
        seq, ts = msg.seq, msg.ts_ns

        if t == "A" or t == "F":
            ref = msg.order_ref
            if ref in self.orders:
                self._error("duplicate_ref", f"{t} ref={ref}", seq)
                return None
            if msg.shares == 0:
                st.zero_share_add += 1
                return None
            is_bid = msg.side == "B"
            self.orders[ref] = [is_bid, msg.price, msg.shares]
            r = self._side(is_bid).add(msg.price, msg.shares)
            st.msgs_applied[t] += 1
            return self._emit(seq, ts, False) if r < LEVELS else None

        if t == "E" or t == "C" or t == "X":
            shares = msg.shares if t == "X" else msg.exec_shares
            r = self._reduce(msg, shares)
            if r is None:
                return None
            st.msgs_applied[t] += 1
            if t == "E" or (t == "C" and msg.printable == "Y"):
                st.trades_printable += 1
                st.trade_shares_printable += msg.exec_shares
            elif t == "C":
                st.nonprintable_exec += 1
            return self._emit(seq, ts, False) if r < LEVELS else None

        if t == "D":
            o = self.orders.pop(msg.order_ref, None)
            if o is None:
                self._error("unknown_ref", f"D ref={msg.order_ref}", seq)
                return None
            is_bid, price, left = o
            r = self._side(is_bid).remove(price, left, 1)
            st.msgs_applied[t] += 1
            return self._emit(seq, ts, False) if r < LEVELS else None

        if t == "U":
            o = self.orders.pop(msg.order_ref, None)
            if o is None:
                # The new order's side is unknown without the original: do not book it.
                self._error("unknown_ref", f"U ref={msg.order_ref}", seq)
                return None
            is_bid, price, left = o
            side = self._side(is_bid)
            r_old = side.remove(price, left, 1)
            r_new = LEVELS
            new = msg.new_order_ref
            if new in self.orders:
                self._error("duplicate_ref", f"U new_ref={new}", seq)
            elif msg.shares == 0:
                st.zero_share_add += 1
            else:
                self.orders[new] = [is_bid, msg.price, msg.shares]
                r_new = side.add(msg.price, msg.shares)
            st.msgs_applied[t] += 1
            return self._emit(seq, ts, False) if min(r_old, r_new) < LEVELS else None

        if t == "P":
            st.msgs_applied[t] += 1
            st.trades_printable += 1
            st.trade_shares_printable += msg.shares
            return self._emit(seq, ts, True)

        if t == "Q":
            st.msgs_applied[t] += 1
            st.cross_prints.append((seq, ts, msg.cross_type, msg.shares, msg.price))
            st.trades_printable += 1
            st.trade_shares_printable += msg.shares
            return self._emit(seq, ts, True)

        if t == "B":
            st.msgs_applied[t] += 1
            st.broken += 1
            return None

        if t == "H":
            st.msgs_applied[t] += 1
            self.trading_state = msg.trading_state
            st.trading_state_changes.append((seq, ts, msg.trading_state))
            return self._emit(seq, ts, True)

        return None  # S, R, Y, L, I, N, K, J, h, V, W, O: no book effect

    def resting_orders(self) -> int:
        return len(self.orders)
