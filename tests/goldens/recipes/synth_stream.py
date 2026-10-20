"""Seeded synthetic ITCH 5.0 trading days for the golden tests (standard library only).

The byte builders below pack big-endian fields at the spec offsets and assert that every payload has the
spec length of its type. ``day_stream(date)`` builds one whole day from a seed; its bytes are pinned by the
``input_sha256`` keys of the goldens and by ``recipe_sha256``, so any change here fails loudly as an input
change. ``coverage_problems`` checks that a built day exercises what the goldens are meant to pin.

Rules of the generator, kept so that the bytes do not depend on the interpreter: all randomness is
``ri(n) = rng.getrandbits(32) % n`` with ``rng = random.Random(DAYS[date])``; no set is ever iterated (dicts
and lists keep insertion order); refs and match numbers come from one counter that starts at 1.
"""

from __future__ import annotations

import gzip
import json
import random
import struct
from pathlib import Path
from typing import Iterable

# Payload length (without the 2-byte prefix) of every type of the ITCH 5.0 spec, section 1.
MSG_LEN: dict[int, int] = {
    ord("S"): 12, ord("R"): 39, ord("H"): 25, ord("Y"): 20, ord("L"): 26, ord("V"): 35,
    ord("W"): 12, ord("K"): 28, ord("J"): 35, ord("h"): 21, ord("A"): 36, ord("F"): 40,
    ord("E"): 31, ord("C"): 36, ord("X"): 23, ord("D"): 19, ord("U"): 35, ord("P"): 44,
    ord("Q"): 40, ord("B"): 19, ord("I"): 50, ord("N"): 20, ord("O"): 48,
}


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


# -- the seeded day ------------------------------------------------------------------------------

TICK = 100
SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "ZZZ")       # locates 1..5; the goldens track the first four
LOCATE = {s: i + 1 for i, s in enumerate(SYMBOLS)}
TRACKED = SYMBOLS[:4]
START_MID = {"AAA": 1_000_000, "BBB": 500_000, "CCC": 2_000_000, "DDD": 300_000, "ZZZ": 700_000}
DAYS = {"2019-01-30": 101, "2019-01-31": 102, "2019-02-01": 103, "2019-11-29": 104}
CLOSE = {"2019-11-29": T(13, 0)}                    # a half day; every other day closes at 16:00
REGULAR_CLOSE = T(16, 0)
STOP = T(16, 0)                                     # the random events end here, on the half day too
SEED_LEVELS = 12
KEEP_LEVELS = 11                                    # refill target: distinct prices per side
CROSS_SHARES = 10_000
SCHEDULED_SHARES = 100                              # size of the crossing and locking asks of the schedule
UNKNOWN_REF = 999_999_999                           # never reached by the counter
MINUTE_NS = 60 * 1_000_000_000


class _Book:
    """The generator's own view of one symbol's resting orders: refs in time order per price level."""

    def __init__(self) -> None:
        self.orders: dict[int, list] = {}                                  # ref -> [is_bid, price, shares]
        self.levels: dict[bool, dict[int, list[int]]] = {True: {}, False: {}}
        self.protected: set[int] = set()                                   # membership tests only, never iterated

    def add(self, ref: int, is_bid: bool, price: int, shares: int) -> None:
        self.orders[ref] = [is_bid, price, shares]
        self.levels[is_bid].setdefault(price, []).append(ref)

    def remove(self, ref: int) -> None:
        is_bid, price, _ = self.orders.pop(ref)
        refs = self.levels[is_bid][price]
        refs.remove(ref)
        if not refs:
            del self.levels[is_bid][price]
        self.protected.discard(ref)

    def reduce(self, ref: int, n: int) -> None:
        o = self.orders[ref]
        o[2] -= n
        if o[2] == 0:
            self.remove(ref)

    def best(self, is_bid: bool) -> int | None:
        lv = self.levels[is_bid]
        if not lv:
            return None
        return max(lv) if is_bid else min(lv)

    def worst(self, is_bid: bool) -> int | None:
        lv = self.levels[is_bid]
        if not lv:
            return None
        return min(lv) if is_bid else max(lv)

    def unprotected(self) -> list[int]:
        if not self.protected:
            return list(self.orders)
        return [r for r in self.orders if r not in self.protected]

    def oldest_unprotected_at(self, is_bid: bool, price: int | None) -> int | None:
        if price is None:
            return None
        for r in self.levels[is_bid].get(price, []):
            if r not in self.protected:
                return r
        return None


class _Day:
    """One trading day: open, seeded books, scheduled actions inside a random event loop, close."""

    def __init__(self, date: str) -> None:
        self.rng = random.Random(DAYS[date])
        self.half = date in CLOSE
        self.close = CLOSE.get(date, REGULAR_CLOSE)
        self.books = {s: _Book() for s in SYMBOLS}
        self.out: list[bytes] = []
        self.last_ts = 0
        self.counter = 0
        self.matches: list[int] = []
        self.refill_from: dict[tuple[str, bool], int] = {}     # (symbol, is_bid) -> no refill before this ns
        self.held: dict[str, int] = {}                          # name of a scheduled order -> its ref
        self.pending: list[tuple[int, int, object]] = []

    # -- primitives ---------------------------------------------------------------------------

    def ri(self, n: int) -> int:
        return self.rng.getrandbits(32) % n

    def next_id(self) -> int:
        self.counter += 1
        return self.counter

    def emit(self, payload: bytes, ts: int | None) -> None:
        self.out.append(payload)
        if ts is not None and ts > self.last_ts:
            self.last_ts = ts

    def bb(self, s: str) -> int:
        b = self.books[s].best(True)
        return START_MID[s] - TICK if b is None else b

    def ba(self, s: str) -> int:
        a = self.books[s].best(False)
        return START_MID[s] + TICK if a is None else a

    def add(self, s: str, is_bid: bool, price: int, shares: int, ts: int, *, kind: str = "A",
            protected: bool = False) -> int:
        ref = self.next_id()
        side = "B" if is_bid else "S"
        if kind == "A":
            payload = msg_A(LOCATE[s], ts, ref, side, shares, s, price)
        else:
            payload = msg_F(LOCATE[s], ts, ref, side, shares, s, price, mpid="SYNT")
        self.books[s].add(ref, is_bid, price, shares)
        if protected:
            self.books[s].protected.add(ref)
        self.emit(payload, ts)
        return ref

    def delete(self, s: str, ref: int, ts: int) -> None:
        self.books[s].remove(ref)
        self.emit(msg_D(LOCATE[s], ts, ref), ts)

    def cross(self, s: str, ts: int, cross_type: str) -> None:
        price = (self.bb(s) + self.ba(s)) // 2 // TICK * TICK          # the mid, rounded down to the tick
        match = self.next_id()
        self.matches.append(match)
        self.emit(msg_Q(LOCATE[s], ts, CROSS_SHARES, s, price, match, cross_type), ts)

    # -- scheduled actions --------------------------------------------------------------------

    def act_ccc_thin(self, ts: int) -> None:
        """CCC asks cut to 9 distinct prices, deleting at the worst price; CCC asks are not refilled for 2 s."""
        book = self.books["CCC"]
        while len(book.levels[False]) > 9:
            for ref in list(book.levels[False][book.worst(False)]):
                self.delete("CCC", ref, ts)
                ts += 1
        self.refill_from[("CCC", False)] = T(10, 30, 2)

    def act_aaa_cross(self, ts: int) -> None:
        self.held["aaa_cross"] = self.add("AAA", False, self.bb("AAA") - TICK, SCHEDULED_SHARES, ts, protected=True)

    def act_aaa_lock(self, ts: int) -> None:
        self.held["aaa_lock"] = self.add("AAA", False, self.bb("AAA"), SCHEDULED_SHARES, ts, protected=True)

    def act_drop(self, s: str, name: str, ts: int) -> None:
        ref = self.held.pop(name, None)
        if ref is not None and ref in self.books[s].orders:
            self.delete(s, ref, ts)

    def act_ccc_flash_cross(self, ts: int) -> None:
        """A crossing CCC ask added and deleted in the same nanosecond."""
        ref = self.add("CCC", False, self.bb("CCC") - TICK, SCHEDULED_SHARES, ts, protected=True)
        self.delete("CCC", ref, ts)

    def act_ddd_unknown_ref(self, ts: int) -> None:
        self.emit(msg_E(LOCATE["DDD"], ts, UNKNOWN_REF, SCHEDULED_SHARES, self.next_id()), ts)

    def schedule(self) -> list[tuple[int, int, object]]:
        c = self.close
        acts = [
            (T(10, 30), self.act_ccc_thin),
            (T(11, 0), self.act_aaa_cross),
            (T(11, 0) + 50_000_000, lambda ts: self.act_drop("AAA", "aaa_cross", ts)),
            (T(11, 30), self.act_aaa_lock),
            (T(11, 30) + 20_000_000, lambda ts: self.act_drop("AAA", "aaa_lock", ts)),
            (T(12, 0), lambda ts: self.emit(msg_H(LOCATE["BBB"], ts, "BBB", "H"), ts)),
            (T(12, 5), lambda ts: self.emit(msg_H(LOCATE["BBB"], ts, "BBB", "T"), ts)),
            (T(12, 30) if self.half else T(13, 0), self.act_ccc_flash_cross),
            (T(14, 0), self.act_ddd_unknown_ref),
            *[(c + 1_000 + LOCATE[s], (lambda ts, s=s: self.cross(s, ts, "C"))) for s in TRACKED],
            (c + 500_000_000, lambda ts: self.emit(msg_S(ts, "M"), ts)),
        ]
        return sorted(((t, i, fn) for i, (t, fn) in enumerate(acts)), key=lambda a: (a[0], a[1]))

    def run_due(self, upto: int | None) -> None:
        """Every scheduled action due at or before ``upto`` (None: all of them), each at its own time, never
        before a message already emitted."""
        while self.pending and (upto is None or self.pending[0][0] <= upto):
            t, _, fn = self.pending.pop(0)
            fn(max(t, self.last_ts))

    # -- random events ------------------------------------------------------------------------

    def ev_add(self, s: str, t: int, kind: str) -> None:
        is_bid = self.ri(2) == 0
        off = self.ri(8)
        bb, ba = self.bb(s), self.ba(s)
        if is_bid:
            price = min(bb + TICK - off * TICK, ba - TICK)
        else:
            price = max(ba - TICK + off * TICK, bb + TICK)
        price = max(price, TICK)
        self.add(s, is_bid, price, 100 * (1 + self.ri(5)), t, kind=kind)

    def ev_delete(self, s: str, t: int) -> None:
        refs = self.books[s].unprotected()
        if refs:
            self.delete(s, refs[self.ri(len(refs))], t)

    def ev_cancel(self, s: str, t: int) -> None:
        book = self.books[s]
        refs = book.unprotected()
        if not refs:
            return
        ref = refs[self.ri(len(refs))]
        sh = book.orders[ref][2]
        if sh <= 1:
            self.delete(s, ref, t)
            return
        n = 1 + self.ri(sh - 1)
        book.reduce(ref, n)
        self.emit(msg_X(LOCATE[s], t, ref, n), t)

    def ev_exec(self, s: str, t: int, kind: str) -> None:
        book = self.books[s]
        is_bid = self.ri(2) == 0
        price = book.best(is_bid)
        ref = book.oldest_unprotected_at(is_bid, price)
        if ref is None:
            return                                   # nothing to execute: emit nothing, draw nothing more
        n = 1 + self.ri(book.orders[ref][2])
        match = self.next_id()
        self.matches.append(match)
        if kind == "E":
            payload = msg_E(LOCATE[s], t, ref, n, match)
        else:
            payload = msg_C(LOCATE[s], t, ref, n, match, "Y" if self.ri(2) == 0 else "N", price)
        book.reduce(ref, n)
        self.emit(payload, t)

    def ev_replace(self, s: str, t: int) -> None:
        book = self.books[s]
        refs = book.unprotected()
        if not refs:
            return
        ref = refs[self.ri(len(refs))]
        is_bid, old_price, _ = book.orders[ref]
        new_price = old_price + TICK * (self.ri(3) - 1)
        new_price = min(new_price, self.ba(s)) if is_bid else max(new_price, self.bb(s))   # a lock is allowed
        shares = 100 * (1 + self.ri(5))
        new_ref = self.next_id()
        book.remove(ref)
        book.add(new_ref, is_bid, new_price, shares)
        self.emit(msg_U(LOCATE[s], t, ref, new_ref, shares, new_price), t)

    def ev_trade(self, s: str, t: int) -> None:
        match = self.next_id()
        self.matches.append(match)
        self.emit(msg_P(LOCATE[s], t, 100, s, self.bb(s), match, side="B", ref=0), t)

    def ev_broken(self, s: str, t: int) -> None:
        if self.matches:
            self.emit(msg_B(LOCATE[s], t, self.matches[self.ri(len(self.matches))]), t)

    def event(self, t: int) -> None:
        s = SYMBOLS[self.ri(5)]
        r = self.ri(1000)
        if r < 380:
            self.ev_add(s, t, "A")
        elif r < 420:
            self.ev_add(s, t, "F")
        elif r < 600:
            self.ev_delete(s, t)
        elif r < 680:
            self.ev_cancel(s, t)
        elif r < 760:
            self.ev_exec(s, t, "E")
        elif r < 780:
            self.ev_exec(s, t, "C")
        elif r < 850:
            self.ev_replace(s, t)
        elif r < 880:
            self.ev_trade(s, t)
        elif r < 882:
            self.ev_broken(s, t)
        elif r < 884:
            self.emit(msg_unknown("Z", 5), None)
        else:
            self.ev_add(s, t, "A")

    def refill(self, t: int) -> None:
        """Every side of every symbol, unless suspended, back to KEEP_LEVELS distinct prices: adds at the worst
        price one tick further out, at the event's ns + 1, + 2, ..."""
        ts = t + 1
        for s in SYMBOLS:
            book = self.books[s]
            for is_bid in (True, False):
                if t < self.refill_from.get((s, is_bid), 0):
                    continue
                while len(book.levels[is_bid]) < KEEP_LEVELS:
                    w = book.worst(is_bid)
                    if w is None:
                        price = START_MID[s] - TICK if is_bid else START_MID[s] + TICK
                    else:
                        price = w - TICK if is_bid else w + TICK
                    self.add(s, is_bid, price, 100 * (1 + self.ri(5)), ts)
                    ts += 1

    # -- the day ------------------------------------------------------------------------------

    def run(self) -> list[bytes]:
        self.emit(msg_S(T(3, 0), "O"), T(3, 0))
        for s in SYMBOLS:
            self.emit(msg_R(LOCATE[s], T(3, 5) + LOCATE[s], s), T(3, 5) + LOCATE[s])
        for s in SYMBOLS:
            self.emit(msg_H(LOCATE[s], T(9, 28) + LOCATE[s], s, "T"), T(9, 28) + LOCATE[s])
        ts = T(9, 29)
        for s in SYMBOLS:
            for k in range(1, SEED_LEVELS + 1):
                self.add(s, True, START_MID[s] - k * TICK, 100 * (1 + self.ri(5)), ts)
                ts += 1
                self.add(s, False, START_MID[s] + k * TICK, 100 * (1 + self.ri(5)), ts)
                ts += 1
        self.emit(msg_S(T(9, 30), "Q"), T(9, 30))
        for s in TRACKED:
            self.cross(s, T(9, 30) + 1_000 + LOCATE[s], "O")
        self.pending = self.schedule()
        t = T(9, 30, 0, 10_000)
        while True:
            gap = 0 if self.ri(8) == 0 else 1 + self.ri(780_000_000)
            cand = t + gap
            self.run_due(cand)
            t = max(cand, self.last_ts)
            if t >= STOP:
                break
            self.event(t)
            self.refill(t)
        self.run_due(None)
        end = T(17, 0) if self.half else T(20, 0)
        self.emit(msg_S(end, "E"), end)
        self.emit(msg_S(end + 5 * MINUTE_NS, "C"), end + 5 * MINUTE_NS)
        return self.out


def day_stream(date: str) -> bytes:
    """The framed bytes of one seeded day (a key of DAYS)."""
    return stream(_Day(date).run())


def day_file_name(date: str) -> str:
    y, m, d = date.split("-")
    return f"{m}{d}{y}.NASDAQ_ITCH50"


def write_day(directory: Path, date: str) -> Path:
    """Write the day as a plain file (no gzip: a gzip header carries a time stamp)."""
    path = Path(directory) / day_file_name(date)
    path.write_bytes(day_stream(date))
    return path


# -- coverage of a built full day ------------------------------------------------------------------


def coverage_problems(log: dict, values: dict) -> list[str]:
    """What a built full day must exercise; an empty list means it does. ``log`` is the day's build log,
    ``values`` the flat chain values that hold the day's ``<sym>/<date>/...`` keys."""
    p: list[str] = []
    date = log.get("date")
    if log.get("status") != "complete":
        p.append(f"{date}: build status {log.get('status')!r}")
    syms = log.get("symbols", {})
    for s in ("AAA", "BBB", "CCC"):
        if syms.get(s, {}).get("status") != "valid":
            p.append(f"{date} {s}: status {syms.get(s, {}).get('status')!r}, not valid")
    ddd = syms.get("DDD", {})
    if ddd.get("status") != "invalid" or not str(ddd.get("first_error") or "").startswith("unknown_ref E"):
        p.append(f"{date} DDD: status {ddd.get('status')!r}, first_error {ddd.get('first_error')!r}")
    aaa = syms.get("AAA", {})
    if aaa.get("crossed_segment_count", 0) < 1 or aaa.get("locked_rows", 0) < 1:
        p.append(f"{date} AAA: crossed_segment_count {aaa.get('crossed_segment_count')}, locked_rows {aaa.get('locked_rows')}")
    bbb = syms.get("BBB", {})
    if bbb.get("halted_rows", 0) < 1 or not any(c[2] == "H" for c in bbb.get("trading_state_changes", [])):
        p.append(f"{date} BBB: halted_rows {bbb.get('halted_rows')}, no H state change")
    parse = log.get("parse", {})
    unknown = parse.get("unknown_type_counts", {})
    if list(unknown) != ["Z"] or unknown.get("Z", 0) < 1:
        p.append(f"{date}: unknown_type_counts {unknown}")
    if parse.get("skipped_untracked", 0) < 1:
        p.append(f"{date}: skipped_untracked {parse.get('skipped_untracked')}")
    counts = parse.get("counts_by_type", {})
    missing = [t for t in "AFECXDUPQBHSR" if counts.get(t, 0) < 1]
    if missing:
        p.append(f"{date}: message types never framed: {missing}")
    for s in ("AAA", "BBB", "CCC"):
        n = values.get(f"{s}/{date}/n_samples")
        if not isinstance(n, int) or n < 2000:
            p.append(f"{date} {s}: n_samples {n}")
        kv = values.get(f"{s}/{date}/k_valid")
        if kv is None or json.loads(kv) != list(range(1, 11)):
            p.append(f"{date} {s}: k_valid {kv}")
        funnel = json.loads(values.get(f"{s}/{date}/flow/cks/d1/funnel", "{}"))
        if funnel.get("dedupe_removed", 0) < 1:
            p.append(f"{date} {s}: dedupe_removed {funnel.get('dedupe_removed')}")
        if s == "CCC" and funnel.get("invalid_intervals", 0) < 2:
            p.append(f"{date} CCC: invalid_intervals {funnel.get('invalid_intervals')}")
        if s == "BBB" and funnel.get("qh_removed", 0) < 2:
            p.append(f"{date} BBB: qh_removed {funnel.get('qh_removed')}")
    return p
