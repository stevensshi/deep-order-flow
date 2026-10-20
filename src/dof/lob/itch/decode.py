"""Decoding of framed ITCH messages for a set of tracked stocks."""

from __future__ import annotations

import os
from typing import Iterable, Iterator, NamedTuple

from dof.lob.itch.framing import ParseStats, _frames, _open
from dof.lob.itch.spec import (
    _ADD,
    _C,
    _CHR,
    _E,
    _H,
    _P,
    _Q,
    _R,
    _REF,
    _T_A,
    _T_B,
    _T_C,
    _T_D,
    _T_E,
    _T_F,
    _T_H,
    _T_P,
    _T_Q,
    _T_R,
    _T_S,
    _T_U,
    _T_X,
    _TS,
    _U,
    _X,
    DEFAULT_CHUNK,
)


class Message(NamedTuple):
    seq: int            # 0-based ordinal of the message in the file, counting EVERY framed message
    ts_ns: int          # u48 nanoseconds since midnight (Eastern), spec p.4
    msg_type: str       # one character
    locate: int
    order_ref: int = 0      # A/F/E/C/X/D/P ref; U original ref
    new_order_ref: int = 0  # U new ref
    side: str = ""          # A/F/P raw "B"/"S" (P is always "B", spec §1.5.1 — never interpreted)
    shares: int = 0         # A/F shares; X cancelled shares; U new shares; P shares; Q cross shares (u64)
    price: int = 0          # A/F/U display price; P match price; Q cross price (1e-4 USD)
    exec_price: int = 0     # C execution price
    exec_shares: int = 0    # E/C executed shares
    printable: str = ""     # C only: "Y"/"N"
    match_number: int = 0   # E/C/P/Q
    cross_type: str = ""    # Q only
    trading_state: str = "" # H only
    broken_match: int = 0   # B only
    symbol: str = ""        # R/H/A/F/P/Q (stripped of right padding)
    event_code: str = ""    # S only
    reason: str = ""        # H only (stripped)


class LocateFilter:
    """Track a symbol set by learning each symbol's locate from the day's R messages.

    Locates are assigned per day (spec p.3), so the filter starts empty and is
    filled as the directory spin streams past. A symbol seen again in a later R
    keeps its first entry and only bumps ``directory_duplicates``.
    """

    def __init__(self, symbols: Iterable[str]) -> None:
        self.symbols: frozenset[str] = frozenset(s.strip() for s in symbols)
        self.locates: set[int] = set()
        self.locate_to_symbol: dict[int, str] = {}
        self.directory: dict[str, dict[str, object]] = {}
        self.directory_size = 0
        self.directory_duplicates = 0

    def observe_directory(self, msg: Message, raw_fields: dict[str, object]) -> bool:
        """Record one R message; return True if its symbol is tracked."""
        self.directory_size += 1
        symbol = msg.symbol
        if symbol not in self.symbols:
            return False
        if symbol in self.directory:
            self.directory_duplicates += 1
            return True
        entry: dict[str, object] = {"locate": msg.locate, "seq": msg.seq, "ts_ns": msg.ts_ns}
        entry.update(raw_fields)
        self.directory[symbol] = entry
        self.locates.add(msg.locate)
        self.locate_to_symbol[msg.locate] = symbol
        return True


def iter_messages(
    path: str | os.PathLike[str],
    locates: LocateFilter | set[int] | None,
    *,
    stats: ParseStats | None = None,
    limit_bytes: int | None = None,
    chunk_size: int = DEFAULT_CHUNK,
) -> Iterator[Message]:
    """Yield decoded messages for tracked locates (``None`` = every locate).

    S is always yielded. R is yielded when its symbol (LocateFilter) or locate
    (set) is tracked; every other type only when its locate is tracked —
    untracked messages are counted in ``stats.skipped_untracked`` without
    decoding the body. ``first_ts_ns``/``last_ts_ns`` cover yielded messages
    (S, which opens and closes every file, is always among them).

    Kept as one function on purpose: it is the hot loop of a build (hundreds of millions of messages a day).
    """
    if stats is None:
        stats = ParseStats()
    filt = locates if isinstance(locates, LocateFilter) else None
    track_all = locates is None
    tracked: set[int] = filt.locates if filt is not None else (locates if locates is not None else set())  # type: ignore[assignment]
    is_tracked = tracked.__contains__
    yielded = [0] * 256
    chr_ = _CHR
    ts_unpack = _TS.unpack_from
    first_ts: int | None = None
    ts = 0
    try:
        with _open(path) as fh:
            for seq, buf, s, _length in _frames(fh, limit_bytes, chunk_size, stats):
                t = buf[s]
                locate = (buf[s + 1] << 8) | buf[s + 2]
                if t == _T_S:
                    hi, lo = ts_unpack(buf, s + 5)
                    ts = (hi << 32) | lo
                    code = chr_[buf[s + 11]]
                    stats.system_events.append((seq, ts, code))
                    msg = Message(seq, ts, "S", locate, event_code=code)
                elif t == _T_R:
                    hi, lo = ts_unpack(buf, s + 5)
                    ts = (hi << 32) | lo
                    (sym, mc, fs, lot, rlo, icl, ist, auth, sst, ipo, luld, etp, lev, inv) = _R.unpack_from(buf, s + 11)
                    msg = Message(seq, ts, "R", locate, symbol=sym.decode("ascii").rstrip())
                    if filt is not None:
                        raw = {
                            "market_category": mc.decode("ascii"),
                            "financial_status": fs.decode("ascii"),
                            "round_lot": lot,
                            "round_lots_only": rlo.decode("ascii"),
                            "issue_classification": icl.decode("ascii"),
                            "issue_subtype": ist.decode("ascii"),
                            "authenticity": auth.decode("ascii"),
                            "short_sale_threshold": sst.decode("ascii"),
                            "ipo_flag": ipo.decode("ascii"),
                            "luld_reference": luld.decode("ascii"),
                            "etp_flag": etp.decode("ascii"),
                            "etp_leverage": lev,
                            "inverse_indicator": inv.decode("ascii"),
                        }
                        if not filt.observe_directory(msg, raw):
                            continue
                    elif not track_all and not is_tracked(locate):
                        continue
                else:
                    if not track_all and not is_tracked(locate):
                        stats.skipped_untracked += 1
                        continue
                    hi, lo = ts_unpack(buf, s + 5)
                    ts = (hi << 32) | lo
                    msg = _decode_body(t, seq, ts, locate, buf, s)
                yielded[t] += 1
                if first_ts is None:
                    first_ts = ts
                    if stats.first_ts_ns is None:
                        stats.first_ts_ns = ts
                stats.last_ts_ns = ts
                yield msg
    finally:
        for i, c in enumerate(yielded):
            if c:
                stats.tracked_by_type[chr_[i]] += c


def _decode_body(t: int, seq: int, ts: int, locate: int, buf: bytes, s: int) -> Message:
    o = s + 11
    if t == _T_A or t == _T_F:
        ref, side, shares, sym, price = _ADD.unpack_from(buf, o)
        return Message(seq, ts, _CHR[t], locate, order_ref=ref, side=side.decode("ascii"),
                       shares=shares, price=price, symbol=sym.decode("ascii").rstrip())
    if t == _T_D:
        return Message(seq, ts, "D", locate, order_ref=_REF.unpack_from(buf, o)[0])
    if t == _T_X:
        ref, shares = _X.unpack_from(buf, o)
        return Message(seq, ts, "X", locate, order_ref=ref, shares=shares)
    if t == _T_E:
        ref, shares, match = _E.unpack_from(buf, o)
        return Message(seq, ts, "E", locate, order_ref=ref, exec_shares=shares, match_number=match)
    if t == _T_U:
        old, new, shares, price = _U.unpack_from(buf, o)
        return Message(seq, ts, "U", locate, order_ref=old, new_order_ref=new, shares=shares, price=price)
    if t == _T_C:
        ref, shares, match, printable, price = _C.unpack_from(buf, o)
        return Message(seq, ts, "C", locate, order_ref=ref, exec_shares=shares, match_number=match,
                       printable=printable.decode("ascii"), exec_price=price)
    if t == _T_P:
        ref, side, shares, sym, price, match = _P.unpack_from(buf, o)
        return Message(seq, ts, "P", locate, order_ref=ref, side=side.decode("ascii"), shares=shares,
                       price=price, match_number=match, symbol=sym.decode("ascii").rstrip())
    if t == _T_Q:
        shares, sym, price, match, cross = _Q.unpack_from(buf, o)
        return Message(seq, ts, "Q", locate, shares=shares, symbol=sym.decode("ascii").rstrip(),
                       price=price, match_number=match, cross_type=cross.decode("ascii"))
    if t == _T_B:
        return Message(seq, ts, "B", locate, broken_match=_REF.unpack_from(buf, o)[0])
    if t == _T_H:
        sym, state, _reserved, reason = _H.unpack_from(buf, o)
        return Message(seq, ts, "H", locate, symbol=sym.decode("ascii").rstrip(),
                       trading_state=state.decode("ascii"), reason=reason.decode("ascii").rstrip())
    # Y, L, V, W, K, J, h, I, N, O: no field of Message applies (the book never
    # reads them), so a tracked one is yielded with its header only.
    return Message(seq, ts, _CHR[t], locate)


def scan_counts(
    path: str | os.PathLike[str], *, limit_bytes: int | None = None, chunk_size: int = DEFAULT_CHUNK
) -> ParseStats:
    """Frame the whole stream and count types without decoding bodies (perf staging)."""
    stats = ParseStats()
    ts_unpack = _TS.unpack_from
    ts = None
    with _open(path) as fh:
        for _seq, buf, s, _length in _frames(fh, limit_bytes, chunk_size, stats):
            hi, lo = ts_unpack(buf, s + 5)
            ts = (hi << 32) | lo
            if stats.first_ts_ns is None:
                stats.first_ts_ns = ts
    stats.last_ts_ns = ts
    return stats
