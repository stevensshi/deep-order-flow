"""Cleaning of the raw state rows: bad rows, raw invalid intervals, the raw price series, the flow branch's row
selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from dof.lob.book import FLAG_CROSSED, FLAG_HALTED, LEVELS, SENTINEL_ASK_PX, SENTINEL_BID_PX
from dof.lob.features.session import Session
from dof.lob.units import PRICE_SCALE

BID_PX = [f"bid_px_{k}" for k in range(1, LEVELS + 1)]
BID_SZ = [f"bid_sz_{k}" for k in range(1, LEVELS + 1)]
ASK_PX = [f"ask_px_{k}" for k in range(1, LEVELS + 1)]
ASK_SZ = [f"ask_sz_{k}" for k in range(1, LEVELS + 1)]


def _missing_levels(frame: pd.DataFrame) -> np.ndarray:
    """Any side, any level at its sentinel; column by column (no (n, 10) temporaries)."""
    missing = np.zeros(len(frame), dtype=bool)
    for cols, sentinel in ((BID_PX, SENTINEL_BID_PX), (ASK_PX, SENTINEL_ASK_PX)):
        for c in cols:
            missing |= frame[c].to_numpy() == sentinel
    return missing


@dataclass(frozen=True)
class RowMasks:
    """Per raw row: crossed, halted, any level at its sentinel. Computed once per call and passed down."""
    crossed: np.ndarray
    halted: np.ndarray
    missing: np.ndarray


def row_masks(frame: pd.DataFrame) -> RowMasks:
    flags = frame["flags"].to_numpy()
    return RowMasks(crossed=(flags & FLAG_CROSSED) != 0, halted=(flags & FLAG_HALTED) != 0,
                    missing=_missing_levels(frame))


REASONS = ("crossed", "halted", "missing_levels")    # code of a bad row = 1 + index here; 0 = a good row


def reason_code(name: str) -> int:
    return 1 + REASONS.index(name)


def _bad_reasons(masks: RowMasks, order: Sequence[str]) -> np.ndarray:
    """Per-row int8 code of the first bad reason in ``order`` (``reason_code``); 0 if the row is good."""
    by_name = {"crossed": masks.crossed, "halted": masks.halted, "missing_levels": masks.missing}
    out = np.zeros(len(masks.crossed), dtype=np.int8)
    for reason in reversed(order):  # earlier reasons overwrite later ones
        out[by_name[reason]] = reason_code(reason)
    return out


_FLOW_BAD_ORDER = ("crossed", "halted", "missing_levels")   # the order of the flow branch's cleaning steps
_MINUTE_BAD_ORDER = ("halted", "crossed", "missing_levels")   # minute branch: halted takes precedence over crossed
OPEN_END_NS = np.iinfo(np.int64).max                         # INT64_MAX: an interval never restored that day


@dataclass(frozen=True)
class InvalidIntervals:
    """Maximal runs of consecutive bad raw rows (crossed / halted / any sentinel level), in seq order.

    Derived from the RAW ordered state table before any session cut, dedupe or
    Q/H drop: a halt carried only by H
    rows, or a crossed state restored within one nanosecond group, still forms
    an interval, and its boundaries are kept as seqs.
    """
    start_seq: np.ndarray   # int64, seq of the first bad raw row of the run
    end_seq: np.ndarray     # int64, seq of the last bad raw row of the run
    close_seq: np.ndarray   # int64, seq of the first good raw row after the run; INT64_MAX if the day ends bad
    start_ts: np.ndarray    # int64 ts of the first bad row
    close_ts: np.ndarray    # int64 ts of the closing good row; INT64_MAX if none (unclosed run)
    reason: np.ndarray      # str: first of "halted" > "crossed" > "missing_levels" over the run's rows

    def overlaps(self, lo_excl: np.ndarray, hi_incl: np.ndarray, reason: str | None = None) -> np.ndarray:
        """Per window (lo, hi]: any interval (of ``reason``) with start_ts <= hi and close_ts > lo."""
        sel = np.ones(len(self.start_ts), dtype=bool) if reason is None else (self.reason == reason)
        st, cl = self.start_ts[sel], self.close_ts[sel]
        hi_incl = np.asarray(hi_incl, dtype=np.int64)
        lo_excl = np.asarray(lo_excl, dtype=np.int64)
        if len(st) == 0:
            return np.zeros(len(hi_incl), dtype=bool)
        order = np.argsort(st, kind="stable")
        st, cl = st[order], np.maximum.accumulate(cl[order])
        k = np.searchsorted(st, hi_incl, "right")          # intervals starting at or before hi
        return (k > 0) & (cl[np.maximum(k - 1, 0)] > lo_excl)


def invalid_intervals(frame: pd.DataFrame, masks: RowMasks | None = None) -> InvalidIntervals:
    """Runs of bad raw rows, found without a Python loop over rows: run starts and ends from the edges of the
    padded bad mask; whether a run holds a reason, from cumulative counts; its reason, the first present in
    ``_MINUTE_BAD_ORDER``. ``reason`` is built from a Python list of str, so its dtype width is that of its longest
    entry ("<U1" when there is no run)."""
    if masks is None:
        masks = row_masks(frame)
    every = {"halted": masks.halted, "crossed": masks.crossed, "missing_levels": masks.missing}
    bad = masks.crossed | masks.halted | masks.missing
    ts = frame["ts_ns"].to_numpy().astype(np.int64)
    sq = frame["seq"].to_numpy().astype(np.int64)
    n = len(frame)
    edges = np.diff(np.concatenate(([False], bad, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)             # first bad row of each run
    ends = np.flatnonzero(edges == -1)              # one past the last bad row of each run
    present = {}
    for r in _MINUTE_BAD_ORDER:
        c = np.concatenate(([0], np.cumsum(every[r], dtype=np.int64)))
        present[r] = (c[ends] - c[starts]) > 0
    reasons = [next(r for r in _MINUTE_BAD_ORDER if present[r][i]) for i in range(len(starts))]
    closed = ends < n
    after = np.minimum(ends, max(n - 1, 0))        # the first good row after a run (when the run is closed)
    return InvalidIntervals(start_seq=sq[starts], end_seq=sq[ends - 1],
                            close_seq=np.where(closed, sq[after], OPEN_END_NS).astype(np.int64),
                            start_ts=ts[starts], close_ts=np.where(closed, ts[after], OPEN_END_NS).astype(np.int64),
                            reason=np.array(reasons, dtype=str))


def breaks_between(iv: InvalidIntervals, seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
    """Per pair: any interval with seq_a < start_seq <= seq_b."""
    st = np.sort(iv.start_seq)
    a = np.asarray(seq_a, dtype=np.int64)
    b = np.asarray(seq_b, dtype=np.int64)
    return (np.searchsorted(st, b, "right") - np.searchsorted(st, a, "right")) > 0


def in_interval(iv: InvalidIntervals, seq: np.ndarray) -> np.ndarray:
    """Per seq: does the raw row lie inside a run of bad rows (start_seq <= seq <= end_seq)?"""
    seq = np.asarray(seq, dtype=np.int64)
    if len(iv.start_seq) == 0:
        return np.zeros(len(seq), dtype=bool)
    k = np.searchsorted(iv.start_seq, seq, "right") - 1
    return (k >= 0) & (seq <= iv.end_seq[np.maximum(k, 0)])


@dataclass(frozen=True)
class PriceSeries:
    """Complete same-day mid series: every raw state row in seq order."""
    ts_ns: np.ndarray
    seq: np.ndarray
    mid: np.ndarray   # float64 dollars; NaN when level 1 is missing on either side or the row is crossed/halted


def price_series(frame: pd.DataFrame) -> PriceSeries:
    fl = frame["flags"].to_numpy()
    bid1 = frame["bid_px_1"].to_numpy()
    ask1 = frame["ask_px_1"].to_numpy()
    bad = (bid1 == SENTINEL_BID_PX) | (ask1 == SENTINEL_ASK_PX) | ((fl & (FLAG_CROSSED | FLAG_HALTED)) != 0)
    mid = mid_px(bid1, ask1)
    mid[bad] = np.nan
    return PriceSeries(frame["ts_ns"].to_numpy().astype(np.int64), frame["seq"].to_numpy().astype(np.int64), mid)


def asof_index(ps: PriceSeries, tau_ns: np.ndarray) -> np.ndarray:
    """Index of the last raw row with ts <= tau (the final state at that instant); -1 if none."""
    return np.searchsorted(ps.ts_ns, np.asarray(tau_ns, dtype=np.int64), "right") - 1


def _cols_at(frame: pd.DataFrame, cols: list[str], pos: np.ndarray) -> np.ndarray:
    """(len(pos), len(cols)) array of ``frame[cols]`` at raw positions, without copying whole columns."""
    out = np.empty((len(pos), len(cols)), dtype=np.int64)
    for j, c in enumerate(cols):
        out[:, j] = frame[c].to_numpy()[pos]
    return out


def mid_px(bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    """Mid price in dollars from level-1 prices in 1e-4 USD: the one mid formula of the feature layer."""
    return (bid.astype(np.float64) + ask.astype(np.float64)) / 2.0 / PRICE_SCALE


def mid_at(frame: pd.DataFrame, pos: np.ndarray) -> np.ndarray:
    return mid_px(frame["bid_px_1"].to_numpy()[pos], frame["ask_px_1"].to_numpy()[pos])


def flow_clean(frame: pd.DataFrame, session: Session, dedupe: bool):
    """Cleaning steps 0-3 of Kolm, Turiel and Westray (2023) on the rows of the trimmed session (``session.start_ns``
    to ``session.end_ns``); returns (kept raw row positions, their seqs, segment ids, intervals, funnel)."""
    return _select_rows(frame, session.start_ns, session.end_ns, dedupe)


def _select_rows(frame: pd.DataFrame, start_ns: int, end_ns: int, dedupe: bool):
    """Cleaning steps 0-3 of Kolm, Turiel and Westray (2023) on the rows with ``start_ns <= ts < end_ns``.

    Invalid intervals and segment ids come from the RAW table first;
    the fixed order (dedupe -> drop Q/H -> drop bad rows) then only selects rows,
    and every kept row carries the raw segment id, so no row pair, history or
    label can span an interval even when its rows were removed by dedupe or Q/H.
    """
    # Memory: the steps select raw row POSITIONS only; the cleaned frame is
    # built once at the end with just the columns the flow branch reads. The old chained
    # .iloc copies of the full 65-column frame held up to three copies at once, and with
    # dedupe=False they stay near full size (the 17.8 GB no-dedupe pass).
    funnel: dict[str, int] = {"rows_in": int(len(frame))}
    masks = row_masks(frame)
    iv = invalid_intervals(frame, masks)             # step 0a: from the RAW table
    funnel["invalid_intervals"] = int(len(iv.start_ts))
    ts = frame["ts_ns"].to_numpy()
    in_session = (ts >= start_ns) & (ts < end_ns)
    funnel["rows_outside_session"] = int((~in_session).sum())
    pos = np.flatnonzero(in_session)                 # raw row positions still kept
    # 1. dedupe: keep the LAST row of each run of equal timestamps.
    if dedupe and len(pos):
        t = ts[pos]
        keep = np.ones(len(pos), dtype=bool)
        keep[:-1] = t[:-1] != t[1:]
        funnel["dedupe_removed"] = int((~keep).sum())
        pos = pos[keep]
    else:
        funnel["dedupe_removed"] = 0
    # 2. Q/H rows: removed (a halt they carry is already an interval from the raw table).
    qh = frame["msg_type"].isin(["Q", "H"]).to_numpy()[pos]
    funnel["qh_removed"] = int(qh.sum())
    t_qh = np.unique(ts[pos[qh]])
    pos = pos[~qh]
    # timestamp groups that vanished entirely because their (kept) last row was Q/H
    funnel["group_lost_to_qh"] = int((~np.isin(t_qh, ts[pos])).sum())
    # 3. bad rows: removed; segment ids are the raw ones.
    reasons = _bad_reasons(masks, _FLOW_BAD_ORDER)[pos]
    bad = reasons != 0
    for r in _FLOW_BAD_ORDER:
        funnel[f"{r}_removed"] = int((reasons == reason_code(r)).sum())
    pos = pos[~bad]
    seq = frame["seq"].to_numpy()[pos].astype(np.int64)
    # segments: a new one wherever an interval starts between two consecutive kept rows
    seg = np.zeros(len(pos), dtype=np.int32)
    if len(pos) > 1:
        seg[1:] = np.cumsum(breaks_between(iv, seq[:-1], seq[1:])).astype(np.int32)
    return pos, seq, seg, iv, funnel
