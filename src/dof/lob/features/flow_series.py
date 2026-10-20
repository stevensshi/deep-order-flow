"""The flow branch, after Kolm, Turiel and Westray (2023): cleaned rows with 20-column order flow, mid-price changes,
history windows, Δt."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from dof.lob.book import LEVELS
from dof.lob.features.cleaning import (
    ASK_PX,
    ASK_SZ,
    BID_PX,
    BID_SZ,
    InvalidIntervals,
    PriceSeries,
    _cols_at,
    _select_rows,
    flow_clean,
    mid_at,
    price_series,
)
from dof.lob.features.order_flow import SignConvention, _ofi_parts
from dof.lob.features.session import Session

DELTA_T_NUMERATOR_MS = 2.34e7   # Kolm, Turiel and Westray (2023), eq. (16): 6.5 trading hours in ms
FLOW_COLUMNS = ["ts_ns", "seq", "flags", "msg_type", *BID_PX, *BID_SZ, *ASK_PX, *ASK_SZ]   # what the flow branch reads
OF_CHUNK_ROWS = 250_000


@dataclass(frozen=True)
class FlowConfig:
    history: int = 100                      # W
    convention: SignConvention = "current_size"
    latency_ns: int = 10_000_000           # δτ = 10 ms
    dedupe: bool = True                     # keep the last row of each run of equal timestamps


@dataclass
class FlowFrame:
    ts_ns: np.ndarray          # int64, cleaned rows
    mid: np.ndarray            # float64 dollars ((bid1+ask1)/2/1e4)
    of: np.ndarray             # float64 (n, 20): bOF_1..10, aOF_1..10; NaN on the first row of each segment
    ofi: np.ndarray            # float64 (n, 10)
    seq: np.ndarray            # int64 source seq of each cleaned row
    segment: np.ndarray        # int32 segment id
    of_valid: np.ndarray       # bool
    intervals: InvalidIntervals  # from the RAW table (step 0a), never from the cleaned rows
    prices: PriceSeries        # complete same-day raw price series used by horizon_labels
    funnel: dict[str, int]
    config: FlowConfig
    session: Session           # the rows lie in [session.start_ns, session.end_ns); labels end before end_ns


def flow_pipeline(frame: pd.DataFrame, session: Session, config: FlowConfig = FlowConfig()) -> FlowFrame:
    pos, seq, seg, iv, funnel = flow_clean(frame, session, config.dedupe)
    n = len(pos)
    of = np.full((n, 2 * LEVELS), np.nan)
    ofi = np.full((n, LEVELS), np.nan)
    of_valid = np.zeros(n, dtype=bool)
    if n > 1:
        of_valid[1:] = seg[1:] == seg[:-1]
    # Order flow of row i against row i-1, in row chunks gathered straight from the raw
    # columns (memory: no full cleaned copy of the 40-column book).
    for a in range(1, n, OF_CHUNK_ROWS):
        b = min(n, a + OF_CHUNK_ROWS)
        p = pos[a - 1:b]
        bpx, bsz = _cols_at(frame, BID_PX, p), _cols_at(frame, BID_SZ, p)
        apx, asz = _cols_at(frame, ASK_PX, p), _cols_at(frame, ASK_SZ, p)
        bof, aof = _ofi_parts(bpx[:-1], bsz[:-1], apx[:-1], asz[:-1], bpx[1:], bsz[1:], apx[1:], asz[1:],
                              config.convention)
        del bpx, bsz, apx, asz
        v = of_valid[a:b]
        of[a:b][v, :LEVELS] = bof[v]
        of[a:b][v, LEVELS:] = aof[v]
        ofi[a:b][v] = bof[v] - aof[v]
        del bof, aof
    funnel["segment_first_rows"] = int(n - of_valid.sum())
    funnel["rows_out"] = int(n)
    funnel["of_valid"] = int(of_valid.sum())
    return FlowFrame(ts_ns=frame["ts_ns"].to_numpy()[pos].astype(np.int64), mid=mid_at(frame, pos), of=of, ofi=ofi, seq=seq,
                    segment=seg.astype(np.int32), of_valid=of_valid, intervals=iv, prices=price_series(frame),
                    funnel=funnel, config=config, session=session)


def count_mid_changes(frame: pd.DataFrame, session: Session, *, dedupe: bool = True) -> int:
    """Non-zero tick mid returns (the denominator of Kolm, Turiel and Westray 2023, eq. 16) over regular trading,
    ``session.open_ns`` to ``session.close_ns``: same-segment adjacent rows with different mid."""
    pos, _, seg, _, _ = _select_rows(frame, session.open_ns, session.close_ns, dedupe)
    if len(pos) < 2:
        return 0
    mid = mid_at(frame, pos)
    return int(((mid[1:] != mid[:-1]) & (seg[1:] == seg[:-1])).sum())


def history_valid(kf: FlowFrame, history: int | None = None) -> np.ndarray:
    """Row t is a sample iff rows t-W+1..t are all ``of_valid`` and in the same segment."""
    w = kf.config.history if history is None else history
    # Length of the of_valid run ending at t. A segment's first row is never
    # of_valid, so a run cannot cross a segment boundary; the explicit segment
    # check below is a guard for frames built by hand.
    ok = np.asarray(kf.of_valid, dtype=bool)
    idx = np.arange(len(ok))
    seg_change = np.zeros(len(ok), dtype=bool)
    seg_change[1:] = kf.segment[1:] != kf.segment[:-1]
    # Marker = index of the last row NOT in the current run: the row itself if
    # not of_valid, the previous row if a new segment starts here.
    marker = np.where(~ok, idx, np.where(seg_change, idx - 1, -1))
    run = idx - np.maximum.accumulate(marker) if len(ok) else idx
    return run >= w


@dataclass(frozen=True)
class TrainWindow:
    train_dates: tuple[datetime.date, ...]
    mid_changes_per_day: float              # per full day of regular trading
    delta_t_ms: float                       # 2.34e7 / mid_changes_per_day


def estimate_delta_t(train_frames: Sequence[pd.DataFrame], train_dates: Sequence[datetime.date],
                     sessions: Iterable[Session]) -> TrainWindow:
    """Δt from TRAINING days only (Kolm, Turiel and Westray 2023, eq. 16); frozen and reused for validation and test.

    Δt is the total regular trading time of the days over their total count of mid-price changes, so a half day
    enters with its own regular trading time. The number of full days the training days are worth is their total
    time over 2.34e7 ms, an exact integer when every day is full: then ``mid_changes_per_day`` is the mean count
    to the last bit, and Δt the same quotient."""
    sessions = list(sessions)
    if len(train_frames) == 0:
        raise ValueError("no training frames")
    if len(sessions) != len(train_frames):
        raise ValueError(f"{len(train_frames)} training frames but {len(sessions)} sessions")
    counts = [count_mid_changes(f, s) for f, s in zip(train_frames, sessions)]
    full_days = sum(s.regular_ms for s in sessions) / DELTA_T_NUMERATOR_MS
    per_day = float(np.sum(counts)) / full_days
    if per_day == 0:
        raise ValueError("no mid changes in training window")
    return TrainWindow(tuple(train_dates), per_day, DELTA_T_NUMERATOR_MS / per_day)
