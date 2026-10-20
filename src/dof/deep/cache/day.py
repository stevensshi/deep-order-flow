"""One stock-day through the order-book library: cleaned rows, order flow, labels with the group's Δt, and the
event index of the row that holds each label's start price."""

from __future__ import annotations

import datetime
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dof.deep.config import RunConfig
from dof.deep.runtime.memory import release_memory
from dof.deep.splits import session_of
from dof.lob.features import (
    FLOW_COLUMNS,
    FlowConfig,
    FlowFrame,
    LabelSet,
    TrainWindow,
    asof_index,
    history_valid,
    horizon_labels,
    flow_pipeline,
)
from dof.lob.tables import load_states


@dataclass
class DayData:
    date: datetime.date
    ts: np.ndarray
    of: np.ndarray
    ofi: np.ndarray
    y: np.ndarray            # (n, len(horizons)) dollars
    mask: np.ndarray         # rows with a full history window (before the label rule)


def day_frame(cfg: RunConfig, book_root: Path, symbol: str, date: datetime.date, tw: TrainWindow,
              funnel: Counter) -> tuple[FlowFrame, DayData, LabelSet, np.ndarray]:
    """(FlowFrame, DayData, LabelSet, label start event index per row). The session is the day's own: a half day
    ends ten minutes before its early close, so no label crosses it."""
    f = cfg.features
    frame = load_states(book_root / date.isoformat(), symbol, columns=FLOW_COLUMNS)
    kf = flow_pipeline(frame, session_of(cfg, date.isoformat()),
                      FlowConfig(history=f.history, convention=f.convention, latency_ns=f.latency_ns, dedupe=f.dedupe))
    del frame
    release_memory()   # hand the frame's pages back before the labels allocate
    labels = horizon_labels(kf, f.horizons, train_window=tw)
    hist = history_valid(kf, f.history)
    for k, v in kf.funnel.items():
        funnel[k] += v
    for k, v in labels.funnel.items():
        funnel[f"label_{k}" if not k.startswith("label_") else k] += v
    # Raw event index of the row that holds the label's start price p(t + latency): the as-of lookup of horizon_labels.
    lab_seq = (kf.prices.seq[asof_index(kf.prices, kf.ts_ns + kf.config.latency_ns)] if len(kf.ts_ns)
               else np.zeros(0, dtype=np.int64))
    return kf, DayData(date, kf.ts_ns, kf.of, kf.ofi, labels.y, hist), labels, lab_seq.astype(np.int64)


def check_windows(z: np.ndarray, rows: np.ndarray, history: int, what: str) -> None:
    """A sample row's window must hold no NaN (the first row of each segment is NaN)."""
    bad = np.cumsum(~np.isfinite(z).all(axis=1))
    if len(rows) and (rows.min() < history or (bad[rows] - bad[rows - history]).any()):
        raise AssertionError(f"{what}: NaN inside a sample window")
