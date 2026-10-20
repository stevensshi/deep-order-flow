"""Labels: mid-price changes over the ten horizons of Kolm, Turiel and Westray (2023), the sample mask."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from dof.lob.features.cleaning import asof_index, breaks_between, in_interval
from dof.lob.features.flow_series import FlowFrame, TrainWindow


@dataclass
class LabelSet:
    y: np.ndarray            # float64 (n, K) dollars, NaN where invalid
    horizons_ms: np.ndarray  # float64 (K,)
    valid: np.ndarray        # bool (n, K)
    funnel: dict[str, int]   # per reason, counted over (row, k) pairs
    k: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))            # (K,) horizon numbers
    h_le_latency: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))    # (K,) whole column invalid


def horizon_labels(frame: FlowFrame, k: Iterable[int] = range(1, 11), *,
               train_window: TrainWindow | None = None) -> LabelSet:
    """y = p(t + h_k) - p(t + latency) in dollars.

    p(τ) = frame.prices.mid[asof_index(frame.prices, τ)]: the COMPLETE raw same-day
    series, last row with ts <= τ — not the
    cleaned sample rows, whose dedupe/Q-H drop can delete the group that holds a
    price update. Invalid, in this priority: h_le_latency; label_crosses_session_end;
    label_crosses_invalid_segment (an interval starts between row t and the far
    as-of row, or the far row lies inside an interval); label_asof_invalid_row
    (a looked-up mid is NaN).
    """
    if train_window is None:
        raise ValueError("train_window required: Δt must be estimated on training days only")
    ks = np.asarray(list(k), dtype=np.int64)
    h_ms = ks / 5.0 * train_window.delta_t_ms
    h_ns = np.round(h_ms * 1e6).astype(np.int64)
    lat_ns = int(frame.config.latency_ns)        # the latency buffer is the frame's own FlowConfig.latency_ns
    ts, seq_t = frame.ts_ns, frame.seq
    ps, iv = frame.prices, frame.intervals
    n, K = len(ts), len(ks)
    y = np.full((n, K), np.nan)
    valid = np.zeros((n, K), dtype=bool)
    funnel = {"h_le_latency": 0, "label_crosses_session_end": 0, "label_crosses_invalid_segment": 0,
              "label_asof_invalid_row": 0, "cross_day": 0, "cross_split": 0, "valid": 0}
    h_le = h_ns <= lat_ns
    if n:
        i_lat = asof_index(ps, ts + lat_ns)                       # >= row t itself (t is a raw row)
        p_lat = ps.mid[i_lat]
        for j in range(K):
            if h_le[j]:
                funnel["h_le_latency"] += n
                continue
            end = ts + h_ns[j]
            crosses_end = end >= frame.session.end_ns
            i_h = asof_index(ps, end)
            far_seq = ps.seq[i_h]
            bad_seg = breaks_between(iv, seq_t, far_seq) | in_interval(iv, far_seq)
            p_h = ps.mid[i_h]
            bad_row = ~np.isfinite(p_h) | ~np.isfinite(p_lat)
            ok = ~crosses_end & ~bad_seg & ~bad_row
            funnel["label_crosses_session_end"] += int(crosses_end.sum())
            funnel["label_crosses_invalid_segment"] += int((~crosses_end & bad_seg).sum())
            funnel["label_asof_invalid_row"] += int((~crosses_end & ~bad_seg & bad_row).sum())
            funnel["valid"] += int(ok.sum())
            y[ok, j] = p_h[ok] - p_lat[ok]
            valid[:, j] = ok
    return LabelSet(y=y, horizons_ms=h_ms, valid=valid, funnel=funnel, k=ks, h_le_latency=h_le)


def k_valid_from(labels: LabelSet) -> list[int]:
    """Horizons k not judged ``h_le_latency`` (a stock is modelled on these only)."""
    return [int(k) for k, bad in zip(labels.k, labels.h_le_latency) if not bad]


def sample_mask(labels: LabelSet, k_valid: Sequence[int], hist_ok: np.ndarray) -> np.ndarray:
    """Rows with a full history AND valid labels for every horizon in ``k_valid``."""
    mask = np.asarray(hist_ok, dtype=bool).copy()
    if len(k_valid) == 0:
        return np.zeros_like(mask)
    cols = [int(np.flatnonzero(labels.k == k)[0]) for k in k_valid]
    return mask & labels.valid[:, cols].all(axis=1)
