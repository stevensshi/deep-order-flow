"""Winsorising and z-scoring fitted on training rows, and the lagged design matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Scaler:
    lo: np.ndarray
    hi: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    zero_std_cols: tuple[int, ...]


def fit_scaler(x: np.ndarray, lo_q: float = 0.005, hi_q: float = 0.995) -> Scaler:
    """Fit on TRAIN rows only (no NaN): per-column quantile bounds, then mean/std of the clipped data."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if np.isnan(x).any():
        raise ValueError("fit_scaler input contains NaN")
    lo = np.quantile(x, lo_q, axis=0, method="linear")
    hi = np.quantile(x, hi_q, axis=0, method="linear")
    c = np.clip(x, lo, hi)
    mean = c.mean(axis=0)
    std = c.std(axis=0, ddof=0)
    return Scaler(lo=lo, hi=hi, mean=mean, std=std, zero_std_cols=tuple(int(i) for i in np.flatnonzero(std == 0)))


def fit_scaler_columnwise(get_column: Callable[[int], np.ndarray], ncols: int, lo_q: float = 0.005,
                          hi_q: float = 0.995) -> Scaler:
    """``fit_scaler`` one column at a time (the fit is column-independent): ``get_column(c)`` gives column c of
    every training row, so the whole (n, ncols) matrix is never held. Equal, column for column, to
    ``fit_scaler(col[:, None])``; a whole-matrix ``fit_scaler`` may differ from it in the last bits."""
    parts = []
    for c in range(ncols):
        x = get_column(c)
        parts.append(fit_scaler(x[:, None], lo_q, hi_q))
        del x
    zero = tuple(c for c, sc in enumerate(parts) if sc.zero_std_cols)
    return Scaler(lo=np.concatenate([sc.lo for sc in parts]), hi=np.concatenate([sc.hi for sc in parts]),
                  mean=np.concatenate([sc.mean for sc in parts]), std=np.concatenate([sc.std for sc in parts]),
                  zero_std_cols=zero)


def apply_scaler(x: np.ndarray, s: Scaler) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    safe = np.where(s.std == 0, 1.0, s.std)
    z = (np.clip(x, s.lo, s.hi) - s.mean) / safe
    if s.zero_std_cols:
        z[:, list(s.zero_std_cols)] = 0.0
    return z[:, 0] if squeeze else z


def inverse_target(z: np.ndarray, s: Scaler) -> np.ndarray:
    """z * std + mean (no un-clipping): model output back in dollars."""
    return np.asarray(z, dtype=np.float64) * s.std + s.mean


def lagged_design(z_of: np.ndarray, rows: np.ndarray, history: int) -> np.ndarray:
    """Row i = z_of[rows[i]-history+1 : rows[i]+1] flattened, oldest first.

    Fixed implementation: preallocate and assign one lag block at a
    time — a sliding_window_view(...).reshape copy measured 18x slower.
    Callers chunk ``rows`` (<= 16,384 per call).
    """
    z_of = np.asarray(z_of)
    rows = np.asarray(rows, dtype=np.int64)
    d = z_of.shape[1]
    if len(rows) and rows.min() < history - 1:
        raise ValueError("rows need history-1 preceding rows")
    out = np.empty((len(rows), history * d), dtype=np.float64)
    for i in range(history):
        out[:, i * d:(i + 1) * d] = z_of[rows - history + 1 + i]
    return out
