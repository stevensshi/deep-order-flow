"""Per-level order flow (bOF, aOF) between two book states, under either sign convention."""

from __future__ import annotations

from typing import Literal

import numpy as np

SignConvention = Literal["current_size", "cks"]


def order_flow(prev_px, prev_sz, cur_px, cur_sz, *, is_bid: bool,
               convention: SignConvention = "current_size") -> np.ndarray:
    """Per-level order flow, shape (n, levels): bOF for the bid side, aOF for the ask side."""
    if convention not in ("current_size", "cks"):
        raise ValueError(f"unknown convention {convention!r}")
    prev_px = np.asarray(prev_px)
    cur_px = np.asarray(cur_px)
    prev_sz = np.asarray(prev_sz, dtype=np.float64)
    cur_sz = np.asarray(cur_sz, dtype=np.float64)
    worse = -cur_sz if convention == "current_size" else -prev_sz
    if is_bid:
        better, worse_mask = cur_px > prev_px, cur_px < prev_px
    else:
        better, worse_mask = cur_px < prev_px, cur_px > prev_px
    return np.where(better, cur_sz, np.where(worse_mask, worse, cur_sz - prev_sz))


def _ofi_parts(bpx0, bsz0, apx0, asz0, bpx1, bsz1, apx1, asz1, convention) -> tuple[np.ndarray, np.ndarray]:
    bof = order_flow(bpx0, bsz0, bpx1, bsz1, is_bid=True, convention=convention)
    aof = order_flow(apx0, asz0, apx1, asz1, is_bid=False, convention=convention)
    return bof, aof
