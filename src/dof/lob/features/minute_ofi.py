"""The minute branch, after Cont, Cucuringu and Zhang (2023): book-change events, one-minute buckets, normalised OFI,
the first principal component."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dof.lob.features.cleaning import (
    _MINUTE_BAD_ORDER,
    ASK_PX,
    ASK_SZ,
    BID_PX,
    BID_SZ,
    InvalidIntervals,
    _bad_reasons,
    asof_index,
    breaks_between,
    invalid_intervals,
    mid_px,
    price_series,
    reason_code,
    row_masks,
)
from dof.lob.features.order_flow import SignConvention, _ofi_parts
from dof.lob.features.session import Session
from dof.lob.units import NS, hms


@dataclass(frozen=True)
class MinuteConfig:
    feature_start_ns: int = hms(9, 30)   # events used for OFI/Q aggregation start here
    analysis_start_ns: int = hms(10, 0)  # Cont, Cucuringu and Zhang (2023), §3.2
    analysis_end_ns: int = hms(15, 30)
    bucket_ns: int = 60 * NS
    M: int = 10
    convention: SignConvention = "current_size"


@dataclass
class ChangeEvents:
    ts_ns: np.ndarray
    bid_px: np.ndarray
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray            # (n,) / (n,10) int64
    mid: np.ndarray               # float64 dollars
    segment: np.ndarray
    has_prev: np.ndarray          # predecessor in same segment
    funnel: dict[str, int]
    intervals: InvalidIntervals   # raw invalid intervals of the day
    seq: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))   # source seq per event


def change_events(frame: pd.DataFrame, config: MinuteConfig = MinuteConfig()) -> ChangeEvents:
    """Observable 10-level change events: not P/Q/H, (px, sz) vector changed vs the previous state row."""
    funnel: dict[str, int] = {"rows_in": int(len(frame))}
    book = frame[BID_PX + BID_SZ + ASK_PX + ASK_SZ].to_numpy()
    changed = np.ones(len(frame), dtype=bool)
    if len(frame) > 1:
        changed[1:] = (book[1:] != book[:-1]).any(axis=1)
    pqh = frame["msg_type"].isin(["P", "Q", "H"]).to_numpy()
    early = frame["ts_ns"].to_numpy() < config.feature_start_ns
    funnel["pqh_removed"] = int(pqh.sum())
    funnel["unchanged_removed"] = int((~pqh & ~changed).sum())
    funnel["before_feature_start"] = int((~pqh & changed & early).sum())
    # Invalid intervals from the RAW table first: an H-only halt or a
    # same-ns crossed->restored pair is an interval even though no event row shows it.
    masks = row_masks(frame)
    iv = invalid_intervals(frame, masks)
    cand = np.flatnonzero(~pqh & changed & ~early)
    reasons = _bad_reasons(masks, _MINUTE_BAD_ORDER)[cand]
    for r in _MINUTE_BAD_ORDER:
        funnel[f"{r}_removed"] = int((reasons == reason_code(r)).sum())
    funnel["invalid_intervals"] = int(len(iv.start_ts))
    good = cand[reasons == 0]
    g = frame.iloc[good]
    ev_seq = g["seq"].to_numpy().astype(np.int64)
    has_prev = np.zeros(len(g), dtype=bool)
    if len(g) > 1:   # an interval start between two consecutive events breaks the segment
        has_prev[1:] = ~breaks_between(iv, ev_seq[:-1], ev_seq[1:])
    seg = (np.cumsum(~has_prev) - 1).astype(np.int32)
    funnel["events"] = int(len(g))
    return ChangeEvents(ts_ns=g["ts_ns"].to_numpy().astype(np.int64),
                     bid_px=g[BID_PX].to_numpy().astype(np.int64), bid_sz=g[BID_SZ].to_numpy().astype(np.int64),
                     ask_px=g[ASK_PX].to_numpy().astype(np.int64), ask_sz=g[ASK_SZ].to_numpy().astype(np.int64),
                     mid=mid_px(g["bid_px_1"].to_numpy(), g["ask_px_1"].to_numpy()), segment=seg, has_prev=has_prev,
                     funnel=funnel, intervals=iv, seq=ev_seq)


def _event_of(ev: ChangeEvents, M: int, convention: SignConvention) -> np.ndarray:
    """Per-event OFI (n, M) against the previous event; 0 where the event has no same-segment predecessor."""
    n = len(ev.ts_ns)
    out = np.zeros((n, M))
    if n > 1:
        bof, aof = _ofi_parts(ev.bid_px[:-1, :M], ev.bid_sz[:-1, :M], ev.ask_px[:-1, :M], ev.ask_sz[:-1, :M],
                              ev.bid_px[1:, :M], ev.bid_sz[1:, :M], ev.ask_px[1:, :M], ev.ask_sz[1:, :M],
                              convention)
        out[1:] = bof - aof
        out[~ev.has_prev] = 0.0
    return out


def window_ofi(ev: ChangeEvents, window_end_ns: np.ndarray, length_ns: int, M: int = 10,
                   convention: SignConvention = "current_size") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """OFI (n, M), Q (n,), dN (n,), broken (n,) over events in (end - length, end].

    ``broken``: a raw invalid interval overlaps (end - length, end], i.e.
    start_ts <= end and close_ts > end - length.

    OFI sums event order flow over events with a same-segment predecessor.
    Q = sum_{n,m}(q^b + q^a) / (2 M dN) over ALL events in the window (Cont, Cucuringu and Zhang
    2023, eq. 3) — one depth shared by every level, not each level's own depth.
    An empty window has dN = 0 and Q = 0.
    """
    ends = np.asarray(window_end_ns, dtype=np.int64)
    of = _event_of(ev, M, convention)
    depth = (ev.bid_sz[:, :M].sum(axis=1) + ev.ask_sz[:, :M].sum(axis=1)).astype(np.float64)
    c_of = np.vstack([np.zeros((1, M)), np.cumsum(of, axis=0)])
    c_depth = np.concatenate([[0.0], np.cumsum(depth)])
    lo = np.searchsorted(ev.ts_ns, ends - length_ns, "right")
    hi = np.searchsorted(ev.ts_ns, ends, "right")
    ofi = c_of[hi] - c_of[lo]
    dn = (hi - lo).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(dn > 0, (c_depth[hi] - c_depth[lo]) / (2.0 * M * np.maximum(dn, 1)), 0.0)
    broken = ev.intervals.overlaps(ends - length_ns, ends)   # any invalid time inside the window
    return ofi, q, dn, broken


def minute_buckets(frame: pd.DataFrame, *, config: MinuteConfig = MinuteConfig(), session: Session | None = None) -> pd.DataFrame:
    """One-minute buckets over regular trading with OFI, Q, normalised ofi, as-of mid and log return: 390 buckets
    ending 09:31..16:00 on a full day (``session`` None), (close - open) / 1 min on the day of ``session``.
    ``config.M`` levels; only one-minute buckets (``config.bucket_ns``) are supported."""
    if config.bucket_ns != 60 * NS:
        raise ValueError(f"only one-minute buckets are supported, got bucket_ns={config.bucket_ns}")
    M = config.M
    open_ns, close_ns = (hms(9, 30), hms(16, 0)) if session is None else (session.open_ns, session.close_ns)
    ev = change_events(frame, config)
    ps = price_series(frame)
    length = 60 * NS
    n_buckets = (close_ns - open_ns) // length
    ends = open_ns + length * np.arange(1, n_buckets + 1, dtype=np.int64)
    ofi_raw, q, dn, _ = window_ofi(ev, ends, length, M, config.convention)
    with np.errstate(invalid="ignore", divide="ignore"):
        ofi_norm = np.where(q[:, None] > 0, ofi_raw / np.where(q > 0, q, 1.0)[:, None], np.nan)
    # as-of mid at t_j and t_{j-1}
    starts = ends - length
    # as-of mid on the complete raw price series
    i_end = asof_index(ps, ends)
    i_start = asof_index(ps, starts)
    has_end, has_start = i_end >= 0, i_start >= 0
    nan = np.full(len(ends), np.nan)
    mid_end = np.where(has_end, ps.mid[np.maximum(i_end, 0)], nan) if len(ps.mid) else nan
    mid_start = np.where(has_start, ps.mid[np.maximum(i_start, 0)], nan) if len(ps.mid) else nan
    if len(ps.mid):
        broken_asof = breaks_between(ev.intervals, ps.seq[np.maximum(i_start, 0)], ps.seq[np.maximum(i_end, 0)])
    else:
        broken_asof = np.zeros(len(ends), dtype=bool)
    no_mid = ~has_end | ~has_start | broken_asof | ~np.isfinite(mid_end) | ~np.isfinite(mid_start)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = np.where(no_mid, np.nan, np.log(mid_end / mid_start))
    # break reasons inside each bucket
    reason = np.full(len(ends), "", dtype=object)
    for r in reversed(_MINUTE_BAD_ORDER + ("empty_bucket", "zero_depth", "no_mid")):
        if r in _MINUTE_BAD_ORDER:
            # any invalid time of this reason inside (t_{j-1}, t_j], also a halt spanning the whole bucket
            hit = ev.intervals.overlaps(starts, ends, r)
        elif r == "empty_bucket":
            hit = dn == 0
        elif r == "zero_depth":
            hit = (dn > 0) & (q == 0)
        else:
            hit = no_mid
        reason[hit] = r
    out = {"bucket_end_ns": ends, "n_events": dn}
    for m in range(M):
        out[f"OFI_{m + 1}"] = ofi_raw[:, m]
    out["Q"] = q
    for m in range(M):
        out[f"ofi_{m + 1}"] = ofi_norm[:, m]
    out["mid_end"] = mid_end
    out["ret"] = ret
    out["valid"] = reason == ""
    out["invalid_reason"] = reason.astype(str)
    out["in_analysis"] = (ends > config.analysis_start_ns) & (ends <= config.analysis_end_ns)
    return pd.DataFrame(out)


# -- PCA -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PcaWeights:
    w: np.ndarray                 # (M,), L1-normalised: w = v / sum(|v|)
    explained_variance_ratio: float
    n_obs: int


def fit_pc1(ofi: np.ndarray) -> PcaWeights:
    """First principal component of the estimation window's normalised ofi.

    Sign: the first component with |v_i| > 1e-12 is positive. No per-column
    standardisation; fit on the estimation window only.
    """
    x = np.asarray(ofi, dtype=np.float64)
    n, m = x.shape
    if n < m:
        raise ValueError(f"n_obs={n} < M={m}")
    cov = np.cov(x - x.mean(axis=0), rowvar=False, ddof=1)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, -1].copy()
    nz = np.flatnonzero(np.abs(v) > 1e-12)
    if len(nz) and v[nz[0]] < 0:
        v = -v
    total = vals.sum()
    return PcaWeights(w=v / np.abs(v).sum(), explained_variance_ratio=float(vals[-1] / total) if total > 0 else float("nan"),
                      n_obs=int(n))


def integrated_ofi(ofi: np.ndarray, pw: PcaWeights) -> np.ndarray:
    """Projection of the RAW ofi (not de-meaned; the regression intercept absorbs it)."""
    return np.asarray(ofi, dtype=np.float64) @ pw.w
