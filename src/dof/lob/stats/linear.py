"""Linear models and statistics for the OFI baselines.

Sufficient statistics let the 2,000-column design of Kolm, Turiel and Westray (2023) be accumulated in chunks and
solved once for every ridge λ (ARX-OF and Ridge-OF share one ``SuffStats``). The intercept is never penalised.

Two out-of-sample R² benchmarks are kept apart on purpose: ``r2_os`` uses the mean of the EVALUATION set (Kolm, Turiel and Westray
2023, eq. 18; the main column of Cont, Cucuringu and Zhang 2023), ``r2_zero`` the zero forecast (their secondary column).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.stats


@dataclass
class SuffStats:
    n: int
    sx: np.ndarray
    sy: np.ndarray
    sxx: np.ndarray
    sxy: np.ndarray
    syy: np.ndarray   # float64


def suff_stats_init(p: int, q: int) -> SuffStats:
    return SuffStats(0, np.zeros(p), np.zeros(q), np.zeros((p, p)), np.zeros((p, q)), np.zeros((q, q)))


def suff_stats_update(s: SuffStats, X: np.ndarray, Y: np.ndarray) -> None:
    """Accumulate one chunk in place; X (n, p), Y (n, q). A NaN or an infinity would spread into every later
    solution of the sums, so a chunk holding one is refused before anything is added."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise ValueError("non-finite value in a sufficient-statistics chunk")
    s.n += X.shape[0]
    s.sx += X.sum(axis=0)
    s.sy += Y.sum(axis=0)
    s.sxx += X.T @ X
    s.sxy += X.T @ Y
    s.syy += Y.T @ Y


def solve_ridge(s: SuffStats, lam: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Centred ridge: (Cxx + lam I) W = Cxy, b = my - mx W; lam = 0 is OLS."""
    if s.n == 0:
        raise ValueError("no observations accumulated")
    mx = s.sx / s.n
    my = s.sy / s.n
    cxx = s.sxx / s.n - np.outer(mx, mx)
    cxy = s.sxy / s.n - np.outer(mx, my)
    a = cxx + lam * np.eye(len(mx))
    info: dict = {"lam": lam, "n": s.n, "cond": float(np.linalg.cond(a)), "fallback": None}
    try:
        w = scipy.linalg.solve(a, cxy, assume_a="pos")
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(a, cxy, rcond=None)[0]
        info["fallback"] = "lstsq"
    b = my - mx @ w
    return w, b, info


def ols_small(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """OLS with intercept via lstsq on [1, X]; returns (coef, intercept, in-sample R²)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=np.float64)
    A = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - A @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = float("nan") if ss_tot == 0 else 1.0 - float((resid ** 2).sum()) / ss_tot
    return beta[1:], float(beta[0]), r2


def r2_vs_eval_mean(y: np.ndarray, yhat: np.ndarray) -> float:
    """1 - Σ(y-ŷ)² / Σ(y-ȳ)² with ȳ the mean of the evaluation set; NaN if the denominator is 0."""
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    den = float(((y - y.mean()) ** 2).sum())
    if den == 0:
        return float("nan")
    return 1.0 - float(((y - yhat) ** 2).sum()) / den


# The name of the main column of both papers (Kolm, Turiel and Westray 2023; Cont, Cucuringu and Zhang 2023); the longer name says
# which benchmark it uses.
r2_os = r2_vs_eval_mean


def r2_zero(y: np.ndarray, yhat: np.ndarray) -> float:
    """1 - Σ(y-ŷ)² / Σy² (zero benchmark); NaN if Σy² == 0."""
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    den = float((y ** 2).sum())
    if den == 0:
        return float("nan")
    return 1.0 - float(((y - yhat) ** 2).sum()) / den


def adj_r2(r2: float, n: int, p: int) -> float:
    """1 - (1-r2)(n-1)/(n-p-1); NaN if n-p-1 <= 0."""
    if n - p - 1 <= 0:
        return float("nan")
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


def block_t_test(loss_diff: np.ndarray, block_id: np.ndarray, min_blocks: int = 20) -> tuple[float, int, int, float]:
    """t-test on block means: (t, df, B, two-sided p); (nan, 0, B, nan) if B < min_blocks."""
    loss_diff = np.asarray(loss_diff, dtype=np.float64)
    block_id = np.asarray(block_id)
    ids, inv = np.unique(block_id, return_inverse=True)
    B = len(ids)
    if B < min_blocks:
        return float("nan"), 0, B, float("nan")
    means = np.bincount(inv, weights=loss_diff, minlength=B) / np.bincount(inv, minlength=B)
    se = means.std(ddof=1) / np.sqrt(B)
    t = float(means.mean() / se) if se > 0 else float("nan")
    df = B - 1
    p = float(2.0 * scipy.stats.t.sf(abs(t), df)) if np.isfinite(t) else float("nan")
    return t, df, B, p
