from __future__ import annotations

import math

import numpy as np
import pytest

from dof.lob.stats import (
    adj_r2,
    block_t_test,
    ols_small,
    r2_os,
    r2_vs_eval_mean,
    r2_zero,
    solve_ridge,
    suff_stats_init,
    suff_stats_update,
)


def _data(n: int = 300, p: int = 5, q: int = 3):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, p)) + rng.normal(size=p)          # non-zero means exercise centring
    W = rng.normal(size=(p, q))
    Y = X @ W + 2.5 + 0.1 * rng.normal(size=(n, q))           # non-zero intercept
    return X, Y


def test_suffstats_ols_equals_lstsq() -> None:
    X, Y = _data()
    s = suff_stats_init(X.shape[1], Y.shape[1])
    for chunk in np.array_split(np.arange(len(X)), 3):
        suff_stats_update(s, X[chunk], Y[chunk])
    w, b, info = solve_ridge(s, 0.0)
    ref = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), Y, rcond=None)[0]
    assert np.abs(w - ref[1:]).max() < 1e-8
    assert np.abs(b - ref[0]).max() < 1e-8
    assert info["n"] == len(X) and info["lam"] == 0.0 and info["fallback"] is None
    coef, intercept, r2 = ols_small(X, Y[:, 0])
    assert np.abs(coef - ref[1:, 0]).max() < 1e-8 and abs(intercept - ref[0, 0]) < 1e-8 and 0.9 < r2 <= 1


def test_ridge_shrinks_norm() -> None:
    X, Y = _data()
    s = suff_stats_init(X.shape[1], Y.shape[1])
    suff_stats_update(s, X, Y)
    norms = [np.linalg.norm(solve_ridge(s, lam)[0]) for lam in (0.0, 1e-3, 1e-2, 1e-1, 1, 10, 100)]
    assert all(a >= b - 1e-12 for a, b in zip(norms, norms[1:]))
    assert norms[-1] < 0.5 * norms[0]


def test_r2_os_benchmark_is_eval_mean() -> None:
    y = np.array([1.0, 2.0, 3.0, 6.0])
    assert r2_os(y, np.full(4, y.mean())) == 0.0
    assert r2_os(y, y) == 1.0
    assert math.isnan(r2_os(np.full(3, 2.0), np.zeros(3)))
    # Not the zero benchmark: predicting 0 is much worse than the mean here.
    assert r2_os(y, np.zeros(4)) < -1


def test_suff_stats_rejects_nan() -> None:
    X, Y = _data()
    s = suff_stats_init(X.shape[1], Y.shape[1])
    X[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite value"):
        suff_stats_update(s, X, Y)
    assert s.n == 0 and not s.sxx.any()


def test_suff_stats_rejects_inf() -> None:
    X, Y = _data()
    s = suff_stats_init(X.shape[1], Y.shape[1])
    Y[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite value"):
        suff_stats_update(s, X, Y)
    assert s.n == 0


def test_r2_vs_eval_mean_alias() -> None:
    assert r2_os is r2_vs_eval_mean
    y = np.array([1.0, 2.0, 3.0, 6.0])
    assert r2_vs_eval_mean(y, np.full(4, y.mean())) == 0.0


def test_r2_zero_benchmark() -> None:
    y = np.array([1.0, 2.0, 3.0, 6.0])
    assert r2_zero(y, np.zeros(4)) == 0.0
    assert r2_zero(y, y) == 1.0
    assert math.isnan(r2_zero(np.zeros(3), np.ones(3)))
    # 1 - Σ(y-ȳ)²/Σy² = 1 - 14/50 for the mean forecast
    assert abs(r2_zero(y, np.full(4, 3.0)) - (1 - 14 / 50)) < 1e-12


def test_adj_r2() -> None:
    assert abs(adj_r2(0.5, 11, 2) - 0.375) < 1e-12        # 1 - 0.5*10/8
    assert abs(adj_r2(0.8, 101, 1) - (1 - 0.2 * 100 / 99)) < 1e-12
    assert math.isnan(adj_r2(0.5, 3, 2))


def test_block_t_test() -> None:
    blocks = np.repeat(np.arange(20), 3)
    # block b holds b, b+1, b+2 -> block mean b+1, i.e. means 1..20
    loss = np.concatenate([[b, b + 1, b + 2] for b in range(20)]).astype(float)
    t, df, B, p = block_t_test(loss, blocks, min_blocks=20)
    want_t = 10.5 / (math.sqrt(35.0) / math.sqrt(20))   # var(1..20, ddof=1) = 20*21/12 = 35
    assert abs(t - want_t) < 1e-10 and df == 19 and B == 20
    assert 0 < p < 1e-6
    t2, df2, B2, p2 = block_t_test(loss[:57], blocks[:57], min_blocks=20)   # 19 blocks
    assert math.isnan(t2) and df2 == 0 and B2 == 19 and math.isnan(p2)


# -- pins: each fails if the behaviour named in its docstring changes --------------------------------


def test_solve_ridge_lstsq_fallback() -> None:
    """A singular system (a column that is always zero, λ = 0) falls back to least squares, with finite weights."""
    X, Y = _data()
    X[:, 2] = 0.0
    s = suff_stats_init(X.shape[1], Y.shape[1])
    suff_stats_update(s, X, Y)
    w, b, info = solve_ridge(s, 0.0)
    assert info["fallback"] == "lstsq"
    assert np.isfinite(w).all() and np.isfinite(b).all()
