from __future__ import annotations

import numpy as np
import pytest

from dof.lob import features as F


def test_scaler_train_only() -> None:
    rng = np.random.default_rng(1)
    train = np.column_stack([rng.normal(size=1000), rng.normal(size=1000) * 3, np.full(1000, 7.0)])
    train[0, 0] = 1e6
    s = F.fit_scaler(train)
    assert s.lo[0] == np.quantile(train[:, 0], 0.005) and s.hi[0] == np.quantile(train[:, 0], 0.995)
    assert s.hi[0] < 10                                   # the 1e6 outlier is clipped away
    clipped = np.clip(train, s.lo, s.hi)
    assert np.allclose(s.mean, clipped.mean(axis=0)) and np.allclose(s.std, clipped.std(axis=0))
    assert s.zero_std_cols == (2,)
    test = np.array([[1e9, -1e9, 7.0], [0.1, 0.2, 7.0]])
    z = F.apply_scaler(test, s)
    assert z[0, 0] == pytest.approx((s.hi[0] - s.mean[0]) / s.std[0])
    assert z[0, 1] == pytest.approx((s.lo[1] - s.mean[1]) / s.std[1])
    assert (z[:, 2] == 0).all()
    back = F.inverse_target(z, s)
    assert back[1, :2] == pytest.approx(test[1, :2])    # within the bounds: exact round trip
    assert back[0, 0] == pytest.approx(s.hi[0])          # clipped values are not un-clipped


def test_lagged_design_order() -> None:
    z = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = F.lagged_design(z, np.array([1, 2]), history=2)
    assert out.tolist() == [[1, 2, 3, 4], [3, 4, 5, 6]]
    with pytest.raises(ValueError):
        F.lagged_design(z, np.array([0]), history=2)


def test_fit_scaler_columnwise_equals_per_column_fit() -> None:
    rng = np.random.default_rng(3)
    x = np.column_stack([rng.normal(size=500), rng.standard_t(3, size=500) * 7, np.full(500, 2.0)])
    s = F.fit_scaler_columnwise(lambda c: x[:, c], 3)
    for c in range(3):
        one = F.fit_scaler(x[:, c][:, None])
        for name in ("lo", "hi", "mean", "std"):
            assert getattr(s, name)[c] == getattr(one, name)[0], (name, c)       # exact, not approximate
    assert s.zero_std_cols == (2,)
    # A whole-matrix fit computes the same statistics, but its column reductions may differ in the last bits.
    whole = F.fit_scaler(x)
    assert np.allclose(s.mean, whole.mean, rtol=1e-12, atol=0) and np.allclose(s.std, whole.std, rtol=1e-12, atol=0)

