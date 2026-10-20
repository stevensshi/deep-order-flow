from __future__ import annotations

import numpy as np
import pytest

from dof.lob import features as F
from dof.lob.book import FLAG_CROSSED
from dof.lob.features import hms
from hand_frames import FULL, MS, T10, _frame, _mid_row, _tw


def _labels_frame() -> F.FlowFrame:
    rows = [_mid_row(T10 + d * MS, m) for d, m in ((0, 100_000), (5, 100_100), (20, 100_200), (50, 100_300))]
    return F.flow_pipeline(_frame(rows), FULL)


def test_labels_asof_latency() -> None:
    lab = F.horizon_labels(_labels_frame(), k=[1, 2], train_window=_tw(200.0))
    assert lab.horizons_ms.tolist() == [40.0, 80.0]
    # row 0: p(0+40ms) = mid at 20ms (10.02); p(0+10ms) = mid at 5ms (10.01)
    assert lab.y[0, 0] == pytest.approx(0.01, abs=1e-12)
    assert lab.valid[0, 0]
    # row 1 (5ms): p(45ms) = 10.02, p(15ms) = 10.01 -> 0.01; k=2: p(85ms) = 10.03 -> 0.02
    assert lab.y[1, 0] == pytest.approx(0.01, abs=1e-12) and lab.y[1, 1] == pytest.approx(0.02, abs=1e-12)


def test_labels_invalid_reasons() -> None:
    end = hms(15, 50)
    rows = [
        _mid_row(T10, 100_000), _mid_row(T10 + 20 * MS, 100_100),
        _mid_row(T10 + 30 * MS, 100_100, flags=FLAG_CROSSED),
        _mid_row(T10 + 60 * MS, 100_200), _mid_row(T10 + 200 * MS, 100_300),
        _mid_row(end - 30 * MS, 100_300), _mid_row(end - 1 * MS, 100_400),
    ]
    kf = F.flow_pipeline(_frame(rows), FULL)
    lab = F.horizon_labels(kf, k=[1], train_window=_tw(200.0))       # h_1 = 40 ms
    # row 0: (0, 40ms] contains the crossed row at 30ms -> invalid segment
    assert not lab.valid[0, 0] and np.isnan(lab.y[0, 0])
    # row 2 (60ms): p(100ms) = row 2 itself, same segment -> valid, 0.0
    assert lab.valid[2, 0] and lab.y[2, 0] == 0.0
    # rows 4, 5 (15:49:59.970 and .999): t + 40ms >= 15:50
    assert not lab.valid[4, 0] and not lab.valid[5, 0]
    f = lab.funnel
    assert f["label_crosses_session_end"] == 2
    assert f["label_crosses_invalid_segment"] == 2       # rows 0 and 1 (break at 30ms within (t, t+40])
    assert f["cross_day"] == 0 and f["cross_split"] == 0
    assert f["valid"] == 2                               # rows 2 and 3
    # Δt small enough that h_1 <= 10 ms: the whole column is invalid
    lab2 = F.horizon_labels(kf, k=[1, 2], train_window=_tw(40.0))   # h_1 = 8 ms, h_2 = 16 ms
    assert not lab2.valid[:, 0].any() and lab2.funnel["h_le_latency"] == len(kf.ts_ns)
    assert lab2.h_le_latency.tolist() == [True, False]


def test_k_valid_excludes_short_horizons() -> None:
    kf = _labels_frame()
    lab = F.horizon_labels(kf, train_window=_tw(40.0))   # h_1 = 8 ms <= 10 ms; h_2 = 16 ms
    assert not lab.valid[:, 0].any()
    kv = F.k_valid_from(lab)
    assert kv == list(range(2, 11))
    hist_ok = np.ones(len(kf.ts_ns), dtype=bool)
    mask = F.sample_mask(lab, kv, hist_ok)
    # row 0 has valid labels for k = 2..10? h_10 = 80ms -> within the session, no breaks -> yes
    assert mask[0]
    assert (mask == (lab.valid[:, 1:].all(axis=1) & hist_ok)).all()
    assert not F.sample_mask(lab, kv, np.zeros_like(hist_ok)).any()
    tiny = F.horizon_labels(kf, train_window=_tw(1.0))   # h_10 = 2 ms
    assert F.k_valid_from(tiny) == []
    assert not F.sample_mask(tiny, [], hist_ok).any()


def test_horizon_labels_requires_train_window() -> None:
    with pytest.raises(ValueError, match="train_window required"):
        F.horizon_labels(_labels_frame())


def test_label_uses_complete_price_series() -> None:
    # Counterexample: 0 ms $100, 5 ms $100.1 and a Q row at the same instant
    # (repeating the book), 20 ms $100.2; h = 20 ms, latency 10 ms. Correct label: $0.10, not $0.20.
    rows = [_mid_row(T10, 1_000_000), _mid_row(T10 + 5 * MS, 1_001_000),
            _mid_row(T10 + 5 * MS, 1_001_000, mt="Q"), _mid_row(T10 + 20 * MS, 1_002_000)]
    kf = F.flow_pipeline(_frame(rows), FULL)
    assert kf.ts_ns.tolist() == [T10, T10 + 20 * MS]          # the 5 ms group vanished from the samples
    lab = F.horizon_labels(kf, k=[1], train_window=_tw(100.0))   # h_1 = 20 ms, latency 10 ms (FlowConfig default)
    assert lab.horizons_ms.tolist() == [20.0]
    assert lab.valid[0, 0]
    assert lab.y[0, 0] == pytest.approx(0.10, abs=1e-12)


# -- pins: each fails if the behaviour named in its docstring changes --------------------------------


def test_label_far_row_inside_interval() -> None:
    """A label whose far as-of row lies inside a run of bad raw rows is invalid even when the run started before
    the sample row, so no interval starts between the two rows. Only the in-interval rule catches this case; the
    frame is built by hand, as ``flow_pipeline`` never keeps a sample row inside a run."""
    raw_ts = np.array([T10, T10 + 2 * MS, T10 + 5 * MS, T10 + 30 * MS, T10 + 60 * MS], dtype=np.int64)
    prices = F.PriceSeries(ts_ns=raw_ts, seq=np.arange(5, dtype=np.int64), mid=np.full(5, 10.0))
    run = F.InvalidIntervals(start_seq=np.array([1]), end_seq=np.array([3]), close_seq=np.array([4]),
                             start_ts=raw_ts[[1]], close_ts=raw_ts[[4]], reason=np.array(["crossed"]))
    kf = F.FlowFrame(ts_ns=raw_ts[[2]], mid=np.array([10.0]), of=np.zeros((1, 20)), ofi=np.zeros((1, 10)),
                    seq=np.array([2], dtype=np.int64), segment=np.zeros(1, dtype=np.int32), of_valid=np.ones(1, dtype=bool),
                    intervals=run, prices=prices, funnel={}, config=F.FlowConfig(), session=FULL)
    lab = F.horizon_labels(kf, k=[1], train_window=_tw(200.0))      # h_1 = 40 ms: the far row is raw seq 3
    assert not lab.valid[0, 0] and np.isnan(lab.y[0, 0])
    assert lab.funnel["label_crosses_invalid_segment"] == 1 and lab.funnel["label_asof_invalid_row"] == 0
