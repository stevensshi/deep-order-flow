"""Trading sessions and half days."""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from dof.lob import features as F
from dof.lob.features import Session, hms, session_for
from hand_frames import FULL, MS, _frame, _mid_row, _tw

HALF = Session.half_day("13:00")


def _changes(k: int):
    """A frame with exactly k mid-price changes, one per minute from 10:00."""
    return _frame([_mid_row(hms(10, i), 1_000_000 + 1_000 * (i % 2)) for i in range(k + 1)])


def test_full_day_constants() -> None:
    assert (FULL.start_ns, FULL.end_ns, FULL.regular_ms) == (hms(9, 40), hms(15, 50), 2.34e7)
    assert (HALF.start_ns, HALF.end_ns, HALF.regular_ms) == (hms(9, 40), hms(12, 50), 1.26e7)


def test_half_day_drops_rows_from_1250() -> None:
    rows = [_mid_row(hms(12, 49, 59), 1_000_000), _mid_row(hms(12, 50), 1_001_000), _mid_row(hms(13, 30), 1_002_000)]
    assert F.flow_pipeline(_frame(rows), HALF).ts_ns.tolist() == [hms(12, 49, 59)]
    assert len(F.flow_pipeline(_frame(rows), FULL).ts_ns) == 3


def test_half_day_labels_end_at_1250() -> None:
    rows = [_mid_row(hms(12, 49), 1_000_000), _mid_row(hms(12, 49, 59) + 970 * MS, 1_001_000),
            _mid_row(hms(12, 50) + 100 * MS, 1_002_000)]
    half = F.horizon_labels(F.flow_pipeline(_frame(rows), HALF), k=[1], train_window=_tw(200.0))   # h_1 = 40 ms
    assert half.funnel["label_crosses_session_end"] == 1 and not half.valid[1, 0]
    full = F.horizon_labels(F.flow_pipeline(_frame(rows), FULL), k=[1], train_window=_tw(200.0))
    assert full.funnel["label_crosses_session_end"] == 0


def test_count_mid_changes_half_day_window() -> None:
    rows = [_mid_row(hms(10, 0), 1_000_000), _mid_row(hms(10, 1), 1_001_000), _mid_row(hms(13, 30), 1_002_000),
            _mid_row(hms(13, 31), 1_003_000)]
    assert F.count_mid_changes(_frame(rows), HALF) == 1        # regular trading ends at 13:00
    assert F.count_mid_changes(_frame(rows), FULL) == 3


def test_estimate_delta_t_full_days_equals_mean() -> None:
    """On full days Δt is 2.34e7 over the mean count, to the last bit."""
    days = tuple(datetime.date(2025, 12, d) for d in (8, 9, 10))
    tw = F.estimate_delta_t([_changes(1), _changes(1), _changes(3)], days, [FULL] * 3)
    mean = float(np.mean([1, 1, 3]))
    assert tw.mid_changes_per_day == mean and tw.delta_t_ms == 2.34e7 / mean and tw.train_dates == days


def test_estimate_delta_t_mixed_group() -> None:
    """A half day enters with its own 3.5 hours: Δt = total regular time / total count."""
    days = (datetime.date(2025, 11, 28), datetime.date(2025, 12, 8))
    tw = F.estimate_delta_t([_changes(4), _changes(2)], days, [HALF, FULL])
    assert tw.delta_t_ms == pytest.approx((1.26e7 + 2.34e7) / 6, rel=1e-15)
    assert tw.mid_changes_per_day == pytest.approx(6 / (3.6e7 / 2.34e7), rel=1e-15)
    with pytest.raises(ValueError, match="sessions"):
        F.estimate_delta_t([_changes(4), _changes(2)], days, [FULL])


def test_minute_buckets_half_day_has_210() -> None:
    frame = _frame([_mid_row(hms(9, 30, 5), 1_000_000)])
    b = F.minute_buckets(frame, session=HALF)
    assert len(b) == 210 and b["bucket_end_ns"].iloc[-1] == hms(13, 0)
    assert len(F.minute_buckets(frame)) == 390


def test_session_for() -> None:
    half_days = {"2025-11-28": "13:00"}
    assert session_for("2025-11-28", half_days) == HALF
    assert session_for("2025-12-08", half_days) == FULL


def test_invalid_session_raises() -> None:
    with pytest.raises(ValueError, match="invalid session"):
        Session(open_ns=hms(9, 30), close_ns=hms(9, 45))       # the trimmed session would end before it starts
    with pytest.raises(ValueError, match="invalid session"):
        Session(margin_ns=0)
