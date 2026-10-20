from __future__ import annotations

import datetime
import sys

import numpy as np
import pytest

from dof.lob import features as F
from dof.lob.book import FLAG_CROSSED
from dof.lob.features import hms
from hand_frames import BASE_ASK, BASE_BID, FULL, T10, _frame, _mid_row, _row


def test_history_valid_needs_w_rows() -> None:
    rows = [_row(T10 + i) for i in range(5)] + [_row(T10 + 5, flags=FLAG_CROSSED)] + [_row(T10 + 6 + i) for i in range(4)]
    kf = F.flow_pipeline(_frame(rows), FULL)
    # Row 0 of a segment has no OF; W=3 needs three of_valid rows, so 0-based row 3 is the first sample.
    assert F.history_valid(kf, 3).tolist() == [False, False, False, True, True, False, False, False, True]
    assert F.history_valid(kf, 1).tolist() == [False, True, True, True, True, False, True, True, True]


def test_count_mid_changes_window() -> None:
    rows = [
        _mid_row(hms(9, 29), 1_000_000),
        _mid_row(hms(9, 31), 1_001_000),
        _mid_row(hms(9, 32), 1_002_000),
        _mid_row(hms(9, 33), 1_009_000, flags=FLAG_CROSSED),
        _mid_row(hms(9, 34), 1_003_000),
        _mid_row(hms(9, 35), 1_003_000),
        _mid_row(hms(16, 0), 1_004_000),
    ]
    # In-window same-segment changes: 9:31 -> 9:32 only (9:32 -> 9:34 crosses a break).
    assert F.count_mid_changes(_frame(rows), FULL) == 1


def test_delta_t_train_only() -> None:
    one = _frame([_mid_row(hms(10, 0), 1_000_000), _mid_row(hms(10, 1), 1_001_000)])
    three = _frame([_mid_row(hms(10, i), 1_000_000 + 1_000 * (i % 2)) for i in range(4)])
    assert (F.count_mid_changes(one, FULL), F.count_mid_changes(three, FULL)) == (1, 3)
    days = (datetime.date(2025, 12, 8), datetime.date(2025, 12, 9))
    tw = F.estimate_delta_t([one, three], days, [FULL, FULL])
    assert tw.delta_t_ms == 2.34e7 / 2.0 and tw.mid_changes_per_day == 2.0 and tw.train_dates == days
    flat = _frame([_mid_row(hms(10, 0), 1_000_000), _mid_row(hms(10, 1), 1_000_000)])
    with pytest.raises(ValueError, match="no mid changes"):
        F.estimate_delta_t([flat], days[:1], [FULL])


# -- pins: each fails if the behaviour named in its docstring changes --------------------------------


def test_flow_chunk_seam(monkeypatch) -> None:
    """Order flow computed in chunks of 3 rows equals the one-chunk result (each chunk reads the row before it)."""
    rows = [_row(T10 + i, b1=(BASE_BID, 100 + 7 * i), a1=(BASE_ASK, 300 - 3 * i)) for i in range(30)]
    rows.append(_row(T10 + 30, flags=FLAG_CROSSED))
    rows += [_row(T10 + 31 + i, b1=(BASE_BID, 50 + 5 * i), a1=(BASE_ASK, 200 + 2 * i)) for i in range(29)]
    frame = _frame(rows)
    ref = F.flow_pipeline(frame, FULL)
    monkeypatch.setattr(sys.modules[F.flow_pipeline.__module__], "OF_CHUNK_ROWS", 3)
    small = F.flow_pipeline(frame, FULL)
    assert len(frame) == 60 and ref.segment.tolist().count(1) == 29
    for name in ("of", "ofi", "of_valid"):
        np.testing.assert_array_equal(getattr(small, name), getattr(ref, name))
