from __future__ import annotations

import numpy as np
import pytest

from dof.lob import features as F
from hand_frames import BASE_ASK, BASE_BID, FULL, T10, _frame, _row


def test_of_hand_example() -> None:
    bof = F.order_flow([[10]], [[10]], [[10]], [[13]], is_bid=True)
    aof = F.order_flow([[12]], [[12]], [[12]], [[10]], is_bid=False)
    assert bof[0, 0] == 3 and aof[0, 0] == -2 and (bof - aof)[0, 0] == 5
    # Same example through the flow pipeline (level 1 sizes; prices unchanged).
    kf = F.flow_pipeline(_frame([_row(T10, b1=(BASE_BID, 10), a1=(BASE_ASK, 12)),
                                _row(T10 + 1, b1=(BASE_BID, 13), a1=(BASE_ASK, 10))]), FULL)
    assert kf.of[1, 0] == 3 and kf.of[1, 10] == -2 and kf.ofi[1, 0] == 5
    assert np.isnan(kf.of[0]).all() and kf.of_valid.tolist() == [False, True]


def test_of_conventions() -> None:
    def flow(prev_px, prev_sz, cur_px, cur_sz, is_bid, conv):
        return F.order_flow([[prev_px]], [[prev_sz]], [[cur_px]], [[cur_sz]], is_bid=is_bid, convention=conv)[0, 0]
    # bid price drops 100.00 -> 99.99, size 20 -> 80
    assert flow(1_000_000, 20, 999_900, 80, True, "current_size") == -80
    assert flow(1_000_000, 20, 999_900, 80, True, "cks") == -20
    # ask price rises, size 20 -> 80
    assert flow(1_000_000, 20, 1_000_100, 80, False, "current_size") == -80
    assert flow(1_000_000, 20, 1_000_100, 80, False, "cks") == -20
    # bid rises / ask falls: +current size under both conventions
    for conv in ("current_size", "cks"):
        assert flow(1_000_000, 20, 1_000_100, 80, True, conv) == 80
        assert flow(1_000_000, 20, 999_900, 80, False, conv) == 80
    with pytest.raises(ValueError):
        flow(1, 1, 1, 1, True, "nope")

