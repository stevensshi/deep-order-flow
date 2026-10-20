"""The scaling-group rule and the group's Δt."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from dof.deep.cache.groups import group_delta_t, scaling_groups
from dof.deep.config import load

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.delenv("DOF_LOCAL_CONFIG", raising=False)
    return load(ROOT / "configs" / "smoke.toml")


def _day(date: datetime.date, mid: float) -> dict:
    return {"date": date.isoformat(), "median_mid": mid}


def test_gap_rule(cfg) -> None:
    d0 = datetime.date(2019, 1, 30)
    same = [_day(d0, 10.0), _day(d0 + datetime.timedelta(days=300), 10.0)]
    assert [len(g) for g in scaling_groups(cfg, same)] == [2]
    apart = [_day(d0, 10.0), _day(d0 + datetime.timedelta(days=301), 10.0)]
    assert [len(g) for g in scaling_groups(cfg, apart)] == [1, 1]


def test_ratio_rule(cfg) -> None:
    d0, d1 = datetime.date(2019, 1, 30), datetime.date(2019, 3, 27)
    for mid in (60.0, 160.0):                       # ratios 0.6 and 1.6: the ends of the range stay inside
        assert len(scaling_groups(cfg, [_day(d0, 100.0), _day(d1, mid)])) == 1
    for mid in (59.99, 160.01):                     # 0.5999 and 1.6001: a new group
        assert len(scaling_groups(cfg, [_day(d0, 100.0), _day(d1, mid)])) == 2


def test_group_delta_t_full_days_is_the_mean(cfg) -> None:
    counts = [1001, 2002, 3004]
    per_day, delta_t = group_delta_t(cfg, counts, ["2019-01-30", "2019-03-27", "2019-07-30"])
    assert per_day == float(sum(counts)) / 3
    assert delta_t == 2.34e7 / per_day


def test_group_delta_t_half_day(cfg) -> None:
    counts = [1000, 500]
    per_day, delta_t = group_delta_t(cfg, counts, ["2025-11-26", "2025-11-28"])     # 2025-11-28 closes at 13:00
    assert per_day == float(sum(counts)) / ((2.34e7 + 1.26e7) / 2.34e7)
    assert delta_t == 2.34e7 / per_day


def test_group_delta_t_no_change_gives_none(cfg) -> None:
    assert group_delta_t(cfg, [0, 0], ["2019-01-30", "2019-03-27"]) == (0.0, None)
