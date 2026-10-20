"""The scaling-group rule: which training days of a stock share one Δt and one pair of scalers.

A group is a run of consecutive held training days of one stock. A new group starts when the gap to the stock's
previous held day is longer than ``features.gap_days``, or when the ratio of the two days' median mid prices lies
outside ``features.ratio_range`` (a split). Pure functions, no I/O.
"""

from __future__ import annotations

import datetime

import numpy as np

from dof.deep.config import RunConfig
from dof.deep.splits import regular_ms
from dof.lob.features import DELTA_T_NUMERATOR_MS


def continuous(cfg: RunConfig, mid_before: float, mid_now: float) -> bool:
    """No price discontinuity between two days: the ratio of the median mid prices lies inside the range."""
    return cfg.features.ratio_lo <= mid_now / mid_before <= cfg.features.ratio_hi


def scaling_groups(cfg: RunConfig, days: list[dict]) -> list[list[dict]]:
    """``days``: one stock's held training days in date order, each with ``date`` and ``median_mid``."""
    groups: list[list[dict]] = []
    for d in days:
        if groups:
            prev = groups[-1][-1]
            gap = (datetime.date.fromisoformat(d["date"]) - datetime.date.fromisoformat(prev["date"])).days
            if gap <= cfg.features.gap_days and continuous(cfg, prev["median_mid"], d["median_mid"]):
                groups[-1].append(d)
                continue
        groups.append([d])
    return groups


def group_delta_t(cfg: RunConfig, counts: list[int], dates: list[str]) -> tuple[float, float | None]:
    """(mid-price changes per full day of regular trading, Δt in ms). Δt is the total regular trading time of the
    group's days over their total count of mid-price changes; a half day enters with its own trading time. The
    number of full days the group is worth is its total time over 2.34e7 ms, an exact integer when every day is
    full, so ``per_day`` is then the mean count and Δt the same quotient as ``estimate_delta_t``. Δt is None when
    there is no change at all."""
    full_days = sum(regular_ms(cfg, d) for d in dates) / DELTA_T_NUMERATOR_MS
    per_day = float(np.sum(counts)) / full_days
    return per_day, (DELTA_T_NUMERATOR_MS / per_day if per_day > 0 else None)
