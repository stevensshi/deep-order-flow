"""Hand-built state frames for the feature tests (shared by several test modules; not a test module)."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from dof.lob.book import FLAG_HALTED
from dof.lob.features import Session, TrainWindow, hms

MS = 1_000_000
BASE_BID, BASE_ASK = 1_000_000, 1_000_100
T10 = hms(10, 0)
FULL = Session.full_day()


def _row(ts: int, *, mt: str = "A", flags: int = 0, bid: list[tuple[int, int]] | None = None,
         ask: list[tuple[int, int]] | None = None, b1: tuple[int, int] | None = None,
         a1: tuple[int, int] | None = None, ct: int = 1) -> dict:
    """One state row. Default book: bid 1,000,000 - 100k, ask 1,000,100 + 100k, 100 shares each."""
    bid = list(bid or [(BASE_BID - 100 * k, 100) for k in range(10)])
    ask = list(ask or [(BASE_ASK + 100 * k, 100) for k in range(10)])
    if b1:
        bid[0] = b1
    if a1:
        ask[0] = a1
    r = {"seq": 0, "ts_ns": ts}
    for side, lv in (("bid", bid), ("ask", ask)):
        for k, (px, sz) in enumerate(lv, 1):
            r[f"{side}_px_{k}"] = px
            r[f"{side}_sz_{k}"] = sz
            r[f"{side}_ct_{k}"] = ct
    r.update(flags=flags, msg_type=mt, trading_state="H" if flags & FLAG_HALTED else "T")
    return r


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["seq"] = np.arange(len(df))
    return df


def _mid_row(ts: int, mid_1e4: int, **kw) -> dict:
    return _row(ts, b1=(mid_1e4 - 100, 100), a1=(mid_1e4 + 100, 100), **kw)


def _tw(delta_t_ms: float) -> TrainWindow:
    return TrainWindow((datetime.date(2025, 12, 8),), 2.34e7 / delta_t_ms, delta_t_ms)
