"""The trading session of a day of a run: a full day, or a half day listed in the configuration's calendar."""

from __future__ import annotations

from dof.deep.config import RunConfig
from dof.lob.features import Session, session_for


def session_of(cfg: RunConfig, date: str) -> Session:
    """The session of ``date`` (ISO): 09:40 to 15:50, or ten minutes before the early close on a half day."""
    return session_for(date, cfg.half_days)


def regular_ms(cfg: RunConfig, date: str) -> float:
    """Regular trading time of ``date`` in ms: 2.34e7 on a full day, 1.26e7 on a half day closing at 13:00."""
    return session_of(cfg, date).regular_ms
