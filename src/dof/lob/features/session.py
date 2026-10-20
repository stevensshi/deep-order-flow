"""Trading sessions: the regular trading hours of a day and the trimmed session inside them.

A full day trades 09:30 to 16:00; a half day (the day after Thanksgiving, for example) closes early. The trimmed
session drops the first and the last ten minutes of regular trading (Kolm, Turiel and Westray 2023, §3.2), so on a half
day that closes at 13:00 it ends at 12:50. Δt counts each day with its own regular trading time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from dof.lob.units import NS, hms


def _hhmm_ns(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return hms(int(h), int(m))


@dataclass(frozen=True)
class Session:
    """Regular trading from ``open_ns`` to ``close_ns`` (nanoseconds since midnight); the trimmed session runs from
    ``start_ns`` to ``end_ns``, ``margin_ns`` inside each end."""
    open_ns: int = hms(9, 30)
    close_ns: int = hms(16, 0)
    margin_ns: int = 10 * 60 * NS

    def __post_init__(self) -> None:
        if not (self.open_ns < self.start_ns < self.end_ns < self.close_ns):
            raise ValueError(f"invalid session: needs open < start < end < close, got {self.open_ns}, {self.start_ns}, "
                             f"{self.end_ns}, {self.close_ns}")

    @property
    def start_ns(self) -> int:
        return self.open_ns + self.margin_ns

    @property
    def end_ns(self) -> int:
        return self.close_ns - self.margin_ns

    @property
    def regular_ms(self) -> float:
        """Regular trading time in ms: 2.34e7 on a full day (the numerator of Kolm, Turiel and Westray 2023, eq. 16)."""
        return (self.close_ns - self.open_ns) / 1e6

    @classmethod
    def full_day(cls) -> Session:
        return cls()

    @classmethod
    def half_day(cls, close: str = "13:00") -> Session:
        return cls(close_ns=_hhmm_ns(close))


def session_for(date: str, half_days: Mapping[str, str]) -> Session:
    """The session of ``date`` (ISO): a half day when ``half_days`` maps the date to its close ("HH:MM"), else a
    full day."""
    close = half_days.get(date)
    return Session.full_day() if close is None else Session.half_day(close)
