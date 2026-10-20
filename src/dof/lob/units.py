"""Units shared by the order-book layers: nanoseconds since midnight and the ITCH price scale."""

from __future__ import annotations

NS = 1_000_000_000
PRICE_SCALE = 10_000


def hms(h: int, m: int, s: int = 0) -> int:
    """Nanoseconds since midnight."""
    return ((h * 60 + m) * 60 + s) * NS
