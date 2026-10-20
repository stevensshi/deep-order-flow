"""Comparison of flat golden values."""

from __future__ import annotations


def _cut(v) -> str:
    s = repr(v)
    return s if len(s) <= 80 else s[:77] + "..."


def _same(a, b) -> bool:
    # The type is part of the value: True and 1 are not the same golden value.
    return type(a) is type(b) and a == b


def diff(expected: dict, got: dict) -> list[str]:
    """Sorted lines "missing k", "extra k" and "differs k: expected=... got=..." (values cut to 80 characters)."""
    out = []
    for k in sorted(set(expected) | set(got)):
        if k not in got:
            out.append(f"missing {k}")
        elif k not in expected:
            out.append(f"extra {k}")
        elif not _same(expected[k], got[k]):
            out.append(f"differs {k}: expected={_cut(expected[k])} got={_cut(got[k])}")
    return out
