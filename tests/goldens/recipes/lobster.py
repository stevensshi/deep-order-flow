"""LOBSTER sample golden recipe (G5): both reconciliation modes on the 10-level and the 50-level sample."""

from __future__ import annotations

from pathlib import Path

from . import hashing as H

LEVELS = (10, 50)
MODES = ("continuous", "transition")


def _one(paths, what: str) -> Path:
    found = sorted(paths)
    if len(found) != 1:
        raise FileNotFoundError(f"expected one {what}, found {len(found)}")
    return found[0]


def run(b, lobster_dir: Path) -> tuple[dict, dict]:
    """Returns (flat values, {(levels, mode): report dict}). The sample files are found under ``lobster_dir`` by
    their LOBSTER names, ``*_message_<L>.csv`` and ``*_orderbook_<L>.csv``."""
    root = Path(lobster_dir)
    values: dict = {}
    reports: dict = {}
    for levels in LEVELS:
        msg = _one(root.rglob(f"*_message_{levels}.csv"), f"*_message_{levels}.csv")
        book = _one(root.rglob(f"*_orderbook_{levels}.csv"), f"*_orderbook_{levels}.csv")
        day = b.read_lobster(msg, book)
        for mode, fn in zip(MODES, (b.reconcile_continuous, b.reconcile_transition)):
            rep = fn(day).to_dict()
            reports[(levels, mode)] = rep
            flat = dict(rep)
            flat["first_diffs"] = H.json_value(rep["first_diffs"])
            values.update(H.flatten(flat, f"lobster/L{levels}/{mode}"))
        del day
    return values, reports
