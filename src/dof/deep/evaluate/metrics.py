"""Per-stock summaries of score cells."""

from __future__ import annotations

import numpy as np


def stock_values(rows: list[dict], ks=None, dates=None) -> dict[str, float]:
    """Per stock, the mean of its cells (days x horizons). A cell without an R² (labels of zero variance) is left
    out of the mean."""
    cells: dict[str, list[float]] = {}
    for r in rows:
        if (ks is None or r["k"] in ks) and (dates is None or r["date"] in dates) and np.isfinite(r["r2_os"]):
            cells.setdefault(r["symbol"], []).append(r["r2_os"])
    return {s: float(np.mean(v)) for s, v in cells.items()}
