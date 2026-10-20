"""The primitives of every score table: stock values, their mean and standard error, paired differences, the
price-ratio filter and the kind of a cell without an R².

A stock's value is the mean of its cells with a finite R² (days x horizons); a table value is the mean over stocks;
the standard error is taken across stocks. The statement order, the numpy calls and the iteration order are those of
the reference implementation, so every float equals the stored one bit for bit (numpy's pairwise summation makes
the last bits of a mean depend on the order of its inputs).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from dof.experiments.errors import AnalysisError

if TYPE_CHECKING:
    import pandas as pd

    from dof.experiments.cells import Rows


def stock_values(rows: Rows, ks=None, dates=None) -> dict[str, float]:
    """Per stock, in the order of first appearance: the mean of its cells with a finite R²."""
    cells: dict[str, list[float]] = {}
    for symbol, date, k, r2 in zip(rows.symbol, rows.date, rows.k, rows.r2):
        if (ks is None or k in ks) and (dates is None or date in dates) and np.isfinite(r2):
            cells.setdefault(symbol, []).append(r2)
    return {s: float(np.mean(v)) for s, v in cells.items()}


def agg(rows: Rows, ks=None, dates=None) -> dict:
    """Mean over stocks of the stock values, its standard error across stocks, and the counts behind it."""
    v = stock_values(rows, ks, dates)
    sel = [r2 for date, k, r2 in zip(rows.date, rows.k, rows.r2) if (ks is None or k in ks) and (dates is None or date in dates)]
    x = np.array(list(v.values()))
    return {"mean": float(x.mean()) if len(x) else float("nan"),
            "se": float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan"), "n_stocks": len(x),
            "n_cells": len(sel), "n_no_r2": sum(not np.isfinite(r2) for r2 in sel)}


def paired(a: Rows, b: Rows, ks=None, dates=None) -> dict:
    """Stock value under a minus stock value under b, over the stocks scored under both (sorted by name)."""
    va, vb = stock_values(a, ks, dates), stock_values(b, ks, dates)
    common = sorted(set(va) & set(vb))
    d = np.array([va[s] - vb[s] for s in common])
    return {"mean": float(d.mean()) if len(d) else float("nan"),
            "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
            "n_stocks": len(d), "n_better": int((d > 0).sum())}


def inside(rows: Rows, stock_days: pd.DataFrame) -> Rows:
    """The cells whose stock-day has its price ratio inside [ratio_lo, ratio_hi] (both ends kept): the secondary
    analysis. A cell whose stock-day has no ratio raises AnalysisError."""
    scored = stock_days[stock_days["status"] == "scored"]
    ratio = {(s, d): (r, lo, hi) for s, d, r, lo, hi in zip(scored["symbol"], scored["date"], scored["price_ratio"],
                                                            scored["ratio_lo"], scored["ratio_hi"])}
    keep = []
    for symbol, date in zip(rows.symbol, rows.date):
        try:
            r, lo, hi = ratio[(symbol, date)]
        except KeyError:
            raise AnalysisError(f"{symbol} {date}: no price ratio for a scored cell") from None
        keep.append(lo <= r <= hi)
    return rows.take(keep)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def finite_kind(r2, sse, sst) -> str:
    """finite: R², sse and sst finite, sst above 0. zero_variance: sst exactly 0, sse finite, R² NaN (labels of zero
    variance). not_finite: sst above 0 and R² or sse not finite (a forecast that is not finite). other: the rest."""
    if not _finite(sst) or sst < 0:
        return "other"
    if sst == 0:
        return "zero_variance" if _finite(sse) and isinstance(r2, float) and math.isnan(r2) else "other"
    return "finite" if _finite(r2) and _finite(sse) else "not_finite"
