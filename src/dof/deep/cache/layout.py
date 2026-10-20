"""Layout of the feature cache of a split.

A training stock-day is ONE file of fixed-size records, one record per cleaned row: the z-scored order flow, the
labels in dollars as float32, and the event indices of the row and of the row that holds the label's start price.
The fields of a row sit side by side, so one read of 100 consecutive records gives a sample's window, its label
and both event indices. ``rows`` (int32, its own file) lists the sample rows. A validation or test stock-day is a
set of whole arrays (``EVAL_ARRAYS``), read in one piece.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dof.deep.config import CachePaths
from dof.lob.features import Scaler

REC = np.dtype([("zof", "<f4", (20,)), ("y", "<f4", (10,)), ("seq", "<i8"), ("labseq", "<i8")])
EVAL_ARRAYS = {"zof": ("float32", 20), "zy": ("float32", 10), "y": ("float64", 10), "seq": ("int64", None),
               "labseq": ("int64", None), "rows": ("int32", None)}


def plan_path(c: CachePaths) -> Path:
    return c.train / "plan.json"


def probe_path(c: CachePaths, symbol: str, date: str) -> Path:
    return c.train / "probe" / f"{symbol}_{date}.json"


def group_path(c: CachePaths, symbol: str, gid: int) -> Path:
    return c.train / symbol / f"g{gid}.json"


def scaler_path(c: CachePaths, symbol: str, gid: int, key: str) -> Path:
    return c.train / symbol / f"g{gid}_scaler_{key}.npz"


def rec_path(c: CachePaths, symbol: str, date: str) -> Path:
    return c.train / symbol / f"train_{date}.rec.npy"


def rows_path(c: CachePaths, symbol: str, date: str) -> Path:
    return c.train / symbol / f"train_{date}.rows.npy"


def eval_meta_path(c: CachePaths, part: str, symbol: str, date: str) -> Path:
    return c.eval / part / symbol / f"{date}.json"


def eval_arr_path(c: CachePaths, part: str, symbol: str, date: str, name: str) -> Path:
    return c.eval / part / symbol / f"{date}_{name}.npy"


def summary_path(c: CachePaths) -> Path:
    return c.train / "summary.json"


def save_scaler(path: Path, sc: Scaler) -> None:
    np.savez(path, lo=sc.lo, hi=sc.hi, mean=sc.mean, std=sc.std, zero_std_cols=np.array(sc.zero_std_cols, dtype=np.int64))


def load_scaler(path: Path) -> Scaler:
    z = np.load(path)
    return Scaler(lo=z["lo"], hi=z["hi"], mean=z["mean"], std=z["std"], zero_std_cols=tuple(int(i) for i in z["zero_std_cols"]))
