"""Validation and test stock-days: read whole, one stock at a time, while the caller works on the stock before.

A test day is loaded only through the test gate (``dof.deep.evaluate.gate``): ``load_eval_stock`` and
``iter_eval`` refuse a test part unless every model being scored holds a token the gate issued for this split.
torch is imported inside the functions that put arrays on a device.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dof.deep.cache import layout as L
from dof.deep.config import CachePaths, RunConfig, Split
from dof.deep.errors import CacheError, TestDayRefused

LOSS_ARRAYS = ("zof", "zy", "seq", "labseq", "rows")       # what a validation reading needs
SCORE_ARRAYS = (*LOSS_ARRAYS, "y")                         # scoring adds the labels in dollars


@dataclass
class EvalDay:
    part: str
    date: str
    tag: str
    n_rows: int
    rows: np.ndarray            # int64 sample rows
    arrays: dict

    def arr(self, name: str) -> np.ndarray:
        return self.arrays[name]


@dataclass
class EvalStock:
    """The horizons and the label scaler of the stock's scaling group, and its days."""
    symbol: str
    meta: dict
    scalers: dict
    days: list[EvalDay]

    def n(self) -> int:
        return sum(len(d.rows) for d in self.days)


def eval_stocks(cfg: RunConfig, split: Split, cache: CachePaths, part: str, symbols=None, skip=()) -> list[tuple[str, list[dict]]]:
    """(stock, meta files of its used days of ``part``) in universe order. Meta files only: no array is read.
    ``skip``: (stock, day) pairs left out by a rule applied at scoring (the corporate actions).

    The used stock-days of every day of the part are counted first and must be the cache summary's count: a missing
    or extra stock-day stops the run instead of being skipped."""
    sp = L.summary_path(cache)
    if not sp.exists():
        raise CacheError(f"{sp} missing: the cache build has not finished")
    summary = json.loads(sp.read_text(encoding="utf-8"))
    if part not in summary["parts"]:
        raise CacheError(f"{part} is not a built part of the cache (the summary has {summary['parts']})")
    plan = json.loads(L.plan_path(cache).read_text(encoding="utf-8"))
    for d in split.days(part):
        got = 0
        for s in plan["stocks"]:
            p = L.eval_meta_path(cache, part, s, d)
            got += p.exists() and json.loads(p.read_text(encoding="utf-8"))["status"] == "used"
        try:
            want = summary["days"][d]["used"]
        except KeyError:
            raise CacheError(f"{part} {d}: the cache summary has no count for the day") from None
        if got != want:
            raise CacheError(f"{part} {d}: {got} used stock-days in the cache, the summary counts {want}")
    out = []
    for s in plan["stocks"]:
        if symbols is not None and s not in symbols:
            continue
        metas = []
        for d in split.days(part):
            p = L.eval_meta_path(cache, part, s, d)
            if p.exists() and (s, d) not in skip:
                m = json.loads(p.read_text(encoding="utf-8"))
                if m["status"] == "used":
                    metas.append(m)
        if metas:
            out.append((s, metas))
    return out


def read_whole(path: Path) -> np.ndarray:
    """One sequential read of a whole file; its pages are then given back (the page cache is for the training
    records)."""
    arr = np.load(path)
    if hasattr(os, "posix_fadvise"):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return arr


def _check_test(gate, tokens, split: Split, part: str, models, what: str) -> None:
    if part != "test":
        return
    if gate is None:
        raise TestDayRefused(f"{what}: no fit that has ended with saved weights is being scored")
    gate.check(tokens, split.name, part, models)


def load_eval_stock(cfg: RunConfig, split: Split, cache: CachePaths, part: str, symbol: str, metas: list[dict],
                    names: tuple[str, ...], *, gate=None, tokens=None, models=(), row_cap: int | None = None) -> EvalStock:
    """The stock's days of ``part``. ``row_cap`` cuts every day to its first sample rows (a dry run), after the
    bounds check."""
    _check_test(gate, tokens, split, part, models, f"{symbol}: refused to load a test day")
    if part not in ("valid", "test"):
        raise ValueError(part)
    groups = {m["group"] for m in metas}
    if len(groups) != 1:
        raise AssertionError(f"{symbol}: the days of {part} use more than one scaling group")
    history = cfg.features.history
    days = []
    for m in metas:
        arrays = {n: read_whole(L.eval_arr_path(cache, part, symbol, m["date"], n)) for n in names}
        for n, arr in arrays.items():
            dtype, cols = L.EVAL_ARRAYS[n]
            want = (m["n_samples"],) if n == "rows" else (m["n_rows"],) if cols is None else (m["n_rows"], cols)
            if arr.dtype != np.dtype(dtype) or arr.shape != want:
                raise RuntimeError(f"{symbol} {m['date']} {n}: {arr.dtype} {arr.shape}, expected {dtype} {want}")
        rows = arrays.pop("rows").astype(np.int64)
        # No window may leave its stock-day.
        if rows.min() < history - 1 or rows.max() >= m["n_rows"]:
            raise AssertionError(f"{symbol} {m['date']}: a sample window leaves the day")
        days.append(EvalDay(part, m["date"], f"{part}_{m['date']}", m["n_rows"], rows, arrays))
    scaler_y = L.load_scaler(L.scaler_path(cache, symbol, groups.pop(), "y"))
    stock = EvalStock(symbol, {"k_valid": metas[0]["k_valid"], "h_ms": metas[0]["h_ms"], "delta_t_ms": metas[0]["delta_t_ms"]},
                      {"y": scaler_y}, days)
    if row_cap is not None:
        for day in days:
            day.rows = day.rows[:row_cap]
    return stock


def iter_eval(cfg: RunConfig, split: Split, cache: CachePaths, part: str, names: tuple[str, ...], *, gate=None,
              tokens=None, models=(), symbols=None, skip=(), row_cap: int | None = None):
    """The stocks of ``part`` one at a time, in universe order. The files of the next stock are read while the
    caller works on the present one."""
    stocks = eval_stocks(cfg, split, cache, part, symbols, skip)
    # Refused here too, before a reader thread is started.
    _check_test(gate, tokens, split, part, models, "refused to load the test days")

    def load(i: int):
        return pool.submit(load_eval_stock, cfg, split, cache, part, *stocks[i], names, gate=gate, tokens=tokens,
                           models=models, row_cap=row_cap)
    with ThreadPoolExecutor(1) as pool:
        nxt = load(0) if stocks else None
        for i in range(len(stocks)):
            cur = nxt.result()
            nxt = load(i + 1) if i + 1 < len(stocks) else None
            yield cur


class RamRows:
    """A per-row array held in memory, indexed with a device tensor; the result goes to the device."""

    def __init__(self, arr: np.ndarray, device) -> None:
        self.arr, self.device = arr, device

    def __getitem__(self, idx):
        import torch
        return torch.from_numpy(self.arr[idx.cpu().numpy()]).to(self.device)


@dataclass
class DeviceDay:
    day: EvalDay
    z: object                # (n_rows, d) float32 tensor: the scaled features of the whole day, on the device
    zy: object               # RamRows: the scaled labels (NaN off the sample rows)
    rows: object             # (m,) int64 tensor of sample rows, on the device
    seq: object              # RamRows: the raw event index of each row
    labseq: object           # RamRows: the raw event index of the row holding the label's start price
    target_rows: object = None   # a leak check's shuffled labels: whose label each sample gets


def to_device(stock: EvalStock, device) -> list[DeviceDay]:
    """The stock's days: the feature matrix and the sample rows on the device, labels and event indices in memory."""
    import torch
    return [DeviceDay(day, torch.from_numpy(day.arr("zof")).to(device), RamRows(day.arr("zy"), device),
                      torch.from_numpy(day.rows).to(device), RamRows(day.arr("seq"), device),
                      RamRows(day.arr("labseq"), device)) for day in stock.days]
