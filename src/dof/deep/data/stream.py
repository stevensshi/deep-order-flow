"""The training stream: the stock-days of a fit, the random order of a pass, and the batches read in that order.

- ``draw_pass`` makes, from the fit's generator, a permutation of the samples of every stock-day, then a
  permutation of all batches. A batch holds ``batch_size`` samples of ONE stock-day (the last batch of a
  stock-day may be smaller).
- ``TrainStream`` is one pass as a map-style dataset: item ``b`` is batch ``b`` of that order, so the batches reach
  the fit in the drawn order whatever the loader processes do.
- A sample is one read of ``history`` consecutive records: its window, its label and both event indices. The
  label and the label's start index come from the window's LAST record, so a window and its label cannot come from
  different rows. Each batch counts the windows whose last row is a later event than the row holding the label's
  start price (``late``), the windows that are not consecutive rows of one day (``broken``), and whether it holds a
  value that is not finite; the fit stops on any of them.

torch is imported inside the functions that need it, never at module level.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dof.deep.cache import layout as L
from dof.deep.config import CachePaths, RunConfig, Split
from dof.deep.data.store import TrainStore, load_store
from dof.deep.data.windows import day_windows
from dof.deep.errors import ConfigError
from dof.lob.features import Scaler, apply_scaler


@dataclass
class TrainDay:
    symbol: str
    date: str
    group: int
    n_rows: int
    rows: np.ndarray            # int32 sample rows, ascending
    scaler_y: Scaler            # the label scaler of the stock's scaling group
    path: Path
    offset: int                 # byte offset of record 0 in the .npy file
    fd: int
    store: TrainStore | None = None     # the compact store this day's records are read from; None: the record file
    slot: int = -1                      # the day's index in that store


def train_days(cfg: RunConfig, split: Split, which: str, cache: CachePaths, symbols: set[str] | None = None, *,
               compact: bool | None = None) -> list[TrainDay]:
    """The training stock-days of ``which`` ("all" or "recent"): stocks in universe order, days in date order.
    Each record file is opened once; a loader process reads through the same descriptor (a positioned read has no
    file position). ``compact``: None takes ``machine.compact_store``."""
    plan = json.loads(L.plan_path(cache).read_text(encoding="utf-8"))
    want = set(split.days(which))
    history = cfg.features.history
    use_store = cfg.machine.compact_store if compact is None else compact
    ts = TrainStore(cache, cfg.machine) if use_store else None
    out = []
    for s, x in plan["stocks"].items():
        if symbols is not None and s not in symbols:
            continue
        for g in x["groups"]:
            gm = json.loads(L.group_path(cache, s, g["id"]).read_text(encoding="utf-8")) if g["status"] == "ok" else g
            if gm["status"] != "ok":
                continue
            sc_y = L.load_scaler(L.scaler_path(cache, s, g["id"], "y"))
            for r in gm["day_rows"]:
                if r["date"] not in want or r["n_samples"] == 0:
                    continue
                path = L.rec_path(cache, s, r["date"])
                mm = np.load(path, mmap_mode="r")
                if mm.dtype != L.REC or mm.shape != (r["n_rows"],):
                    raise RuntimeError(f"{path}: not {r['n_rows']} records of the cache layout")
                offset = mm.offset
                del mm
                rows = np.load(L.rows_path(cache, s, r["date"]))
                if rows.dtype != np.int32 or len(rows) != r["n_samples"]:
                    raise RuntimeError(f"{path}: the sample-row file does not match the group record")
                # No window may leave its stock-day: the window of sample row t is rows t-history+1..t of THIS file.
                if rows.min() < history - 1 or rows.max() >= r["n_rows"]:
                    raise AssertionError(f"{s} {r['date']}: a sample window leaves the day")
                fd = os.open(path, os.O_RDONLY)
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)       # windows are read at random: no read-ahead
                out.append(TrainDay(s, r["date"], g["id"], r["n_rows"], rows, sc_y, path, offset, fd, ts,
                                    -1 if ts is None else ts.add(path, r["n_rows"])))
    return out


def close_days(days: list[TrainDay]) -> None:
    for d in days:
        os.close(d.fd)


@dataclass
class PassPlan:
    order: list[np.ndarray]                 # per stock-day: its sample rows in this pass's order (int32)
    label_rows: list[np.ndarray] | None     # placebo only: the row whose label each sample gets
    batches: np.ndarray                     # (n_batches, 3) int64: stock-day, start, end in ``order``


def draw_targets(days: list[TrainDay], gen) -> list[np.ndarray]:
    """Placebo "shuffled labels": each sample gets the label of another sample row of the same day. Drawn once,
    before the first pass, from the fit's generator."""
    import torch
    return [day.rows[torch.randperm(len(day.rows), generator=gen).numpy()] for day in days]


def draw_pass(days: list[TrainDay], gen, targets: list[np.ndarray] | None = None, *, batch_size: int) -> PassPlan:
    """The draws of one pass from ``gen`` (the fit's ``torch.Generator``), in a fixed order of generator calls."""
    import torch
    order, label_rows, table = [], [], []
    for i, day in enumerate(days):
        m = len(day.rows)
        perm = torch.randperm(m, generator=gen).numpy()
        order.append(day.rows[perm])
        if targets is not None:
            label_rows.append(targets[i][perm])
        starts = np.arange(0, m, batch_size, dtype=np.int64)
        table.append(np.stack([np.full(len(starts), i, dtype=np.int64), starts, np.minimum(starts + batch_size, m)], axis=1))
    batches = np.concatenate(table)
    batches = batches[torch.randperm(len(batches), generator=gen).numpy()]
    return PassPlan(order, label_rows if targets is not None else None, batches)


def steps_per_pass(days: list[TrainDay], batch_size: int) -> int:
    return sum(-(-len(d.rows) // batch_size) for d in days)


class TrainStream:
    """One pass over the training set, as a map-style dataset: item ``b`` is batch ``b`` of the pass."""

    def __init__(self, days: list[TrainDay], plan: PassPlan, history: int) -> None:
        self.days, self.plan, self.history = days, plan, history
        load_store(days)        # a compact store is in memory before a loader process is forked; once per store

    def __len__(self) -> int:
        return len(self.plan.batches)

    def __getitem__(self, b: int) -> dict:
        di, a, e = (int(v) for v in self.plan.batches[b])
        day = self.days[di]
        t = self.plan.order[di][a:e].astype(np.int64)
        rec = day_windows(day, t - (self.history - 1), self.history)       # rows t-history+1..t: window_index
        seq = rec["seq"]
        late = int((seq[:, -1] > rec["labseq"][:, -1]).sum())
        broken = int((np.diff(seq, axis=1) <= 0).any(axis=1).sum())
        x = np.ascontiguousarray(rec["zof"])
        if self.plan.label_rows is None:
            y = rec["y"][:, -1]                                       # the label of sample row t
        else:
            y = day_windows(day, self.plan.label_rows[di][a:e].astype(np.int64), 1)["y"][:, 0]
        # z-scored with the label scaler of the stock's scaling group, held in float32 as the stored labels
        zy = apply_scaler(y.astype(np.float64), day.scaler_y).astype(np.float32)
        bad = int(not (np.isfinite(x).all() and np.isfinite(zy).all()))
        return {"x": x, "zy": zy, "late": late, "broken": broken, "not_finite": bad}


def _worker_init(_worker: int) -> None:
    import torch
    torch.set_num_threads(1)        # a loader process reads and copies; the fitting process keeps the cores


def make_loader(days: list[TrainDay], plan: PassPlan, *, history: int, workers: int, prefetch: int, start: int = 0,
                generator=None):
    """The batches of one pass from batch ``start`` on, in order (a torch DataLoader). ``start`` is where a fit that
    was interrupted continues its pass. Loader processes are forked, so they share the pass's order arrays without a
    copy; they never touch CUDA. ``generator`` (when given) is the DataLoader's own generator; otherwise the
    DataLoader draws its worker seed from the global generator."""
    import torch
    ds = TrainStream(days, plan, history)
    todo = range(start, len(ds))
    extra = {} if generator is None else {"generator": generator}
    if workers == 0:
        return torch.utils.data.DataLoader(ds, batch_size=None, sampler=todo, num_workers=0, **extra)
    if "fork" not in mp.get_all_start_methods():
        raise ConfigError("loader processes need the fork start method, which this platform lacks: set "
                          "machine.loader_workers = 0")
    return torch.utils.data.DataLoader(ds, batch_size=None, sampler=todo, num_workers=workers, prefetch_factor=prefetch,
                                       multiprocessing_context="fork", persistent_workers=False,
                                       worker_init_fn=_worker_init, **extra)
