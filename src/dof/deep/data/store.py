"""The compact backend seen from the training stream: which record files one ``train_days`` call holds and, once
loaded, the store itself (``dof.deep.data.compact``).

``load`` runs in the fitting process before a loader process is forked. A loader process inherits the loaded store
and never loads one: ``windows`` refuses. A compact file that is missing or is not the one of its manifest stops
the fit; the record files are not read in its place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dof.deep.cache import layout as L
from dof.deep.config import CachePaths, MachineConfig


class TrainStore:
    def __init__(self, cache: CachePaths, machine: MachineConfig) -> None:
        self.root, self.train_cache = cache.compact_root, cache.train
        self.context = {p.name: p for p in (L.plan_path(cache), L.summary_path(cache))}
        self.min_available_after = machine.compact_min_available_after
        self.min_cgroup_room_after = machine.compact_min_cgroup_room_after
        self.sources: list[Path] = []
        self.rows: list[int] = []
        self.store = None
        self.facts: dict | None = None

    def add(self, path: Path, n_rows: int) -> int:
        self.sources.append(path)
        self.rows.append(n_rows)
        return len(self.sources) - 1

    def load(self, log=print) -> dict:
        if self.store is None:
            from dof.deep.data import compact as CS        # here: only a process that loads a store needs zstandard
            keys = [CS.key_of(p.relative_to(self.train_cache)) for p in self.sources]
            self.store = CS.CompactStore.load(self.root, keys, context=self.context, dtype=L.REC, rows=self.rows,
                                              sources=self.sources, min_available_after=self.min_available_after,
                                              min_cgroup_room_after=self.min_cgroup_room_after)
            f = self.facts = self.store.facts()
            log(f"training records: compact store {self.root}: {f['files']} stock-days, {f['rows']:,} rows, "
                f"{f['compressed_bytes'] / 1e9:.2f} GB in memory for {f['source_bytes'] / 1e9:.1f} GB of record files "
                f"({f['bytes_per_row']:.2f} bytes per row), read and checked in {f['load_seconds']:.0f} s; manifest sha256 "
                f"{f['manifest_sha256']}; cgroup memory.high "
                f"{'max' if f['cgroup_memory_high'] is None else f['cgroup_memory_high']}, memory.current before the load "
                f"{f['cgroup_memory_current_before']}")
        return self.facts

    def windows(self, slot: int, first: np.ndarray, width: int) -> np.ndarray:
        if self.store is None:
            raise RuntimeError("the compact store is not in memory: load_store must run before a batch is asked for")
        return self.store.windows(slot, first, width)


def load_store(days: list, log=print) -> dict | None:
    """Puts the compact store of ``days`` into this process's memory, once, and returns its facts; None when the
    days are read from their record files. It must run before a loader process is forked."""
    stores = {id(d.store): d.store for d in days if d.store is not None}
    if not stores:
        return None
    if len(stores) != 1 or any(d.store is None for d in days):
        raise RuntimeError("the days of one stream do not come from one train_days() call")
    return next(iter(stores.values())).load(log)
