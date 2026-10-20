"""Windows of consecutive rows: the one definition of a sample's window, and the reads of record windows."""

from __future__ import annotations

import os

import numpy as np

from dof.deep.cache.layout import REC


def window_index(rows: np.ndarray, history: int, shift: int = 0) -> np.ndarray:
    """(len(rows), history) row indices of the windows of sample ``rows``: rows t-history+1..t, oldest first.
    ``shift`` is for the leak checks only: +1 moves the window one row into the future."""
    return np.asarray(rows, dtype=np.int64)[:, None] + np.arange(-history + 1, 1, dtype=np.int64)[None, :] + shift


def read_windows(day, first: np.ndarray, width: int) -> np.ndarray:
    """(len(first), width) records of a training stock-day's record file: for each entry the ``width``
    consecutive rows from row ``first``. One positioned read per window, straight into one buffer."""
    if len(first) and (int(first.min()) < 0 or int(first.max()) + width > day.n_rows):
        raise AssertionError(f"{day.symbol} {day.date}: a window leaves its stock-day")
    size = REC.itemsize
    n = width * size
    buf = bytearray(len(first) * n)
    view = memoryview(buf)
    starts = first.tolist()
    # Ask for all windows of the batch before the first is read: the reads that miss the page cache then wait on
    # the disk together, not one after the other.
    if hasattr(os, "posix_fadvise"):
        for t in starts:
            os.posix_fadvise(day.fd, day.offset + t * size, n, os.POSIX_FADV_WILLNEED)
    for i, t in enumerate(starts):
        if hasattr(os, "preadv"):
            got = os.preadv(day.fd, [view[i * n:(i + 1) * n]], day.offset + t * size)
        else:
            data = os.pread(day.fd, n, day.offset + t * size)
            view[i * n:i * n + len(data)] = data
            got = len(data)
        if got != n:
            raise OSError(f"{day.path}: short read")
    return np.frombuffer(buf, dtype=REC).reshape(len(first), width)


def day_windows(day, first: np.ndarray, width: int) -> np.ndarray:
    """``read_windows`` for a day read from its record file; the same records from memory for a day that has a
    compact store. The two give the same bytes."""
    if day.store is None:
        return read_windows(day, first, width)
    return day.store.windows(day.slot, first, width)
