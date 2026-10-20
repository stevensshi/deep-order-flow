"""Memory: hand freed heap back to the system between large steps, read the peak and the available memory."""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import gc
import sys
from pathlib import Path


@functools.lru_cache(maxsize=1)
def _libc():
    name = ctypes.util.find_library("c")
    if name is None:
        return None
    try:
        return ctypes.CDLL(name)
    except OSError:
        return None


def release_memory() -> None:
    """Collect cycles, release pyarrow's unused pool and trim the C heap (glibc only)."""
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:  # noqa: BLE001 — best effort only
        pass
    libc = _libc()
    trim = getattr(libc, "malloc_trim", None) if libc is not None else None
    if trim is not None:
        trim(0)


def peak_rss_kb() -> int | None:
    """Peak resident memory of this process in KiB; None where the ``resource`` module is missing."""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak // 1024 if sys.platform == "darwin" else peak        # bytes on macOS, KiB on Linux


def mem_available_bytes() -> int | None:
    """``MemAvailable`` of /proc/meminfo; None when the file cannot be read."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None
