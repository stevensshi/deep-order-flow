"""The memory side of the compact store: free disk, available memory, this process's cgroup, and the one place
where the store's memory is asked for. Linux only: on another system the readers refuse (fail closed).

The other modules call these functions through this module (``memory._allocate(...)``), so a test can replace
them here.
"""

from __future__ import annotations

import mmap
import os

import numpy as np

from dof.deep.data.compact.errors import CompactError


def _free_bytes(path) -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def _mem_available(path: str = "/proc/meminfo") -> int:
    """MemAvailable in bytes. No file or no line: CompactError, so that a load stops (fail closed)."""
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024       # the kernel reports kB
    except OSError as e:
        raise CompactError(f"{path}: cannot be read ({e}); the memory rule cannot be checked") from e
    raise CompactError(f"{path}: no MemAvailable line; the memory rule cannot be checked")


def _cgroup_memory(proc: str = "/proc/self/cgroup", root: str = "/sys/fs/cgroup") -> tuple[str, int | None, int, int, int]:
    """(folder, memory.high, memory.current, inactive_file, active_file) of this process's cgroup, in bytes.
    memory.high is None for ``max``: no limit. inactive_file and active_file are the two lines of memory.stat that
    hold the page cache the kernel can give back; ``file`` also holds tmpfs and shared memory, which it cannot.

    No cgroup v2 entry (cgroup v1), or one of memory.high, memory.current and memory.stat missing, unreadable or
    without its number (a hybrid layout, the root cgroup, a cgroup without the memory controller): CompactError, so
    that a load stops (fail closed). This holds for a cgroup without a limit too: the three files exist together,
    and a load records all of them."""
    try:
        with open(proc) as fh:
            entries = [line.rstrip("\n") for line in fh if line.startswith("0::")]
    except OSError as e:
        raise CompactError(f"{proc}: cannot be read ({e}); the cgroup rule cannot be checked") from e
    if not entries:
        raise CompactError(f"{proc}: no cgroup v2 entry (a line that starts with 0::); the cgroup rule cannot be checked")
    folder = os.path.join(root, entries[0][len("0::"):].lstrip("/"))
    texts = {}
    for name in ("memory.high", "memory.current", "memory.stat"):
        path = os.path.join(folder, name)
        try:
            with open(path) as fh:
                texts[name] = fh.read()
        except OSError as e:
            raise CompactError(f"{path}: cannot be read ({e}); the cgroup rule cannot be checked") from e
    values: dict[str, int | None] = {}
    for name in ("memory.high", "memory.current"):
        text = texts[name].strip()
        if name == "memory.high" and text == "max":
            values[name] = None
        elif text.isascii() and text.isdigit():
            values[name] = int(text)
        else:
            raise CompactError(f"{os.path.join(folder, name)}: holds {text!r}, not a number; the cgroup rule cannot be checked")
    stat = dict(line.split()[:2] for line in texts["memory.stat"].splitlines() if len(line.split()) >= 2)
    for name in ("inactive_file", "active_file"):
        text = stat.get(name, "")
        if not (text.isascii() and text.isdigit()):
            raise CompactError(f"{os.path.join(folder, 'memory.stat')}: no line {name} with a number; the cgroup rule "
                               "cannot be checked")
        values[name] = int(text)
    return folder, values["memory.high"], values["memory.current"], values["inactive_file"], values["active_file"]


def _allocate(nbytes: int) -> np.ndarray:
    """The only place where the store's memory is asked for: ``nbytes`` of private anonymous memory that the kernel
    never backs with 2 MB pages (MADV_NOHUGEPAGE), as a writable uint8 array.

    Why not numpy's own allocation: with transparent huge pages set to ``madvise``, numpy asks for them on a large
    array, and the kernel then reclaims and compacts memory at each page fault to find 2 MB blocks; in a virtual
    machine short of such blocks that stalls the whole system. A decoded block is 35 kB, so 4 kB pages cost nothing
    that can be measured. Every other array of a fit keeps numpy's default.

    The array is a view of the mapping and holds it (``array.base.obj`` is the ``mmap.mmap``); the memory goes back
    to the system when the last view of the array is gone. A forked process shares the pages copy-on-write, and the
    store only reads them. Zero bytes: an empty array; there is no page to mark. A platform that cannot mark the
    memory: CompactError; the store is not put on other memory in its place."""
    if nbytes == 0:
        return np.empty(0, np.uint8)
    mapping = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    try:
        mapping.madvise(mmap.MADV_NOHUGEPAGE)
    except (AttributeError, OSError) as e:
        mapping.close()
        raise CompactError(f"the store's memory cannot be marked MADV_NOHUGEPAGE on this platform "
                           f"({type(e).__name__}: {e}); the store is not allocated") from e
    return np.frombuffer(mapping, np.uint8)
