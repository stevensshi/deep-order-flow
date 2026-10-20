"""Disk guard: a step that writes a lot stops before the guarded volume falls under its floor."""

from __future__ import annotations

import shutil
from pathlib import Path

from dof.deep.errors import DiskFloor


def free_gb(path: Path) -> int:
    return shutil.disk_usage(path).free // 2**30


def check_free(path: Path | None, floor_gb: int, where: str, log=print) -> int | None:
    """Free GiB on ``path``; ``DiskFloor`` below ``floor_gb``. No path: no check, None."""
    if path is None:
        return None
    avail = free_gb(path)
    log(f"disk check {where}: {path} {avail} G available, floor {floor_gb} G")
    if avail < floor_gb:
        raise DiskFloor(f"disk check {where}: {path} has {avail} G available, below {floor_gb} G: stopped")
    return avail
