"""Writers that leave either the old file or the whole new one: each writes ``<name>.tmp`` and renames it."""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import numpy as np


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def write_json_atomic(path: Path, doc, *, indent: int | None = None) -> None:
    """Cache files are written with ``indent=1``, records with no indent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=indent) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, np.ascontiguousarray(arr))
    os.replace(tmp, path)


def torch_save_atomic(path: Path, obj) -> None:
    """The only writer of ``.pt`` files. torch names the records inside the zip archive after the file it writes
    to, minus the last suffix, so ``<name>.tmp`` gives the archive name ``<name>``: the bytes of the file do not
    depend on the folder or on a random temporary name."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
