"""Hashes, canonical forms and the committed form of golden values.

numpy is imported here; pandas and pyarrow only inside the functions that need them.

Conventions every golden relies on:
- an array is pinned by its dtype string, its shape and its bytes; float NaNs are set to the canonical NaN
  before hashing, ``-0.0`` is not changed (its bit pattern is part of the contract);
- a float is written as ``float.hex`` ("nan", "inf" and "-inf" for the special values);
- a list is written as canonical JSON; a dict is flattened into "/"-joined keys.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
from pathlib import Path

import numpy as np

# "prereg_sha256" is the hash of a study-protocol document. The document can be edited after a cache is written,
# and a rebuild then stores the new hash, so the stored and the rebuilt cache differ only there. Hashes of data,
# code and weights are not volatile.
VOLATILE = frozenset({"created", "written", "seconds", "source_sha256", "provenance", "bytes_train_cache",
                      "bytes_eval_cache", "prereg_sha256"})
CELL_LEAVES = frozenset({"r2_os", "r2_zero", "sse", "sst", "n_rows", "h_ms", "late_windows"})
DROPPED_PREFIXES = ("crosscheck_", "legacy_")
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED")
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_MACHINE_PATH_RE = re.compile(r"/(home|mnt|Users)/")
_DRIVE_RE = re.compile(r"[A-Za-z]:\\")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _as_numpy(a) -> np.ndarray:
    if hasattr(a, "detach"):            # a torch tensor
        a = a.detach().cpu().numpy()
    return np.asarray(a)


def array_sha256(a) -> str:
    """sha256 over "dtype|shape|" and the array's bytes (C order).

    A structured dtype is written by its field list, because its dtype string only gives the record size.
    An object array is hashed as the JSON list of ``str`` of its items."""
    a = _as_numpy(a)
    dtype = a.dtype.str if a.dtype.names is None else str(a.dtype.descr)
    head = f"{dtype}|{a.shape}|".encode()
    if a.dtype.kind == "O":
        return _sha(head + json.dumps([str(v) for v in a.ravel().tolist()]).encode())
    if a.dtype.kind == "f":
        a = a.copy()
        a[np.isnan(a)] = np.nan
    return _sha(head + np.ascontiguousarray(a).tobytes())


def frame_sha256(df) -> str:
    """sha256 over one line "name:dtype:hash" per column, in column order. A categorical column is hashed as
    the JSON of its values plus the JSON of its categories."""
    import pandas as pd
    lines = []
    for name in df.columns:
        col = df[name]
        if isinstance(col.dtype, pd.CategoricalDtype):
            values = [None if pd.isna(v) else str(v) for v in col.tolist()]
            cats = [str(c) for c in col.cat.categories]
            h = _sha((json.dumps(values) + "|" + json.dumps(cats)).encode())
        else:
            h = array_sha256(col.to_numpy())
        lines.append(f"{name}:{col.dtype}:{h}")
    return _sha("\n".join(lines).encode())


def table_content_sha256(t) -> str:
    """Content hash of a pyarrow Table: per field its name, type and values (numeric columns without nulls by
    their bytes, every other column by the JSON of its Python values), then the sorted schema metadata
    (pyarrow's own "ARROW:" keys left out)."""
    import pyarrow as pa
    h = hashlib.sha256()
    for f in t.schema:
        col = t.column(f.name)
        h.update(f"{f.name}|{f.type}|".encode())
        numeric = pa.types.is_integer(f.type) or pa.types.is_floating(f.type) or pa.types.is_boolean(f.type)
        if numeric and col.null_count == 0:
            h.update(array_sha256(col.to_numpy()).encode())
        else:
            h.update(json.dumps(col.to_pylist(), default=str).encode())
        h.update(b"\n")
    meta = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items() if not k.startswith(b"ARROW:")}
    h.update(canonical_json(meta).encode())
    return h.hexdigest()


def canonical_json(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


def json_value(o) -> str:
    """canonical_json of a value that may hold numpy types or tuples."""
    return canonical_json(plain(o))


def sha256_file(path: Path, bufsize: int = 8 * 2**20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(bufsize), b""):
            h.update(block)
    return h.hexdigest()


def fhex(x) -> str:
    return float(x).hex()


def plain(o):
    """A JSON-ready copy: numpy values become Python values, tuples become lists, dict keys become str."""
    if isinstance(o, dict):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return plain(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def _put(out: dict, key: str, value) -> None:
    if key in out:
        raise ValueError(f"duplicate golden key {key}")
    out[key] = value


def _flat(o, prefix: str, out: dict) -> None:
    if isinstance(o, dict):
        if not o and prefix:
            _put(out, prefix, "{}")
        for k, v in o.items():
            _flat(v, f"{prefix}/{k}" if prefix else str(k), out)
        return
    if not prefix:
        raise ValueError("a golden leaf needs a key")
    if isinstance(o, np.ndarray):
        v = array_sha256(o)
    elif isinstance(o, (list, tuple)):
        v = canonical_json(plain(o))
    elif isinstance(o, (bool, np.bool_)):
        v = bool(o)
    elif isinstance(o, (int, np.integer)):
        v = int(o)
    elif isinstance(o, (float, np.floating)):
        v = fhex(o)
    elif o is None or isinstance(o, str):
        v = o
    else:
        raise TypeError(f"{prefix}: no golden form for {type(o).__name__}")
    _put(out, prefix, v)


def flatten(o, prefix: str = "") -> dict:
    """dict -> keys joined with "/"; list -> canonical JSON; array -> array_sha256; float -> fhex; an empty
    dict -> "{}"; int, bool, str and None as they are."""
    out: dict = {}
    _flat(o, prefix, out)
    return out


def scrub(o, drop: frozenset):
    """A copy of ``o`` without the keys in ``drop``, at any depth."""
    if isinstance(o, dict):
        return {k: scrub(v, drop) for k, v in o.items() if k not in drop}
    if isinstance(o, list):
        return [scrub(v, drop) for v in o]
    return o


def _strings(o, where: str = ""):
    """(location, text) for every key and every string value."""
    if isinstance(o, dict):
        for k, v in o.items():
            loc = f"{where}/{k}"
            yield loc, str(k)
            yield from _strings(v, loc)
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from _strings(v, f"{where}[{i}]")
    elif isinstance(o, str):
        yield where, o


def assert_no_machine_paths(o) -> None:
    """Raise ValueError if a key or a string value starts with "/", names a home or mount folder, or holds a
    drive letter path."""
    for where, s in _strings(o):
        if s.startswith("/") or _MACHINE_PATH_RE.search(s) or _DRIVE_RE.search(s):
            raise ValueError(f"machine path at {where}: {s[:80]!r}")


def scan_committable(obj, patterns) -> None:
    """``assert_no_machine_paths`` plus: no key or string value matches one of ``patterns`` (case-insensitive).
    The patterns are passed in by the caller; this file holds none."""
    assert_no_machine_paths(obj)
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    for where, s in _strings(obj):
        for rx in compiled:
            if rx.search(s):
                raise ValueError(f"pattern {rx.pattern!r} matches at {where}")


def committed_form(values: dict, real_data: bool) -> dict:
    """The committed form of flat golden values. Keys starting "crosscheck_" or "legacy_" are dropped. For a
    real-data golden every value that is not a sha256 hex string, not a bool and not a score-cell leaf is
    replaced by "h:" + sha256 of its canonical JSON."""
    out = {}
    for k, v in values.items():
        if k.startswith(DROPPED_PREFIXES):
            continue
        if real_data:
            keep = (isinstance(v, bool) or (isinstance(v, str) and _SHA_RE.fullmatch(v) is not None)
                    or k.rsplit("/", 1)[-1] in CELL_LEAVES)
            if not keep:
                v = "h:" + _sha(canonical_json(v).encode())
        out[k] = v
    return out


def recipe_sha256() -> str:
    """sha256 over (relative path, length, bytes) of every *.py of this folder, in path order."""
    root = Path(__file__).resolve().parent
    rels = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py")
                  if "__pycache__" not in p.relative_to(root).parts)
    h = hashlib.sha256()
    for rel in rels:
        data = (root / rel).read_bytes()
        h.update(f"{rel}\n{len(data)}\n".encode())
        h.update(data)
    return h.hexdigest()


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def blas_info() -> dict:
    """Name and version of numpy's BLAS, from ``numpy.show_config(mode="dicts")``."""
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    except (KeyError, TypeError, ValueError):
        return {"name": None, "version": None}
    return {"name": blas.get("name"), "version": blas.get("version")}


def env_block(extra: dict | None = None) -> dict:
    """Versions and thread settings a golden was recorded with; ``extra`` adds fields (torch, module hashes)."""
    env = {"python": platform.python_version(), **{d: _version(d) for d in ("numpy", "pandas", "pyarrow", "scipy")},
           "blas": blas_info(), "threads": {v: os.environ.get(v) for v in THREAD_VARS}}
    env.update(extra or {})
    return env
