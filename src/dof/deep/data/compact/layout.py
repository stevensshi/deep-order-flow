"""Names, constants and file helpers of the compact store: the key of a file, the paths of its data file and
sidecar, the manifest, atomic writes, the writer's lock, and the dtype in JSON."""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
import platform
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import zstandard
from numpy.lib import format as npy_format

from dof.deep.data.compact.errors import CompactError, StoreMismatch, StoreMissing

FORMAT = 1
BLOCK_ROWS = 256             # the smallest and fastest of the block lengths measured
ZSTD_LEVEL = 3               # ZSTD_CLEVEL_DEFAULT, written out so that a library default cannot move it
DATA_SUFFIX = ".zst"
SIDECAR_SUFFIX = ".zst.json"
MANIFEST_NAME = "MANIFEST.json"
LOCK_NAME = ".convert.lock"
VERIFY_WIDTH = 1000          # rows per window of the full-file check; on purpose not a multiple of BLOCK_ROWS
VERIFY_SLAB = 32             # windows per call in that check
DEFAULT_RATE = 10_000_000    # bytes per second
DEFAULT_PAUSE = 0.05         # seconds after every chunk
DEFAULT_MIN_AVAILABLE_AFTER = 4 * 2**30
DEFAULT_MIN_CGROUP_ROOM_AFTER = 4 * 2**30
CONVERT_RSS_STOP = 2_500_000_000


# -- names --------------------------------------------------------------------------------------
def key_of(relative: str | Path) -> str:
    """"UAL/train_2019-01-30.rec.npy" -> "UAL/train_2019-01-30.rec". The only place that holds the naming rule."""
    text = Path(relative).as_posix()
    if not text.endswith(".npy"):
        raise ValueError(f"{text}: not a .npy file, so it has no key")
    return text[:-len(".npy")]


def data_path(root: Path, key: str) -> Path:
    return Path(root) / (key + DATA_SUFFIX)


def sidecar_path(root: Path, key: str) -> Path:
    return Path(root) / (key + SIDECAR_SUFFIX)


def manifest_path(root: Path) -> Path:
    return Path(root) / MANIFEST_NAME


# -- small helpers ------------------------------------------------------------------------------
def _codec() -> dict:
    """The codec entry of every sidecar and manifest this package writes; a store with another one is not its own."""
    return {"name": "zstd", "level": ZSTD_LEVEL, "block_rows": BLOCK_ROWS, "content_size": True, "checksum": False}


def _libzstd() -> str:
    return ".".join(map(str, zstandard.ZSTD_VERSION))


def _versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "zstandard": zstandard.__version__,
            "libzstd": _libzstd()}


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _stat_path(path: Path) -> tuple[int, int] | None:
    """(st_size, st_mtime_ns) of a path; None when it is not there, which no recorded value equals."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return st.st_size, st.st_mtime_ns


def _hash_context(paths: Mapping[str, Path], error: type[CompactError]) -> dict[str, str]:
    """name -> sha256 of each context file. A context file that is not there fails the check it is hashed for,
    with that check's error type."""
    out = {}
    for name, path in paths.items():
        try:
            out[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except FileNotFoundError:
            raise error(f"{path}: the context file {name} is not there") from None
    return out


def _read_manifest(root: Path) -> tuple[dict, str]:
    """(manifest, sha256 of the file's bytes). A store without a manifest is not finished and cannot be read."""
    path = manifest_path(root)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise StoreMissing(f"{path}: no manifest; the store is not finished") from None
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _read_sidecar(root: Path, key: str) -> tuple[dict, str]:
    """(sidecar, sha256 of the file's bytes) of one key."""
    path = sidecar_path(root, key)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise StoreMissing(f"{key}: the sidecar {path} is missing") from None
    try:
        doc = json.loads(raw)
    except ValueError as e:
        raise StoreMismatch(f"{key}: the sidecar {path} is not JSON ({e})") from e
    return doc, hashlib.sha256(raw).hexdigest()


def _write_atomic(path: Path, data: bytes, *, give_back: bool) -> None:
    """``data`` to ``path`` through ``<path>.tmp``, fsync and rename: a reader sees the old file or the whole new
    one. ``give_back``: the new file's pages leave the page cache; it was written once and nobody reads it soon."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)                 # a stale one, from a run that was stopped in the middle of a file
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())                   # posix_fadvise(2): dirty pages are not dropped, so sync first
        os.replace(tmp, path)
        if give_back and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


@contextlib.contextmanager
def _locked(dst_root: Path) -> Iterator[None]:
    """One writer per store: a non-blocking lock on ``<dst_root>/.convert.lock``, held until the block ends."""
    import fcntl
    fd = os.open(Path(dst_root) / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CompactError(f"{dst_root}: another process holds {LOCK_NAME}; two writers on one store") from None
        yield
    finally:
        os.close(fd)                            # closing the descriptor releases the lock


# -- dtypes -------------------------------------------------------------------------------------
def _dtype_to_json(dtype: np.dtype):
    """A dtype as the JSON form of numpy.lib.format.dtype_to_descr (what a .npy header holds)."""
    return json.loads(json.dumps(npy_format.dtype_to_descr(dtype)))


def _descr_from_json(descr):
    """The inverse step JSON needs: field lists become tuples, shape lists become tuples, nested field lists recurse."""
    if isinstance(descr, str):
        return descr
    out = []
    for field in descr:
        name, sub, *shape = field
        name = tuple(name) if isinstance(name, list) else name           # a (title, name) pair
        out.append((name, _descr_from_json(sub), *[tuple(s) if isinstance(s, list) else s for s in shape]))
    return out


def _dtype_from_json(descr) -> np.dtype:
    return npy_format.descr_to_dtype(_descr_from_json(descr))


def read_npy_header(path: Path) -> tuple[np.dtype, int, int]:
    """(dtype, rows, data_offset) of a .npy file, read as the training stream reads it. ValueError unless the array
    is one-dimensional, has rows, has no object dtype, and has a dtype that survives the sidecar's JSON."""
    try:
        mm = np.load(path, mmap_mode="r")
    except ValueError as e:                     # numpy itself refuses to map an object dtype
        raise ValueError(f"{path}: {e}") from e
    dtype, shape, offset = mm.dtype, mm.shape, int(mm.offset)
    del mm
    if len(shape) != 1:
        raise ValueError(f"{path}: the array has {len(shape)} dimensions, not one")
    if shape[0] == 0:
        raise ValueError(f"{path}: the array has no rows")
    if dtype.hasobject:
        raise ValueError(f"{path}: the dtype {dtype} holds Python objects; its records have no fixed size")
    try:
        back = _dtype_from_json(_dtype_to_json(dtype))
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path}: the dtype {dtype} cannot be written to a sidecar ({e})") from e
    if back != dtype or back.itemsize != dtype.itemsize:
        raise ValueError(f"{path}: the dtype {dtype} comes back from a sidecar as {back}")
    return dtype, int(shape[0]), offset
