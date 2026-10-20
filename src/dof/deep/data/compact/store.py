"""The compact store in memory: the compressed blocks of a set of files in ONE buffer of anonymous memory, and the
window reader a training batch uses."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import zstandard

from dof.deep.data.compact import memory
from dof.deep.data.compact.errors import NotEnoughMemory, StoreMismatch, StoreMissing
from dof.deep.data.compact.layout import (
    BLOCK_ROWS,
    DEFAULT_MIN_AVAILABLE_AFTER,
    DEFAULT_MIN_CGROUP_ROOM_AFTER,
    FORMAT,
    ZSTD_LEVEL,
    _codec,
    _dtype_from_json,
    _hash_context,
    _libzstd,
    _read_manifest,
    _read_sidecar,
    _stat_path,
    data_path,
    manifest_path,
    sidecar_path,
)


def encode_blocks(payload, itemsize: int, block_rows: int = BLOCK_ROWS) -> tuple[bytes, list[int]]:
    """One zstd frame per block of ``block_rows`` records, the last block shorter; the record bytes as stored (byte
    transposition was worse for every codec and block length measured). Returns the frames joined back to back and
    the list of frame sizes.

    Each frame carries its content size, so a block decodes without a size hint. No checksum: it would cost about
    1 ms per batch, and the sha256 of every file is checked at each load."""
    data = np.frombuffer(payload, np.uint8)
    if len(data) % itemsize:
        raise ValueError(f"the payload has {len(data)} bytes: not a multiple of the item size {itemsize}")
    comp = zstandard.ZstdCompressor(level=ZSTD_LEVEL, write_content_size=True, write_checksum=False,
                                    write_dict_id=False, threads=0)
    step = block_rows * itemsize
    frames = [comp.compress(data[a:a + step]) for a in range(0, len(data), step)]
    return b"".join(frames), [len(f) for f in frames]


class CompactStore:
    """The compressed blocks of a set of files in ONE buffer of anonymous memory (``memory._allocate``), never a
    mapping of a file: pages mapped from a file can be reclaimed, and the wait on the disk would come back.

    ``keys[i]``, ``dtypes[i]`` and ``n_rows[i]`` describe entry ``i``; ``windows(i, ...)`` reads it."""

    def __init__(self, keys: list[str], dtypes: list[np.dtype], n_rows: list[int], block_rows: int,
                 buf: np.ndarray, ends: list[np.ndarray], loaded: dict) -> None:
        self.keys, self.dtypes, self.n_rows, self.block_rows = keys, dtypes, n_rows, block_rows
        self._buf = buf
        self._mv = memoryview(buf)
        self._ends = ends               # per entry: int64, blocks + 1 absolute byte offsets into _buf
        self._dctx, self._dctx_pid = None, -1
        self._loaded = loaded           # the facts that differ between from_blocks and load

    @classmethod
    def from_blocks(cls, parts: Sequence[tuple[str, np.dtype, int, bytes, Sequence[int]]],
                    block_rows: int = BLOCK_ROWS) -> "CompactStore":
        """A store from compressed bytes already in memory. Each part is (key, dtype, rows, frames, frame sizes).
        The converter checks a file through this before it writes it; the tests use it."""
        keys, dtypes, n_rows, blobs, ends, pos = [], [], [], [], [], 0
        for key, dtype, rows, frames, sizes in parts:
            dtype, rows, sizes = np.dtype(dtype), int(rows), [int(s) for s in sizes]
            want = -(-rows // block_rows)
            if len(sizes) != want:
                raise StoreMismatch(f"{key}: {len(sizes)} block sizes, but {rows} rows in blocks of {block_rows} "
                                    f"make {want} blocks")
            if sum(sizes) != len(frames):
                raise StoreMismatch(f"{key}: the block sizes add up to {sum(sizes)} bytes, the frames have {len(frames)}")
            keys.append(key)
            dtypes.append(dtype)
            n_rows.append(rows)
            blobs.append(frames)
            ends.append(np.cumsum([pos, *sizes], dtype=np.int64))
            pos += len(frames)
        buf, at = memory._allocate(pos), 0
        for blob in blobs:
            buf[at:at + len(blob)] = np.frombuffer(blob, np.uint8)
            at += len(blob)
        return cls(keys, dtypes, n_rows, block_rows, buf, ends,
                   {"source_bytes": sum(r * d.itemsize for r, d in zip(n_rows, dtypes))})

    @classmethod
    def load(cls, root: Path, keys: Sequence[str], *, context: Mapping[str, Path],
             dtype: np.dtype | None = None, rows: Sequence[int] | None = None,
             sources: Sequence[Path] | None = None,
             min_available_after: int | None = DEFAULT_MIN_AVAILABLE_AFTER,
             min_cgroup_room_after: int | None = DEFAULT_MIN_CGROUP_ROOM_AFTER) -> "CompactStore":
        """The files ``keys`` of the sealed store ``root`` into this process's memory, in the order given.

        Every check runs before the memory is asked for: the manifest, the context files (``context``: name ->
        path; their sha256 must be the manifest's), each sidecar against the manifest, and, when given, ``dtype``,
        ``rows[i]`` and the size and mtime_ns of ``sources[i]``. Then two memory rules. ``MemAvailable`` less the
        store must leave ``min_available_after``. And when this process's cgroup has a ``memory.high``, that limit
        less what the cgroup uses outside its page cache (``memory.current`` less ``inactive_file`` and
        ``active_file`` of ``memory.stat``) must hold the store and ``min_cgroup_room_after``. A cgroup without a
        limit passes; one whose three files cannot all be read stops the load. A rule given as None is not checked,
        and the facts it would record are None. A missing or differing file stops the load; nothing falls back to
        another reader."""
        t0 = time.perf_counter()
        root, keys = Path(root), list(keys)
        if not keys:
            raise ValueError("no key was given: a store holds at least one file")
        if len(set(keys)) != len(keys):
            raise ValueError(f"a key was given twice ({len(keys)} keys, {len(set(keys))} different)")
        for name, given in (("rows", rows), ("sources", sources)):
            if given is not None and len(given) != len(keys):
                raise ValueError(f"{name} has {len(given)} entries for {len(keys)} keys")

        manifest, manifest_sha = _read_manifest(root)
        if manifest.get("format") != FORMAT or manifest.get("codec") != _codec():
            raise StoreMismatch(f"{manifest_path(root)}: format {manifest.get('format')} and codec {manifest.get('codec')}; "
                                f"this package reads format {FORMAT} and codec {_codec()}")

        recorded = manifest["context"]
        if set(context) != set(recorded):
            raise StoreMismatch(f"{root}: the context files given are {sorted(context)}, the manifest has {sorted(recorded)}")
        for name, digest in _hash_context(context, StoreMismatch).items():
            if digest != recorded[name]:
                raise StoreMismatch(f"{context[name]}: sha256 {digest}, the manifest of {root} has {recorded[name]} "
                                    f"for {name}; the store was made from other data")

        sides, dtypes, n_rows = [], [], []
        for i, key in enumerate(keys):
            entry = manifest["entries"].get(key)
            if entry is None:
                raise StoreMissing(f"{key}: not in the manifest of {root}")
            side, digest = _read_sidecar(root, key)
            if digest != entry["sidecar_sha256"]:
                raise StoreMismatch(f"{key}: the sidecar {sidecar_path(root, key)} has sha256 {digest}, the manifest has "
                                    f"{entry['sidecar_sha256']}")
            n, sizes, blocks = side["rows"], side["block_bytes"], -(-side["rows"] // BLOCK_ROWS)
            if not (len(sizes) == side["blocks"] == blocks):
                raise StoreMismatch(f"{key}: {len(sizes)} block sizes and blocks={side['blocks']} in the sidecar, but "
                                    f"{n} rows in blocks of {BLOCK_ROWS} make {blocks}")
            if sum(sizes) != side["compressed"]["bytes"]:
                raise StoreMismatch(f"{key}: the block sizes add up to {sum(sizes)} bytes, the sidecar gives the data "
                                    f"file {side['compressed']['bytes']}")
            dt = _dtype_from_json(side["dtype"])
            if dtype is not None and dt != np.dtype(dtype):
                raise StoreMismatch(f"{key}: the store has dtype {dt}, the caller expects {np.dtype(dtype)}")
            if rows is not None and int(rows[i]) != n:
                raise StoreMismatch(f"{key}: the store has {n} rows, the caller expects {int(rows[i])}")
            if sources is not None:
                have, want = _stat_path(sources[i]), (side["source"]["bytes"], side["source"]["mtime_ns"])
                if have != want:
                    raise StoreMismatch(f"{key}: the source {sources[i]} has size and mtime_ns {have}, it had {want} "
                                        "when it was converted")
            sides.append(side)
            dtypes.append(dt)
            n_rows.append(n)

        total = sum(side["compressed"]["bytes"] for side in sides)
        available = None
        if min_available_after is not None:
            available = memory._mem_available()
            if available - total < min_available_after:
                raise NotEnoughMemory(f"{root}: the store needs {total} bytes and MemAvailable is {available}; "
                                      f"{min_available_after} bytes must stay available after the load")
        # The cgroup rule. A fit in a scope with too low a memory.high is swapped out and nothing says so. The page
        # cache of the cgroup is not counted as used: under memory.high the kernel gives cold page cache back, and the
        # warm copy of this very store is cold cache when a fit loads it. The two LRU lines are subtracted, not
        # ``file``: that one holds tmpfs and shared memory, which reclaim cannot free. Read last, just before the
        # memory is asked for.
        cg_folder = cg_high = cg_current = cg_inactive = cg_active = cg_room = None
        if min_cgroup_room_after is not None:
            cg_folder, cg_high, cg_current, cg_inactive, cg_active = memory._cgroup_memory()
            cg_room = None if cg_high is None else cg_high - (cg_current - cg_inactive - cg_active)
            if cg_room is not None and cg_room < total + min_cgroup_room_after:
                raise NotEnoughMemory(f"{root}: the store needs {total} bytes; the cgroup {cg_folder} has memory.high "
                                      f"{cg_high} and memory.current {cg_current}, of which {cg_inactive + cg_active} bytes "
                                      f"are page cache (inactive_file {cg_inactive}, active_file {cg_active}) and are not "
                                      f"counted; that leaves {cg_room} bytes, and {min_cgroup_room_after} bytes must stay "
                                      "free under memory.high after the load")

        buf = memory._allocate(total)
        mv, ends, pos = memoryview(buf), [], 0
        for key, side in zip(keys, sides):
            path, size = data_path(root, key), side["compressed"]["bytes"]
            try:
                fh = open(path, "rb", buffering=0)
            except FileNotFoundError:
                raise StoreMissing(f"{key}: the data file {path} is missing") from None
            with fh:
                have = os.fstat(fh.fileno()).st_size
                if have != size:
                    raise StoreMismatch(f"{key}: {path} has {have} bytes, its sidecar has {size}")
                got = 0
                while got < size:
                    n = fh.readinto(mv[pos + got:pos + size])
                    if not n:
                        raise StoreMismatch(f"{key}: {path} ended after {got} bytes, its sidecar has {size}")
                    got += n
                if fh.read(1):
                    raise StoreMismatch(f"{key}: {path} has more than the {size} bytes of its sidecar")
                # The bytes are in this process now. Left in the page cache they would be a second copy that fills
                # the machine's memory.
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            digest = hashlib.sha256(mv[pos:pos + size]).hexdigest()
            if digest != side["compressed"]["sha256"]:
                raise StoreMismatch(f"{key}: {path} has sha256 {digest}, its sidecar has {side['compressed']['sha256']}")
            ends.append(np.cumsum([pos, *side["block_bytes"]], dtype=np.int64))
            pos += size

        store = cls(keys, dtypes, n_rows, BLOCK_ROWS, buf, ends, {})
        for i in range(len(keys)):              # the block table fits the data: first and last block of every file decode
            for block in sorted({0, len(ends[i]) - 2}):
                store._decompress(i, block)
        entries = "".join(f"{key} {manifest['entries'][key]['sidecar_sha256']}\n" for key in keys)
        store._loaded = {"source_bytes": sum(manifest["entries"][key]["source_bytes"] for key in keys),
                         "root": str(root), "manifest_sha256": manifest_sha,
                         "entries_sha256": hashlib.sha256(entries.encode()).hexdigest(),
                         "context_sha256": dict(recorded), "mem_available_before": available,
                         "load_seconds": time.perf_counter() - t0,
                         "cgroup": cg_folder, "cgroup_memory_high": cg_high, "cgroup_memory_current_before": cg_current,
                         "cgroup_inactive_file_before": cg_inactive, "cgroup_active_file_before": cg_active,
                         "cgroup_room_before": cg_room}
        return store

    def __len__(self) -> int:
        return len(self.keys)

    def _decompress(self, index: int, block: int) -> np.ndarray:
        """The records of one block. A forked loader process inherits the store's bytes and only reads them, but it
        makes its own decompressor: python-zstandard instances are not to be shared."""
        pid = os.getpid()
        if pid != self._dctx_pid:
            self._dctx, self._dctx_pid = zstandard.ZstdDecompressor(), pid
        ends, dtype, key = self._ends[index], self.dtypes[index], self.keys[index]
        a, b = int(ends[block]), int(ends[block + 1])
        try:
            # allow_extra_data=False is stricter than the library's default: a block table that does not fit the
            # data must fail here, not give the first frame of a longer slice.
            raw = self._dctx.decompress(self._mv[a:b], allow_extra_data=False)
        except zstandard.ZstdError as e:
            raise StoreMismatch(f"{key}: block {block} (bytes {a} to {b} of the store) is not one zstd frame: {e}") from e
        want = min(self.block_rows, self.n_rows[index] - block * self.block_rows)
        if len(raw) != want * dtype.itemsize:
            raise StoreMismatch(f"{key}: block {block} decodes to {len(raw)} bytes, not to {want} records of "
                                f"{dtype.itemsize} bytes")
        return np.frombuffer(raw, dtype)

    def windows(self, index: int, first: np.ndarray, width: int) -> np.ndarray:
        """(len(first), width) records of entry ``index``: for each start row the ``width`` rows from it on. The
        bytes equal ``source[first[:, None] + arange(width)]``. The result is C-contiguous, writable and owns its
        data. A block that several windows of the call need is decoded once (this loop with a per-call cache was
        faster than a vectorised gather)."""
        first = np.asarray(first)
        if first.ndim != 1:
            raise ValueError(f"first has {first.ndim} dimensions, not one")
        if first.dtype.kind not in "iu":
            raise TypeError(f"first has dtype {first.dtype}: start rows are integers")
        if width < 1:
            raise ValueError(f"width {width}: a window has at least one row")
        dtype, rows = self.dtypes[index], self.n_rows[index]
        out = np.empty((len(first), width), dtype)
        if len(first) == 0:
            return out
        if int(first.min()) < 0 or int(first.max()) + width > rows:
            raise IndexError(f"{self.keys[index]}: a window of {width} rows from {int(first.min())} or "
                             f"{int(first.max())} leaves the {rows} rows of the file")
        cache: dict[int, np.ndarray] = {}
        n = self.block_rows
        for i, s in enumerate(first.tolist()):
            pos = 0
            for b in range(s // n, (s + width - 1) // n + 1):
                arr = cache.get(b)
                if arr is None:
                    arr = cache[b] = self._decompress(index, b)
                lo, hi = max(s - b * n, 0), min(s + width - b * n, len(arr))
                out[i, pos:pos + hi - lo] = arr[lo:hi]
                pos += hi - lo
        return out

    def facts(self) -> dict:
        """What a fit record keeps of its store: 23 keys, JSON-serialisable. The twelve that exist only for a store
        read from disk are None for a store made by ``from_blocks``. For a store read from disk in a cgroup without
        a limit (``max``), ``cgroup_memory_high`` and ``cgroup_room_before`` are None too; ``cgroup`` is then still
        its folder. ``cgroup_room_before`` is memory.high - (memory.current - inactive_file - active_file)."""
        rows, nbytes = sum(self.n_rows), int(len(self._buf))
        out = {"module": "dof.deep.data.compact", "format": FORMAT, "block_rows": self.block_rows,
               "zstd_level": ZSTD_LEVEL, "zstandard": zstandard.__version__, "libzstd": _libzstd(),
               "files": len(self.keys), "rows": rows, "compressed_bytes": nbytes,
               "bytes_per_row": nbytes / rows if rows else 0.0, "source_bytes": None,
               "root": None, "manifest_sha256": None, "entries_sha256": None, "context_sha256": None,
               "mem_available_before": None, "load_seconds": None,
               "cgroup": None, "cgroup_memory_high": None, "cgroup_memory_current_before": None,
               "cgroup_inactive_file_before": None, "cgroup_active_file_before": None, "cgroup_room_before": None}
        out.update(self._loaded)
        if out["context_sha256"] is not None:
            out["context_sha256"] = dict(out["context_sha256"])
        return out
