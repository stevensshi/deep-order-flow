"""Conversion of record files into a compact store, at a fixed pace, and the checks of a store on disk.

``convert`` writes one data file and one sidecar per source file (a file whose sidecar is valid is skipped, so the
same call continues a stopped run); ``finalize`` seals the store with ``MANIFEST.json``; ``verify`` reads a sealed
store against its manifest without loading it; ``show`` prints its totals. Every source file is read at the
pacer's rate and its pages are given back to the page cache as soon as they are read.

``READ_CHUNK``, ``Pacer`` and ``_stat_id`` are read from this module's globals when a function runs, and the disk
and memory readers are called through ``memory``, so a test can replace them.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from dof.deep.data.compact import memory
from dof.deep.data.compact.errors import CompactError, StoreMismatch, StoreMissing, VerificationError
from dof.deep.data.compact.layout import (
    CONVERT_RSS_STOP,
    DEFAULT_PAUSE,
    DEFAULT_RATE,
    FORMAT,
    LOCK_NAME,
    MANIFEST_NAME,
    SIDECAR_SUFFIX,
    VERIFY_SLAB,
    VERIFY_WIDTH,
    _codec,
    _dtype_to_json,
    _hash_context,
    _locked,
    _now,
    _read_manifest,
    _read_sidecar,
    _stat_path,
    _versions,
    _write_atomic,
    data_path,
    key_of,
    manifest_path,
    read_npy_header,
    sidecar_path,
)
from dof.deep.data.compact.store import CompactStore, encode_blocks

READ_CHUNK = 4 * 2**20


def _give_back(fd: int, offset: int, length: int) -> None:
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)


def _stat_id(fd: int) -> tuple[int, int]:
    """(st_size, st_mtime_ns) of an open file: what must not move while a source is read."""
    st = os.fstat(fd)
    return st.st_size, st.st_mtime_ns


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024         # Linux reports kB


class Pacer:
    """Holds a reader to ``rate`` bytes per second with at least ``pause`` seconds after every chunk. No credit
    carries over: after a stretch of other work the next chunk still waits the pause, so the reads never come in
    a burst. The disk may belong to a running training job."""

    def __init__(self, rate: float = DEFAULT_RATE, pause: float = DEFAULT_PAUSE,
                 clock=time.monotonic, sleep=time.sleep) -> None:
        self.rate, self.pause, self.clock, self.sleep = rate, pause, clock, sleep
        self.last = clock()

    def after(self, nbytes: int) -> None:
        wait = self.pause
        if self.rate > 0:
            wait = max(wait, nbytes / self.rate - (self.clock() - self.last))
        if wait > 0:
            self.sleep(wait)
        self.last = self.clock()


def _hash_paced(path: Path, pacer: Pacer) -> tuple[int, str]:
    """(bytes read, sha256) of a file, read at the pacer's rate; every page goes back as soon as it is hashed."""
    digest, pos = hashlib.sha256(), 0
    with open(path, "rb", buffering=0) as fh:
        fd = fh.fileno()
        while True:
            piece = fh.read(READ_CHUNK)
            if not piece:
                break
            digest.update(piece)
            _give_back(fd, pos, len(piece))
            pacer.after(len(piece))
            pos += len(piece)
        _give_back(fd, 0, 0)                    # posix_fadvise(2): a partial page is not dropped per chunk
    return pos, digest.hexdigest()


def _summary(dst_root: Path) -> dict:
    """What the command line prints of a sealed store: the one reader of the manifest for finalize, verify and show."""
    manifest, digest = _read_manifest(dst_root)
    totals = manifest["totals"]
    return {"root": str(dst_root), "manifest_sha256": digest, "context": dict(manifest["context"]),
            "files": totals["files"], "rows": totals["rows"], "source_bytes": totals["source_bytes"],
            "compressed_bytes": totals["compressed_bytes"], "bytes_per_row": totals["bytes_per_row"],
            "verified_rows": totals["verified_rows"]}


def verify_windows(store: CompactStore, index: int, payload) -> int:
    """The full-file check, through the function a fit reads with: every row of entry ``index`` is rebuilt by
    ``store.windows`` and compared with ``payload`` byte for byte (NaN bit patterns included: bytes are compared,
    never values). Returns the number of rows compared, which is the entry's row count."""
    data = np.frombuffer(payload, np.uint8)
    key, rows, size = store.keys[index], store.n_rows[index], store.dtypes[index].itemsize
    if len(data) != rows * size:
        raise VerificationError(f"{key}: the source has {len(data)} bytes, the store has {rows} rows of {size} bytes")
    width = VERIFY_WIDTH
    full, done = rows // width, 0
    for k in range(0, full, VERIFY_SLAB):
        starts = np.arange(k, min(k + VERIFY_SLAB, full), dtype=np.int64) * width
        out = store.windows(index, starts, width)
        a, b = int(starts[0]), int(starts[-1]) + width
        if not np.array_equal(out.view(np.uint8).reshape(-1), data[a * size:b * size]):
            raise VerificationError(f"{key}: rows {a} to {b - 1} read through windows() are not the source bytes")
        done += len(starts) * width
    rest = rows - full * width
    if rest:
        out = store.windows(index, np.array([full * width], dtype=np.int64), rest)
        if not np.array_equal(out.view(np.uint8).reshape(-1), data[full * width * size:]):
            raise VerificationError(f"{key}: rows {full * width} to {rows - 1} read through windows() are not the source bytes")
        done += rest
    if done != rows:
        raise VerificationError(f"{key}: {done} rows were compared, the file has {rows}")
    return done


def convert_file(src: Path, key: str, dst_root: Path, context: Mapping[str, str], pacer: Pacer) -> dict:
    """One source file to ``<key>.zst`` and its sidecar. Nothing is written before the compressed blocks, read
    back through ``CompactStore.windows``, equal the source bytes. The sidecar is written last: its presence means
    the file is converted and checked. Returns the sidecar."""
    src, dst_root = Path(src), Path(dst_root)
    dtype, rows, data_offset = read_npy_header(src)
    size = os.stat(src).st_size
    if data_offset + rows * dtype.itemsize != size:
        raise ValueError(f"{src}: {size} bytes, but a header of {data_offset} bytes and {rows} rows of {dtype.itemsize} "
                         f"bytes make {data_offset + rows * dtype.itemsize}")
    buf = bytearray(size)
    view, pos = memoryview(buf), 0
    with open(src, "rb", buffering=0) as fh:
        fd = fh.fileno()
        before = _stat_id(fd)
        while pos < size:
            n = fh.readinto(view[pos:pos + READ_CHUNK])
            if not n:
                break
            # Give the pages back at once, then wait: the source may be a file a training job reads, and a converter
            # that kept what it read would fill the page cache.
            _give_back(fd, pos, n)
            pacer.after(n)
            pos += n
        after = _stat_id(fd)
        if after != before or pos != size:
            raise CompactError(f"{src}: changed while it was read (size and mtime_ns {before} before, {after} after; "
                               f"{pos} of {size} bytes read)")
        _give_back(fd, 0, 0)
    payload = view[data_offset:]
    source_sha = hashlib.sha256(buf).hexdigest()
    payload_sha = hashlib.sha256(payload).hexdigest()
    frames, sizes = encode_blocks(payload, dtype.itemsize)
    store = CompactStore.from_blocks([(key, dtype, rows, frames, sizes)])
    verified = verify_windows(store, 0, payload)        # a VerificationError leaves here, before any file is written
    del store

    path = data_path(dst_root, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, frames, give_back=True)
    doc = {"format": FORMAT, "key": key,
           "source": {"relative": key + ".npy", "path": os.path.abspath(src), "bytes": size, "mtime_ns": before[1],
                      "sha256": source_sha, "data_offset": data_offset, "payload_sha256": payload_sha},
           "dtype": _dtype_to_json(dtype), "itemsize": dtype.itemsize, "rows": rows, "codec": _codec(),
           "blocks": len(sizes), "block_bytes": sizes,
           "compressed": {"file": path.name, "bytes": len(frames), "sha256": hashlib.sha256(frames).hexdigest()},
           "verified": {"through": "CompactStore.windows", "width": VERIFY_WIDTH, "rows": verified, "equal": True},
           "context": dict(context), "versions": _versions(), "written": _now()}
    _write_atomic(sidecar_path(dst_root, key), (json.dumps(doc, sort_keys=True) + "\n").encode(), give_back=False)
    return doc


def _valid_sidecar(src: Path, key: str, dst_root: Path, context: Mapping[str, str]) -> tuple[dict | None, str | None]:
    """(sidecar, None) when ``key`` is converted and its source has not changed since. Else (None, why); why is
    None when there is no sidecar at all, which needs no explanation in the log."""
    try:
        doc = json.loads(sidecar_path(dst_root, key).read_bytes())
    except FileNotFoundError:
        return None, None
    except ValueError:
        return None, "the sidecar is not JSON"
    try:
        if doc["format"] != FORMAT or doc["codec"] != _codec():
            return None, f"format {doc['format']} or codec {doc['codec']} is not this package's"
        if doc["context"] != dict(context):
            return None, "the context files have other hashes than at its conversion"
        have, want = _stat_path(src), (doc["source"]["bytes"], doc["source"]["mtime_ns"])
        if have != want:
            return None, f"the source has size and mtime_ns {have}, it had {want}"
        path = data_path(dst_root, key)
        if not path.is_file() or path.stat().st_size != doc["compressed"]["bytes"]:
            return None, "the data file is missing or has another size"
    except (KeyError, TypeError):
        return None, "the sidecar lacks a field"
    return doc, None


def convert(src_root: Path, dst_root: Path, pattern: str, context_names: Sequence[str], *,
            rate: float = DEFAULT_RATE, pause: float = DEFAULT_PAUSE,
            free_path: Path | None = None, min_free_bytes: int = 0, log=print) -> dict:
    """Every file of ``src_root`` that matches ``pattern`` into the store ``dst_root``, one after the other, at
    ``rate`` bytes per second. A file whose sidecar is valid is skipped, so the same call continues a stopped run.
    ``context_names``: files of ``src_root`` whose sha256 goes into every sidecar; they must not change during
    the run. ``free_path`` with ``min_free_bytes``: the disk floor, checked before each file."""
    src_root, dst_root = Path(src_root), Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    with _locked(dst_root):
        if manifest_path(dst_root).exists():
            raise CompactError(f"{dst_root} is sealed: it has a {MANIFEST_NAME}, and a sealed store is never converted again")
        names = {name: src_root / name for name in context_names}
        context = _hash_context(names, CompactError)
        files = sorted(src_root.glob(pattern))
        if not files:
            raise CompactError(f"{src_root}: no file matches {pattern}")
        total = len(files)
        t0 = time.monotonic()                   # before the pacer starts its own clock: rate= can then never read above the pace
        pacer = Pacer(rate, pause)
        docs, converted, skipped, read = [], 0, 0, 0
        for i, src in enumerate(files, 1):
            key = key_of(src.relative_to(src_root))
            doc, why = _valid_sidecar(src, key, dst_root, context)
            if doc is not None:
                log(f"[{i}/{total}] skipped {key}")
                docs.append(doc)
                skipped += 1
                continue
            if why is not None:
                log(f"[{i}/{total}] redo {key}: {why}")
            if free_path is not None:
                free = memory._free_bytes(free_path)
                if free < min_free_bytes:
                    raise CompactError(f"{free_path}: {free} bytes free, below the floor of {min_free_bytes}; stopped "
                                       f"before {key}")
            t1 = time.monotonic()
            doc = convert_file(src, key, dst_root, context, pacer)
            docs.append(doc)
            converted += 1
            read += doc["source"]["bytes"]
            now = time.monotonic()
            elapsed = max(now - t0, 1e-9)
            log(f"[{i}/{total}] converted {key} rows={doc['rows']} source_bytes={doc['source']['bytes']} "
                f"compressed_bytes={doc['compressed']['bytes']} bytes_per_row={doc['compressed']['bytes'] / doc['rows']:.2f} "
                f"seconds={now - t1:.1f} total_read={read} elapsed={elapsed:.0f} rate={read / elapsed / 1e6:.2f}")
            peak = _peak_rss_bytes()
            if peak is not None and peak > CONVERT_RSS_STOP:
                raise CompactError(f"peak resident memory {peak} bytes after {key}, above the stop of {CONVERT_RSS_STOP}; "
                                   "the same call continues the run")
        if _hash_context(names, CompactError) != context:
            raise CompactError(f"{src_root}: a context file ({', '.join(names)}) changed during the conversion; the "
                               "sidecars written carry the old hashes and the next run converts them again")
        seconds = max(time.monotonic() - t0, 1e-9)
        mean = read / seconds / 1e6 if converted else None
        log(f"convert: files={total} converted={converted} skipped={skipped} source_bytes_read={read} "
            f"seconds={seconds:.0f} mean_rate_mb_s={'none' if mean is None else format(mean, '.2f')}")
        rows = sum(d["rows"] for d in docs)
        compressed = sum(d["compressed"]["bytes"] for d in docs)
        return {"files": total, "converted": converted, "skipped": skipped, "rows": rows,
                "source_bytes": sum(d["source"]["bytes"] for d in docs), "compressed_bytes": compressed,
                "bytes_per_row": compressed / rows, "source_bytes_read": read, "seconds": seconds,
                "mean_rate_mb_s": mean}


def finalize(src_root: Path, dst_root: Path, pattern: str, context_names: Sequence[str], *,
             rate: float = DEFAULT_RATE, pause: float = DEFAULT_PAUSE, log=print) -> dict:
    """Seals a converted store: checks that it holds exactly the files of ``src_root`` that match ``pattern``,
    converted from the present data, reads every data file once more against its sidecar, and writes
    ``MANIFEST.json``. A store without that file cannot be loaded. It creates no folder."""
    src_root, dst_root = Path(src_root), Path(dst_root)
    if not dst_root.is_dir():
        raise StoreMissing(f"{dst_root}: no such folder; nothing was converted")
    with _locked(dst_root):
        if manifest_path(dst_root).exists():
            raise CompactError(f"{dst_root} is sealed: it has a {MANIFEST_NAME} already")
        context = _hash_context({name: src_root / name for name in context_names}, StoreMismatch)
        sources = {key_of(p.relative_to(src_root)): p for p in sorted(src_root.glob(pattern))}
        if not sources:
            raise CompactError(f"{src_root}: no file matches {pattern}")
        for key, src in sources.items():
            if not sidecar_path(dst_root, key).is_file():
                raise StoreMissing(f"{key}: no sidecar; the source {src} is not converted")
            if not data_path(dst_root, key).is_file():
                raise StoreMissing(f"{key}: the data file {data_path(dst_root, key)} is missing")
        own = {dst_root / LOCK_NAME}
        for key in sources:
            own |= {data_path(dst_root, key), sidecar_path(dst_root, key)}
        for path in sorted(dst_root.rglob("*")):
            if path.is_dir() or path in own:
                continue
            if path.name.endswith(SIDECAR_SUFFIX):
                raise StoreMismatch(f"{path}: a sidecar without a source file that matches {pattern} in {src_root}")
            raise StoreMismatch(f"{path}: not the lock, a data file or a sidecar of this store")
        docs, digests = {}, {}
        for key, src in sources.items():
            doc, digests[key] = _read_sidecar(dst_root, key)
            if doc.get("context") != context:
                raise StoreMismatch(f"{key}: the sidecar has the context hashes {doc.get('context')}, the files have {context} now")
            checked = doc.get("verified", {})
            if checked.get("equal") is not True or checked.get("rows") != doc.get("rows"):
                raise StoreMismatch(f"{key}: the sidecar's full-file check is {checked} for {doc.get('rows')} rows")
            have, want = _stat_path(src), (doc["source"]["bytes"], doc["source"]["mtime_ns"])
            if have != want:
                raise StoreMismatch(f"{key}: the source {src} has size and mtime_ns {have}, it had {want} when it was converted")
            docs[key] = doc
        pacer = Pacer(rate, pause)
        t0, read = time.monotonic(), 0
        for key, doc in docs.items():
            got = _hash_paced(data_path(dst_root, key), pacer)
            want = (doc["compressed"]["bytes"], doc["compressed"]["sha256"])
            if got != want:
                raise StoreMismatch(f"{key}: the data file has size and sha256 {got}, its sidecar has {want}")
            read += got[0]
        rows = sum(d["rows"] for d in docs.values())
        compressed = sum(d["compressed"]["bytes"] for d in docs.values())
        manifest = {"format": FORMAT, "codec": _codec(), "source_root": os.path.abspath(src_root), "pattern": pattern,
                    "context": context,
                    "entries": {key: {"sidecar_sha256": digests[key], "rows": d["rows"], "itemsize": d["itemsize"],
                                      "source_bytes": d["source"]["bytes"], "source_sha256": d["source"]["sha256"],
                                      "compressed_bytes": d["compressed"]["bytes"],
                                      "compressed_sha256": d["compressed"]["sha256"],
                                      "verified_rows": d["verified"]["rows"]} for key, d in docs.items()},
                    "totals": {"files": len(docs), "rows": rows,
                               "source_bytes": sum(d["source"]["bytes"] for d in docs.values()),
                               "compressed_bytes": compressed, "bytes_per_row": compressed / rows,
                               "verified_rows": sum(d["verified"]["rows"] for d in docs.values())},
                    "versions": _versions(), "written": _now()}
        _write_atomic(manifest_path(dst_root), (json.dumps(manifest, sort_keys=True, indent=1) + "\n").encode(),
                      give_back=False)
        log(f"finalize: {len(docs)} data files, {read} bytes read in {time.monotonic() - t0:.0f} s")
    return _summary(dst_root)


def verify(dst_root: Path, *, src_root: Path | None = None,
           rate: float = DEFAULT_RATE, pause: float = DEFAULT_PAUSE, log=print) -> dict:
    """Checks a sealed store on disk: every sidecar against the manifest and every data file against its sidecar,
    read at ``rate``. With ``src_root`` also the context files and each source's size and mtime_ns. It never loads
    the store into memory (a store of several GB of anonymous memory beside a fit would press the machine)."""
    dst_root = Path(dst_root)
    manifest, _ = _read_manifest(dst_root)
    docs = {}
    for key, entry in manifest["entries"].items():
        doc, digest = _read_sidecar(dst_root, key)
        if digest != entry["sidecar_sha256"]:
            raise StoreMismatch(f"{key}: the sidecar has sha256 {digest}, the manifest has {entry['sidecar_sha256']}")
        if not data_path(dst_root, key).is_file():
            raise StoreMissing(f"{key}: the data file {data_path(dst_root, key)} is missing")
        docs[key] = doc
    if src_root is not None:
        src_root = Path(src_root)
        recorded = manifest["context"]
        present = _hash_context({name: src_root / name for name in recorded}, StoreMismatch)
        if present != recorded:
            raise StoreMismatch(f"{src_root}: the context files have the hashes {present}, the manifest has {recorded}")
        for key, doc in docs.items():
            src = src_root / doc["source"]["relative"]
            have, want = _stat_path(src), (doc["source"]["bytes"], doc["source"]["mtime_ns"])
            if have != want:
                raise StoreMismatch(f"{key}: the source {src} has size and mtime_ns {have}, it had {want} when it was converted")
    pacer = Pacer(rate, pause)
    t0, read = time.monotonic(), 0
    for key, doc in docs.items():
        got = _hash_paced(data_path(dst_root, key), pacer)
        want = (doc["compressed"]["bytes"], doc["compressed"]["sha256"])
        if got != want:
            raise StoreMismatch(f"{key}: the data file has size and sha256 {got}, its sidecar has {want}")
        read += got[0]
    log(f"verify: {len(docs)} data files, {read} bytes read in {time.monotonic() - t0:.0f} s")
    return _summary(dst_root)


def show(dst_root: Path) -> dict:
    """The totals, the context and the manifest sha256 of a sealed store."""
    return _summary(dst_root)
