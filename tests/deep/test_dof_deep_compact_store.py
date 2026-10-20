"""The compact store in memory: the window function, any fixed dtype, load and its refusals, the memory rules,
forked readers, the frames; and the compact backend against the record files of the synthetic run."""

from __future__ import annotations

import hashlib
import io
import json
import mmap
import os
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

zstandard = pytest.importorskip("zstandard")

from compact_helpers import (  # noqa: E402
    CONTEXT,
    DISK_ONLY_FACTS,
    DONTNEED,
    FACT_KEYS,
    FAKE_CGROUP,
    NAN_A,
    NAN_B,
    NEG_ZERO,
    NESTED,
    REC,
    _built,
    _cgroup_without_limit,  # noqa: F401  (the autouse fixture, bound in this module)
    _convert,
    _count_decompress,
    _ctx,
    _finalize,
    _flip,
    _load,
    _parts,
    _rec,
    _sha,
    _slices,
    _smaps,
    _store,
    _tree,
)
from dof.deep.data import compact  # noqa: E402
from dof.deep.data.compact import CompactError, CompactStore, NotEnoughMemory, StoreMismatch, StoreMissing, memory  # noqa: E402


# -- the window function ------------------------------------------------------------------------
def test_roundtrip_bytes_equal_with_nan_payloads() -> None:
    src = _rec(1000)
    store = _store(src)
    got = store.windows(0, [0], len(src))
    assert got.tobytes() == src.tobytes()
    bits = got["zof"].view("<u4")
    assert bits[0, 0, 0] == NAN_A and bits[0, 500, 3] == NAN_B and bits[0, 333, 1] == NEG_ZERO
    assert got["y"].view("<u4")[0, 999, 9] == NAN_A
    facts = store.facts()
    assert set(facts) == FACT_KEYS and len(facts) == 23
    assert all(facts[k] is None for k in DISK_ONLY_FACTS)
    assert all(facts[k] is not None for k in FACT_KEYS - set(DISK_ONLY_FACTS))
    assert (facts["files"], facts["rows"], facts["source_bytes"]) == (1, 1000, src.nbytes)
    assert facts["module"] == "dof.deep.data.compact"
    json.dumps(facts)


def test_windows_match_slices_random_and_boundary_starts() -> None:
    src = _rec(256 * 4 + 37)
    store = _store(src)
    rng = np.random.default_rng(2)
    # 156 ends on the last row of block 0; 157 is the first start that reaches block 1
    first = np.concatenate([[0, 155, 156, 157, 255, 256, len(src) - 100], rng.integers(0, len(src) - 100 + 1, 250)])
    got = store.windows(0, first, 100)
    assert got.tobytes() == _slices(src, first, 100)
    assert got.shape == (len(first), 100) and got.dtype == REC
    assert got.flags.c_contiguous and got.flags.writeable and got.flags.owndata


def test_windows_width_one_and_width_over_many_blocks() -> None:
    src = _rec(1500)
    store = _store(src)
    every = np.arange(len(src))
    assert store.windows(0, every, 1).tobytes() == _slices(src, every, 1)
    rng = np.random.default_rng(3)
    first = np.concatenate([[0, 1, 255, 256, len(src) - 700], rng.integers(0, len(src) - 700 + 1, 20)])
    assert store.windows(0, first, 700).tobytes() == _slices(src, first, 700)       # each window spans 3 or 4 blocks


def test_windows_short_last_block_exact_multiple_and_tiny_arrays() -> None:
    for rows in (256 * 3 + 5, 256 * 3, 70, 1):
        src = _rec(rows, rows)
        key, dtype, n, frames, sizes = _parts(src)
        assert len(sizes) == -(-rows // 256)
        store = CompactStore.from_blocks([(key, dtype, n, frames, sizes)])
        for width in sorted({1, min(rows, 100), rows}):
            first = np.array([rows - width])                           # the window ends on the last row
            assert store.windows(0, first, width).tobytes() == _slices(src, first, width)
        assert store.windows(0, [0], rows).tobytes() == src.tobytes()


def test_windows_empty_first(monkeypatch) -> None:
    store = _store(_rec(600))
    calls = _count_decompress(monkeypatch)
    got = store.windows(0, np.empty(0, np.int64), 100)
    assert got.shape == (0, 100) and got.dtype == REC
    assert calls == []


def test_windows_rejects_bad_arguments() -> None:
    store = _store(_rec(600))
    with pytest.raises(IndexError):
        store.windows(0, np.array([-1]), 10)
    with pytest.raises(IndexError):
        store.windows(0, np.array([0, 591]), 10)                       # 591 + 10 > 600
    assert store.windows(0, np.array([590]), 10).shape == (1, 10)      # the last window that fits
    with pytest.raises(ValueError):
        store.windows(0, np.array([0]), 0)
    with pytest.raises(ValueError):
        store.windows(0, np.zeros((2, 2), np.int64), 10)
    with pytest.raises(TypeError):
        store.windows(0, np.array([0.0]), 10)
    with pytest.raises(TypeError):
        store.windows(0, np.asarray([]), 10)                           # empty, and float: rejected before the early return


def test_windows_decompresses_each_block_once_per_call(monkeypatch) -> None:
    store = _store(_rec(1000))
    calls = _count_decompress(monkeypatch)
    store.windows(0, np.arange(256), 100)                              # rows 0 to 354: blocks 0 and 1
    assert sorted(calls) == [(0, 0), (0, 1)]


@pytest.mark.parametrize("dtype", [np.dtype("<f4"), np.dtype("<i8"), np.dtype("S5"), NESTED, REC],
                         ids=["f4", "i8", "S5", "nested", "REC"])
def test_any_fixed_dtype(tmp_path: Path, dtype: np.dtype) -> None:
    rows = 256 * 2 + 44
    arr = np.frombuffer(np.random.default_rng(8).bytes(rows * dtype.itemsize), dtype).copy()     # any bytes are a value
    src, dst, key = tmp_path / "src", tmp_path / "dst", "A/train_2019-01-30.rec"
    (src / "A").mkdir(parents=True)
    np.save(src / (key + ".npy"), arr)
    for name in CONTEXT:
        (src / name).write_text("{}\n")
    _convert(src, dst)
    _finalize(src, dst)
    store = _load(src, dst, [key], dtype=dtype, rows=[rows])
    assert store.dtypes == [dtype] and store.n_rows == [rows]
    assert store.windows(0, [0], rows).tobytes() == arr.tobytes()


# -- load ---------------------------------------------------------------------------------------
def test_load_returns_windows_of_the_right_key_and_subset(tmp_path: Path) -> None:
    src, dst, arrays = _built(tmp_path)
    keys = sorted(arrays, reverse=True)
    store = _load(src, dst, keys)
    assert len(store) == 3 and store.keys == keys
    for i, key in enumerate(keys):
        arr = arrays[key]
        assert store.n_rows[i] == len(arr) and store.dtypes[i] == REC
        assert store.windows(i, [0], len(arr)).tobytes() == arr.tobytes()
    path = compact.manifest_path(dst)
    facts = store.facts()
    assert set(facts) == FACT_KEYS
    assert facts["context_sha256"] == json.loads(path.read_text())["context"]
    assert facts["manifest_sha256"] == _sha(path)
    assert facts["compressed_bytes"] == sum(compact.data_path(dst, k).stat().st_size for k in keys)
    json.dumps(facts)
    assert (facts["cgroup"], facts["cgroup_memory_high"], facts["cgroup_memory_current_before"],
            facts["cgroup_inactive_file_before"], facts["cgroup_active_file_before"]) == FAKE_CGROUP
    assert facts["cgroup_room_before"] is None                          # no limit: there is no room to name
    one = _load(src, dst, [keys[1]])
    assert one.facts()["files"] == 1
    assert one.facts()["compressed_bytes"] == compact.data_path(dst, keys[1]).stat().st_size
    assert one.windows(0, [0], len(arrays[keys[1]])).tobytes() == arrays[keys[1]].tobytes()


def test_load_with_rules_off_records_none(tmp_path: Path, monkeypatch) -> None:
    """A memory rule given as None is not checked: its reader is not called, and its facts are None."""
    src, dst, arrays = _built(tmp_path)

    def must_not_run(*args):
        raise AssertionError("the reader of a rule that is off was called")
    monkeypatch.setattr(memory, "_mem_available", must_not_run)
    monkeypatch.setattr(memory, "_cgroup_memory", must_not_run)
    store = _load(src, dst, sorted(arrays), min_available_after=None, min_cgroup_room_after=None)
    facts = store.facts()
    assert facts["mem_available_before"] is None
    assert all(facts[k] is None for k in ("cgroup", "cgroup_memory_high", "cgroup_memory_current_before",
                                          "cgroup_inactive_file_before", "cgroup_active_file_before",
                                          "cgroup_room_before"))
    assert facts["manifest_sha256"] == _sha(compact.manifest_path(dst)) and len(store) == 3


def test_load_refuses_changed_data_file(tmp_path: Path) -> None:
    src, dst, arrays = _built(tmp_path)
    key = "A/train_2019-03-27.rec"
    _flip(compact.data_path(dst, key))
    with pytest.raises(StoreMismatch, match=re.escape(key)):
        _load(src, dst, sorted(arrays))


def test_load_refuses_changed_context(tmp_path: Path) -> None:
    src, dst, arrays = _built(tmp_path)
    keys = sorted(arrays)
    with pytest.raises(StoreMismatch):
        _load(src, dst, keys, context={"plan.json": src / "plan.json"})                      # a name is missing
    with pytest.raises(StoreMismatch):
        _load(src, dst, keys, context={**_ctx(src), "other.json": src / "plan.json"})        # a name too many
    (src / "plan.json").write_text('{"plan": 2}\n')
    with pytest.raises(StoreMismatch):
        _load(src, dst, keys)


def test_load_refuses_changed_sidecar(tmp_path: Path) -> None:
    src, dst, arrays = _built(tmp_path)
    key = "A/train_2019-01-30.rec"
    path = compact.sidecar_path(dst, key)
    side = json.loads(path.read_text())
    side["block_bytes"][0] += 1
    path.write_text(json.dumps(side, sort_keys=True) + "\n")
    with pytest.raises(StoreMismatch, match=re.escape(key)):
        _load(src, dst, sorted(arrays))


def test_load_refuses_missing_manifest_or_key(tmp_path: Path) -> None:
    src, arrays = _tree(tmp_path)
    dst = tmp_path / "dst"
    _convert(src, dst)
    with pytest.raises(StoreMissing):                                   # converted, not sealed
        _load(src, dst, sorted(arrays))
    _finalize(src, dst)
    with pytest.raises(StoreMissing):
        _load(src, dst, ["A/train_2019-01-30.rec", "A/not_there.rec"])


def test_load_refuses_other_rows_dtype_or_changed_source(tmp_path: Path) -> None:
    src, dst, arrays = _built(tmp_path)
    key = "A/train_2019-01-30.rec"
    source, rows = src / (key + ".npy"), len(arrays[key])
    assert len(_load(src, dst, [key], dtype=REC, rows=[rows], sources=[source])) == 1
    with pytest.raises(StoreMismatch):
        _load(src, dst, [key], rows=[rows + 1])
    with pytest.raises(StoreMismatch):
        _load(src, dst, [key], dtype=np.dtype("<f4"))
    st = source.stat()
    os.utime(source, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    with pytest.raises(StoreMismatch):
        _load(src, dst, [key], sources=[source])


def test_load_refuses_when_memory_is_short(tmp_path: Path, monkeypatch) -> None:
    src, dst, arrays = _built(tmp_path)
    keys, real = sorted(arrays), memory._mem_available
    with pytest.raises(NotEnoughMemory):
        _load(src, dst, keys, min_available_after=2**60)
    monkeypatch.setattr(memory, "_mem_available", lambda: 0)
    with pytest.raises(NotEnoughMemory):                                # the store's own bytes exceed what is available
        _load(src, dst, keys, min_available_after=0)
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       24609884 kB\nMemFree:          161168 kB\n")
    with pytest.raises(CompactError):
        real(str(meminfo))
    meminfo.write_text("MemTotal:       24609884 kB\nMemAvailable:       2048 kB\n")
    assert real(str(meminfo)) == 2048 * 1024


def test_load_gives_pages_back_and_holds_no_file(tmp_path: Path, monkeypatch) -> None:
    src, dst, arrays = _built(tmp_path)
    keys = sorted(arrays)
    calls, real = [], os.posix_fadvise

    def spy(fd, offset, length, advice):
        calls.append((os.fstat(fd).st_ino, offset, length, advice))
        return real(fd, offset, length, advice)

    with monkeypatch.context() as m:
        m.setattr(os, "posix_fadvise", spy)
        store = _load(src, dst, keys)
    assert calls == [(compact.data_path(dst, k).stat().st_ino, 0, 0, DONTNEED) for k in keys]
    assert type(store._buf) is np.ndarray and isinstance(store._buf.base.obj, mmap.mmap)     # no np.memmap
    region = _smaps(store._buf)
    assert region["path"] == "" and "nh" in region["VmFlags"]           # no file behind it, and never a 2 MB page
    for key in keys:
        compact.data_path(dst, key).unlink()
    for i, key in enumerate(keys):
        assert store.windows(i, [0], len(arrays[key])).tobytes() == arrays[key].tobytes()


def test_load_refuses_when_the_cgroup_has_no_room(tmp_path: Path, monkeypatch) -> None:
    src, dst, arrays = _built(tmp_path)
    keys = sorted(arrays)
    total = sum(compact.data_path(dst, k).stat().st_size for k in keys)
    folder = "/sys/fs/cgroup/user.slice/unit.scope"
    high = total + 5000
    monkeypatch.setattr(memory, "_cgroup_memory", lambda: (folder, high, 4300, 0, 0))            # 700 bytes beyond the store
    with pytest.raises(NotEnoughMemory, match="memory.high"):
        _load(src, dst, keys, min_cgroup_room_after=1000)
    # the same limit and the same memory.current, but 300 bytes of it are page cache: only that makes the room enough
    monkeypatch.setattr(memory, "_cgroup_memory", lambda: (folder, high, 4300, 200, 100))
    store = _load(src, dst, keys, min_cgroup_room_after=1000)           # exactly the room asked for is enough
    facts = store.facts()
    assert (facts["cgroup"], facts["cgroup_memory_high"], facts["cgroup_memory_current_before"],
            facts["cgroup_inactive_file_before"], facts["cgroup_active_file_before"],
            facts["cgroup_room_before"]) == (folder, high, 4300, 200, 100, total + 1000)
    with pytest.raises(NotEnoughMemory) as refused:
        _load(src, dst, keys, min_cgroup_room_after=1001)
    for part in (folder, f"memory.high {high}", "memory.current 4300", "300 bytes are page cache", "inactive_file 200",
                 "active_file 100", f"needs {total} bytes", f"leaves {total + 1000} bytes", "1001 bytes"):
        assert part in str(refused.value)
    monkeypatch.setattr(memory, "_cgroup_memory", lambda: (folder, None, 2**60, 0, 0))           # max: no limit
    free = _load(src, dst, keys, min_cgroup_room_after=2**60)
    assert len(free) == 3 and free.facts()["cgroup"] == folder and free.facts()["cgroup_room_before"] is None

    asked, allocate = [], memory._allocate
    monkeypatch.setattr(memory, "_allocate", lambda n: asked.append(n) or allocate(n))
    monkeypatch.setattr(memory, "_cgroup_memory", lambda: (folder, total - 1, 0, 0, 0))
    with pytest.raises(NotEnoughMemory, match="memory.high"):
        _load(src, dst, keys)
    assert asked == []                                                  # refused before the memory was asked for

    def unreadable():
        raise CompactError("memory.stat: cannot be read; the cgroup rule cannot be checked")

    monkeypatch.setattr(memory, "_cgroup_memory", unreadable)
    with pytest.raises(CompactError, match="cannot be checked") as stopped:
        _load(src, dst, keys)
    assert not isinstance(stopped.value, NotEnoughMemory) and asked == []
    monkeypatch.setattr(memory, "_mem_available", lambda: 0)            # both rules refuse: the first one speaks
    monkeypatch.setattr(memory, "_cgroup_memory", lambda: (folder, 1, 0, 0, 0))
    with pytest.raises(NotEnoughMemory, match="MemAvailable"):
        _load(src, dst, keys)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork does not exist here")
def test_windows_in_forked_processes(tmp_path: Path) -> None:
    src, dst, arrays = _built(tmp_path)
    keys = sorted(arrays)
    store = _load(src, dst, keys)

    def digest() -> str:
        rng, h = np.random.default_rng(25), hashlib.sha256()
        for _ in range(20):
            i = int(rng.integers(len(keys)))
            h.update(store.windows(i, rng.integers(0, store.n_rows[i] - 50 + 1, 64), 50).tobytes())
        return h.hexdigest()

    want = digest()                         # the parent has its decompressor now; a child must make its own
    children = []
    for _ in range(2):
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:                        # the child: never back into pytest
            code = 1
            try:
                os.close(r)
                os.write(w, digest().encode())
                code = 0
            finally:
                os._exit(code)
        os.close(w)
        children.append((pid, r))
    got = []
    for pid, r in children:
        with os.fdopen(r) as fh:
            got.append(fh.read())
        assert os.waitpid(pid, 0)[1] == 0
    assert got == [want, want]
    assert digest() == want


def test_frames_have_content_size_and_decode_as_one_stream(tmp_path: Path) -> None:
    src, dst, arrays = _built(tmp_path)
    key = "A/train_2019-01-30.rec"
    blob = compact.data_path(dst, key).read_bytes()
    side = json.loads(compact.sidecar_path(dst, key).read_text())
    params = zstandard.get_frame_parameters(blob[:side["block_bytes"][0]])
    assert params.content_size == 256 * REC.itemsize and not params.has_checksum and params.dict_id == 0
    reader = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(blob), read_across_frames=True)
    out = b""
    while chunk := reader.read(1 << 16):                                # no block table is used here
        out += chunk
    assert out == arrays[key].tobytes()


# -- the backend of the training stream ----------------------------------------------------------
def test_compact_backend_equals_record_files(synth_run, tmp_path: Path) -> None:
    """The records of the synthetic run converted into a store: the window of every sample row, read from the store
    through the stream's reader, has the bytes read from the record file. A pass is an order of these windows."""
    from dof.deep.config import CachePaths
    from dof.deep.data.store import load_store
    from dof.deep.data.stream import close_days, train_days
    from dof.deep.data.windows import day_windows, read_windows
    cfg, c = synth_run.cfg, synth_run.cache
    root = tmp_path / "compact"
    compact.convert(c.train, root, "*/train_*.rec.npy", ["plan.json", "summary.json"], rate=0, pause=0, log=lambda m: None)
    compact.finalize(c.train, root, "*/train_*.rec.npy", ["plan.json", "summary.json"], rate=0, pause=0, log=lambda m: None)
    cache = CachePaths(c.train, c.eval, compact=root)
    machine = replace(cfg.machine, compact_store=True, compact_min_available_after=0, compact_min_cgroup_room_after=0)
    cfg2 = replace(cfg, machine=machine)
    stored = train_days(cfg2, cfg2.split, "all", cache)
    files = train_days(cfg2, cfg2.split, "all", cache, compact=False)
    try:
        facts = load_store(stored, log=lambda m: None)
        assert facts["files"] == len(stored) and facts["module"] == "dof.deep.data.compact"
        history = cfg.features.history
        for a, b in zip(stored, files, strict=True):
            assert (a.symbol, a.date, a.store is not None, b.store) == (b.symbol, b.date, True, None)
            rows = np.sort(a.rows).astype(np.int64)
            for start in range(0, len(rows), 256):
                first = rows[start:start + 256] - (history - 1)
                assert day_windows(a, first, history).tobytes() == read_windows(b, first, history).tobytes()
    finally:
        close_days(stored)
        close_days(files)
