"""Shared constants, builders and fixtures of the compact-store tests (``test_dof_deep_compact_*.py``).

Hermetic: every test writes under ``tmp_path`` only. ``load`` is called with ``min_available_after=0`` unless the
test is about memory, so no test depends on the machine's free memory. No test depends on the cgroup it runs in:
the autouse fixture ``_cgroup_without_limit`` (each test file imports it) gives ``load`` a cgroup without a limit.
The reader of the cgroup files is tested on a fake tree under ``tmp_path``; one test reads this process's real
cgroup.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dof.deep.cache.layout import REC
from dof.deep.data import compact
from dof.deep.data.compact import CompactStore, memory
from dof.deep.data.compact.layout import LOCK_NAME

convert_mod = importlib.import_module("dof.deep.data.compact.convert")   # the module; compact.convert is the function

NESTED = np.dtype([("a", "<i4"), ("b", [("c", "<f8", (2,)), ("d", "S3")])])
NAN_A, NAN_B, NEG_ZERO = 0x7FC00001, 0xFFC12345, 0x80000000        # two NaN bit patterns and -0.0, as uint32
PATTERN = "*/train_*.rec.npy"
CONTEXT = ("plan.json", "summary.json")
DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)
SUMMARY_KEYS = {"root", "manifest_sha256", "context", "files", "rows", "source_bytes", "compressed_bytes",
                "bytes_per_row", "verified_rows"}
FACT_KEYS = {"module", "format", "block_rows", "zstd_level", "zstandard", "libzstd", "files", "rows",
             "compressed_bytes", "bytes_per_row", "source_bytes", "root", "manifest_sha256", "entries_sha256",
             "context_sha256", "mem_available_before", "load_seconds", "cgroup", "cgroup_memory_high",
             "cgroup_memory_current_before", "cgroup_inactive_file_before", "cgroup_active_file_before",
             "cgroup_room_before"}
DISK_ONLY_FACTS = ("root", "manifest_sha256", "entries_sha256", "context_sha256", "mem_available_before", "load_seconds",
                   "cgroup", "cgroup_memory_high", "cgroup_memory_current_before", "cgroup_inactive_file_before",
                   "cgroup_active_file_before", "cgroup_room_before")
REAL_CGROUP_MEMORY = memory._cgroup_memory          # the function itself; the fixture below replaces the module's name
FAKE_CGROUP = ("/fake/cgroup/unit.scope", None, 0, 0, 0)           # memory.high is max: no limit


def _quiet(msg: str) -> None:
    pass


@pytest.fixture(autouse=True)
def _cgroup_without_limit(monkeypatch):
    """No test depends on the cgroup it runs in: ``load`` sees a cgroup without a limit unless the test puts
    another one in place."""
    monkeypatch.setattr(memory, "_cgroup_memory", lambda: FAKE_CGROUP)


def _rec(n: int, seed: int = 0) -> np.ndarray:
    """Records with what a lossless store must keep and a value comparison would lose: two NaN bit patterns,
    -0.0, both infinities; and a rising ``seq``. Every special value has its own cell, also when ``n`` is 1."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n, REC)
    out["zof"] = rng.standard_normal((n, 20), dtype=np.float32)
    out["y"] = rng.standard_normal((n, 10), dtype=np.float32)
    out["seq"] = np.cumsum(rng.integers(1, 50, n))
    out["labseq"] = out["seq"] + rng.integers(0, 1000, n)
    zu, yu = out["zof"].view("<u4"), out["y"].view("<u4")              # the bits, written as bits
    zu[0, 0], zu[n // 2, 3], yu[n - 1, 9] = NAN_A, NAN_B, NAN_A
    zu[n // 3, 1] = NEG_ZERO
    out["zof"][n // 4, 2], out["y"][n // 5, 0] = np.inf, -np.inf
    return out


def _parts(src: np.ndarray, key: str = "A/x.rec") -> tuple:
    frames, sizes = compact.encode_blocks(src.tobytes(), src.dtype.itemsize)
    return key, src.dtype, len(src), frames, sizes


def _store(src: np.ndarray) -> CompactStore:
    return CompactStore.from_blocks([_parts(src)])


def _slices(src: np.ndarray, first, width: int) -> bytes:
    return src[np.asarray(first)[:, None] + np.arange(width)].tobytes()


def _tree(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    """Three source files in two folders (1000, 256 * 3 and 70 rows) and the two context files."""
    src = tmp_path / "src"
    arrays = {"A/train_2019-01-30.rec": _rec(1000, 1), "A/train_2019-03-27.rec": _rec(256 * 3, 2),
              "B/train_2019-01-30.rec": _rec(70, 3)}
    for key, arr in arrays.items():
        path = src / (key + ".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
    (src / "plan.json").write_text('{"plan": 1}\n')
    (src / "summary.json").write_text('{"summary": 1}\n')
    return src, arrays


def _convert(src: Path, dst: Path, **kw) -> dict:
    kw.setdefault("log", _quiet)
    return compact.convert(src, dst, PATTERN, CONTEXT, rate=0, pause=0, **kw)


def _finalize(src: Path, dst: Path) -> dict:
    return compact.finalize(src, dst, PATTERN, CONTEXT, rate=0, pause=0, log=_quiet)


def _built(tmp_path: Path) -> tuple[Path, Path, dict[str, np.ndarray]]:
    src, arrays = _tree(tmp_path)
    dst = tmp_path / "dst"
    _convert(src, dst)
    _finalize(src, dst)
    return src, dst, arrays


def _ctx(src: Path) -> dict[str, Path]:
    return {name: src / name for name in CONTEXT}


def _load(src: Path, dst: Path, keys, **kw) -> CompactStore:
    kw.setdefault("min_available_after", 0)
    kw.setdefault("min_cgroup_room_after", 0)
    return CompactStore.load(dst, keys, context=kw.pop("context", _ctx(src)), **kw)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flip(path: Path) -> None:
    """One byte of the file changed, its size kept."""
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    path.write_bytes(raw)


def _store_files(dst: Path) -> list[Path]:
    return sorted(p for p in dst.rglob("*") if p.is_file() and p.name != LOCK_NAME) if dst.exists() else []


def _count_decompress(monkeypatch) -> list:
    calls, real = [], CompactStore._decompress

    def counting(self, index, block):
        calls.append((index, block))
        return real(self, index, block)

    monkeypatch.setattr(CompactStore, "_decompress", counting)
    return calls


def _spies(monkeypatch):
    """One event list for a spy on os.posix_fadvise and a stand-in for the converter's Pacer that never waits: the
    order of the events is the order of the calls."""
    events, real = [], os.posix_fadvise

    def fadvise(fd, offset, length, advice):
        events.append(("advise", offset, length, advice))
        return real(fd, offset, length, advice)

    class SpyPacer:
        def __init__(self, rate=0, pause=0) -> None:
            self.rate, self.pause = rate, pause

        def after(self, nbytes: int) -> None:
            events.append(("pace", nbytes))

    monkeypatch.setattr(os, "posix_fadvise", fadvise)
    return events, SpyPacer


def _check_paced(events: list, sizes: list[int]) -> None:
    """Every byte of the data files went through the pacer in 4096-byte pieces, each piece given back to the
    page cache before the wait, and each file given back whole once."""
    paced = [e[1] for e in events if e[0] == "pace"]
    assert sum(paced) == sum(sizes)
    assert len(paced) == sum(math.ceil(s / 4096) for s in sizes)
    for i, e in enumerate(events):
        if e[0] == "pace":
            assert events[i - 1][0] == "advise" and events[i - 1][2] == e[1] and events[i - 1][3] == DONTNEED
    assert [e for e in events if e[:3] == ("advise", 0, 0)] == [("advise", 0, 0, DONTNEED)] * len(sizes)


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", "-m", "dof.deep.data.compact", *args], capture_output=True, text=True,
                          timeout=300)


def _smaps_at(address: int) -> dict | None:
    """The mapping of /proc/self/smaps that holds ``address``: its path ("" for anonymous memory) and its fields."""
    cur = None
    with open("/proc/self/smaps") as fh:
        for line in fh:
            head = line.split()
            if "-" in head[0] and not head[0].endswith(":"):
                a, b = (int(x, 16) for x in head[0].split("-"))
                cur = {"path": head[5] if len(head) > 5 else ""} if a <= address < b else None
            elif cur is not None:
                key, _, value = line.partition(":")
                cur[key] = value.split()
                if key == "VmFlags":
                    return cur
    return None


def _smaps(arr: np.ndarray) -> dict | None:
    return _smaps_at(arr.__array_interface__["data"][0])


def _fake_cgroup(tmp_path: Path, high: str | None = "max", current: str | None = "1000",
                 stat: str | None = "anon 400\ninactive_file 300\nactive_file 200\n",
                 entry: str = "0::/user.slice/unit.scope\n") -> tuple[str, str, str]:
    """A stand-in for /proc/self/cgroup and /sys/fs/cgroup under ``tmp_path``: (proc file, root, the unit's folder).
    None for the text of a file: the file is not there."""
    proc, root = tmp_path / "proc_self_cgroup", tmp_path / "sys_fs_cgroup"
    folder = root / "user.slice" / "unit.scope"
    folder.mkdir(parents=True, exist_ok=True)
    proc.write_text(entry)
    for name, text in (("memory.high", high), ("memory.current", current), ("memory.stat", stat)):
        if text is None:
            (folder / name).unlink(missing_ok=True)
        else:
            (folder / name).write_text(text if text.endswith("\n") else text + "\n")
    return str(proc), str(root), str(folder)


def _own_cgroup_has(name: str) -> bool:
    """Whether this process's own cgroup v2 folder holds the file ``name``. The folder is found as
    ``memory._cgroup_memory`` finds it: the ``0::`` line of /proc/self/cgroup, under /sys/fs/cgroup. In the root
    cgroup, or in one without the memory controller, the test of the real cgroup has nothing to read and is
    skipped, not failed."""
    try:
        with open("/proc/self/cgroup") as fh:
            entries = [line.rstrip("\n") for line in fh if line.startswith("0::")]
    except OSError:
        return False
    return bool(entries) and os.path.exists(os.path.join("/sys/fs/cgroup", entries[0][len("0::"):].lstrip("/"), name))
