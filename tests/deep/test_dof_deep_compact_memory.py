"""The store's memory (anonymous, never huge pages) and the reader of this process's cgroup."""

from __future__ import annotations

import gc
import mmap
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zstandard")

from compact_helpers import (  # noqa: E402
    REAL_CGROUP_MEMORY,
    _cgroup_without_limit,  # noqa: F401  (the autouse fixture, bound in this module)
    _fake_cgroup,
    _own_cgroup_has,
    _rec,
    _smaps,
    _smaps_at,
    _store,
)
from dof.deep.data.compact import CompactError, memory  # noqa: E402


@pytest.mark.skipif(not os.path.exists("/proc/self/smaps"), reason="/proc/self/smaps does not exist here")
def test_allocate_is_anonymous_memory_without_huge_pages() -> None:
    n = 8 * 2**20 + 123                                                 # four 2 MB pages could back it; not a whole number of pages
    buf = memory._allocate(n)
    assert type(buf) is np.ndarray and buf.dtype == np.uint8 and buf.shape == (n,)
    assert buf.flags.writeable and buf.flags.c_contiguous and not buf.flags.owndata
    assert isinstance(buf.base.obj, mmap.mmap)                          # the array holds its mapping
    buf[:] = 7                                                          # every page is in memory now
    region = _smaps(buf)
    assert "nh" in region["VmFlags"] and int(region["AnonHugePages"][0]) == 0
    # no file behind it, and all of it resident or in swap: a page swapped out since the write is not a fault
    assert region["path"] == "" and (int(region["Rss"][0]) + int(region["Swap"][0])) * 1024 >= n
    address = buf.__array_interface__["data"][0]
    view = buf[5:9]
    del buf, region
    gc.collect()
    assert view.tolist() == [7, 7, 7, 7] and "nh" in _smaps_at(address)["VmFlags"]      # a view keeps the mapping
    del view
    gc.collect()
    gone = _smaps_at(address)
    assert gone is None or "nh" not in gone["VmFlags"]                  # the last reference gave the memory back


def test_allocate_zero_bytes_and_a_platform_without_the_mark(monkeypatch) -> None:
    empty = memory._allocate(0)
    assert empty.shape == (0,) and empty.dtype == np.uint8
    monkeypatch.delattr(mmap, "MADV_NOHUGEPAGE")
    with pytest.raises(CompactError, match="MADV_NOHUGEPAGE"):
        memory._allocate(4096)
    with pytest.raises(CompactError, match="MADV_NOHUGEPAGE"):
        _store(_rec(10))                                                # no store is built on other memory in its place


def test_cgroup_memory_reads_a_fake_tree(tmp_path: Path) -> None:
    proc, root, folder = _fake_cgroup(tmp_path)
    assert REAL_CGROUP_MEMORY(proc, root) == (folder, None, 1000, 300, 200)             # max: no limit
    _fake_cgroup(tmp_path, high="17179869184")
    assert REAL_CGROUP_MEMORY(proc, root) == (folder, 17179869184, 1000, 300, 200)
    for bad in ({"high": None}, {"current": None}, {"high": "lots"}, {"current": "max"},
                {"stat": None},                                         # memory.high is max here: the three files go together
                {"high": "17179869184", "stat": None},
                {"stat": "anon 400\ninactive_file 300\n"},              # no active_file line
                {"stat": "anon 400\ninactive_file many\nactive_file 200\n"}):
        _fake_cgroup(tmp_path, **bad)
        with pytest.raises(CompactError, match="cannot be checked"):
            REAL_CGROUP_MEMORY(proc, root)
    _fake_cgroup(tmp_path, entry="12:memory:/user.slice\n1:name=systemd:/user.slice/unit.scope\n")     # cgroup v1: no 0:: line
    with pytest.raises(CompactError, match="no cgroup v2 entry"):
        REAL_CGROUP_MEMORY(proc, root)
    _fake_cgroup(tmp_path, entry="12:memory:/user.slice\n0::/not/there.scope\n")        # hybrid: no memory files under the v2 path
    with pytest.raises(CompactError, match="cannot be checked"):
        REAL_CGROUP_MEMORY(proc, root)
    with pytest.raises(CompactError, match="cannot be checked"):
        REAL_CGROUP_MEMORY(str(tmp_path / "no_such_file"), root)
    _fake_cgroup(tmp_path, entry="0::/\n")                                               # the cgroup is the root of the tree
    (Path(root) / "memory.high").write_text("5000\n")
    (Path(root) / "memory.current").write_text("7\n")
    with pytest.raises(CompactError, match="cannot be checked"):                        # no memory.stat beside them
        REAL_CGROUP_MEMORY(proc, root)
    (Path(root) / "memory.stat").write_text("inactive_file 3\nactive_file 4\n")
    assert REAL_CGROUP_MEMORY(proc, root) == (os.path.join(root, ""), 5000, 7, 3, 4)


@pytest.mark.skipif(not os.path.exists("/sys/fs/cgroup/cgroup.controllers"), reason="no cgroup v2 here")
@pytest.mark.skipif(not _own_cgroup_has("memory.current"),
                    reason="this process's own cgroup has no memory.current (the root cgroup, or no memory controller)")
def test_cgroup_memory_of_this_process() -> None:
    folder, high, current, inactive, active = REAL_CGROUP_MEMORY()      # the real files, with this machine's layout
    assert folder.startswith("/sys/fs/cgroup") and os.path.isdir(folder)
    assert current > 0 and (high is None or high > 0) and inactive >= 0 and active >= 0
