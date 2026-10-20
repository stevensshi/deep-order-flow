"""Conversion into a compact store: data files and sidecars, resume, the checks before anything is written, the
seal, and the pace and page cache of every read."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zstandard")

from compact_helpers import (  # noqa: E402
    CONTEXT,
    DONTNEED,
    REC,
    SUMMARY_KEYS,
    _built,
    _cgroup_without_limit,  # noqa: F401  (the autouse fixture, bound in this module)
    _check_paced,
    _convert,
    _finalize,
    _flip,
    _quiet,
    _rec,
    _sha,
    _spies,
    _store_files,
    _tree,
    convert_mod,
)
from dof.deep.data import compact  # noqa: E402
from dof.deep.data.compact import CompactError, CompactStore, StoreMismatch, StoreMissing, VerificationError, memory  # noqa: E402


def test_source_must_be_one_dimensional_fixed_size(tmp_path: Path) -> None:
    cases = {"two_d": np.zeros((4, 3), "<f4"), "objects": np.array([{"a": 1}, None], dtype=object),
             "empty": np.zeros(0, REC)}
    for name, arr in cases.items():
        path = tmp_path / f"{name}.npy"
        np.save(path, arr)
        with pytest.raises(ValueError):
            compact.read_npy_header(path)
    good = tmp_path / "good.npy"
    np.save(good, _rec(5))
    assert compact.read_npy_header(good) == (REC, 5, 192)


def test_convert_writes_data_and_sidecar(tmp_path: Path) -> None:
    src, arrays = _tree(tmp_path)
    dst = tmp_path / "deep" / "dst"
    assert not dst.exists() and not dst.parent.exists()
    _convert(src, dst)
    context = {name: _sha(src / name) for name in CONTEXT}
    for key, arr in arrays.items():
        raw = (src / (key + ".npy")).read_bytes()
        data, side = compact.data_path(dst, key), json.loads(compact.sidecar_path(dst, key).read_text())
        assert data.is_file()
        assert side["key"] == key and side["rows"] == len(arr)
        assert side["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
        offset = side["source"]["data_offset"]
        assert raw[offset:] == arr.tobytes()
        assert side["source"]["payload_sha256"] == hashlib.sha256(raw[offset:]).hexdigest()
        assert sum(side["block_bytes"]) == data.stat().st_size == side["compressed"]["bytes"]
        assert side["compressed"]["sha256"] == _sha(data)
        assert side["blocks"] == len(side["block_bytes"]) == math.ceil(len(arr) / 256)
        assert side["context"] == context
        assert side["verified"]["rows"] == len(arr) and side["verified"]["equal"] is True


def test_convert_is_resumable_and_redoes_only_what_is_missing(tmp_path: Path) -> None:
    src, arrays = _tree(tmp_path)
    dst = tmp_path / "dst"
    first = _convert(src, dst)
    assert first["converted"] == 3 and first["skipped"] == 0 and isinstance(first["mean_rate_mb_s"], float)
    stat = lambda: {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in _store_files(dst)}        # noqa: E731
    before = stat()
    assert len(before) == 6
    second = _convert(src, dst)
    assert second["converted"] == 0 and second["skipped"] == 3 and second["mean_rate_mb_s"] is None
    assert stat() == before                                             # nothing was written again
    gone = "A/train_2019-03-27.rec"
    compact.sidecar_path(dst, gone).unlink()
    lines: list[str] = []
    third = _convert(src, dst, log=lines.append)
    assert third["converted"] == 1 and third["skipped"] == 2
    assert [line.split()[1:3] for line in lines[:3]] == [["skipped", "A/train_2019-01-30.rec"], ["converted", gone],
                                                         ["skipped", "B/train_2019-01-30.rec"]]
    after = stat()
    theirs = {compact.data_path(dst, gone), compact.sidecar_path(dst, gone)}
    assert set(after) == set(before)
    assert {p: v for p, v in after.items() if p not in theirs} == {p: v for p, v in before.items() if p not in theirs}
    # the data file was replaced by a new file: the old one still existed when the new one was made
    assert after[compact.data_path(dst, gone)][0] != before[compact.data_path(dst, gone)][0]


def test_convert_writes_nothing_when_windows_differ(tmp_path: Path, monkeypatch) -> None:
    src, _ = _tree(tmp_path)
    dst = tmp_path / "dst"
    real = CompactStore.windows

    def flipped(self, index, first, width):
        out = real(self, index, first, width)
        out.view(np.uint8).reshape(-1)[7] ^= 1
        return out

    monkeypatch.setattr(CompactStore, "windows", flipped)
    with pytest.raises(VerificationError):
        _convert(src, dst)
    assert _store_files(dst) == []                                      # no .zst, no sidecar, no .tmp


def test_convert_stops_below_disk_floor(tmp_path: Path, monkeypatch) -> None:
    src, _ = _tree(tmp_path)
    dst = tmp_path / "dst"
    monkeypatch.setattr(memory, "_free_bytes", lambda path: 10)
    with pytest.raises(CompactError, match="floor"):
        _convert(src, dst, free_path=tmp_path, min_free_bytes=11)
    assert _store_files(dst) == []
    assert _convert(src, dst, free_path=tmp_path, min_free_bytes=10)["converted"] == 3      # at the floor is not below it


def test_convert_stops_when_source_changes_while_read(tmp_path: Path, monkeypatch) -> None:
    src, _ = _tree(tmp_path)
    dst, key = tmp_path / "dst", "A/train_2019-01-30.rec"
    calls, real = [], convert_mod._stat_id

    def moved(fd):
        size, mtime_ns = real(fd)
        calls.append(fd)
        return (size, mtime_ns + 1) if len(calls) == 2 else (size, mtime_ns)

    monkeypatch.setattr(convert_mod, "_stat_id", moved)
    with pytest.raises(CompactError, match="changed while it was read"):
        compact.convert_file(src / (key + ".npy"), key, dst, {}, compact.Pacer(0, 0))
    assert len(calls) == 2
    assert _store_files(dst) == [] and not dst.exists()


def test_finalize_writes_manifest_and_seals(tmp_path: Path) -> None:
    src, arrays = _tree(tmp_path)
    dst = tmp_path / "dst"
    _convert(src, dst)
    summary = _finalize(src, dst)
    path = compact.manifest_path(dst)
    manifest = json.loads(path.read_text())
    assert sorted(manifest["entries"]) == sorted(arrays)
    rows = sum(len(a) for a in arrays.values())
    source = sum((src / (k + ".npy")).stat().st_size for k in arrays)
    compressed = sum(compact.data_path(dst, k).stat().st_size for k in arrays)
    assert manifest["totals"] == {"files": 3, "rows": rows, "source_bytes": source, "compressed_bytes": compressed,
                                  "bytes_per_row": compressed / rows, "verified_rows": rows}
    assert manifest["context"] == {name: _sha(src / name) for name in CONTEXT}
    for key in arrays:
        assert manifest["entries"][key]["sidecar_sha256"] == _sha(compact.sidecar_path(dst, key))
    assert set(summary) == SUMMARY_KEYS and len(summary) == 9
    assert summary["manifest_sha256"] == _sha(path) and summary["files"] == 3 and summary["rows"] == rows
    assert compact.show(dst) == summary
    with pytest.raises(CompactError, match="sealed"):
        _convert(src, dst)
    with pytest.raises(CompactError, match="sealed"):
        _finalize(src, dst)


def test_finalize_refuses_missing_extra_stale_or_corrupt(tmp_path: Path) -> None:
    key = "A/train_2019-03-27.rec"

    def case(name: str) -> tuple[Path, Path]:
        src, _ = _tree(tmp_path / name)
        dst = tmp_path / name / "dst"
        _convert(src, dst)
        return src, dst

    src, dst = case("sidecar_removed")
    compact.sidecar_path(dst, key).unlink()
    with pytest.raises(StoreMissing):
        _finalize(src, dst)
    assert not compact.manifest_path(dst).exists()

    src, dst = case("source_not_converted")
    np.save(src / "B/train_2019-03-27.rec.npy", _rec(10))
    with pytest.raises(StoreMissing):
        _finalize(src, dst)
    assert not compact.manifest_path(dst).exists()

    src, dst = case("context_changed")
    (src / "plan.json").write_text('{"plan": 2}\n')
    with pytest.raises(StoreMismatch):
        _finalize(src, dst)
    assert not compact.manifest_path(dst).exists()

    src, dst = case("data_corrupt")
    _flip(compact.data_path(dst, key))
    with pytest.raises(StoreMismatch):
        _finalize(src, dst)
    assert not compact.manifest_path(dst).exists()


# -- pacing -------------------------------------------------------------------------------------
class _Clock:
    """A clock that moves only when somebody sleeps or says that time has passed."""

    def __init__(self) -> None:
        self.t, self.sleeps = 100.0, []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def test_pacer_holds_the_rate_without_bursts() -> None:
    chunk = 4 * 2**20
    c = _Clock()
    pacer = compact.Pacer(10_000_000, 0.05, clock=c.now, sleep=c.sleep)
    for _ in range(10):
        pacer.after(chunk)
    assert c.sleeps == pytest.approx([0.4194304] * 10)                  # 4 MiB at 10 MB/s
    c.t += 5.0                                                          # other work
    pacer.after(chunk)
    assert c.sleeps[-1] == 0.05                                         # only the pause
    pacer.after(chunk)
    assert c.sleeps[-1] == pytest.approx(0.4194304)                     # and no credit from the 5 s: no burst follows
    c = _Clock()
    free = compact.Pacer(0, 0.05, clock=c.now, sleep=c.sleep)
    free.after(chunk)
    free.after(chunk)
    assert c.sleeps == [0.05, 0.05]                                     # rate 0: only the pause
    c = _Clock()
    compact.Pacer(0, 0, clock=c.now, sleep=c.sleep).after(chunk)
    assert c.sleeps == []


# -- the pace and the page cache of every read --------------------------------------------------
def test_convert_file_paces_every_chunk_and_gives_pages_back(tmp_path: Path, monkeypatch) -> None:
    src, _ = _tree(tmp_path)
    key = "A/train_2019-01-30.rec"
    path = src / (key + ".npy")
    assert path.stat().st_size == 136_192                               # a 192-byte header and 1000 records of 136 bytes
    monkeypatch.setattr(convert_mod, "READ_CHUNK", 4096)
    events, SpyPacer = _spies(monkeypatch)
    compact.convert_file(path, key, tmp_path / "dst", {}, SpyPacer(0, 0))
    paced = [e[1] for e in events if e[0] == "pace"]
    assert paced == [4096] * 33 + [1024] and sum(paced) == path.stat().st_size
    offset = 0
    for i, e in enumerate(events):
        if e[0] == "pace":                                              # the pages go back before the wait
            assert events[i - 1] == ("advise", offset, e[1], DONTNEED)
            offset += e[1]
    assert all(e[3] == DONTNEED for e in events if e[0] == "advise")
    assert [e for e in events if e[:3] == ("advise", 0, 0)] == [("advise", 0, 0, DONTNEED)] * 2   # the source, the new .zst


def test_finalize_reads_data_files_through_the_pacer(tmp_path: Path, monkeypatch) -> None:
    src, arrays = _tree(tmp_path)
    dst = tmp_path / "dst"
    _convert(src, dst)
    sizes = [compact.data_path(dst, k).stat().st_size for k in sorted(arrays)]
    monkeypatch.setattr(convert_mod, "READ_CHUNK", 4096)
    events, SpyPacer = _spies(monkeypatch)
    monkeypatch.setattr(convert_mod, "Pacer", SpyPacer)
    _finalize(src, dst)
    _check_paced(events, sizes)


def test_verify_reads_data_files_through_the_pacer(tmp_path: Path, monkeypatch) -> None:
    src, dst, arrays = _built(tmp_path)
    sizes = [compact.data_path(dst, k).stat().st_size for k in sorted(arrays)]
    monkeypatch.setattr(convert_mod, "READ_CHUNK", 4096)
    events, SpyPacer = _spies(monkeypatch)
    monkeypatch.setattr(convert_mod, "Pacer", SpyPacer)
    assert set(compact.verify(dst, src_root=src, rate=0, pause=0, log=_quiet)) == SUMMARY_KEYS
    _check_paced(events, sizes)
