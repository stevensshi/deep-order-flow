"""The cache build on the synthetic run: layout, resume, stale group records, the latest group's scalers, the
test-day refusal, no machine path in any cache file."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from dof.deep.cache import build
from dof.deep.cache import layout as L
from dof.deep.cache.build import eval_task, load_plan
from dof.deep.config import CachePaths
from dof.deep.errors import CacheError, TestDayRefused
from recipes import hashing as H

SYMBOLS = ("AAA", "BBB", "CCC")


def _copy(synth_run, tmp_path: Path) -> tuple:
    """A writable copy of the synthetic cache and the configuration that points at it."""
    cache = CachePaths(tmp_path / "train", tmp_path / "eval")
    shutil.copytree(synth_run.cache.train, cache.train)
    shutil.copytree(synth_run.cache.eval, cache.eval)
    cfg = replace(synth_run.cfg, machine=replace(synth_run.cfg.machine, cache={"synth": cache}))
    return cfg, cache


def _quiet(msg: str) -> None:
    pass


def test_build_layout(synth_run) -> None:
    c, cfg = synth_run.cache, synth_run.cfg
    assert L.plan_path(c).is_file() and L.summary_path(c).is_file()
    assert sorted(p.name for p in (c.train / "probe").iterdir()) == sorted(
        f"{s}_{d}.json" for s in SYMBOLS for d in cfg.split.train_days)
    plan = load_plan(c)
    for s in SYMBOLS:
        groups = plan["stocks"][s]["groups"]
        assert len(groups) == 1 and groups[0]["status"] == "ok"
        assert L.group_path(c, s, 0).is_file()
        assert L.scaler_path(c, s, 0, "of").is_file() and L.scaler_path(c, s, 0, "y").is_file()
        for d in cfg.split.train_days:
            assert L.rec_path(c, s, d).is_file() and L.rows_path(c, s, d).is_file()
        for part in ("valid", "test"):
            d = cfg.split.days(part)[0]
            meta = json.loads(L.eval_meta_path(c, part, s, d).read_text(encoding="utf-8"))
            assert meta["status"] == "used"
            for name in L.EVAL_ARRAYS:
                assert L.eval_arr_path(c, part, s, d, name).is_file()
    summary = json.loads(L.summary_path(c).read_text(encoding="utf-8"))
    assert sorted(summary["days"]) == ["2019-01-30", "2019-01-31", "2019-02-01", "2019-02-04"]
    assert summary["parts"] == ["train", "valid", "test"] and "prereg_sha256" not in summary


def _times(cache: CachePaths) -> dict:
    return {p.as_posix(): p.stat().st_mtime_ns for root in (cache.train, cache.eval) for p in root.rglob("*") if p.is_file()}


def test_build_resumes_without_rewriting(synth_run, tmp_path: Path) -> None:
    cfg, cache = _copy(synth_run, tmp_path)
    before = _times(cache)
    first = json.loads(L.summary_path(cache).read_text(encoding="utf-8"))
    build(cfg, cfg.split, cache, workers=0, log=_quiet)
    after = _times(cache)
    summary = L.summary_path(cache).as_posix()
    assert set(after) == set(before)
    assert {k: v for k, v in after.items() if k != summary} == {k: v for k, v in before.items() if k != summary}
    again = json.loads(L.summary_path(cache).read_text(encoding="utf-8"))
    volatile = ("written", "bytes_train_cache", "bytes_eval_cache")
    assert {k: v for k, v in again.items() if k not in volatile} == {k: v for k, v in first.items() if k not in volatile}


def test_stale_group_record_refused(synth_run, tmp_path: Path) -> None:
    cfg, cache = _copy(synth_run, tmp_path)
    p = L.group_path(cache, "AAA", 0)
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["delta_t_ms"] += 1.0
    p.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    with pytest.raises(CacheError, match="AAA g0"):
        build(cfg, cfg.split, cache, workers=0, log=_quiet)


def test_eval_uses_latest_group_scalers(synth_run) -> None:
    c, cfg = synth_run.cache, synth_run.cfg
    plan = load_plan(c)
    for s in SYMBOLS:
        meta = json.loads(L.eval_meta_path(c, "valid", s, cfg.split.valid_days[0]).read_text(encoding="utf-8"))
        assert meta["group"] == plan["stocks"][s]["groups"][-1]["id"]


def test_test_days_refused_without_document(synth_run, tmp_path: Path) -> None:
    cfg, cache = _copy(synth_run, tmp_path)
    shutil.rmtree(cache.eval / "test")
    strict = replace(cfg, protocol=replace(cfg.protocol, test_days_need_document=True, document=None))
    with pytest.raises(TestDayRefused):
        eval_task(strict, strict.split, cache, "test", "AAA", strict.split.test_days[0])
    missing = replace(cfg, protocol=replace(cfg.protocol, test_days_need_document=True,
                                            document=tmp_path / "no_such_document.md"))
    with pytest.raises(TestDayRefused):
        eval_task(missing, missing.split, cache, "test", "AAA", missing.split.test_days[0])


def test_cache_json_has_no_absolute_path(synth_run) -> None:
    c = synth_run.cache
    files = [p for root in (c.train, c.eval) for p in root.rglob("*.json")]
    assert files
    for p in files:
        H.assert_no_machine_paths(json.loads(p.read_text(encoding="utf-8")))
