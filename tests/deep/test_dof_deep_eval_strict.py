"""The evaluation days are checked against the cache summary: a missing or extra stock-day stops the run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from dof.deep.cache import layout as L
from dof.deep.config import CachePaths
from dof.deep.data.eval import eval_stocks
from dof.deep.errors import CacheError


def _copy(synth_run, tmp_path: Path) -> CachePaths:
    cache = CachePaths(tmp_path / "train", tmp_path / "eval")
    shutil.copytree(synth_run.cache.train, cache.train)
    shutil.copytree(synth_run.cache.eval, cache.eval)
    return cache


def _edit_summary(cache: CachePaths, change) -> None:
    p = L.summary_path(cache)
    doc = json.loads(p.read_text(encoding="utf-8"))
    change(doc)
    p.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def test_complete_cache_lists_every_used_stock(synth_run) -> None:
    cfg = synth_run.cfg
    assert [s for s, _ in eval_stocks(cfg, cfg.split, synth_run.cache, "valid")] == ["AAA", "BBB", "CCC"]


def test_deleted_meta_file_refused(synth_run, tmp_path: Path) -> None:
    cfg, cache = synth_run.cfg, _copy(synth_run, tmp_path)
    L.eval_meta_path(cache, "valid", "BBB", cfg.split.valid_days[0]).unlink()
    with pytest.raises(CacheError, match="2 used stock-days in the cache, the summary counts 3"):
        eval_stocks(cfg, cfg.split, cache, "valid")


def test_summary_count_raised_refused(synth_run, tmp_path: Path) -> None:
    cfg, cache = synth_run.cfg, _copy(synth_run, tmp_path)
    day = cfg.split.valid_days[0]
    _edit_summary(cache, lambda doc: doc["days"][day].update(used=doc["days"][day]["used"] + 1))
    with pytest.raises(CacheError, match="the summary counts 4"):
        eval_stocks(cfg, cfg.split, cache, "valid")


def test_summary_without_part_refused(synth_run, tmp_path: Path) -> None:
    cfg, cache = synth_run.cfg, _copy(synth_run, tmp_path)
    _edit_summary(cache, lambda doc: doc.update(parts=["train", "test"]))
    with pytest.raises(CacheError, match="valid"):
        eval_stocks(cfg, cfg.split, cache, "valid")
    L.summary_path(cache).unlink()
    with pytest.raises(CacheError, match="missing"):
        eval_stocks(cfg, cfg.split, cache, "test")
