"""Scoring the test days once, on the synthetic run: every scored model must have ended, the corporate-action
file leaves CCC out, nothing is scored twice, a diverged fit is never scored."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dof.deep.config import Seeds  # noqa: E402
from dof.deep.errors import RecordConflict, TestDayRefused  # noqa: E402
from dof.deep.evaluate.scoring import score_test  # noqa: E402
from dof.deep.linear.ridge import fit_ridge  # noqa: E402
from dof.deep.train.fit import fit  # noqa: E402


def _quiet(msg: str) -> None:
    pass


@pytest.fixture(scope="module")
def fitted(synth_run, tmp_path_factory) -> Path:
    """Records of every scored model, no test file yet."""
    cfg = synth_run.cfg
    root = tmp_path_factory.mktemp("fitted")
    for name in cfg.train.fit_order:
        fit(cfg, cfg.split, name, Seeds.same(0), root / name, log=_quiet)
    fit_ridge(cfg, cfg.split, root, workers=0, log=_quiet)
    return root


@pytest.fixture(scope="module")
def scored(synth_run, fitted, tmp_path_factory) -> tuple[Path, dict]:
    root = tmp_path_factory.mktemp("scored") / "results"
    shutil.copytree(fitted, root)
    return root, score_test(synth_run.cfg, synth_run.cfg.split, root, workers=0, log=_quiet)


def _copy(src: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "results"
    shutil.copytree(src, dst)
    return dst


def test_score_test_needs_every_record(synth_run, fitted, tmp_path: Path) -> None:
    root = _copy(fitted, tmp_path)
    (root / "Ridge-recent" / "fit_seed0.json").unlink()
    with pytest.raises(TestDayRefused, match="Ridge-recent has not ended"):
        score_test(synth_run.cfg, synth_run.cfg.split, root, workers=0, log=_quiet)
    assert not list(root.glob("*/test_seed0.json"))


def test_ca_left_out_stock_day(synth_run, scored) -> None:
    root, out = scored
    assert sorted(out) == sorted(synth_run.cfg.scored_models)
    for name in synth_run.cfg.scored_models:
        doc = json.loads((root / name / "test_seed0.json").read_text(encoding="utf-8"))
        left = doc["corporate_actions"]["left_out"]
        assert [(x["symbol"], x["date"]) for x in left] == [("CCC", "2019-02-04")]
        assert {r["symbol"] for r in doc["scores"]} == {"AAA", "BBB"}
        assert sorted(doc["price_ratio"]) == ["AAA 2019-02-04", "BBB 2019-02-04"]


def test_model_not_scored_twice(synth_run, scored, tmp_path: Path) -> None:
    root = _copy(scored[0], tmp_path)
    before = {p: p.stat().st_mtime_ns for p in root.glob("*/test_seed0.json")}
    assert score_test(synth_run.cfg, synth_run.cfg.split, root, workers=0, log=_quiet) == {}
    assert {p: p.stat().st_mtime_ns for p in root.glob("*/test_seed0.json")} == before
    t = root / "TinyMLP-all" / "test_seed0.json"
    doc = json.loads(t.read_text(encoding="utf-8"))
    doc["corporate_actions"]["sha256"] = "0" * 64
    t.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(RecordConflict):
        score_test(synth_run.cfg, synth_run.cfg.split, root, workers=0, log=_quiet)


def test_diverged_model_not_scored(synth_run, fitted, tmp_path: Path) -> None:
    root = _copy(fitted, tmp_path)
    p = root / "TinyLSTM-recent" / "fit_seed0.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["status"] = "diverged"
    p.write_text(json.dumps(doc), encoding="utf-8")
    out = score_test(synth_run.cfg, synth_run.cfg.split, root, workers=0, log=_quiet)
    assert "TinyLSTM-recent" not in out and not (root / "TinyLSTM-recent" / "test_seed0.json").exists()
    assert sorted(out) == ["Ridge-all", "Ridge-recent", "TinyMLP-all"]
