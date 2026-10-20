"""The fit loop on the synthetic run: files and record, reuse, restart, divergence, the placebo, mixed precision."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dof.deep.config import FitSpec, Seeds, identity_sha256  # noqa: E402
from dof.deep.data.stream import close_days, draw_targets, train_days  # noqa: E402
from dof.deep.errors import CheckpointConflict, ConfigError, RecordConflict  # noqa: E402
from dof.deep.models import FAMILIES  # noqa: E402
from dof.deep.train.fit import FitHooks, fit  # noqa: E402
from recipes.train_values import CHECK_FIELDS, FIT_FIELDS  # noqa: E402

SEEDS = Seeds.same(0)


class _Stop(Exception):
    """Stands for a fit that is stopped between two checks."""


def _stop_at(n: int) -> FitHooks:
    def hook(check: int) -> None:
        if check == n:
            raise _Stop(f"stopped before the validation reading of check {n}")
    return FitHooks(before_validation=hook)


def _quiet(msg: str) -> None:
    pass


def test_fit_writes_weights_then_record(synth_run, tmp_path: Path) -> None:
    cfg = synth_run.cfg
    out = tmp_path / "TinyMLP-all"
    rec = fit(cfg, cfg.split, "TinyMLP-all", SEEDS, out, log=_quiet)
    assert rec.status == "finished" and rec.weights_path == out / "best_weights_seed0.pt" and rec.weights_path.is_file()
    assert (out / "fit_seed0.json").is_file() and not (out / "checkpoint_seed0.pt").exists()
    assert set(FIT_FIELDS) <= set(rec.doc) and all(set(CHECK_FIELDS) <= set(c) for c in rec.doc["checks"])
    assert len(rec.doc["checks"]) == 2 and rec.doc["stopped_by"] == "max checks"
    assert rec.doc["identity_sha256"] == identity_sha256(cfg, cfg.split, SEEDS, "TinyMLP-all")
    assert rec.doc["restore_check"]["abs_diff"] == 0.0
    assert rec.doc["device_settings"]["type"] == "cpu" and rec.doc["seeds"] == {"init": 0, "data": 0}


def test_existing_record_is_reused(synth_run, tmp_path: Path) -> None:
    cfg = synth_run.cfg
    out = tmp_path / "TinyMLP-all"
    first = fit(cfg, cfg.split, "TinyMLP-all", SEEDS, out, log=_quiet)
    mtime = (out / "fit_seed0.json").stat().st_mtime_ns
    again = fit(cfg, cfg.split, "TinyMLP-all", SEEDS, out, log=_quiet)
    assert again.doc == first.doc and (out / "fit_seed0.json").stat().st_mtime_ns == mtime


def test_existing_record_of_other_identity_refused(synth_run, tmp_path: Path) -> None:
    cfg = synth_run.cfg
    out = tmp_path / "TinyMLP-all"
    fit(cfg, cfg.split, "TinyMLP-all", SEEDS, out, log=_quiet)
    other = replace(cfg, train=replace(cfg.train, check_steps=5))
    with pytest.raises(RecordConflict):
        fit(other, other.split, "TinyMLP-all", SEEDS, out, log=_quiet)


@pytest.mark.parametrize("name", ["TinyMLP-all", "TinyLSTM-recent"])     # Adam, and Adam in the Keras form
def test_resume_equals_uninterrupted(synth_run, tmp_path: Path, name: str) -> None:
    cfg = synth_run.cfg
    whole = fit(cfg, cfg.split, name, SEEDS, tmp_path / "whole" / name, log=_quiet)
    out = tmp_path / "stopped" / name
    with pytest.raises(_Stop):
        fit(cfg, cfg.split, name, SEEDS, out, hooks=_stop_at(2), log=_quiet)
    assert (out / "checkpoint_seed0.pt").is_file() and not (out / "fit_seed0.json").exists()
    resumed = fit(cfg, cfg.split, name, SEEDS, out, log=_quiet)
    assert [r["after_check"] for r in resumed.doc["resumes"]] == [1]
    a = torch.load(whole.weights_path, map_location="cpu", weights_only=True)
    b = torch.load(resumed.weights_path, map_location="cpu", weights_only=True)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert resumed.weights_sha256 == whole.weights_sha256


def test_checkpoint_of_other_identity_refused(synth_run, tmp_path: Path) -> None:
    cfg = synth_run.cfg
    out = tmp_path / "TinyMLP-all"
    with pytest.raises(_Stop):
        fit(cfg, cfg.split, "TinyMLP-all", SEEDS, out, hooks=_stop_at(2), log=_quiet)
    other = replace(cfg, train=replace(cfg.train, check_steps=5))
    with pytest.raises(CheckpointConflict):
        fit(other, other.split, "TinyMLP-all", SEEDS, out, log=_quiet)


class _NanNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(2000, 10)

    def forward(self, x):
        return self.lin(x.flatten(1)) * float("nan")


def _with_model(cfg, family: str, **model_changes):
    m = replace(cfg.models["TinyMLP"], name="Test", family=family, **model_changes)
    return replace(cfg, models={**cfg.models, "Test": m},
                   fits={**cfg.fits, "Test-all": FitSpec("Test-all", "Test", "adam", 1e-3, 1e-7, "all", None)})


def test_diverged_fit_is_recorded(synth_run, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(FAMILIES, "nan_net", lambda c: _NanNet())
    cfg = _with_model(synth_run.cfg, "nan_net", init="torch")
    cfg = replace(cfg, machine=replace(cfg.machine, progress_every=1))
    out = tmp_path / "Test-all"
    rec = fit(cfg, cfg.split, "Test-all", SEEDS, out, log=_quiet)
    assert rec.status == "diverged" and "not finite at step 1" in rec.doc["error"]
    assert (out / "fit_seed0.json").is_file()
    assert not (out / "checkpoint_seed0.pt").exists() and not (out / "best_weights_seed0.pt").exists()


def test_placebo_needs_its_own_folder(synth_run, tmp_path: Path) -> None:
    cfg = synth_run.cfg
    with pytest.raises(ValueError, match="own result folder"):
        fit(cfg, cfg.split, "TinyMLP-all", SEEDS, tmp_path / "TinyMLP-all", placebo=True, log=_quiet)


def test_placebo_targets_same_day(synth_run) -> None:
    cfg = synth_run.cfg
    days = train_days(cfg, cfg.split, "all", synth_run.cache, compact=False)
    try:
        targets = draw_targets(days, torch.Generator().manual_seed(0))
        for day, t in zip(days, targets, strict=True):
            assert sorted(t.tolist()) == sorted(day.rows.tolist()) and not (t == day.rows).all()
    finally:
        close_days(days)


def test_amp_on_cpu_refused(synth_run, tmp_path: Path) -> None:
    cfg = _with_model(synth_run.cfg, "mlp", precision="amp")
    with pytest.raises(ConfigError, match="mixed precision"):
        fit(cfg, cfg.split, "Test-all", SEEDS, tmp_path / "Test-all", log=_quiet)
