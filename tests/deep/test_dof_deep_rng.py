"""Random-number state across a restart: checkpoints hold the CPU (and CUDA) generator state, and the loader draws
its worker seed from its own generator, so a model with dropout resumes to the weights of an uninterrupted fit."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dof.deep.config import FitSpec, Seeds  # noqa: E402
from dof.deep.errors import CheckpointConflict  # noqa: E402
from dof.deep.models import FAMILIES, build  # noqa: E402
from dof.deep.runtime.io import torch_save_atomic  # noqa: E402
from dof.deep.train.fit import FitHooks, fit  # noqa: E402

SEEDS = Seeds.same(0)


class _Stop(Exception):
    """Stands for a fit that is stopped between two checks."""


def _stop_at_two() -> FitHooks:
    def hook(check: int) -> None:
        if check == 2:
            raise _Stop("stopped before the validation reading of check 2")
    return FitHooks(before_validation=hook)


def _quiet(msg: str) -> None:
    pass


class _DropoutMlp(torch.nn.Module):
    def __init__(self, d: int, history: int, n_out: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(history * d, 16), torch.nn.ReLU(), torch.nn.Dropout(0.5),
                                       torch.nn.Linear(16, n_out))

    def forward(self, x):
        return self.net(x.flatten(1))


class _Probe(torch.nn.Module):
    """Keeps the global generator's state at its first forward in training mode."""

    first_state = None

    def __init__(self, d: int, history: int, n_out: int) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(history * d, n_out)

    def forward(self, x):
        if self.training and _Probe.first_state is None:
            _Probe.first_state = torch.get_rng_state()
        return self.lin(x.flatten(1))


def _with_family(cfg, family: str, cls, monkeypatch):
    monkeypatch.setitem(FAMILIES, family, lambda c: cls(c.input_dim, c.history, c.n_out))
    m = replace(cfg.models["TinyMLP"], name="Test", family=family, params={})
    return replace(cfg, models={**cfg.models, "Test": m},
                   fits={**cfg.fits, "Test-all": FitSpec("Test-all", "Test", "adam", 1e-3, 1e-7, "all", None)})


def test_resume_equals_uninterrupted_with_dropout(synth_run, tmp_path: Path, monkeypatch) -> None:
    cfg = _with_family(synth_run.cfg, "dropout_mlp", _DropoutMlp, monkeypatch)
    whole = fit(cfg, cfg.split, "Test-all", SEEDS, tmp_path / "whole" / "Test-all", log=_quiet)
    out = tmp_path / "stopped" / "Test-all"
    with pytest.raises(_Stop):
        fit(cfg, cfg.split, "Test-all", SEEDS, out, hooks=_stop_at_two(), log=_quiet)
    resumed = fit(cfg, cfg.split, "Test-all", SEEDS, out, log=_quiet)
    assert [r["after_check"] for r in resumed.doc["resumes"]] == [1]
    a = torch.load(whole.weights_path, map_location="cpu", weights_only=True)
    b = torch.load(resumed.weights_path, map_location="cpu", weights_only=True)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_checkpoint_holds_rng_state(synth_run, tmp_path: Path) -> None:
    cfg = synth_run.cfg
    out = tmp_path / "TinyMLP-all"
    with pytest.raises(_Stop):
        fit(cfg, cfg.split, "TinyMLP-all", SEEDS, out, hooks=_stop_at_two(), log=_quiet)
    ck = torch.load(out / "checkpoint_seed0.pt", map_location="cpu", weights_only=True)
    assert ck["format"] == 2 and ck["rng_cpu"].dtype == torch.uint8 and ck["rng_cuda"] is None


def test_fit_loader_leaves_global_generator_alone(synth_run, tmp_path: Path, monkeypatch) -> None:
    cfg = _with_family(synth_run.cfg, "probe", _Probe, monkeypatch)
    monkeypatch.setattr(_Probe, "first_state", None)
    fit(cfg, cfg.split, "Test-all", SEEDS, tmp_path / "Test-all", log=_quiet)
    seen = _Probe.first_state
    torch.manual_seed(SEEDS.init)
    build(cfg.models["Test"])
    assert seen is not None and torch.equal(seen, torch.get_rng_state())


def test_old_checkpoint_refused(synth_run, tmp_path: Path) -> None:
    cfg = synth_run.cfg
    out = tmp_path / "TinyMLP-all"
    with pytest.raises(_Stop):
        fit(cfg, cfg.split, "TinyMLP-all", SEEDS, out, hooks=_stop_at_two(), log=_quiet)
    path = out / "checkpoint_seed0.pt"
    ck = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("format", "rng_cpu", "rng_cuda"):
        del ck[key]
    torch_save_atomic(path, ck)
    with pytest.raises(CheckpointConflict, match="format 1"):
        fit(cfg, cfg.split, "TinyMLP-all", SEEDS, out, log=_quiet)
