"""A Transformer family added as one model file and one ``[models.*]`` entry: it is found by the registry, it
trains, and a fit stopped at its second check resumes to the weights of an uninterrupted fit with dropout on.

The fit specification that trains it is made here, in the test; the package needs no other change. The two tests
on the smoke split read the stored smoke cache (``DOF_GOLDEN_SMOKE_CACHE``, read only) on the CPU with two torch
threads, and are skipped without it.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dof.deep.config import CachePaths, FitSpec, MachineConfig, Seeds, load  # noqa: E402
from dof.deep.models import FAMILIES, build  # noqa: E402
from dof.deep.train.fit import FitHooks, fit  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SMOKE_ENV = "DOF_GOLDEN_SMOKE_CACHE"
FIT = "Transformer-OF-smoke"


class _Stop(Exception):
    """Stands for a fit that is stopped between two checks."""


def _stop_at_two() -> FitHooks:
    def hook(check: int) -> None:
        if check == 2:
            raise _Stop("stopped before the validation reading of check 2")
    return FitHooks(before_validation=hook)


def _quiet(msg: str) -> None:
    pass


def _with_fit(cfg, models_from):
    """``cfg`` with the catalogue's Transformer entry and a fit specification that trains it."""
    return replace(cfg, models={**cfg.models, "Transformer-OF": models_from["Transformer-OF"]},
                   fits={**cfg.fits, FIT: FitSpec(FIT, "Transformer-OF", "adam", 1e-4, 1e-7, "recent", None)})


@pytest.fixture(scope="module")
def catalogue():
    return load(ROOT / "configs" / "smoke.toml").models


def _equal_weights(a, b) -> bool:
    wa = torch.load(a.weights_path, map_location="cpu", weights_only=True)
    wb = torch.load(b.weights_path, map_location="cpu", weights_only=True)
    return wa.keys() == wb.keys() and all(torch.equal(wa[k], wb[k]) for k in wa)


def _resume_check(cfg, split, out: Path, cache=None) -> None:
    whole = fit(cfg, split, FIT, Seeds.same(0), out / "whole", cache=cache, log=_quiet)
    with pytest.raises(_Stop):
        fit(cfg, split, FIT, Seeds.same(0), out / "stopped", cache=cache, hooks=_stop_at_two(), log=_quiet)
    resumed = fit(cfg, split, FIT, Seeds.same(0), out / "stopped", cache=cache, log=_quiet)
    assert [r["after_check"] for r in resumed.doc["resumes"]] == [1]
    assert _equal_weights(whole, resumed)


def test_transformer_is_registered_by_its_file(catalogue) -> None:
    assert "transformer" in FAMILIES
    m = build(catalogue["Transformer-OF"])
    x = torch.randn(4, 100, 20)
    assert tuple(m(x).shape) == (4, 10)
    m.train()
    assert not torch.equal(m(x), m(x))                      # dropout on
    m.eval()
    with torch.no_grad():
        assert torch.equal(m(x), m(x))


def test_transformer_resume_equals_uninterrupted_synthetic(synth_run, catalogue, tmp_path: Path) -> None:
    cfg = _with_fit(synth_run.cfg, catalogue)          # the synthetic run has the smoke split's history and horizons
    _resume_check(cfg, cfg.split, tmp_path)


@pytest.fixture(scope="module")
def smoke():
    """The smoke configuration with the dry run's settings (two checks of 12 steps, validation days cut to 1,024
    samples), the CPU with two threads, and the stored smoke cache, read only."""
    root = os.environ.get(SMOKE_ENV)
    if not root:
        pytest.skip(f"needs {SMOKE_ENV}")
    torch.set_num_threads(2)
    cfg = load(ROOT / "configs" / "smoke.toml")
    cfg = replace(cfg, machine=MachineConfig(device="cpu", torch_threads=2, loader_workers=0),
                  train=replace(cfg.train, check_steps=12, max_checks=2, eval_row_cap=1024))
    cache = CachePaths(Path(root) / "train", Path(root) / "eval")
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(Path(root).rglob("*")) if p.is_file()}
    yield _with_fit(cfg, cfg.models), cache
    after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(Path(root).rglob("*")) if p.is_file()}
    assert after == before, "a file of the stored smoke cache changed"


@pytest.mark.local
def test_transformer_trains_on_smoke_split(smoke, tmp_path: Path) -> None:
    cfg, cache = smoke
    rec = fit(cfg, cfg.split, FIT, Seeds.same(0), tmp_path / FIT, cache=cache, log=_quiet)
    assert rec.status == "finished" and len(rec.doc["checks"]) == 2
    assert all(math.isfinite(c["train_loss"]) and math.isfinite(c["valid_loss"]) for c in rec.doc["checks"])


@pytest.mark.local
def test_transformer_resume_equals_uninterrupted_on_smoke_split(smoke, tmp_path: Path) -> None:
    cfg, cache = smoke
    _resume_check(cfg, cfg.split, tmp_path, cache)
