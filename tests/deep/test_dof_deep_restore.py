"""The restore check: saved weights that do not give back the best validation loss stop the fit before its record."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dof.deep.config import Seeds  # noqa: E402
from dof.deep.errors import RestoreMismatch  # noqa: E402
from dof.deep.train import fit as fit_module  # noqa: E402


def test_restore_mismatch_stops_before_the_record(synth_run, tmp_path: Path, monkeypatch) -> None:
    real = fit_module.score_part

    def off_by_one(*args, **kwargs):
        rows, per = real(*args, **kwargs)
        return rows, {s: v + 1.0 for s, v in per.items()}
    monkeypatch.setattr(fit_module, "score_part", off_by_one)
    cfg = synth_run.cfg
    out = tmp_path / "TinyMLP-all"
    with pytest.raises(RestoreMismatch, match="saved weights give validation loss"):
        fit_module.fit(cfg, cfg.split, "TinyMLP-all", Seeds.same(0), out, log=lambda msg: None)
    assert (out / "best_weights_seed0.pt").is_file()
    assert not (out / "fit_seed0.json").exists()
    assert (out / "checkpoint_seed0.pt").is_file()
