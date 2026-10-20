"""The command line of the training package on the synthetic run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _cli(*args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "DOF_LOCAL_CONFIG"}
    return subprocess.run([sys.executable, "-B", "-m", "dof.deep", *args], env=env, capture_output=True, text=True,
                          timeout=600)


def _local(path: Path, book: Path, train: Path, eval_: Path) -> Path:
    path.write_text(f'[machine]\nbook_root = "{book.as_posix()}"\n\n[machine.cache.synth]\ntrain = "{train.as_posix()}"\n'
                    f'eval = "{eval_.as_posix()}"\n', encoding="utf-8")
    return path


def test_help() -> None:
    proc = _cli("--help")
    assert proc.returncode == 0 and "build" in proc.stdout, proc.stdout + proc.stderr


def test_build_on_synthetic_config(synth_run, tmp_path: Path) -> None:
    local = _local(tmp_path / "local.toml", synth_run.root / "book", tmp_path / "train", tmp_path / "eval")
    proc = _cli("build", "--config", str(synth_run.root / "configs" / "synth.toml"), "--local", str(local),
                "--workers", "1", "--mem-gb", "2")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.splitlines()[-1] == "BUILD_DONE"
    assert json.loads((tmp_path / "train" / "summary.json").read_text(encoding="utf-8"))["parts"] == ["train", "valid", "test"]


def test_conflict_exits_1(synth_run, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    local = _local(tmp_path / "local.toml", synth_run.root / "book", synth_run.cache.train, synth_run.cache.eval)
    out = tmp_path / "TinyMLP-all"
    out.mkdir()
    (out / "fit_seed0.json").write_text(json.dumps({"status": "finished", "identity_sha256": "0" * 64,
                                                    "protocol_sha256": None}), encoding="utf-8")
    proc = _cli("fit", "--config", str(synth_run.root / "configs" / "synth.toml"), "--local", str(local),
                "--fit", "TinyMLP-all", "--seed", "0", "--out", str(out))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert proc.stdout.splitlines()[-1].startswith("FIT_FAILED: RecordConflict")
