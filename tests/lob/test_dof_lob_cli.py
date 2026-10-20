from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import dof
from dof.lob.cli import main
from dof.lob.testing import synth as ls

GZ_NAME = "01302019.NASDAQ_ITCH50.gz"


def _gz(tmp_path: Path, payloads: list[bytes] = ls.SCENARIO_FULL) -> Path:
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    return ls.write_stream(src / GZ_NAME, ls.stream(payloads), gz=True)


def test_cli_smoke(tmp_path: Path) -> None:
    gz = _gz(tmp_path)
    syms = tmp_path / "u.csv"
    syms.write_text("ticker,updates_k\nTEST,1\nNOPE,2\n")
    out = tmp_path / "cli"
    env = {**os.environ, "PYTHONPATH": str(Path(dof.__file__).resolve().parents[1])}
    proc = subprocess.run([sys.executable, "-m", "dof.lob", "build-day", "--itch", str(gz),
                           "--symbols-file", str(syms), "--out", str(out), "--no-manifest-check"],
                          capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["status"] == "complete"
    assert summary["day_dir"] == str(out / "2019-01-30")
    assert (summary["symbols_valid"], summary["symbols_invalid"], summary["symbols_missing"]) == (0, 1, 1)
    # A second run without --overwrite is refused with exit code 1.
    again = subprocess.run([sys.executable, "-m", "dof.lob", "build-day", "--itch", str(gz),
                            "--symbols-file", str(syms), "--out", str(out), "--no-manifest-check"],
                           capture_output=True, text=True, timeout=120, env=env)
    assert again.returncode == 1
    assert json.loads(again.stdout.strip().splitlines()[-1])["status"] == "failed"


def test_cli_exit_code_partial(tmp_path: Path, capsys) -> None:
    """A build that is not complete exits 1."""
    gz = _gz(tmp_path, ls.SCENARIO_FULL[:29])
    syms = tmp_path / "u.csv"
    syms.write_text("ticker\nTEST\n")
    code = main(["build-day", "--itch", str(gz), "--symbols-file", str(syms), "--out", str(tmp_path / "out"),
                 "--no-manifest-check"])
    assert code == 1
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["status"] == "partial"


def test_cli_requires_manifest_choice(tmp_path: Path) -> None:
    """build-day needs exactly one of --manifest PATH and --no-manifest-check."""
    base = ["build-day", "--itch", str(tmp_path / "x.itch"), "--symbols-file", str(tmp_path / "u.csv"),
            "--out", str(tmp_path / "out")]
    for extra in ([], ["--manifest", str(tmp_path / "m.csv"), "--no-manifest-check"]):
        with pytest.raises(SystemExit) as exc:
            main(base + extra)
        assert exc.value.code == 2
    assert not (tmp_path / "out").exists()
