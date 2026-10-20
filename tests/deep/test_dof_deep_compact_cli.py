"""The command line of the compact store."""

from __future__ import annotations

import datetime
import re
from pathlib import Path

import pytest

pytest.importorskip("zstandard")

from compact_helpers import (  # noqa: E402
    CONTEXT,
    PATTERN,
    _cgroup_without_limit,  # noqa: F401  (the autouse fixture, bound in this module)
    _cli,
    _flip,
    _sha,
    _tree,
    convert_mod,
)
from dof.deep.data import compact  # noqa: E402
from dof.deep.data.compact import cli  # noqa: E402


def test_cli_build_show_verify(tmp_path: Path) -> None:
    src, arrays = _tree(tmp_path)
    dst = tmp_path / "dst"
    build = ("build", "--src", str(src), "--dst", str(dst), "--pattern", PATTERN, "--context", *CONTEXT,
             "--rate-mb-s", "0", "--pause-s", "0")
    done = (r"\S+ BUILD_DONE files=3 rows=1838 source_bytes=\d+ compressed_bytes=\d+ bytes_per_row=\d+\.\d{3} "
            r"verified_rows=1838 manifest_sha256=[0-9a-f]{64}")
    first = _cli(*build)
    assert first.returncode == 0, first.stdout + first.stderr
    lines = first.stdout.splitlines()
    assert re.fullmatch(done, lines[-1])
    assert any(" convert: files=3 converted=3 skipped=0 " in line for line in lines)
    datetime.datetime.fromisoformat(lines[-1].split(" ", 1)[0])         # every line starts with an ISO time

    second = _cli(*build)                                               # sealed: it verifies, then the same verdict
    assert second.returncode == 0, second.stdout + second.stderr
    again = second.stdout.splitlines()
    assert any(" VERIFY_OK files=3 rows=1838 " in line for line in again)
    assert not any(" converted " in line for line in again)
    assert re.fullmatch(done, again[-1]) and again[-1].split(" ", 1)[1] == lines[-1].split(" ", 1)[1]

    shown = _cli("show", "--dst", str(dst))
    assert shown.returncode == 0, shown.stdout + shown.stderr
    assert "files 3" in shown.stdout.splitlines() and "rows 1838" in shown.stdout.splitlines()
    assert f"context plan.json {_sha(src / 'plan.json')}" in shown.stdout.splitlines()

    checked = _cli("verify", "--dst", str(dst), "--rate-mb-s", "0", "--pause-s", "0")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert " VERIFY_OK files=3 rows=1838 " in checked.stdout.splitlines()[-1]

    _flip(compact.data_path(dst, "A/train_2019-03-27.rec"))
    broken = _cli(*build)
    assert broken.returncode == 1
    assert " BUILD_FAILED: " in broken.stdout.splitlines()[-1]


def test_cli_unexpected_error_is_reported_and_fails(tmp_path: Path, monkeypatch, capsys) -> None:
    src, _ = _tree(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(convert_mod, "convert", boom)
    code = cli.main(["build", "--src", str(src), "--dst", str(tmp_path / "dst"), "--pattern", PATTERN,
                     "--context", *CONTEXT, "--rate-mb-s", "0", "--pause-s", "0"])
    captured = capsys.readouterr()
    assert code == 1
    stamp, line = captured.out.splitlines()[-1].split(" ", 1)           # a line of build starts with an ISO time
    datetime.datetime.fromisoformat(stamp)
    assert line == "BUILD_FAILED: unexpected RuntimeError: boom"
    assert "RuntimeError: boom" in captured.err                         # the traceback
