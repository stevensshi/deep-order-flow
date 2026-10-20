"""The command line: export, totals, report, run; exit codes and verdict lines."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import dof.experiments.cli as cli
from fake_lab import fake_specs, write_fake_lab


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """A fake lab, and the command line's specs replaced by the fake-pinned ones."""
    lab = write_fake_lab(tmp_path / "lab")
    specs = fake_specs(lab)
    monkeypatch.setattr(cli, "_specs", lambda names: [specs[n] for n in (names or cli.ORDER)])
    return lab


def test_export_cli(lab, tmp_path, capsys) -> None:
    out = tmp_path / "tables"
    assert cli.main(["export", "--results-root", str(lab.root), "--out", str(out)]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "EXPORT_DONE"
    assert (out / "MANIFEST.json").is_file()
    assert cli.main(["export", "--results-root", str(lab.root), "--out", str(out)]) == 2
    assert capsys.readouterr().out.strip().splitlines()[-1] == "EXPORT_FAILED"


def test_totals_cli_equal_and_differ(lab, tmp_path, capsys) -> None:
    out = tmp_path / "tables"
    assert cli.main(["export", "--results-root", str(lab.root), "--out", str(out)]) == 0
    expect = tmp_path / "totals.json"
    expect.write_text(json.dumps({"format": 2, "totals": lab.totals}))
    capsys.readouterr()
    assert cli.main(["totals", "--results", str(out), "--expect", str(expect)]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == f"TOTALS EQUAL {len(lab.totals)} keys"
    changed = dict(lab.totals)
    changed["alldays/test"] = {**changed["alldays/test"], "cells": 0}
    expect.write_text(json.dumps({"format": 2, "totals": changed}))
    assert cli.main(["totals", "--results", str(out), "--expect", str(expect)]) == 1
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1].startswith("TOTALS DIFFER alldays/test: ")
    assert cli.main(["totals", "--results", str(out)]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == f"TOTALS {len(lab.totals)} keys"


def test_report_cli_all(lab, tmp_path, capsys) -> None:
    out = tmp_path / "tables"
    assert cli.main(["export", "--results-root", str(lab.root), "--out", str(out)]) == 0
    reports = tmp_path / "reports"
    capsys.readouterr()
    assert cli.main(["report", "--results", str(out), "--all", "--out-dir", str(reports)]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "REPORT_DONE"
    assert sorted(p.name for p in reports.iterdir()) == ["adjacent.md", "alldays.md", "decay.md", "earlier.md",
                                                          "seeds.md"]
    first = (reports / "alldays.md").read_text(encoding="utf-8")
    assert cli.main(["report", "--results", str(out), "--spec", "alldays", "--out-dir", str(reports)]) == 0
    assert (reports / "alldays.md").read_text(encoding="utf-8") == first


def test_run_cli_refuses_stored_only(tmp_path, capsys) -> None:
    config = Path(__file__).resolve().parents[2] / "configs" / "smoke.toml"
    for name in ("adjacent", "decay"):
        rc = cli.main(["run", "--spec", name, "--config", str(config), "--out", str(tmp_path / name)])
        out = capsys.readouterr().out.strip().splitlines()
        assert rc == 1 and out[-1] == "RUN_FAILED" and "does not run its fit step" in out[-2]
        assert not (tmp_path / name).exists()
    assert cli.main(["run", "--spec", "alldays", "--config", str(config), "--out", str(tmp_path / "x"),
                     "--input", "no-equals-sign"]) == 2


def test_export_cli_root_argument(lab, tmp_path, capsys) -> None:
    assert cli.main(["export", "--results-root", str(lab.root), "--out", str(tmp_path / "t"), "--root", "earlier"]) == 2
    assert capsys.readouterr().out.strip().splitlines()[-1] == "EXPORT_FAILED"
    assert cli.main(["export", "--results-root", str(tmp_path / "nowhere"), "--out", str(tmp_path / "t"),
                     "--root", f"earlier={lab.root}"]) == 1
    assert capsys.readouterr().out.strip().splitlines()[-1] == "EXPORT_FAILED"
