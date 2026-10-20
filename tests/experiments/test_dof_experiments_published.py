"""The published result tables in results/: their totals equal the committed totals, the manifest names their bytes,
the four reports regenerate byte for byte, every headline resolves and no table is empty, and no value is a path."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TOTALS = ROOT / "tests" / "goldens" / "data" / "results_totals.json"
MACHINE = re.compile(r"/(home|mnt|Users)/")

if not (RESULTS / "cells.parquet").exists():
    pytest.skip("the results are published after the seed experiment ends", allow_module_level=True)

from dof.experiments.cells import TABLES, read_tables, totals  # noqa: E402
from dof.experiments.render import render  # noqa: E402
from dof.experiments.report import build  # noqa: E402
from dof.experiments.specs import SPECS  # noqa: E402


@pytest.fixture(scope="module")
def tables():
    return read_tables(RESULTS)


def test_published_totals_equal_committed_totals(tables) -> None:
    assert totals(tables) == json.loads(TOTALS.read_text(encoding="utf-8"))["totals"]


def test_published_manifest_matches_files() -> None:
    import pyarrow.parquet as pq
    manifest = json.loads((RESULTS / "MANIFEST.json").read_text(encoding="utf-8"))
    assert set(manifest["outputs"]) == {f"{n}.parquet" for n in TABLES}
    for name, info in manifest["outputs"].items():
        path = RESULTS / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"], name
        assert pq.read_metadata(path).num_rows == info["rows"], name


def test_published_reports_regenerate(tables) -> None:
    for name, spec in SPECS.items():
        assert render(tables, spec) == (RESULTS / "reports" / f"{name}.md").read_text(encoding="utf-8"), name


def test_published_every_headline_and_no_empty_table(tables) -> None:
    def walk(sections):
        for s in sections:
            yield s
            yield from walk(s.subsections)
    for name, spec in SPECS.items():
        report = build(tables, spec, strict=True)
        sections = list(walk(report.sections))
        assert sum(len(s.headlines) for s in sections) == sum(len(s.headlines) for s in walk(spec.sections)), name
        for s in sections:
            for t in s.tables:
                assert t.rows, (name, t.id)


def test_published_tables_hold_no_path() -> None:
    for name in TABLES:
        df = pd.read_parquet(RESULTS / f"{name}.parquet")
        for col in df.columns:
            for v in df[col].dropna():
                if isinstance(v, str):              # a bool column with nulls also reads back as object
                    assert not v.startswith("/") and not MACHINE.search(v), (name, col, v)
    text = (RESULTS / "MANIFEST.json").read_text(encoding="utf-8")
    assert not MACHINE.search(text) and '"/' not in text


def test_published_manifest_names_the_spec_identities() -> None:
    from dof.experiments.spec import spec_identity_sha256
    manifest = json.loads((RESULTS / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["specs"] == {name: spec_identity_sha256(spec) for name, spec in SPECS.items()}
