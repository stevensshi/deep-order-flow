"""Report goldens: every number of the score and fit tables of the five stored reports, printed and as float hex,
rebuilt from the result tables by the analyses and compared with the recorded goldens (see README.md here).

The goldens are recorded by the recorder of the reference implementation from the stored report files and the
stored score files (the earlier stage's golden by its own recorder, from that stage's stored report and score files);
both sides name their values with ``report_keys.py``, whose sha256 every golden records.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import report_keys as K
from recipes import compare as C
from recipes import hashing as H

REPORTS = ("alldays", "adjacent", "decay", "seeds", "earlier")
DATA_DIR_ENV = "DOF_GOLDEN_DATA_DIR"
FULL_DIR_ENV = "DOF_GOLDEN_FULL_DIR"
RESULTS_ENV = "DOF_GOLDEN_RESULTS_DIR"


@pytest.fixture(scope="module")
def tables():
    from dof.experiments.cells import read_tables
    folder = os.environ.get(RESULTS_ENV)
    if not folder:
        pytest.skip(f"needs {RESULTS_ENV}")
    return read_tables(Path(folder))


def _walk(sections):
    for s in sections:
        yield from s.tables
        yield from _walk(s.subsections)


def report_values(report, name: str, tables=None) -> dict:
    """The golden values of a built report: the compared columns of every table with a golden id (with "seed=*"
    expanded to the spec's seeds), the two keys of the decay list, and for the seeds report the rule's values."""
    from dof.experiments.render import fmt_cell, outside_lines
    from dof.experiments.specs import SPECS
    spec = SPECS[name]
    values: dict = {}

    def put(key, value):
        if key in values:
            raise AssertionError(f"duplicate golden key {key}")
        values[key] = value
    for t in _walk(report.sections):
        if t.golden is None:
            continue
        if name == "decay" and t.golden == "outside":
            put(K.key("decay", "outside", ("count",), "value"), str(len(t.rows)))
            put(K.key("decay", "outside", ("lines",), "value"), "\n".join(outside_lines(t)))
            continue
        compared = set(K.expand(K.TABLES[name][t.golden], spec.seeds))
        for row in t.rows:
            for col, cell in zip(t.columns, row.cells):
                if not col.compare or col.key not in compared:
                    continue
                s = fmt_cell(cell)
                if s == "":
                    continue
                key = K.key(name, t.golden, row.key, col.key)
                put(key, s)
                if cell.raw is not None:
                    put(key + ".hex", H.fhex(cell.raw))
    if name == "seeds":
        from dof.experiments.analyses.seeds import rule_values
        v = rule_values(tables, spec)
        for s, d in v["differences"].items():
            if d is not None:
                put(K.key("seeds", "rule", ("difference", K.seed_label(s)), "value.hex"), H.fhex(d))
        for s, n in v["stocks"].items():
            if n is not None:
                put(K.key("seeds", "rule", ("stocks", K.seed_label(s)), "value"), int(n))
        for q in ("mean", "sd", "half_abs_mean"):
            if v[q] is not None:
                put(K.key("seeds", "rule", (q,), "value.hex"), H.fhex(v[q]))
        put(K.key("seeds", "rule", ("more",), "value"), bool(v["more"]))
    return values


def _golden_path(env_path, name: str):
    data = env_path(DATA_DIR_ENV)
    if data is None:
        pytest.skip(f"needs {DATA_DIR_ENV}")
    path = data / "reports" / f"{name}.json"
    if not path.is_file():
        pytest.skip(f"reports/{name}.json is not recorded yet")
    return path


@pytest.mark.golden
@pytest.mark.local
@pytest.mark.parametrize("name", REPORTS)
def test_report_numbers(name, tables, load_golden, env_path) -> None:
    from dof.experiments.report import build
    from dof.experiments.specs import SPECS
    golden = load_golden(_golden_path(env_path, name))
    assert golden["values"]["meta/report_keys_sha256"] == K.sha256_self(), "report_keys.py changed since recording"
    want = {k: v for k, v in golden["values"].items() if not k.startswith("meta/")}
    values = report_values(build(tables, SPECS[name]), name, tables)
    got = H.committed_form(values, golden["real_data"]) if golden["form"] == "committed" else values
    d = C.diff(want, got)
    full_dir = env_path(FULL_DIR_ENV)
    if d and full_dir is not None:
        full = json.loads((full_dir / "reports" / f"{name}.json").read_text(encoding="utf-8"))
        clear = {k: v for k, v in full["values"].items() if not k.startswith("meta/")}
        print("full-form differences:\n" + "\n".join(C.diff(clear, values)[:50]))
    assert not d, f"{len(d)} differences:\n" + "\n".join(d[:50])


@pytest.mark.golden
@pytest.mark.local
def test_report_keys_file_unchanged(load_golden, env_path) -> None:
    data = env_path(DATA_DIR_ENV)
    if data is None:
        pytest.skip(f"needs {DATA_DIR_ENV}")
    paths = sorted((data / "reports").glob("*.json"))
    if not paths:
        pytest.skip("no report golden is recorded yet")
    for path in paths:
        assert load_golden(path)["values"]["meta/report_keys_sha256"] == K.sha256_self(), path.name
