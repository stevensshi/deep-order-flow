"""The renderer and the report rules: cells format as the stored reports did; the renderer computes nothing; the
reports of the fake lab render the same twice; results cite test tables only; no table mixes parts; every table
labels its rows; every headline names its days, stocks and horizon; strict reports refuse a missing headline cell."""

from __future__ import annotations

import ast
import re
from dataclasses import replace
from pathlib import Path

import pytest

from dof.experiments.analyses import ANALYSES
from dof.experiments.errors import ReportError
from dof.experiments.render import fmt_cell, format_table, render
from dof.experiments.report import build
from dof.experiments.spec import HeadlineSpec, SectionSpec
from dof.experiments.tables import Cell, Column, Row, Table, pct, sci, steps
from fake_lab import ORDER, exported_tables

RENDER = Path(__file__).resolve().parents[2] / "src" / "dof" / "experiments" / "render.py"
BANNED_OPS = (ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.MatMult)
BANNED_CALLS = {"sum", "min", "max", "round", "abs", "len", "sorted", "float", "int"}
BANNED_MODULES = {"math", "np", "numpy", "pd", "pandas"}


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    return exported_tables(tmp_path_factory.mktemp("render"))


def _walk_specs(sections):
    for s in sections:
        yield s
        yield from _walk_specs(s.subsections)


def _tables(report):
    def walk(sections):
        for s in sections:
            yield from s.tables
            yield from walk(s.subsections)
    return {t.id: t for t in walk(report.sections)}


def test_fmt_cell() -> None:
    assert fmt_cell(pct(-1e-7)) == "-0.000"
    assert fmt_cell(Cell(None)) == ""
    assert fmt_cell(pct(float("nan"), missing="not defined")) == "not defined"
    assert fmt_cell(steps(1268856)) == "1,268,856"
    assert fmt_cell(sci(1.289e-06)) == "1.289e-06"
    assert fmt_cell(pct(0.00369)) == "0.369" and pct(0.00369).raw == 0.00369


def test_markdown_table_shape() -> None:
    t = Table("t", None, "Title", (Column("a", "A", compare=False, align="left"), Column("b", "B")),
              (Row(("x",), (Cell("x"), Cell(1.5, ".2f"))), Row(("y",), (Cell("|y|"), Cell(None)))), note="One note.")
    assert format_table(t) == ["#### Title", "", "| A | B |", "|---|---:|", "| x | 1.50 |", "| \\|y\\| |  |", "",
                               "One note.", ""]
    assert format_table(replace(t, rows=())) == ["#### Title", "", "No score yet.", ""]


def test_render_computes_nothing() -> None:
    tree = ast.parse(RENDER.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.append(f"import {[a.name for a in node.names]}")
        elif isinstance(node, ast.ImportFrom):
            ok = node.module in ("__future__", "dof.experiments.tables") or (
                node.module == "dof.experiments" and [a.name for a in node.names] == ["report"])
            if not ok:
                found.append(f"from {node.module} import ...")
        elif isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, BANNED_OPS):
            found.append(f"line {node.lineno}: {type(node.op).__name__}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            found.append(f"line {node.lineno}: {node.func.id}()")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in BANNED_MODULES:
            found.append(f"line {node.lineno}: {node.value.id}.{node.attr}")
    assert not found, found


def test_fake_reports_render_deterministically(fake) -> None:
    _lab, specs, rt = fake
    for name in ORDER:
        a, b = render(rt, specs[name]), render(rt, specs[name])
        assert a == b and a.endswith("\n") and not a.endswith("\n\n")
        for s in _walk_specs(specs[name].sections):
            for t in s.tables:
                assert f"#### {t.title}\n" in a, (name, t.title)
        assert "Not scored yet." not in a


def _headline_tables(spec):
    for s in _walk_specs([x for x in spec.sections if x.id == "result"]):
        for h in s.headlines:
            yield h, [ref[1] for ref in h.refs]


def test_result_sections_cite_test_tables_only(fake) -> None:
    _lab, specs, rt = fake
    for name in ORDER:
        built = _tables(build(rt, specs[name]))
        heads = list(_headline_tables(specs[name]))
        assert heads, name
        allowed = ("test",) if name in ("alldays", "seeds") else ("test", None)
        for _h, tids in heads:
            for tid in tids:
                assert built[tid].part in allowed, (name, tid, built[tid].part)


class Spy:
    """ResultTables that records the record ids whose cells or stock-days an analysis reads."""

    def __init__(self, rt) -> None:
        self._rt, self.read = rt, []

    def __getattr__(self, name):
        return getattr(self._rt, name)

    def rows(self, rid):
        self.read.append(rid)
        return self._rt.rows(rid)

    def stock_days_of(self, rid):
        self.read.append(rid)
        return self._rt.stock_days_of(rid)


def test_tables_never_mix_parts(fake) -> None:
    _lab, specs, rt = fake
    for name in ORDER:
        spec = specs[name]
        for s in _walk_specs(spec.sections):
            for ts in s.tables:
                spy = Spy(rt)
                t = ANALYSES[ts.analysis](spy, spec, id=ts.id, title=ts.title, **dict(ts.args))
                assert t.part in ("valid", "test", None), (name, ts.id)
                if t.part is not None:
                    parts = {rid.split("/")[4] for rid in spy.read}
                    assert parts <= {t.part}, (name, ts.id, parts)


def test_every_table_labels_its_rows(fake) -> None:
    _lab, specs, rt = fake
    for name in ORDER:
        for t in _tables(build(rt, specs[name])).values():
            labels = [tuple(fmt_cell(c) for c, col in zip(r.cells, t.columns) if not col.compare) for r in t.rows]
            assert t.columns[0].compare is False, (name, t.id)
            assert len(labels) == len(set(labels)), (name, t.id)


def test_headlines_name_days_stocks_horizon() -> None:
    from dof.experiments.specs import SPECS
    for name, spec in SPECS.items():
        heads = [h for s in _walk_specs(spec.sections) for h in s.headlines]
        assert heads, name
        for h in heads:
            assert re.search(r"\d{4}-\d{2}-\d{2}|days", h.template), (name, h.template)
            assert "stocks" in h.template and "k = " in h.template, (name, h.template)


def test_non_strict_leaves_out_missing_headline(fake) -> None:
    _lab, specs, rt = fake
    spec = specs["alldays"]
    bad = HeadlineSpec("R² {r2} %", (("r2", "test_primary:CNN-LSTM-OF-alldays", ("no", "such", "row"), "r2"),))
    sec = replace(spec.sections[0], headlines=(bad,), subsections=())
    broken = replace(spec, sections=(sec, *spec.sections[1:]))
    with pytest.raises(ReportError, match=r"section result: table test_primary:CNN-LSTM-OF-alldays has no row"):
        build(rt, broken)
    report = build(rt, broken, strict=False)
    assert report.sections[0].headlines == () and report.sections[0].paragraphs[-1] == "Not scored yet."
    missing = replace(spec, sections=(SectionSpec("x", "X", headlines=(replace(bad, refs=(
        ("r2", "no_table", ("a",), "r2"),)),)), *spec.sections[1:]))
    with pytest.raises(ReportError, match="table no_table is not built"):
        build(rt, missing)
