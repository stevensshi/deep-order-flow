"""A spec's report built from the result tables: every section's tables from their analyses, every headline from
cells of those tables, the protocol hashes filled into the prose.

With ``strict`` (the default), a headline that names a table, row or column the report does not build, or a cell
without a value, raises ReportError naming the section, table, row and column. Without it, such a headline is left
out, and a section left with no headline says "Not scored yet."
"""

from __future__ import annotations

from dof.experiments.errors import ReportError
from dof.experiments.spec import ExperimentSpec, SectionSpec
from dof.experiments.tables import Headline, Report, Section, Table


def _tables(tables, spec: ExperimentSpec, sections: tuple[SectionSpec, ...], built: dict[str, Table]) -> None:
    from dof.experiments.analyses import ANALYSES
    for s in sections:
        for t in s.tables:
            built[t.id] = ANALYSES[t.analysis](tables, spec, id=t.id, title=t.title, **dict(t.args))
        _tables(tables, spec, s.subsections, built)


def _cell(built: dict[str, Table], section: str, tid: str, row: tuple[str, ...], col: str):
    """The cell, or the reason it cannot be used."""
    t = built.get(tid)
    if t is None:
        return None, f"section {section}: table {tid} is not built by this report"
    keys = [c.key for c in t.columns]
    if col not in keys:
        return None, f"section {section}: table {tid} has no column {col}"
    hit = [r for r in t.rows if r.key == tuple(row)]
    if not hit:
        return None, f"section {section}: table {tid} has no row {tuple(row)}"
    cell = hit[0].cells[keys.index(col)]
    if cell.value is None:
        return None, f"section {section}: table {tid}, row {tuple(row)}, column {col} has no value"
    return cell, None


def _section(s: SectionSpec, built: dict[str, Table], fill: dict, strict: bool) -> Section:
    heads = []
    for h in s.headlines:
        cells, why = {}, None
        for placeholder, tid, row, col in h.refs:
            cell, why = _cell(built, s.id, tid, row, col)
            if cell is None:
                break
            cells[placeholder] = cell
        if why is not None:
            if strict:
                raise ReportError(why)
            continue
        heads.append(Headline(h.template, cells))
    paragraphs = tuple(p.format(**fill) for p in s.paragraphs)
    if s.headlines and not heads:
        paragraphs += ("Not scored yet.",)
    return Section(s.id, s.title, paragraphs, tuple(heads), tuple(built[t.id] for t in s.tables),
                   tuple(_section(x, built, fill, strict) for x in s.subsections))


def build(tables, spec: ExperimentSpec, *, strict: bool = True) -> Report:
    """The report of ``spec`` from the result tables ``tables`` (``cells.ResultTables``)."""
    built: dict[str, Table] = {}
    _tables(tables, spec, spec.sections, built)
    p = spec.protocol
    fill = {"main": p.main_sha256 or "not pinned", "extension": p.extension_sha256 or "not pinned",
            "ca": p.corporate_actions_sha256 or "not pinned"}
    return Report(spec.title, spec.intro, tuple(_section(s, built, fill, strict) for s in spec.sections))
