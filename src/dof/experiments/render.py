"""A report as Markdown. The renderer computes nothing: it formats the cells the analyses made (a test checks that
this file holds no arithmetic and no numeric call).

Layout: "# title", the intro, then "## section" with its paragraphs, its headlines as a bullet list, its tables
under "####" headings and its subsections under "###". A table with no row prints "No score yet." The decay list
of price ratios outside the range (table id "outside") prints as a code block, one line per stock-day. A "|" inside
a table cell is escaped. No time stamp and no version: the text depends only on the tables and the spec.
"""

from __future__ import annotations

from dof.experiments import report
from dof.experiments.tables import Cell, Report, Section, Table

HEADING = {2: "##", 3: "###", 4: "####"}
LIST_TABLES = ("outside",)


def fmt_cell(c: Cell) -> str:
    return "" if c.value is None else format(c.value, c.fmt)


def _md(s: str) -> str:
    return s.replace("|", "\\|")


def format_table(t: Table) -> list[str]:
    out = [f"#### {t.title}", ""]
    if not t.rows:
        return out + ["No score yet.", ""]
    out.append("| " + " | ".join(_md(c.title) for c in t.columns) + " |")
    out.append("|" + "|".join("---:" if c.align == "right" else "---" for c in t.columns) + "|")
    for r in t.rows:
        out.append("| " + " | ".join(_md(fmt_cell(c)) for c in r.cells) + " |")
    if t.note:
        out += ["", t.note]
    return out + [""]


def outside_lines(t: Table) -> list[str]:
    return [" ".join(fmt_cell(c) for c in r.cells) for r in t.rows]


def _section(s: Section, level: int, deeper: int) -> list[str]:
    out = [f"{HEADING[level]} {s.title}", ""]
    for p in s.paragraphs:
        out += [p, ""]
    if s.headlines:
        out += ["- " + h.template.format(**{k: fmt_cell(c) for k, c in h.cells.items()}) for h in s.headlines]
        out.append("")
    for t in s.tables:
        if t.id in LIST_TABLES:
            out += [f"#### {t.title}", "", "```", *(outside_lines(t) or ["none"]), "```", ""]
        else:
            out += format_table(t)
    for x in s.subsections:
        out += _section(x, deeper, deeper)
    return out


def format_report(r: Report) -> str:
    out = [f"# {r.title}", ""]
    for p in r.intro:
        out += [p, ""]
    for s in r.sections:
        out += _section(s, 2, 3)
    return "\n".join(out).rstrip("\n") + "\n"


def render(tables, spec, *, strict: bool = True) -> str:
    """The Markdown report of ``spec`` from the result tables."""
    return format_report(report.build(tables, spec, strict=strict))
