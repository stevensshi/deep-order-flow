"""What an analysis returns: tables of cells, sections and a report. Standard library only.

A ``Cell`` holds the value that is printed and, for a float, the unscaled float behind it (``raw``), which the golden
tests compare bit for bit. The helpers below are the only places where a float becomes a printable value; the
renderer only formats cells.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Cell:
    value: float | int | str | None     # what is printed; a percent value is already multiplied by 100
    fmt: str = ""                       # format spec for format(value, fmt); "" means str(value); None prints ""
    raw: float | None = None            # the unscaled float behind a printed float


@dataclass(frozen=True)
class Column:
    key: str                            # stable id used in golden keys: "stocks", "r2", "seed=3089492870", ...
    title: str                          # printed header
    compare: bool = True                # False for labels and prose columns
    align: str = "right"                # "left" | "right"


@dataclass(frozen=True)
class Row:
    key: tuple[str, ...]                # stable row id, built with the label rules below
    cells: tuple[Cell, ...]             # one per column, same order


@dataclass(frozen=True)
class Table:
    id: str                             # unique within one report
    golden: str | None                  # the stored table it reproduces (a table id of the golden keys), or None
    title: str                          # printed as a "####" heading above the table
    columns: tuple[Column, ...]
    rows: tuple[Row, ...]
    part: str | None = None             # "valid" | "test" | None: the one part all its numbers come from
    note: str = ""                      # one fixed sentence under the table, no number


@dataclass(frozen=True)
class Headline:
    template: str                       # str.format template; placeholders are names of cells
    cells: Mapping[str, Cell]


@dataclass(frozen=True)
class Section:
    id: str
    title: str                          # "##" heading (top level) or "###" heading (subsection)
    paragraphs: tuple[str, ...] = ()
    headlines: tuple[Headline, ...] = ()
    tables: tuple[Table, ...] = ()
    subsections: tuple[Section, ...] = ()


@dataclass(frozen=True)
class Report:
    title: str
    intro: tuple[str, ...]
    sections: tuple[Section, ...]


def label(title: str, key: str | None = None) -> Column:
    """A label column: text, not compared, left-aligned."""
    return Column(key or title, title, compare=False, align="left")


# -- the cell helpers ----------------------------------------------------------------------------------------


def finite(x) -> bool:
    return x is not None and math.isfinite(float(x))


def pct(x, missing: str | None = None) -> Cell:
    """A fraction printed as a percent with three decimals; ``missing`` is printed when ``x`` is not finite."""
    return Cell(100 * x, ".3f", raw=x) if finite(x) else Cell(missing, "")


def loss(x) -> Cell:
    return Cell(x, ".6f", raw=x) if finite(x) else Cell(None)


def passes(x) -> Cell:
    return Cell(x, ".2f", raw=x) if finite(x) else Cell(None)


def sci(x) -> Cell:
    return Cell(x, ".3e", raw=x) if finite(x) else Cell(None)


def fixed(x, nd: int) -> Cell:
    return Cell(x, f".{nd}f", raw=x) if finite(x) else Cell(None)


def count(n) -> Cell:
    return Cell(None) if n is None else Cell(int(n), "")


def steps(n) -> Cell:
    return Cell(None) if n is None else Cell(int(n), ",")


def text(s) -> Cell:
    return Cell(s, "")


def flag(b) -> Cell:
    return Cell("yes" if b else "", "")


def general(x) -> Cell:
    return Cell(x, "g", raw=x) if finite(x) else Cell(None)


# -- the label rules (the golden key builder repeats them) ------------------------------------------------------


def days_label(dates) -> str:
    return ",".join(sorted(dates))


def horizon_label(ks) -> str:
    return "k=1..10" if ks is None else "k=" + ",".join(str(k) for k in ks)


def seed_label(s) -> str:
    return f"seed={s}"
