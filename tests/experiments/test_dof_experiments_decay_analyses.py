"""The decay tables on the fake lab: a cell selects and counts units, the difference reads the inside units of both
bases, the days table counts held units and reasons, the outside list keeps days, bases, low before high and the
file-name order."""

from __future__ import annotations

import pytest

from dof.experiments.analyses.decay import _reason_kind, decay_cell, decay_days, decay_diff, decay_outside, decay_table
from fake_lab import exported_tables

LSTM = "LSTM-MLP-OF"


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    return exported_tables(tmp_path_factory.mktemp("decay"))


def test_decay_cell_selects_and_counts(fake) -> None:
    _lab, _specs, rt = fake
    units = rt.decay_units(LSTM, "2021-08-13", "frozen")          # BBB has the ratio 0.5 there
    a, i = decay_cell(rt, units, "all"), decay_cell(rt, units, "inside")
    assert (a["stock_days"], a["n_stocks"], i["stock_days"], i["n_stocks"]) == (3, 3, 2, 2)
    assert a["stock_days_no_r2"] == 0
    ref = rt.decay_units(LSTM, "2025-12-12", "refreshed")         # CCC is left out
    assert decay_cell(rt, ref, "all")["stock_days"] == 2
    assert decay_cell(rt, [], "all")["n_stocks"] == 0


def test_decay_diff_inside_on_both(fake) -> None:
    _lab, _specs, rt = fake
    fr, re = rt.decay_units(LSTM, "2025-12-12", "frozen"), rt.decay_units(LSTM, "2025-12-12", "refreshed")
    assert decay_diff(rt, re, fr, "all")["n_stocks"] == 2           # AAA, BBB scored on both
    assert decay_diff(rt, re, fr, "inside")["n_stocks"] == 1        # AAA is outside on both, CCC left out of (ii)


def test_days_table_counts(fake) -> None:
    _lab, specs, rt = fake
    t = decay_days(rt, specs["decay"], id="days", title="D", golden="days")
    rows = {r.key: dict(zip((c.key for c in t.columns), r.cells)) for r in t.rows}
    ref = rows[("2025-12-12", "refreshed")]
    assert (ref["held"].value, ref["scored"].value, ref["reused"].value, ref["group"].value) == (3, 2, 1, 0)
    froz = rows[("2025-12-12", "frozen")]
    assert (froz["held"].value, froz["above"].value, froz["inside"].value) == (3, 1, 2)
    assert rows[("2020-01-30", "frozen")]["held"].value is None          # no unit of that day in the fake lab
    assert [_reason_kind(w) for w in ("reused ticker (rule 1)", "rule 4: no group", "no cleaned row", "x")] == \
        ["reused", "group", "sample", "other"]
    assert "other" not in [c.key for c in t.columns]


def test_outside_order(fake) -> None:
    _lab, specs, rt = fake
    t = decay_outside(rt, specs["decay"], id="outside", title="O", golden="outside")
    assert [r.key for r in t.rows] == [("frozen", "2021-08-13", "BBB"), ("refreshed", "2021-08-13", "BBB"),
                                       ("frozen", "2025-12-12", "AAA"), ("refreshed", "2025-12-12", "AAA")]
    assert [c.value for c in t.rows[0].cells[:3]] + [format(t.rows[0].cells[3].value, t.rows[0].cells[3].fmt)] == \
        ["frozen", "2021-08-13", "BBB", "0.500"]
    tab = decay_table(rt, specs["decay"], model=LSTM, ks=None, id="d", title="T", golden="table")
    assert len(tab.rows) == 2 * len(specs["decay"].decay.days)
    first = dict(zip((c.key for c in tab.columns), tab.rows[0].cells))
    assert first["days_since"].value == 80 and first["months"].raw == 80 / 30.44 and first["ii_r2"].value is None
