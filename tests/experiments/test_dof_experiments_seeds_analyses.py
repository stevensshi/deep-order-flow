"""Statistics over seeds, the rule for more seeds (both branches and a missing record), the seed tables with the
ensembles, and the kinds of a cell without an R²."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from dof.experiments.analyses.seeds import finite_table, rule_values, seed_older, seed_scores, seed_stats
from dof.experiments.analyses.stocks import finite_kind
from dof.experiments.cells import Rows
from dof.experiments.spec import SeedPolicy
from dof.experiments.specs import SPECS, TEST_DAYS_LABEL
from fake_lab import exported_tables

NAN = float("nan")


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    return exported_tables(tmp_path_factory.mktemp("seeds"))


class StubTables:
    """Just enough of ResultTables for the rule: fit records by (model, seed)."""

    def __init__(self, rows: dict) -> None:
        self.by = rows

    def rid(self, *, experiment, record, model, seed=None, members=None, part=None):
        return f"{model}/{seed}" if (model, seed) in self.by else None

    def rows(self, rid):
        model, seed = rid.split("/")
        return self.by[(model, int(seed))]


def _two_stocks(base: float, diff: float) -> tuple[Rows, Rows]:
    older = Rows(("AAA", "BBB"), ("d", "d"), (1, 1), (base + diff, base + diff), (1.0, 1.0), (1.0, 1.0))
    recent = Rows(("AAA", "BBB"), ("d", "d"), (1, 1), (base, base), (1.0, 1.0), (1.0, 1.0))
    return older, recent


def _rule_spec():
    pol = SeedPolicy(first=(0, 1, 2), drawn=(), older="OLD", recent="NEW", half=0.5, more=2)
    return replace(SPECS["seeds"], seeds=(0, 1, 2), seed_policy=pol, seed0_from=None, rounds=())


def _stub(diffs: dict) -> StubTables:
    rows = {}
    for s, d in diffs.items():
        rows[("OLD", s)], rows[("NEW", s)] = _two_stocks(0.01, d)
    return StubTables(rows)


def test_seed_stats() -> None:
    s = seed_stats({1: 0.1, 2: 0.3, 3: NAN, 4: 0.2})
    v = np.array([0.1, 0.3, 0.2])
    assert s == {"n": 3, "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": 0.1, "max": 0.3}
    one = seed_stats({1: 0.5})
    assert one["n"] == 1 and one["mean"] == 0.5 and math.isnan(one["sd"])


def test_rule_values_both_branches() -> None:
    spec = _rule_spec()
    agree = rule_values(_stub({0: 0.010, 1: 0.011, 2: 0.009}), spec)
    v = [agree["differences"][s] for s in (0, 1, 2)]
    assert agree["mean"] == float(np.mean(v)) and agree["sd"] == float(np.std(v, ddof=1))
    assert agree["more"] is False and agree["half_abs_mean"] == 0.5 * abs(agree["mean"])
    assert agree["stocks"] == {0: 2, 1: 2, 2: 2}
    scatter = rule_values(_stub({0: 0.01, 1: -0.01, 2: 0.03}), spec)
    assert scatter["more"] is True
    missing = rule_values(_stub({0: 0.01, 2: 0.01}), spec)
    assert missing["more"] is True and missing["mean"] is None and missing["sd"] is None
    assert missing["half_abs_mean"] is None and set(missing["differences"]) == {0, 2}


def test_seed_scores_ensemble_columns(fake) -> None:
    _lab, specs, rt = fake
    t = seed_scores(rt, specs["seeds"], record="test", fit="CNN-LSTM-OF-alldays", id="s", title="S")
    keys = [c.key for c in t.columns]
    assert keys == ["days", "horizon", "stocks", "seed=0", "seed=3089492870", "seed=771810125", "mean", "sd", "min",
                    "max", "seeds", "ensemble", "ensemble_stocks"]
    first = dict(zip(keys, t.rows[0].cells))
    assert t.rows[0].key == ("CNN-LSTM-OF-alldays", TEST_DAYS_LABEL, "k=1..10")
    assert first["seeds"].value == 3 and first["ensemble"].raw is not None and first["ensemble_stocks"].value == 3
    assert first["stocks"].value == "3" and t.part == "test"
    sec = seed_scores(rt, specs["seeds"], record="test", fit="CNN-LSTM-OF-alldays", secondary=True, id="x", title="X")
    assert dict(zip(keys, sec.rows[0].cells))["stocks"].value == "3"
    assert dict(zip(keys, sec.rows[3].cells))["stocks"].value == "2"          # BBB is outside the range on 2026-05-18


def test_seed_older(fake) -> None:
    _lab, specs, rt = fake
    t = seed_older(rt, specs["seeds"], record="fit_valid", id="o", title="O")
    assert [r.key for r in t.rows][:2] == [("2025-12-11,2025-12-12", "k=1..10"), ("2025-12-11,2025-12-12", "k=10")]
    cells = dict(zip((c.key for c in t.columns), t.rows[0].cells))
    assert cells["seeds"].value == 3 and cells["ensembles"].raw is not None and t.part == "valid"


def test_finite_kinds(fake) -> None:
    assert finite_kind(0.01, 0.9, 1.0) == "finite"
    assert finite_kind(NAN, 0.0, 0.0) == "zero_variance"
    assert finite_kind(NAN, 1.0, 2.0) == "not_finite"
    assert finite_kind(0.1, float("inf"), 2.0) == "not_finite"
    assert finite_kind(0.1, 1.0, None) == "other"
    assert finite_kind(0.1, 1.0, -1.0) == "other"
    assert finite_kind(NAN, float("inf"), 0.0) == "other"
    assert finite_kind(0.0, 0.0, 0.0) == "other"
    _lab, specs, rt = fake
    t = finite_table(rt, specs["seeds"], id="f", title="F")
    rows = {r.key[0]: dict(zip((c.key for c in t.columns), r.cells)) for r in t.rows}
    assert list(rows) == ["LSTM-MLP-OF-recent", "LSTM-MLP-OF-alldays", "CNN-LSTM-OF-alldays"]
    assert rows["LSTM-MLP-OF-recent"]["files"].value == 8 and rows["LSTM-MLP-OF-recent"]["zero_variance"].value == 1
    assert rows["CNN-LSTM-OF-alldays"]["zero_variance"].value == 0
