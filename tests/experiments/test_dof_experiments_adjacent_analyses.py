"""The adjacent-days tables: a part over the seeds and its uncertainty, the paired difference (seeds in numeric
order), the stocks left out of the pair, and the network and ridge part tables."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dof.experiments.analyses.adjacent import _names, paired_difference, pair_table, part_table, seed_table
from dof.experiments.cells import Rows
from fake_lab import ADJ_SEEDS, exported_tables


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    return exported_tables(tmp_path_factory.mktemp("adjacent"))


def _rows(values: dict) -> Rows:
    """values: symbol -> r2, one cell each."""
    syms = tuple(values)
    n = len(syms)
    return Rows(syms, ("d",) * n, (1,) * n, tuple(values.values()), (1.0,) * n, (1.0,) * n)


def test_seed_table_common_stocks_and_u() -> None:
    t = seed_table({7: _rows({"AAA": 0.1, "BBB": 0.2, "CCC": 0.5}), 3: _rows({"AAA": 0.3, "BBB": 0.0})})
    assert t["n_stocks"] == 2 and t["n_seeds"] == 2 and list(t["per_seed"]) == [7, 3]
    vbar = np.array([np.mean([0.1, 0.3]), np.mean([0.2, 0.0])])
    se = float(vbar.std(ddof=1) / np.sqrt(2))
    means = np.array([np.mean([0.1, 0.2, 0.5]), np.mean([0.3, 0.0])])
    sd = float(means.std(ddof=1))
    assert t["se_stocks"] == se and t["sd"] == sd and t["u"] == math.sqrt(se**2 + sd**2 / 2)


def test_paired_difference_numeric_seed_order() -> None:
    aft = {20: _rows({"AAA": 0.4, "BBB": 0.1}), 3: _rows({"AAA": 0.2, "BBB": 0.3, "CCC": 0.0})}
    far = {3: _rows({"AAA": 0.1, "BBB": 0.1}), 20: _rows({"AAA": 0.0, "BBB": 0.2, "DDD": 0.1})}
    p = paired_difference(aft, far)
    assert p["seeds"] == [3, 20] and p["pair"] == ["AAA", "BBB"] and list(p["per_seed"]) == [3, 20]
    d3, d20 = float(np.mean([0.2 - 0.1, 0.3 - 0.1])), float(np.mean([0.4 - 0.0, 0.1 - 0.2]))
    assert p["per_seed"] == {3: d3, 20: d20} and p["d"] == float(np.array([d3, d20]).mean())
    assert p["only_afternoon"] == ["CCC"] and p["only_far"] == ["DDD"] and p["partial"] == []


def test_pair_left_out_names(fake) -> None:
    assert _names(["BKNG", "HOLX"]) == "2 (BKNG, HOLX)" and _names([]) == "0"
    _lab, specs, rt = fake
    t = pair_table(rt, specs["adjacent"], id="pair", title="P", golden="pair")
    rows = {r.key: dict(zip((c.key for c in t.columns), r.cells)) for r in t.rows}
    assert rows[("only_afternoon",)]["value"].value == "0"             # CCC keeps two later days
    assert rows[("only_far",)]["value"].value == "0"
    assert rows[("n_pairs",)]["value"].value == 3 and ("partial",) not in rows
    assert rows[("relation",)]["value"].value is None and rows[("relation",)]["relation"].value is not None
    d, u = rows[("d",)]["value"].raw, rows[("u",)]["value"].raw
    assert rows[("d_over_u",)]["value"].raw == d / u
    assert [k for k in rows if k[0] == "diff"] == [("diff", f"seed={s}") for s in ADJ_SEEDS]


def test_part_tables_net_and_ridge(fake) -> None:
    _lab, specs, rt = fake
    spec = specs["adjacent"]
    net = part_table(rt, spec, part="far", model="LSTM-MLP-OF-recent", id="f", title="F")
    assert [r.key[1:] for r in net.rows] == [("all", "k=1..10"), ("all", "k=10"), ("2026-05-15", "k=1..10"),
                                            ("2026-05-18", "k=1..10"), ("2026-06-12", "k=1..10"), ("secondary", "k=1..10")]
    keys = [c.key for c in net.columns]
    assert keys == ["scored", "horizon", "stocks", *(f"seed={s}" for s in ADJ_SEEDS), "mean", "sd", "se"]
    assert all(c.raw is not None for c in net.rows[0].cells[3:6]) and net.part == "test"
    ridge = part_table(rt, spec, part="valid", model="Ridge-OF-recent", id="v", title="V")
    assert len(ridge.rows) == 2 and ridge.part == "valid"
    cells = dict(zip(keys, ridge.rows[0].cells))
    assert all(cells[f"seed={s}"].value is None for s in ADJ_SEEDS) and cells["sd"].value is None
    assert cells["mean"].raw is not None and cells["se"].raw is not None
