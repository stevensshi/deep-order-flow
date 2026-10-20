"""The primitives (stock values, mean and s.e., paired differences, the price-ratio filter) on hand cells, and the
main-run score, older-data, fit and check tables on the fake lab."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dof.experiments.analyses.fits import checks_table, fits_table
from dof.experiments.analyses.rescore import rescore_table
from dof.experiments.analyses.scores import older_rows, score_rows
from dof.experiments.analyses.stocks import agg, inside, paired, stock_values
from dof.experiments.cells import Rows
from dof.experiments.errors import AnalysisError
from dof.experiments.specs import TEST_DAYS, TEST_DAYS_LABEL
from fake_lab import exported_tables

NAN = float("nan")


def _rows(cells) -> Rows:
    """cells: (symbol, date, k, r2)."""
    sym, date, k, r2 = zip(*cells)
    return Rows(sym, date, k, r2, tuple(1.0 for _ in cells), tuple(1.0 for _ in cells))


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    return exported_tables(tmp_path_factory.mktemp("analyses"))


def test_stock_values_order_and_nan() -> None:
    rows = _rows([("BBB", "d1", 1, 0.1), ("AAA", "d1", 1, 0.2), ("BBB", "d1", 2, NAN), ("AAA", "d2", 1, 0.4),
                  ("BBB", "d2", 10, 0.3), ("CCC", "d1", 10, NAN)])
    v = stock_values(rows)
    assert list(v) == ["BBB", "AAA"]
    assert v == {"BBB": float(np.mean([0.1, 0.3])), "AAA": float(np.mean([0.2, 0.4]))}
    assert stock_values(rows, ks=(10,)) == {"BBB": 0.3}
    assert stock_values(rows, dates={"d1"}) == {"BBB": 0.1, "AAA": 0.2}


def test_agg_counts() -> None:
    rows = _rows([("AAA", "d1", 1, 0.1), ("AAA", "d1", 2, NAN), ("BBB", "d1", 1, 0.3)])
    a = agg(rows)
    x = np.array([0.1, 0.3])
    assert (a["n_cells"], a["n_no_r2"], a["n_stocks"]) == (3, 1, 2)
    assert a["mean"] == float(x.mean()) and a["se"] == float(x.std(ddof=1) / np.sqrt(2))
    one = agg(_rows([("AAA", "d1", 1, 0.1)]))
    assert one["mean"] == 0.1 and math.isnan(one["se"])
    assert math.isnan(agg(_rows([("AAA", "d1", 1, NAN)]))["mean"])


def test_paired_uses_sorted_common_stocks() -> None:
    a = _rows([("CCC", "d", 1, 0.5), ("AAA", "d", 1, 0.1), ("BBB", "d", 1, 0.2)])
    b = _rows([("BBB", "d", 1, 0.25), ("AAA", "d", 1, 0.0), ("DDD", "d", 1, 0.9)])
    p = paired(a, b)
    d = np.array([0.1 - 0.0, 0.2 - 0.25])                       # AAA, BBB: the sorted common stocks
    assert p["n_stocks"] == 2 and p["n_better"] == 1
    assert p["mean"] == float(d.mean()) and p["se"] == float(d.std(ddof=1) / np.sqrt(2))


def test_inside_closed_range_and_missing_ratio() -> None:
    rows = _rows([("AAA", "d", 1, 0.1), ("BBB", "d", 1, 0.2), ("CCC", "d", 1, 0.3), ("AAA", "d", 2, 0.4)])
    sd = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC", "DDD"], "date": ["d"] * 4, "status": ["scored"] * 3 + ["left_out"],
                       "price_ratio": [0.6, 1.6, 1.7, 1.0], "ratio_lo": [0.6] * 4, "ratio_hi": [1.6] * 4})
    kept = inside(rows, sd)
    assert kept.symbol == ("AAA", "BBB", "AAA") and kept.r2 == (0.1, 0.2, 0.4)
    with pytest.raises(AnalysisError, match="EEE d: no price ratio"):
        inside(_rows([("EEE", "d", 1, 0.1)]), sd)


def test_score_rows_layout(fake) -> None:
    _lab, specs, rt = fake
    t = score_rows(rt, specs["alldays"], record="test", model="CNN-LSTM-OF-alldays", id="t", title="T", golden="g")
    keys = [r.key for r in t.rows]
    assert keys == [("CNN-LSTM-OF-alldays", TEST_DAYS_LABEL, "k=1..10"), ("CNN-LSTM-OF-alldays", TEST_DAYS_LABEL, "k=10"),
                    *[("CNN-LSTM-OF-alldays", d, "k=1..10") for d in TEST_DAYS]]
    assert t.part == "test" and t.golden == "g"
    assert [c.key for c in t.columns] == ["days", "horizon", "stocks", "r2", "se"]
    assert t.rows[-1].cells[2].value == 2                       # CCC is left out of the last day
    v = score_rows(rt, specs["alldays"], record="fit_valid", model="Ridge-OF-alldays", id="v", title="V")
    assert v.part == "valid" and len(v.rows) == 4
    none = score_rows(rt, specs["alldays"], record="test", model="No-Such-Model", id="n", title="N")
    assert none.rows == ()


def test_older_rows_skip_missing_model(fake) -> None:
    _lab, specs, rt = fake
    spec = replace(specs["alldays"], pairs=(("LSTM-MLP-OF-alldays", "LSTM-MLP-OF-recent"), ("Gone", "Ridge-OF-recent")))
    t = older_rows(rt, spec, record="test", id="o", title="O")
    assert [r.key for r in t.rows] == [("LSTM-MLP-OF-alldays", "LSTM-MLP-OF-recent", "k=1..10"),
                                       ("LSTM-MLP-OF-alldays", "LSTM-MLP-OF-recent", "k=10")]
    assert t.rows[0].cells[1].value == ", ".join(TEST_DAYS)


def test_fits_and_checks_tables(fake) -> None:
    _lab, specs, rt = fake
    f = fits_table(rt, specs["seeds"], models=("LSTM-MLP-OF-recent",), seeds=(0, 3089492870, 771810125), id="f",
                   title="F", golden="fits")
    assert [r.key for r in f.rows] == [("LSTM-MLP-OF-recent", f"seed={s}") for s in (0, 3089492870, 771810125)]
    cells = dict(zip((c.key for c in f.columns), f.rows[1].cells))
    assert cells["status"].value == "finished" and cells["checks"].value == 3 and cells["best_check"].value == 2
    assert cells["steps"].fmt == "," and cells["passes"].raw == 7.5 and cells["weights"].value.endswith("…")
    c = checks_table(rt, specs["seeds"], model="LSTM-MLP-OF-recent", seed=0, id="c", title="C")
    assert [r.key for r in c.rows] == [("LSTM-MLP-OF-recent", "seed=0", f"check={n}") for n in (1, 2, 3)]
    flags = [dict(zip((x.key for x in c.columns), r.cells))["new_best"].value for r in c.rows]
    assert flags == ["yes", "yes", ""] and c.part == "valid"


def test_score_rows_single_day_part_has_no_day_rows(fake) -> None:
    _lab, specs, rt = fake
    t = score_rows(rt, specs["earlier"], record="valid", model="LSTM-OF", id="v", title="V")
    assert [r.key for r in t.rows] == [("LSTM-OF", "2019-08-30", "k=1..10"), ("LSTM-OF", "2019-08-30", "k=10")]
    assert t.part == "valid"


def test_rescore_table_pairs_both_scorings(fake) -> None:
    _lab, specs, rt = fake
    spec = specs["earlier"]
    t = rescore_table(rt, spec, other="decay", basis="frozen", models=("LSTM-OF", "LSTM-MLP-OF"), id="r", title="R")
    assert [r.key for r in t.rows] == [("LSTM-MLP-OF", d) for d in ("2019-10-18", "2019-10-30", "2019-12-30")]
    keys = [c.key for c in t.columns]
    for row in t.rows:
        cells = dict(zip(keys, row.cells))
        assert cells["cells"].value == 30 and cells["stocks"].value == 3 and cells["again_stocks"].value == 3
        stored = rt.rows(rt.rid(experiment="earlier", record="test", model="LSTM-MLP-OF", seed=0))
        again = [u for u in rt.decay_units("LSTM-MLP-OF", row.key[1], "frozen") if u["status"] == "scored"]
        pairs = {(s, k): r2 for s, d, k, r2 in zip(stored.symbol, stored.date, stored.k, stored.r2) if d == row.key[1]}
        same = sum(pairs[(s, k)] == r2 for u in again for s, k, r2 in zip(rt.rows(u["record_id"]).symbol,
                                                                          rt.rows(u["record_id"]).k,
                                                                          rt.rows(u["record_id"]).r2))
        assert cells["equal"].value == same
        assert math.isclose(cells["diff"].raw, cells["again_r2"].raw - cells["r2"].raw, abs_tol=1e-15)
    with pytest.raises(AnalysisError, match="only the decay measurement"):
        rescore_table(rt, spec, other="adjacent", basis="frozen", models=("LSTM-MLP-OF",), id="r", title="R")
