"""Score cells, the pooled validation loss and the row cap on the synthetic run."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dof.deep.data.eval import SCORE_ARRAYS, DeviceDay, RamRows, eval_stocks, load_eval_stock, to_device  # noqa: E402
from dof.deep.evaluate.metrics import stock_values  # noqa: E402
from dof.deep.evaluate.parts import score_part, validation_loss  # noqa: E402
from dof.deep.evaluate.predict import score_rows, valid_loss  # noqa: E402
from dof.deep.models import FAMILIES, build  # noqa: E402
from dof.lob.features import inverse_target  # noqa: E402
from dof.lob.stats import r2_os  # noqa: E402

CPU = torch.device("cpu")


class _Zeros(torch.nn.Module):
    def __init__(self, n_out: int) -> None:
        super().__init__()
        self.n_out = n_out
        self.unused = torch.nn.Linear(1, 1)

    def forward(self, x):
        return torch.zeros(len(x), self.n_out)


@pytest.fixture
def zeros_model(synth_run, monkeypatch):
    monkeypatch.setitem(FAMILIES, "zeros", lambda cfg: _Zeros(cfg.n_out))
    return build(replace(synth_run.cfg.models["TinyMLP"], family="zeros", init="torch"))


def _offs(cfg):
    return torch.arange(-cfg.features.history + 1, 1)


def test_score_rows_match_numpy(synth_run, zeros_model) -> None:
    cfg, cache = synth_run.cfg, synth_run.cache
    symbol, metas = eval_stocks(cfg, cfg.split, cache, "valid")[0]
    stock = load_eval_stock(cfg, cfg.split, cache, "valid", symbol, metas, SCORE_ARRAYS)
    rows = score_rows(zeros_model, stock, to_device(stock, CPU), _offs(cfg), eval_batch=512, n_out=10, amp=False)
    assert len(rows) == 10 * len(stock.days)
    for day in stock.days:
        y = day.arr("y")[day.rows]
        yh = inverse_target(np.zeros((len(day.rows), 10)), stock.scalers["y"])
        for j, r in enumerate(x for x in rows if x["date"] == day.date):
            assert r["k"] == j + 1 and r["n_rows"] == len(day.rows) and r["late_windows"] == 0
            assert r["sse"] == float(((y[:, j] - yh[:, j]) ** 2).sum())
            assert r["sst"] == float(((y[:, j] - y[:, j].mean()) ** 2).sum())
            assert r["r2_os"] == r2_os(y[:, j], yh[:, j])


def test_stock_values_skip_nan_cells() -> None:
    rows = [{"symbol": "AAA", "k": 1, "date": "d", "r2_os": 0.1}, {"symbol": "AAA", "k": 2, "date": "d", "r2_os": math.nan},
            {"symbol": "BBB", "k": 1, "date": "d", "r2_os": -0.2}, {"symbol": "CCC", "k": 1, "date": "d", "r2_os": math.nan}]
    assert stock_values(rows) == {"AAA": 0.1, "BBB": -0.2}
    assert stock_values(rows, ks=(2,)) == {}


def test_validation_loss_is_equal_weight_over_stocks(synth_run) -> None:
    cfg, cache = synth_run.cfg, synth_run.cache
    torch.manual_seed(0)
    model = build(cfg.models["TinyMLP"])
    vl, per, n = validation_loss(model, cfg, cfg.split, cache, CPU, _offs(cfg), amp=False)
    assert sorted(per) == ["AAA", "BBB", "CCC"]
    assert vl == float(np.mean(list(per.values())))
    assert n == sum(m["n_samples"] for _, ms in eval_stocks(cfg, cfg.split, cache, "valid") for m in ms)


def test_row_cap_cuts_rows(synth_run) -> None:
    cfg, cache = synth_run.cfg, synth_run.cache
    torch.manual_seed(0)
    model = build(cfg.models["TinyMLP"])
    rows, per = score_part(model, cfg, cfg.split, cache, "valid", CPU, _offs(cfg), name="TinyMLP-all", n_train=1,
                           seed=0, amp=False, row_cap=512)
    assert rows and all(r["n_rows"] <= 512 for r in rows) and sorted(per) == ["AAA", "BBB", "CCC"]
    assert {(r["model"], r["seed"], r["n_train"]) for r in rows} == {("TinyMLP-all", 0, 1)}


def test_late_window_raises(synth_run, zeros_model) -> None:
    cfg, cache = synth_run.cfg, synth_run.cache
    symbol, metas = eval_stocks(cfg, cfg.split, cache, "valid")[0]
    stock = load_eval_stock(cfg, cfg.split, cache, "valid", symbol, metas, SCORE_ARRAYS)
    dd = to_device(stock, CPU)[0]
    early = DeviceDay(dd.day, dd.z, dd.zy, dd.rows, dd.seq, RamRows(dd.day.arr("seq") - 1, CPU))
    with pytest.raises(AssertionError, match="end after their label starts"):
        valid_loss(zeros_model, [early], _offs(cfg), eval_batch=512, n_out=10, amp=False)
