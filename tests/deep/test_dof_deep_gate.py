"""The test-day gate. No torch: the records and weight files are written by the test, with their sha256; the
test-day loads read the synthetic run's cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dof.deep.data.eval import SCORE_ARRAYS, eval_stocks, load_eval_stock
from dof.deep.errors import TestDayRefused
from dof.deep.evaluate import gate as gate_module
from dof.deep.evaluate.gate import TestGate
from dof.deep.linear.ridge import _test_fits
from dof.deep.train.record import FitRecord

NAME = "TinyMLP-all"


def _record(folder: Path, model: str = NAME, *, status: str = "finished", protocol=None,
            weights: str = "best_weights_seed0.pt", data: bytes = b"weights") -> FitRecord:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / weights).write_bytes(data)
    doc = {"model": model, "status": status, "seed": 0, "protocol_sha256": protocol,
           "weights": {"file": weights, "sha256": hashlib.sha256(data).hexdigest()}, "finished": "now"}
    (folder / "fit_seed0.json").write_text(json.dumps(doc), encoding="utf-8")
    return FitRecord.load(folder / "fit_seed0.json")


def test_token_refusals(synth_run, tmp_path: Path) -> None:
    cfg, split = synth_run.cfg, synth_run.cfg.split
    gate = TestGate(cfg, split)
    with pytest.raises(TestDayRefused, match="not a model the configuration scores"):
        gate.token(_record(tmp_path / "other", model="Other"))
    with pytest.raises(TestDayRefused, match="no fit record"):
        gate.token(None)
    with pytest.raises(TestDayRefused, match="diverged"):
        gate.token(_record(tmp_path / "div", status="diverged"))
    changed = _record(tmp_path / "changed")
    changed.weights_path.write_bytes(b"other weights")
    with pytest.raises(TestDayRefused, match="saved weights"):
        gate.token(changed)
    no_ca = TestGate(replace(cfg, protocol=replace(cfg.protocol, corporate_actions=tmp_path / "missing.csv")), split)
    with pytest.raises(TestDayRefused, match="missing"):
        no_ca.token(_record(tmp_path / "ok"))
    with pytest.raises(TestDayRefused, match="another protocol document"):
        gate.token(_record(tmp_path / "prot", protocol="0" * 64))


def test_token_refused_without_protocol_document(synth_run, tmp_path: Path) -> None:
    cfg = synth_run.cfg
    strict = replace(cfg, protocol=replace(cfg.protocol, test_days_need_document=True, document=None))
    with pytest.raises(TestDayRefused, match="no protocol document"):
        TestGate(strict, strict.split).token(_record(tmp_path / "ok"))


def _test_stock(synth_run):
    cfg = synth_run.cfg
    return eval_stocks(cfg, cfg.split, synth_run.cache, "test")[0]


def test_test_day_load_needs_token(synth_run, tmp_path: Path) -> None:
    cfg, split, cache = synth_run.cfg, synth_run.cfg.split, synth_run.cache
    symbol, metas = _test_stock(synth_run)
    rec = _record(tmp_path / NAME)

    def load(gate, tokens, models=(NAME,)):
        return load_eval_stock(cfg, split, cache, "test", symbol, metas, SCORE_ARRAYS, gate=gate, tokens=tokens,
                               models=models)
    gate, other = TestGate(cfg, split), TestGate(cfg, split)
    with pytest.raises(TestDayRefused):
        load(None, None)
    with pytest.raises(TestDayRefused):
        load(gate, {NAME: other.token(rec)})
    elsewhere = TestGate(cfg, replace(split, name="other"))
    with pytest.raises(TestDayRefused):
        load(elsewhere, {NAME: elsewhere.token(rec)})
    with pytest.raises(TestDayRefused):
        load(gate, {}, models=())
    stock = load(gate, {NAME: gate.token(rec)})
    assert stock.symbol == symbol and stock.days[0].part == "test"


def test_token_carries_binding(synth_run, tmp_path: Path) -> None:
    rec = _record(tmp_path / NAME)
    tok = TestGate(synth_run.cfg, synth_run.cfg.split).token(rec)
    assert (tok.stage, tok.part, tok.model, tok.seed, tok.weights_sha256) == ("synth", "test", NAME, 0, rec.weights_sha256)


def test_token_of_other_model_refused(synth_run, tmp_path: Path) -> None:
    cfg, split, cache = synth_run.cfg, synth_run.cfg.split, synth_run.cache
    symbol, metas = _test_stock(synth_run)
    gate = TestGate(cfg, split)
    tok = gate.token(_record(tmp_path / NAME))
    for tokens in ({"TinyLSTM-recent": tok}, {NAME: tok}):
        with pytest.raises(TestDayRefused, match="TinyLSTM-recent"):
            load_eval_stock(cfg, split, cache, "test", symbol, metas, SCORE_ARRAYS, gate=gate, tokens=tokens,
                            models=("TinyLSTM-recent",))


def test_weights_changed_after_token_refused(synth_run, tmp_path: Path, monkeypatch) -> None:
    cfg = synth_run.cfg
    buf = tmp_path / "w.npz"
    np.savez(buf, W=np.zeros((2000, 10)), b=np.zeros(10), lam=1.0)
    _record(tmp_path / "Ridge-all", model="Ridge-all", weights="weights_seed0.npz", data=buf.read_bytes())
    issue = gate_module.TestGate.token

    def token_then_change(self, record):
        tok = issue(self, record)
        record.weights_path.write_bytes(b"changed after the token")
        return tok
    monkeypatch.setattr(gate_module.TestGate, "token", token_then_change)
    with pytest.raises(TestDayRefused, match="changed after the test gate"):
        _test_fits(cfg, cfg.split, ["Ridge-all"], tmp_path, 0)
