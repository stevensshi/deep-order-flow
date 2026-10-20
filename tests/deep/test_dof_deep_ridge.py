"""The pooled ridge references on the synthetic run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dof.deep.cache import layout as L
from dof.deep.linear.ridge import fit_ridge, load_weights


def _quiet(msg: str) -> None:
    pass


@pytest.fixture(scope="module")
def inline(synth_run, tmp_path_factory):
    out = tmp_path_factory.mktemp("ridge_inline")
    return out, fit_ridge(synth_run.cfg, synth_run.cfg.split, out, workers=0, log=_quiet)


def test_ridge_n_train_equals_cache_samples(synth_run, inline) -> None:
    _out, recs = inline
    cfg, cache = synth_run.cfg, synth_run.cache
    plan = json.loads(L.plan_path(cache).read_text(encoding="utf-8"))
    for name, spec in cfg.linear.items():
        want = 0
        for s, x in plan["stocks"].items():
            g = json.loads(L.group_path(cache, s, x["groups"][0]["id"]).read_text(encoding="utf-8"))
            want += sum(r["n_samples"] for r in g["day_rows"] if r["date"] in cfg.split.days(spec.days))
        assert recs[name].doc["train"]["samples"] == want > 0, name


def test_lambda_is_the_lowest_mean_valid_mse(inline) -> None:
    _out, recs = inline
    for rec in recs.values():
        by_lam = rec.doc["valid_loss_by_lambda"]
        best = min(by_lam, key=by_lam.get)
        assert f"{rec.doc['lambda']:g}" == best and rec.doc["best_valid_loss"] == by_lam[best]


def test_pool_equals_inline(synth_run, inline, tmp_path: Path) -> None:
    out0, _recs = inline
    fit_ridge(synth_run.cfg, synth_run.cfg.split, tmp_path, workers=1, log=_quiet)
    for name in synth_run.cfg.linear:
        w0, b0 = load_weights(out0, name)
        w1, b1 = load_weights(tmp_path, name)
        np.testing.assert_allclose(w1, w0, rtol=1e-10, atol=0)
        np.testing.assert_allclose(b1, b0, rtol=1e-10, atol=0)


def test_weights_saved_before_record(inline) -> None:
    out, recs = inline
    for name, rec in recs.items():
        w = out / name / "weights_seed0.npz"
        r = out / name / "fit_seed0.json"
        assert hashlib.sha256(w.read_bytes()).hexdigest() == rec.weights_sha256
        assert w.stat().st_mtime_ns <= r.stat().st_mtime_ns
        assert rec.doc["status"] == "finished" and rec.doc["valid_scores"]
