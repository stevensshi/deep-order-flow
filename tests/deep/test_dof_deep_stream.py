"""The training stream on the synthetic run: one pass covers every sample once, batches stay inside one day, the
order follows the seed, the loader's alignment checks fire on bad records, loader processes give the batches of
the in-process reader."""

from __future__ import annotations

import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dof.deep.cache.layout import REC  # noqa: E402
from dof.deep.config import CachePaths  # noqa: E402
from dof.deep.data.stream import (  # noqa: E402
    PassPlan,
    TrainDay,
    TrainStream,
    close_days,
    draw_pass,
    make_loader,
    train_days,
)
from dof.deep.data.windows import read_windows  # noqa: E402
from dof.lob.features import Scaler  # noqa: E402
from recipes import train_values as TV  # noqa: E402

HISTORY = 100


@pytest.fixture
def days(synth_run):
    ds = train_days(synth_run.cfg, synth_run.cfg.split, "all", synth_run.cache, compact=False)
    yield ds
    close_days(ds)


def _batches(loader, n=None) -> list[dict]:
    out = []
    for i, bt in enumerate(loader):
        if n is not None and i == n:
            break
        out.append({k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in bt.items()})
    return out


def test_draw_pass_every_sample_once_one_day_per_batch(days) -> None:
    plan = draw_pass(days, torch.Generator().manual_seed(0), batch_size=256)
    seen = [[] for _ in days]
    for di, a, e in plan.batches:
        seen[di].append(plan.order[di][a:e])
        m = len(days[di].rows)
        assert e - a == 256 or e == m, (di, a, e)
    for i, day in enumerate(days):
        got = np.sort(np.concatenate(seen[i]))
        assert np.array_equal(got, np.sort(day.rows)), day.symbol
    assert len(plan.batches) == sum(-(-len(d.rows) // 256) for d in days)


def test_draw_pass_seeds(days) -> None:
    def sha(seed: int) -> str:
        p = draw_pass(days, torch.Generator().manual_seed(seed), batch_size=256)
        return TV.pass_sha256(p.batches, p.order)
    assert sha(0) == sha(0)
    assert sha(0) != sha(1)


def test_alignment_counts_are_zero(days) -> None:
    plan = draw_pass(days, torch.Generator().manual_seed(0), batch_size=256)
    for bt in make_loader(days, plan, history=HISTORY, workers=0, prefetch=4):
        assert (bt["late"], bt["broken"], bt["not_finite"]) == (0, 0, 0)


def _hand_day(tmp_path: Path, seq: np.ndarray, labseq: np.ndarray) -> TrainDay:
    n = len(seq)
    rec = np.zeros(n, dtype=REC)
    rec["seq"], rec["labseq"] = seq, labseq
    path = tmp_path / "train_2019-01-30.rec.npy"
    np.save(path, rec)
    offset = np.load(path, mmap_mode="r").offset
    rows = np.arange(HISTORY - 1, n, dtype=np.int32)
    sc = Scaler(lo=np.full(10, -1.0), hi=np.full(10, 1.0), mean=np.zeros(10), std=np.ones(10), zero_std_cols=())
    return TrainDay("AAA", "2019-01-30", 0, n, rows, sc, path, offset, os.open(path, os.O_RDONLY))


def _one_batch(day: TrainDay) -> dict:
    plan = PassPlan([day.rows], None, np.array([[0, 0, len(day.rows)]], dtype=np.int64))
    try:
        return TrainStream([day], plan, HISTORY)[0]
    finally:
        os.close(day.fd)


def test_shifted_labels_fire_late(tmp_path: Path) -> None:
    seq = np.arange(300, dtype=np.int64) * 2
    bt = _one_batch(_hand_day(tmp_path, seq, seq - 1))
    assert bt["late"] == 300 - (HISTORY - 1) and bt["broken"] == 0


def test_broken_window_fires_broken(tmp_path: Path) -> None:
    seq = np.arange(300, dtype=np.int64) * 2
    seq[150] = seq[149] - 1
    bt = _one_batch(_hand_day(tmp_path, seq, seq))
    assert bt["broken"] >= 1 and bt["late"] == 0


def test_workers_equal_inline(days) -> None:
    plan = draw_pass(days, torch.Generator().manual_seed(3), batch_size=256)
    w2 = TV.batch_values(_batches(make_loader(days, plan, history=HISTORY, workers=2, prefetch=2), 20))
    w0 = TV.batch_values(_batches(make_loader(days, plan, history=HISTORY, workers=0, prefetch=2), 20))
    assert w2 == w0


def test_start_skips_batches(days) -> None:
    plan = draw_pass(days, torch.Generator().manual_seed(4), batch_size=256)
    whole = _batches(make_loader(days, plan, history=HISTORY, workers=0, prefetch=2), 8)
    later = _batches(make_loader(days, plan, history=HISTORY, workers=0, prefetch=2, start=3), 5)
    assert TV.batch_values(later) == TV.batch_values(whole[3:8])


def test_train_days_refuses_wrong_layout(synth_run, tmp_path: Path) -> None:
    cache = CachePaths(tmp_path / "train", tmp_path / "eval")
    shutil.copytree(synth_run.cache.train, cache.train)
    victim = cache.train / "AAA" / "train_2019-01-30.rec.npy"      # the first file train_days opens
    rec = np.load(victim)
    np.save(victim, rec["zof"])                         # the right rows, another dtype
    cfg = replace(synth_run.cfg, machine=replace(synth_run.cfg.machine, cache={"synth": cache}))
    with pytest.raises(RuntimeError, match="cache layout"):
        train_days(cfg, cfg.split, "all", cache, compact=False)


def test_read_windows_refuses_out_of_day(days) -> None:
    with pytest.raises(AssertionError, match="leaves its stock-day"):
        read_windows(days[0], np.array([-1], dtype=np.int64), HISTORY)
    with pytest.raises(AssertionError, match="leaves its stock-day"):
        read_windows(days[0], np.array([days[0].n_rows - HISTORY + 1], dtype=np.int64), HISTORY)
