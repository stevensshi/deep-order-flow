"""The pooled linear references, on the CPU: one ridge regression per ``[linear.<NAME>]`` entry.

- Pass A, training days: per stock the sufficient statistics of the lagged design (``history`` rows x 20 columns)
  against the z-scored labels, added into the pooled ones (sufficient statistics are sums). Each stock gives two
  sets, its recent days and its older days, so one pass serves a reference on all days and one on the recent days.
- The ridge closed form for every λ of ``ridge.lambdas``.
- Pass B, validation days: λ by the networks' rule, the lowest equal-weight mean over stocks of the stock's MSE on
  z-scored labels over its validation days; then the validation scores of the chosen λ (R² in dollars).
The inputs are the networks': the same record files, the same scaling groups, the same z-scored labels in float32.
The weights of the chosen λ are saved before the record; the test days are scored at the very end.

Stocks are worked on in spawned processes with one BLAS thread each, and the statistics are added in universe
order whatever the order the processes end in. ``workers=0`` runs every task in this process, in the same order.
This module imports no torch.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import multiprocessing as mp
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dof.deep.cache import layout as L
from dof.deep.config import CachePaths, RunConfig, Seeds, Split, identity, identity_sha256, legacy_config_dict, protocol_sha256
from dof.deep.data.eval import LOSS_ARRAYS, SCORE_ARRAYS, eval_stocks, load_eval_stock
from dof.deep.data.stream import TrainDay, close_days, train_days
from dof.deep.data.windows import window_index
from dof.deep.errors import TestDayRefused
from dof.deep.runtime.io import now, write_json_atomic
from dof.deep.runtime.memory import peak_rss_kb
from dof.deep.runtime.provenance import provenance
from dof.deep.train.record import FitRecord, existing
from dof.lob.features import apply_scaler, inverse_target
from dof.lob.provenance import sha256_file
from dof.lob.stats import SuffStats, r2_os, solve_ridge, suff_stats_init, suff_stats_update

BUCKET = {"all": ("recent", "older"), "recent": ("recent",)}      # which of a stock's two sets a reference sums
BLAS_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _chunks(rows: np.ndarray, chunk: int):
    for i in range(0, len(rows), chunk):
        yield rows[i:i + chunk]


def _sizes(cfg: RunConfig) -> tuple[int, int, int]:
    """(history, design columns, label columns)."""
    return cfg.features.history, cfg.features.history * L.REC["zof"].shape[0], len(cfg.features.horizons)


def design(zof, ch: np.ndarray, history: int) -> np.ndarray:
    """(len(ch), history*d) float64 design of the sample rows ``ch`` from the float32 feature rows: their windows,
    oldest row first."""
    lo = int(ch[0]) - history + 1
    block = np.asarray(zof[lo:int(ch[-1]) + 1], dtype=np.float64)
    return block[window_index(ch, history) - lo].reshape(len(ch), -1)


class RecordsView:
    """A training stock-day's record file: ``arr("zof")`` is the feature rows (memory-mapped)."""

    def __init__(self, day: TrainDay) -> None:
        self.rec = np.load(day.path, mmap_mode="r")

    def arr(self, name: str) -> np.ndarray:
        return self.rec[name]


def _add(dst: SuffStats, src: SuffStats) -> None:
    """Sufficient statistics are sums, so pooling the stocks is adding their statistics."""
    dst.n += src.n
    for name in ("sx", "sy", "sxx", "sxy", "syy"):
        getattr(dst, name).__iadd__(getattr(src, name))


def stats_task(cfg: RunConfig, split: Split, cache: CachePaths, symbol: str) -> dict:
    """The sufficient statistics of one stock's training days: its recent days and its older days apart."""
    history, p, k = _sizes(cfg)
    recent = set(split.recent_days)
    out: dict[str, SuffStats | None] = {"recent": None, "older": None}
    for day in train_days(cfg, split, "all", cache, {symbol}, compact=False):
        key = "recent" if day.date in recent else "older"
        if out[key] is None:
            out[key] = suff_stats_init(p, k)
        src = RecordsView(day)
        for ch in _chunks(day.rows.astype(np.int64), cfg.ridge.chunk_rows):
            # The networks' target: the label in dollars (float32 on disk) z-scored with the group's label scaler and
            # held in float32.
            zy = apply_scaler(np.asarray(src.rec["y"][ch], dtype=np.float64), day.scaler_y).astype(np.float32)
            suff_stats_update(out[key], design(src.arr("zof"), ch, history), zy.astype(np.float64))
        os.close(day.fd)
    return out


def valid_mse_task(cfg: RunConfig, split: Split, cache: CachePaths, symbol: str, metas: list[dict], cands: dict) -> dict:
    """Per candidate (reference, λ): the stock's MSE on z-scored labels over the sample rows of its validation days."""
    history = cfg.features.history
    stock = load_eval_stock(cfg, split, cache, "valid", symbol, metas, LOSS_ARRAYS)
    se = {c: 0.0 for c in cands}
    n = 0
    for day in stock.days:
        zy = np.asarray(day.arr("zy"), dtype=np.float64)[day.rows]
        i = 0
        for ch in _chunks(day.rows, cfg.ridge.chunk_rows):
            X = design(day.arr("zof"), ch, history)
            for c, (W, b) in cands.items():
                se[c] += float(((X @ W + b - zy[i:i + len(ch)]) ** 2).sum())
            i += len(ch)
        n += zy.size
    return {c: v / n for c, v in se.items()}


def _test_fits(cfg: RunConfig, split: Split, names, results_root: Path, seed: int) -> tuple:
    """For the test days, in this process: one token per reference, and its weights read once from the bytes the
    token's hash was checked against."""
    from dof.deep.evaluate.gate import TestGate
    gate, tokens, fits = TestGate(cfg, split), {}, {}
    for name in names:
        token = gate.token(FitRecord.load(Path(results_root) / name / f"fit_seed{seed}.json"))
        data = (Path(results_root) / name / f"weights_seed{seed}.npz").read_bytes()
        if hashlib.sha256(data).hexdigest() != token.weights_sha256 or token.seed != seed:
            raise TestDayRefused(f"{name}: the saved weights changed after the test gate checked them")
        with np.load(io.BytesIO(data)) as z:
            fits[name] = (z["W"], z["b"])
        tokens[name] = token
    return gate, tokens, fits


def score_task(cfg: RunConfig, split: Split, cache: CachePaths, part: str, symbol: str, metas: list[dict], fits: dict,
               n_train: dict, results_root: Path, seed: int) -> list[dict]:
    """Score cells of one stock on ``part`` for the references of ``fits``: per day and horizon the R² in dollars
    against the day's mean label, with SSE and SST. A test day is read only with tokens asked for in this process,
    one per reference scored."""
    history, _p, k = _sizes(cfg)
    gate = tokens = None
    if part == "test":
        gate, tokens, fits = _test_fits(cfg, split, list(fits), results_root, seed)
    stock = load_eval_stock(cfg, split, cache, part, symbol, metas, SCORE_ARRAYS, gate=gate, tokens=tokens,
                            models=tuple(fits))
    rows = []
    for day in stock.days:
        y = np.asarray(day.arr("y"))[day.rows]
        zh = {name: np.empty((len(day.rows), k)) for name in fits}
        i = 0
        for ch in _chunks(day.rows, cfg.ridge.chunk_rows):
            X = design(day.arr("zof"), ch, history)
            for name, (W, b) in fits.items():
                zh[name][i:i + len(ch)] = X @ W + b
            i += len(ch)
        for name in fits:
            yh = inverse_target(zh[name], stock.scalers["y"])
            rows += [{"symbol": symbol, "model": name, "seed": seed, "part": part, "date": day.date, "k": int(kk),
                      "h_ms": stock.meta["h_ms"][str(kk)], "n_train": n_train[name], "n_rows": len(y),
                      "r2_os": r2_os(y[:, j], yh[:, j]), "sse": float(((y[:, j] - yh[:, j]) ** 2).sum()),
                      "sst": float(((y[:, j] - y[:, j].mean()) ** 2).sum())} for j, kk in enumerate(stock.meta["k_valid"])]
    return rows


@contextlib.contextmanager
def one_blas_pool(workers: int):
    """Spawned worker processes with one BLAS thread each (the variables are read when numpy is loaded in the
    worker); the variables are restored when the pool ends."""
    old = {v: os.environ.get(v) for v in BLAS_VARS}
    for v in BLAS_VARS:
        os.environ[v] = "1"
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            yield ex
    finally:
        for v, value in old.items():
            if value is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = value


def in_order(ex: ProcessPoolExecutor, fn, arg_list: list[tuple], ahead: int):
    """Results in the order of ``arg_list``, with at most ``ahead`` tasks submitted beyond the one being waited for:
    memory stays bounded and the order of every sum is fixed."""
    waiting: deque = deque()
    for args in arg_list:
        waiting.append(ex.submit(fn, *args))
        if len(waiting) >= ahead:
            yield waiting.popleft().result()
    while waiting:
        yield waiting.popleft().result()


def _ordered(fn, arg_list: list[tuple], workers: int, ahead: int):
    if workers == 0:
        for args in arg_list:
            yield fn(*args)
        return
    with one_blas_pool(workers) as ex:
        yield from in_order(ex, fn, arg_list, ahead)


def score_part_linear(cfg: RunConfig, split: Split, cache: CachePaths, part: str, fits: dict, n_train: dict,
                      workers: int, *, results_root: Path, seed: int = 0, log=print, skip=()) -> list[dict]:
    """``skip``: (stock, day) pairs left out by a rule applied at scoring (the corporate actions on the test days)."""
    rows: list[dict] = []
    stocks = eval_stocks(cfg, split, cache, part, skip=skip)
    t0 = time.perf_counter()
    args = [(cfg, split, cache, part, s, m, fits, n_train, results_root, seed) for s, m in stocks]
    for i, r in enumerate(_ordered(score_task, args, workers, 2 * max(workers, 1)), start=1):
        rows += r
        if i % 20 == 0 or i == len(stocks):
            log(f"linear scores on {part}: {i}/{len(stocks)} stocks, {time.perf_counter() - t0:.0f} s")
    return rows


def load_weights(results_root: Path, name: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(results_root) / name / f"weights_seed{seed}.npz") as z:
        return z["W"], z["b"]


def fit_ridge(cfg: RunConfig, split: Split, out_root: Path, *, cache: CachePaths | None = None, workers: int = 6,
              seed: int = 0, log=print) -> dict[str, FitRecord]:
    """Every linear reference of ``cfg``; a reference whose record exists keeps its record and its weights."""
    cache = cache if cache is not None else cfg.machine.cache_for(split.name)
    out_root = Path(out_root)
    seeds = Seeds.same(seed)
    _history, p, k = _sizes(cfg)
    protocol = protocol_sha256(cfg)
    idents = {name: identity_sha256(cfg, split, seeds, name) for name in cfg.linear}
    done = {}
    for name in cfg.linear:
        rec = existing(out_root / name / f"fit_seed{seed}.json", idents[name], protocol)
        if rec is not None:
            done[name] = rec
    if len(done) == len(cfg.linear):
        log(f"{split.name} linear references: fit records exist, skipped")
        return done
    t0, started = time.perf_counter(), now()
    plan = json.loads(L.plan_path(cache).read_text(encoding="utf-8"))
    symbols = list(plan["stocks"])
    log(f"=== {split.name} linear references: {', '.join(cfg.linear)}; {len(symbols)} stocks, {workers} processes; start {started}")

    # Pass A: training days.
    pooled = {name: suff_stats_init(p, k) for name in cfg.linear}
    info = {name: {"stocks": 0} for name in cfg.linear}
    args = [(cfg, split, cache, s) for s in symbols]
    for i, (s, stats) in enumerate(zip(symbols, _ordered(stats_task, args, workers, 3 * max(workers, 1)), strict=True), start=1):
        for name, spec in cfg.linear.items():
            parts = [stats[b] for b in BUCKET[spec.days] if stats[b] is not None]
            for x in parts:
                _add(pooled[name], x)
            info[name]["stocks"] += bool(parts)
        if i % 10 == 0 or i == len(symbols):
            log(f"linear A {i}/{len(symbols)} ({s}): n = " + ", ".join(f"{name} {pooled[name].n:,}" for name in cfg.linear)
                + f"; {time.perf_counter() - t0:.0f} s")
    n_train = {name: int(pooled[name].n) for name in cfg.linear}
    for name, spec in cfg.linear.items():
        days = train_days(cfg, split, spec.days, cache, compact=False)
        want = sum(len(d.rows) for d in days)
        close_days(days)
        if n_train[name] != want:
            raise AssertionError(f"{name}: {n_train[name]:,} rows in the statistics, {want:,} sample rows in the cache")
    lambdas = cfg.ridge.lambdas
    solved = {(name, lam): solve_ridge(pooled[name], lam) for name in cfg.linear for lam in lambdas}
    del pooled
    cands = {c: (f[0], f[1]) for c, f in solved.items()}

    # Pass B: λ on the validation days, by the networks' rule.
    stocks = eval_stocks(cfg, split, cache, "valid")
    per_stock: dict = {c: [] for c in cands}
    args = [(cfg, split, cache, s, m, cands) for s, m in stocks]
    for i, mse in enumerate(_ordered(valid_mse_task, args, workers, 2 * max(workers, 1)), start=1):
        for c, v in mse.items():
            per_stock[c].append(v)
        if i % 20 == 0 or i == len(stocks):
            log(f"linear B {i}/{len(stocks)} stocks, {time.perf_counter() - t0:.0f} s")
    score = {c: float(np.mean(v)) for c, v in per_stock.items()}
    if not all(np.isfinite(list(score.values()))):
        raise RuntimeError("linear references: a validation loss is not finite")
    lam_best = {name: min(lambdas, key=lambda lam, name=name: score[(name, lam)]) for name in cfg.linear}
    for name in cfg.linear:
        log(f"{name}: lambda {lam_best[name]:g}; validation loss by lambda "
            + ", ".join(f"{lam:g}: {score[(name, lam)]:.6f}" for lam in lambdas))

    # λ is fixed: the weights are saved, then the validation scores of the saved weights.
    sha = {}
    for name in cfg.linear:
        if name in done:
            continue
        W, b = cands[(name, lam_best[name])]
        w_path = out_root / name / f"weights_seed{seed}.npz"
        w_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = w_path.with_name("tmp_" + w_path.name)          # ends in .npz: np.savez would append it otherwise
        np.savez(tmp, W=W, b=b, lam=lam_best[name])
        os.replace(tmp, w_path)
        sha[name] = sha256_file(w_path)
    fits = {name: load_weights(out_root, name, seed) for name in cfg.linear}
    rows = score_part_linear(cfg, split, cache, "valid", fits, n_train, workers, results_root=out_root, seed=seed, log=log)
    prov = provenance()
    config = legacy_config_dict(cfg, split, seeds)
    out = dict(done)
    for name, spec in cfg.linear.items():
        if name in done:
            continue
        best = solved[(name, lam_best[name])][2]
        doc = {"model": name, "status": "finished", "stage": split.name, "seed": seed, "config": config,
               "protocol_sha256": protocol, "spec": {"days": spec.days}, "provenance": prov, "lambda": lam_best[name],
               "lambdas": list(lambdas), "valid_loss_by_lambda": {f"{lam:g}": score[(name, lam)] for lam in lambdas},
               "best_valid_loss": score[(name, lam_best[name])], "cond": best["cond"], "fallback": best["fallback"],
               "train": {"days": spec.days, "dates": list(split.days(spec.days)), "stocks": info[name]["stocks"],
                         "samples": n_train[name]},
               "weights": {"file": f"weights_seed{seed}.npz", "sha256": sha[name], "saved_before_scoring": True},
               "valid_scores": [r for r in rows if r["model"] == name], "test_scored": False,
               "wall_seconds": time.perf_counter() - t0, "peak_rss_kb": peak_rss_kb(), "started": started,
               "finished": now(), "identity": identity(cfg, split, seeds, name), "identity_sha256": idents[name]}
        rec_path = out_root / name / f"fit_seed{seed}.json"
        write_json_atomic(rec_path, doc)      # written last: its existence means the reference has ended
        out[name] = FitRecord(doc, rec_path.parent)
        v = [r["r2_os"] for r in doc["valid_scores"] if np.isfinite(r["r2_os"])]
        log(f"{split.name} {name}: n_train {n_train[name]:,}, lambda {lam_best[name]:g}, validation loss "
            f"{doc['best_valid_loss']:.6f}; VALIDATION R2 % mean over all cells {100 * float(np.mean(v)):.3f}")
    log(f"{split.name} linear references: done in {time.perf_counter() - t0:.0f} s")
    return out
