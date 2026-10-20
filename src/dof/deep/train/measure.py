"""The speed of a fit: ms per training step on the real rows and validation rows per second, without a fit result.

The batch order is drawn from ``seed`` (default: the clock), not from a fit's seed: a second measurement on the
same batches would find their rows in the page cache and look faster than a fit. Each measurement is appended to
the JSON list in ``out_file``.
"""

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path

import torch

from dof.deep.cache.layout import REC
from dof.deep.config import CachePaths, RunConfig, Split
from dof.deep.data.eval import eval_stocks
from dof.deep.data.stream import close_days, draw_pass, make_loader, steps_per_pass, train_days
from dof.deep.evaluate.parts import validation_loss
from dof.deep.models import build
from dof.deep.optim import make_optimizer
from dof.deep.runtime.device import setup_device
from dof.deep.runtime.io import now, write_json_atomic
from dof.deep.runtime.memory import peak_rss_kb


def _meminfo() -> dict:
    """MemTotal and Cached of /proc/meminfo in bytes; None where the file does not exist."""
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
    except OSError:
        return {"MemTotal": None, "Cached": None}
    mem = {ln.split(":")[0]: int(ln.split()[1]) * 1024 for ln in lines if ln.split(":")[0] in ("MemTotal", "Cached")}
    return {"MemTotal": mem.get("MemTotal"), "Cached": mem.get("Cached")}


def measure(cfg: RunConfig, split: Split, fit_name: str, *, cache: CachePaths | None = None, steps: int = 1500,
            valid_stocks: int = 6, valid_symbols=None, loader_workers: int | None = None, seed: int | None = None,
            note: str = "", out_file: Path, log=print) -> dict:
    """Training steps on training rows after 100 warm-up steps, and a validation reading on the first stocks (or
    ``valid_symbols``): the time of a check."""
    spec = cfg.fits[fit_name]
    cache = cache if cache is not None else cfg.machine.cache_for(split.name)
    workers = cfg.machine.loader_workers if loader_workers is None else loader_workers
    device = setup_device(cfg.machine, log)
    cuda = device.type == "cuda"
    days = train_days(cfg, split, spec.days, cache)
    try:
        n_train, per_pass = sum(len(d.rows) for d in days), steps_per_pass(days, cfg.train.batch_size)
        seed = seed if seed is not None else int(time.time())
        torch.manual_seed(0)
        gen = torch.Generator().manual_seed(seed)
        model = build(cfg.models[spec.arch]).to(device)
        opt = make_optimizer(model, spec, cfg.train.adam_betas)
        offs = torch.arange(-cfg.features.history + 1, 1, device=device)
        plan = draw_pass(days, gen, batch_size=cfg.train.batch_size)
        model.train()
        warm = 100
        t0 = time.perf_counter()
        for i, batch in enumerate(make_loader(days, plan, history=cfg.features.history, workers=workers,
                                              prefetch=cfg.machine.loader_prefetch)):
            if i == warm:
                if cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            loss = torch.nn.functional.mse_loss(model(batch["x"].to(device)).float(), batch["zy"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if i + 1 == warm + steps:
                break
        if cuda:
            torch.cuda.synchronize()
        ms = 1e3 * (time.perf_counter() - t0) / steps
        valid = eval_stocks(cfg, split, cache, "valid")
        subset = set(valid_symbols) if valid_symbols else {s for s, _ in valid[:valid_stocks]}
        t0 = time.perf_counter()
        _, _, n_sub = validation_loss(model, cfg, split, cache, device, offs, amp=False, symbols=subset)
        rate = n_sub / (time.perf_counter() - t0)
        n_valid = sum(m["n_samples"] for _, ms_ in valid for m in ms_)
        sec_check = cfg.train.check_steps * ms / 1e3
        mem = _meminfo()
        gpu_alloc = torch.cuda.max_memory_allocated() if cuda else 0
        gpu_res = torch.cuda.max_memory_reserved() if cuda else 0
        rec = {"fit": fit_name, "stage": split.name, "measured": now(), "steps_timed": steps, "loader_workers": workers,
               "ms_per_step": ms, "train_samples": n_train, "steps_per_pass": per_pass, "stock_days": len(days),
               "record_bytes": sum(d.n_rows for d in days) * REC.itemsize, "valid_stocks_timed": sorted(subset),
               "valid_rows_timed": n_sub, "valid_rows_per_s": rate, "valid_rows": n_valid,
               "min_per_check_train": sec_check / 60, "min_valid": n_valid / rate / 60,
               "gpu_peak_allocated_bytes": gpu_alloc, "gpu_peak_reserved_bytes": gpu_res, "peak_rss_kb": peak_rss_kb(),
               "mem_total_bytes": mem["MemTotal"], "page_cache_bytes_after": mem["Cached"], "note": note}
        out_file = Path(out_file)
        earlier = json.loads(out_file.read_text(encoding="utf-8")) if out_file.exists() else []
        write_json_atomic(out_file, [*earlier, rec], indent=1)        # every measurement is kept
        log(f"measure {fit_name} ({split.name}): {len(days)} stock-days, {n_train:,} training samples, {per_pass:,} steps "
            f"per pass; {workers} loader processes, batch order seed {seed}; {ms:.2f} ms per step over {steps} steps -> "
            f"{sec_check / 60:.1f} min per check of {cfg.train.check_steps:,} steps, {per_pass * ms / 6e4:.1f} min per pass; "
            f"validation {n_valid:,} rows at {rate:,.0f} rows per s ({len(subset)} stocks timed, load included) -> "
            f"{n_valid / rate / 60:.1f} min; peak GPU allocated {gpu_alloc / 2**30:.2f} GiB, reserved "
            f"{gpu_res / 2**30:.2f} GiB; peak RSS {(peak_rss_kb() or 0) / 1e6:.2f} GB; "
            f"{datetime.datetime.now():%H:%M:%S}")
        return rec
    finally:
        close_days(days)
