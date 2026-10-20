"""One fit of a network: training steps from the stream, a validation reading every ``check_steps`` steps (a
"check"), early stopping over checks, the best check's weights saved before any scoring, then the record.

The batch stream does not restart at a check; the order is drawn again when a pass ends. After every check the
whole state is saved (``checkpoint``), and a fit that finds it continues after that check with the batches an
uninterrupted fit would have used. On the CPU the continued fit gives the same weights to the last bit; on a GPU
with cuDNN's algorithm search two runs of the same steps may differ in the last digits.

Files in the fit folder ``out``: ``best_weights_<tag>.pt`` (before any scoring), then ``fit_<tag>.json``;
``checkpoint_<tag>.pt`` between checks (removed when the record is written); ``progress.json``. Two processes
fitting into the same folder are not guarded against.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from dof.deep.config import (
    CachePaths,
    RunConfig,
    Seeds,
    Split,
    identity,
    identity_sha256,
    legacy_config_dict,
    legacy_fit_dict,
    protocol_sha256,
)
from dof.deep.data.store import load_store
from dof.deep.data.stream import close_days, draw_pass, draw_targets, make_loader, steps_per_pass, train_days
from dof.deep.errors import ConfigError, Diverged, RestoreMismatch
from dof.deep.evaluate.metrics import stock_values
from dof.deep.evaluate.parts import score_part, validation_loss
from dof.deep.models import build, flat_weights, n_params
from dof.deep.optim import make_optimizer, optimizer_facts
from dof.deep.runtime.device import device_facts, setup_device
from dof.deep.runtime.io import now, torch_save_atomic, write_json_atomic
from dof.deep.runtime.memory import peak_rss_kb
from dof.deep.runtime.provenance import provenance
from dof.deep.train.checkpoint import checkpoint_state, load_checkpoint, save_checkpoint
from dof.deep.train.early_stop import EarlyStopping
from dof.deep.train.loop import check_weights, train_step
from dof.deep.train.record import FitRecord, base_doc, diverged_doc, existing, finished_doc, progress
from dof.lob.provenance import sha256_file


@dataclass(frozen=True)
class FitHooks:
    before_validation: Callable[[int], None] | None = None     # called with the check number, before its validation reading


def fit(cfg: RunConfig, split: Split, model: str, seeds: Seeds, out: Path, *, cache: CachePaths | None = None,
        placebo: bool = False, hooks: FitHooks | None = None, log=print) -> FitRecord:
    """The fit ``model`` of ``cfg.fits`` on ``split``. ``placebo``: every training sample gets the label of another
    sample of its stock-day (a leak check); it needs a folder of its own. A record that exists is returned."""
    spec = cfg.fits[model]
    arch = cfg.models[spec.arch]
    cache = cache if cache is not None else cfg.machine.cache_for(split.name)
    out = Path(out)
    tag = seeds.tag
    rec_path, w_path, ck_path, prog_path = (out / f"fit_{tag}.json", out / f"best_weights_{tag}.pt",
                                            out / f"checkpoint_{tag}.pt", out / "progress.json")
    ident_sha = identity_sha256(cfg, split, seeds, model)
    protocol = protocol_sha256(cfg)
    done = existing(rec_path, ident_sha, protocol)
    if done is not None:
        log(f"{split.name} {out.name}: fit record exists ({done.status}), skipped")
        return done
    if placebo and out.name == model:
        raise ValueError("a placebo fit needs its own result folder")
    if arch.input != "of":
        raise ConfigError(f"fits.{model}: {spec.arch} reads {arch.input!r}; the cache holds only the 20-column order flow")
    amp = arch.precision == "amp"
    if amp and cfg.machine.device != "cuda":
        raise ConfigError("mixed precision needs a CUDA device")
    t_start, wall_start = time.perf_counter(), now()
    device = setup_device(cfg.machine, log)
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    tr = cfg.train
    days = train_days(cfg, split, spec.days, cache)
    try:
        n_train, per_pass = sum(len(d.rows) for d in days), steps_per_pass(days, tr.batch_size)
        log(f"=== {split.name} {out.name}: {spec.arch}, {spec.optimizer} lr {spec.lr:g} eps {spec.eps:g}; training days "
            f"'{spec.days}': {len(days)} stock-days of {len({d.symbol for d in days})} stocks, {n_train:,} samples, "
            f"{per_pass:,} steps per pass; a check every {tr.check_steps:,} steps, patience {tr.patience}, at most "
            f"{tr.max_checks} checks{'; PLACEBO: shuffled training labels' if placebo else ''}; start {wall_start}")
        # None unless the compact backend is on: the training records in this process's memory, checked against their
        # manifest, before the first loader process is forked.
        train_store = load_store(days, log=lambda msg: log(f"  {msg}"))

        torch.manual_seed(seeds.init)
        gen = torch.Generator().manual_seed(seeds.data)
        net = build(arch).to(device)
        opt = make_optimizer(net, spec, tr.adam_betas)
        scaler = torch.amp.GradScaler(device.type, enabled=amp)
        offs = torch.arange(-cfg.features.history + 1, 1, device=device)
        facts = optimizer_facts(opt)
        log(f"  optimizer made: {facts['class']} ({' <- '.join(facts['bases'])}), Keras form {facts['keras_form']}, "
            f"lr {facts['lr']:g}, eps {facts['eps']:g}, betas {tuple(facts['betas'])}")
        # Drawn from the fresh generator, before the first pass: the same draw in a fit that continues.
        targets = draw_targets(days, gen) if placebo else None

        stopper = EarlyStopping(tr.patience, tr.max_checks)
        checks: list[dict] = []
        steps, stop = 0, None
        first, resumes, sec_before = 0, [], 0.0      # batch of the pass to start at; the restarts so far; earlier seconds
        ident = {"fit": model, "out": out.name, "identity_sha256": ident_sha, "protocol_sha256": protocol,
                 "placebo_shuffled_labels": placebo, "n_train": n_train, "steps_per_pass": per_pass,
                 "seeds": {"init": seeds.init, "data": seeds.data}}
        ck = load_checkpoint(ck_path, ident)
        if ck is not None:
            net.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            if facts["keras_form"]:
                opt.keras_t = ck["keras_t"]             # Keras' step count: the epsilon of the next step depends on it
            scaler.load_state_dict(ck["scaler"])
            stopper.load_state_dict(ck, device)
            checks, steps, stop = ck["checks"], ck["steps"], ck["stop"]
            gen.set_state(ck["gen_state_pass_start"])   # the pass that was under way is drawn again from this state
            first = steps - ck["pass_start_steps"]
            sec_before, wall_start = ck["sec_fit"], ck["started"]
            resumes = [*ck["resumes"], {"after_check": len(checks), "steps": steps, "batch_of_pass": first,
                                        "checkpoint_saved": ck["saved"], "resumed": now()}]
            # The global generators as they were after the check: dropout draws the masks it would have drawn.
            torch.set_rng_state(ck["rng_cpu"])
            if cuda and ck["rng_cuda"] is not None:
                torch.cuda.set_rng_state_all(ck["rng_cuda"])
            log(f"  {out.name}: CONTINUES after check {len(checks)} ({steps:,} steps; saved {ck['saved']}): batch {first:,} "
                f"of {per_pass:,} of the pass under way; best validation loss so far {stopper.best:.6f}, {stopper.wait} "
                "checks without a new best" + (f"; training had ended ({stop}): only the scoring is left" if stop else ""))
        added = {"identity": identity(cfg, split, seeds, model), "identity_sha256": ident_sha,
                 "seeds": {"init": seeds.init, "data": seeds.data}, "device_settings": device_facts(device)}
        base = base_doc(out_name=out.name, fit=model, split_name=split.name, config=legacy_config_dict(cfg, split, seeds),
                        protocol=protocol, spec=legacy_fit_dict(spec), placebo=placebo)
        loss_sum = torch.zeros((), device=device)
        n_seen, late, broken, not_finite, tick_steps = 0, 0, 0, 0, steps
        t_check = t_tick = time.perf_counter()
        try:
            while stop is None:
                gen_state = gen.get_state()                  # this pass's order is drawn from it: saved with each check
                plan = draw_pass(days, gen, targets, batch_size=tr.batch_size)
                pass_start = steps - first
                net.train()
                # The loader draws its worker seed from its own generator, not from the global one.
                for batch in make_loader(days, plan, history=cfg.features.history, workers=cfg.machine.loader_workers,
                                         prefetch=cfg.machine.loader_prefetch, start=first,
                                         generator=torch.Generator().manual_seed(seeds.data)):
                    last = (steps + 1) % tr.check_steps == 0
                    if last:
                        w0 = flat_weights(net).clone()
                    loss = train_step(net, batch, opt, scaler, device, amp)
                    loss_sum += loss.detach() * len(batch["x"])
                    n_seen += len(batch["x"])
                    late += int(batch["late"])
                    broken += int(batch["broken"])
                    not_finite += int(batch["not_finite"])
                    steps += 1
                    if steps % cfg.machine.progress_every == 0 or last:
                        train_loss = float(loss_sum) / n_seen
                        t_now = time.perf_counter()
                        progress(prog_path, {"fit": out.name, "phase": "validation" if last else "training", "steps": steps,
                                             "checks": len(checks), "passes": steps / per_pass,
                                             "train_loss_since_check": train_loss,
                                             "ms_per_step": 1e3 * (t_now - t_tick) / (steps - tick_steps), "time": now()})
                        t_tick, tick_steps = t_now, steps
                        # The loader's alignment counts must be 0; a failure stops the fit here, not at the check.
                        if late or broken or not_finite:
                            raise AssertionError(f"{out.name}: {late} training windows end after their label starts, "
                                                 f"{broken} windows are not consecutive rows, {not_finite} batches hold a "
                                                 "value that is not finite")
                        if not np.isfinite(train_loss):
                            raise Diverged(f"{out.name}: training loss is not finite at step {steps}")
                    if not last:
                        continue
                    # -- a check ---------------------------------------------------------------------------------------
                    if cuda:
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    dw = check_weights(net, w0)
                    if hooks and hooks.before_validation:
                        hooks.before_validation(len(checks) + 1)
                    vl, per_stock, n_valid = validation_loss(net, cfg, split, cache, device, offs, amp=amp,
                                                             row_cap=tr.eval_row_cap)
                    net.train()
                    t2 = time.perf_counter()
                    if not np.isfinite(vl):     # no best check could be chosen on it: the fit is recorded as diverged
                        raise Diverged(f"{out.name}: validation loss is not finite at check {len(checks) + 1} (step {steps})")
                    improved = stopper.update(vl, net)
                    checks.append({"check": len(checks) + 1, "steps": steps, "passes_completed": steps // per_pass,
                                   "passes": steps / per_pass, "train_loss": train_loss, "rows_seen": n_seen,
                                   "valid_loss": vl, "valid_loss_by_stock": per_stock, "valid_rows": n_valid,
                                   "new_best": improved, "median_abs_dw": dw["median_abs_dw"],
                                   "share_weights_unchanged": dw["share_weights_unchanged"],
                                   "median_abs_w": dw["median_abs_w"], "eps_pytorch": float(opt.param_groups[0]["eps"]),
                                   "sec_train": t1 - t_check, "sec_valid": t2 - t1, "time": now()})
                    log(f"  {out.name} check {len(checks):2d} ({steps:,} steps, {steps / per_pass:.2f} passes): train "
                        f"{train_loss:.6f} valid {vl:.6f}{' *' if improved else ''}  median |dw| {dw['median_abs_dw']:.3e}  "
                        f"({t1 - t_check:.0f} s train, {t2 - t1:.0f} s valid)")
                    del w0
                    loss_sum.zero_()
                    n_seen = 0
                    stop = stopper.stop_reason(len(checks))
                    # Everything a restart needs, saved after the check and before the next step.
                    held = stopper.state_dict()
                    save_checkpoint(ck_path, checkpoint_state(
                        format=2, rng_cpu=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state_all() if cuda else None,
                        ident=ident, model={k: v.detach().cpu() for k, v in net.state_dict().items()},
                        optimizer=opt.state_dict(), keras_t=int(opt.keras_t) if facts["keras_form"] else None,
                        scaler=scaler.state_dict(), best_state=held["best_state"], best=held["best"], wait=held["wait"],
                        stop=stop, checks=checks, steps=steps, pass_start_steps=pass_start, gen_state_pass_start=gen_state,
                        sec_fit=sec_before + time.perf_counter() - t_start, started=wall_start, resumes=resumes,
                        saved=now()))
                    t_check = t_tick = time.perf_counter()
                    tick_steps = steps
                    if stop:
                        break
                first = 0                                    # the next pass starts at its first batch
        except Diverged as exc:
            doc = diverged_doc(base, error=str(exc), checks=checks, steps=steps, resumes=resumes, started=wall_start,
                               finished=now(), added=added)
            write_json_atomic(rec_path, doc)
            ck_path.unlink(missing_ok=True)
            log(f"{split.name} {out.name}: DIVERGED ({exc}); recorded; this fit is never scored on the test days")
            return FitRecord(doc, out)

        # The weights of the best check are restored and saved BEFORE any scoring.
        net.load_state_dict(stopper.best_state)
        torch_save_atomic(w_path, {k: v.detach().cpu() for k, v in net.state_dict().items()})
        w_sha = sha256_file(w_path)
        best_check = min(checks, key=lambda c: c["valid_loss"])["check"]
        sec_fit = sec_before + time.perf_counter() - t_start
        # Validation scores of the saved weights, and their validation loss again: the restore must reproduce the
        # best check's loss.
        rows, per = score_part(net, cfg, split, cache, "valid", device, offs, name=out.name, n_train=n_train,
                               seed=seeds.init, amp=amp, row_cap=tr.eval_row_cap)
        again = float(np.mean(list(per.values())))
        # Enforced: weights that do not give back the best loss are not the best check's; no record is written and the
        # checkpoint is kept, so the test gate refuses the fit.
        if abs(again - stopper.best) > tr.restore_rtol * max(1.0, abs(stopper.best)):
            raise RestoreMismatch(f"the saved weights give validation loss {again!r}, the best check gave {stopper.best!r}")
        facts = optimizer_facts(opt)
        expected = best_check + tr.patience if stop == "patience" else tr.max_checks
        doc = finished_doc(
            base, seed=seeds.init,
            run={"provenance": provenance(), "torch": torch.__version__,
                 "device": torch.cuda.get_device_name(0) if cuda else "cpu"},
            train={"days": spec.days, "dates": sorted({d.date for d in days}), "stock_days": len(days),
                   "stocks": len({d.symbol for d in days}), "samples": n_train, "steps_per_pass": per_pass},
            train_store=train_store, optimizer=facts, n_params=n_params(net), checks=checks, best_check=best_check,
            best=stopper.best, stop=stop, steps=steps, per_pass=per_pass, resumes=resumes, patience=tr.patience,
            max_checks=tr.max_checks, expected=expected, again=again,
            weights={"file": w_path.name, "sha256": w_sha, "saved_before_scoring": True}, rows=rows,
            timing={"sec_fit": sec_fit, "wall_seconds": sec_before + time.perf_counter() - t_start,
                    "started": wall_start, "finished": now()},
            memory={"peak_rss_kb": peak_rss_kb(),
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else 0,
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else 0,
                    "gpu_memory_limit_bytes": cfg.machine.gpu_memory_limit_bytes},
            added=added)
        write_json_atomic(rec_path, doc)        # written last: its existence means the fit has ended and its weights are saved
        ck_path.unlink(missing_ok=True)         # the record holds what the checkpoint held; a finished fit is never continued
        progress(prog_path, {"fit": out.name, "phase": "ended", "steps": steps, "checks": len(checks),
                             "passes": steps / per_pass, "train_loss_since_check": checks[-1]["train_loss"],
                             "ms_per_step": 0.0, "time": now()})
        log(f"{split.name} {out.name}: {valid_line(doc)}")
        return FitRecord(doc, out)
    finally:
        close_days(days)


def valid_line(doc: dict) -> str:
    """Validation numbers only: no test number exists when a fit ends."""
    def mean_k(ks) -> float:
        v = stock_values(doc["valid_scores"], ks)
        return 100 * float(np.mean(list(v.values())))
    return (f"{len(doc['checks'])} checks (best {doc['best_check']}), stopped by {doc['stopped_by']}, {doc['steps']:,} steps = "
            f"{doc['passes']:.2f} passes; {doc['sec_train_per_check']:.0f} s per check train + {doc['sec_valid_per_check']:.0f} "
            f"s valid, wall {doc['wall_seconds']:.0f} s; best validation loss {doc['best_valid_loss']:.6f}, saved weights give "
            f"{doc['restore_check']['valid_loss_of_saved_weights']:.6f}; VALIDATION R2 % (mean over stocks) k=1 {mean_k((1,)):.3f} "
            f"k=5 {mean_k((5,)):.3f} k=10 {mean_k((10,)):.3f} mean k {mean_k(None):.3f}; peak RSS {(doc['peak_rss_kb'] or 0) / 1e6:.2f} "
            f"GB, peak GPU allocated {doc['gpu_peak_allocated_bytes'] / 2**30:.2f} GiB reserved "
            f"{doc['gpu_peak_reserved_bytes'] / 2**30:.2f} GiB")
