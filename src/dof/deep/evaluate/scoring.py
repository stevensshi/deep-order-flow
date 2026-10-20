"""The test days, scored once, at the very end.

- Nothing is scored until EVERY scored model has ended (its record exists, finished or diverged), so the test days
  cannot be read while a fit is still to run.
- Nothing is scored without the corporate-action file; its sha256 and the stock-days it leaves out go into every
  test file. Every other test stock-day of the cache is scored; the price ratio of each is recorded, so a subset by
  ratio needs no second scoring.
- A test-day array is read only with a token of the test gate. The weights scored are the saved files, read once
  and checked against the token's hash.
- A model that has a test file is not scored again; a diverged fit is never scored. No test number is printed.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

from dof.deep.config import CachePaths, RunConfig, Split, protocol_sha256
from dof.deep.data.eval import eval_stocks
from dof.deep.errors import RecordConflict, TestDayRefused
from dof.deep.evaluate.gate import TestGate
from dof.deep.linear.ridge import load_weights, score_part_linear
from dof.deep.runtime.disk import check_free
from dof.deep.runtime.io import now, write_json_atomic
from dof.deep.train.record import FitRecord
from dof.deep.universe import ca_excluded, corporate_actions


def score_test(cfg: RunConfig, split: Split, results_root: Path, *, cache: CachePaths | None = None, workers: int = 6,
               seed: int = 0, log=print) -> dict[str, dict]:
    """Score the test days for the scored models that have ended and have no test file; returns the test files
    written, by model."""
    cache = cache if cache is not None else cfg.machine.cache_for(split.name)
    results_root = Path(results_root)
    protocol = protocol_sha256(cfg)
    records = {}
    for name in cfg.scored_models:
        p = results_root / name / f"fit_seed{seed}.json"
        if not p.exists():
            raise TestDayRefused(f"{name} has not ended ({p.name} missing): the test days are scored at the very end, "
                                 f"after all {len(cfg.scored_models)} scored models")
        records[name] = FitRecord.load(p)
        if records[name].doc.get("protocol_sha256") != protocol:
            raise TestDayRefused(f"{name} was fitted under another protocol document: nothing is scored")
    ca = corporate_actions(cfg)                 # refuses here when the file is missing or is not the pinned one
    skip = ca_excluded(cfg, split, ca)
    # The stock-days the file leaves out, among those the cache holds; and the price ratio of every scored stock-day.
    built = {(s, m["date"]): m for s, ms in eval_stocks(cfg, split, cache, "test") for m in ms}
    left_out = [{"symbol": s, "date": d, **ev} for (s, d), ev in sorted(skip.items()) if (s, d) in built]
    scored = {k: m for k, m in built.items() if k not in skip}
    ratios = {f"{s} {d}": m["ratio_to_group_last_day"] for (s, d), m in sorted(scored.items())}
    ca_doc = {"file": ca["file"], "sha256": ca["sha256"], "rows": len(ca["rows"]),
              "rows_that_apply": sum(r["applies"] for r in ca["rows"]), "left_out": left_out,
              "tickers_not_in_the_test_cache": sorted({s for (s, _d) in skip} - {s for (s, _d) in built})}
    todo = []
    for n in cfg.scored_models:
        t = results_root / n / f"test_seed{seed}.json"
        if records[n].status != "finished":
            log(f"{split.name} {n}: status {records[n].status}; it is not scored on the test days")
        elif t.exists():
            if json.loads(t.read_text(encoding="utf-8"))["corporate_actions"]["sha256"] != ca["sha256"]:
                raise RecordConflict(f"{t} was scored with another corporate-action file; a model is not scored twice")
            log(f"{split.name} {n}: test file exists; it is not scored again")
        else:
            todo.append(n)
    check_free(cfg.machine.disk_guard_path, cfg.machine.disk_floor_gb, "before the test scoring", log)
    log(f"=== {split.name} test scoring {now()}: {todo}; protocol document sha256 {protocol}; corporate-action file "
        f"{ca['file']} sha256 {ca['sha256']}: {len(ca['rows'])} rows, {ca_doc['rows_that_apply']} confirmed events of an "
        f"excluding type; test stock-days in the cache {len(built)}, left out by a corporate action {len(left_out)}, "
        f"scored {len(scored)}")
    for x in left_out:
        log(f"  left out: {x['symbol']} {x['date']}: {x['event']} effective {x['effective_date']}")
    out: dict[str, dict] = {}

    def done(name: str, rows: list[dict], t0: float) -> None:
        rec = records[name]
        got = {(r["symbol"], r["date"]) for r in rows}
        if got != set(scored):
            raise AssertionError(f"{name}: {len(got)} stock-days were scored, {len(scored)} were to be scored")
        doc = {"model": name, "stage": split.name, "seed": seed, "protocol_sha256": protocol,
               "weights_sha256": rec.weights_sha256, "fit_finished": rec.doc["finished"],
               "test_days": list(split.test_days), "corporate_actions": ca_doc, "price_ratio": ratios,
               "ratio_range": [cfg.features.ratio_lo, cfg.features.ratio_hi], "scores": rows,
               "seconds": time.perf_counter() - t0, "scored": now(), "identity": rec.doc.get("identity"),
               "identity_sha256": rec.doc.get("identity_sha256")}
        write_json_atomic(results_root / name / f"test_seed{seed}.json", doc)
        out[name] = doc
        log(f"{split.name} {name}: {len(rows)} test cells of {len(got)} stock-days written (the numbers stay in the file)")

    nets = [n for n in todo if n in cfg.fits]
    if nets:
        import torch

        from dof.deep.evaluate.parts import score_part
        from dof.deep.models import build
        from dof.deep.runtime.device import setup_device
        device = setup_device(cfg.machine, log)
        offs = torch.arange(-cfg.features.history + 1, 1, device=device)
        gate = TestGate(cfg, split)
        for name in nets:
            t0 = time.perf_counter()
            rec = records[name]
            token = gate.token(rec)             # refuses unless the fit has ended and its saved weights are intact
            data = rec.weights_path.read_bytes()
            if hashlib.sha256(data).hexdigest() != token.weights_sha256 or token.seed != seed:
                raise TestDayRefused(f"{name}: the saved weights changed after the test gate checked them")
            model = build(cfg.models[cfg.fits[name].arch]).to(device)
            model.load_state_dict(torch.load(io.BytesIO(data), map_location=device, weights_only=True))
            rows, _ = score_part(model, cfg, split, cache, "test", device, offs, name=name,
                                 n_train=rec.doc["train"]["samples"], seed=seed, amp=False, gate=gate,
                                 tokens={name: token}, skip=skip)
            done(name, rows, t0)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    lins = [n for n in todo if n in cfg.linear]
    if lins:
        t0 = time.perf_counter()
        gate = TestGate(cfg, split)
        for name in lins:
            gate.token(records[name])           # checked here first; each worker process asks again before it reads
        fits = {name: load_weights(results_root, name, seed) for name in lins}
        rows = score_part_linear(cfg, split, cache, "test", fits, {n: records[n].doc["train"]["samples"] for n in lins},
                                 workers, results_root=results_root, seed=seed, log=log, skip=set(skip))
        for name in lins:
            done(name, [r for r in rows if r["model"] == name], t0)
    log(f"=== {split.name} test scoring end {now()}")
    return out
