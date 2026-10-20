"""Command line of the training package.

  python -m dof.deep build   --config PATH [--local PATH] [--parts train valid test] [--workers N] [--mem-gb G]
  python -m dof.deep summary --config PATH [--local PATH]
  python -m dof.deep fit     --config PATH [--local PATH] --fit NAME --seed S [--data-seed D] --out DIR [--placebo]
  python -m dof.deep ridge   --config PATH [--local PATH] --out DIR [--workers N] [--seed S]
  python -m dof.deep score   --config PATH [--local PATH] --results DIR [--workers N] [--seed S]
  python -m dof.deep measure --config PATH [--local PATH] --fit NAME --out-file FILE [--steps N] [--valid-stocks N]
                             [--valid-symbols SYM ...] [--loader-workers N] [--seed S] [--note TEXT]

The machine settings (book root, cache folders, device, loader processes) come from the local file (``--local``,
else ``DOF_LOCAL_CONFIG``); the cache of a split is ``machine.cache.<split name>``. The last line of build, fit,
ridge and score is the verdict: ``BUILD_DONE``, ``FIT_DONE <status>``, ``RIDGE_DONE``, ``SCORE_DONE``, or
``<CMD>_FAILED: <error>``. Exit status: 0 done; 1 a refusal or a failed check; 2 a bad argument.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from dof.deep.config import Seeds, load
from dof.deep.errors import CacheError, ConfigError

FAILURES = (ConfigError, AssertionError, RuntimeError)     # every exception of dof.deep.errors is one of these


def _cfg(args):
    return load(args.config, local=args.local)


def cmd_build(args) -> str:
    from dof.deep.cache import build
    from dof.deep.cache.summary import print_summary
    cfg = _cfg(args)
    doc = build(cfg, cfg.split, cfg.machine.cache_for(cfg.split.name), parts=tuple(args.parts), workers=args.workers,
                mem_gb=args.mem_gb)
    print_summary(doc)
    return "BUILD_DONE"


def cmd_summary(args) -> str | None:
    from dof.deep.cache.layout import summary_path
    from dof.deep.cache.summary import print_summary
    cfg = _cfg(args)
    p = summary_path(cfg.machine.cache_for(cfg.split.name))
    if not p.exists():
        raise CacheError(f"{p} missing: the build has not finished")
    print_summary(json.loads(p.read_text(encoding="utf-8")))
    return None


def cmd_fit(args) -> str:
    from dof.deep.train.fit import fit
    cfg = _cfg(args)
    seeds = Seeds(args.seed, args.seed if args.data_seed is None else args.data_seed)
    rec = fit(cfg, cfg.split, args.fit, seeds, args.out, placebo=args.placebo)
    return f"FIT_DONE {rec.status}"


def cmd_ridge(args) -> str:
    from dof.deep.linear.ridge import fit_ridge
    cfg = _cfg(args)
    fit_ridge(cfg, cfg.split, args.out, workers=args.workers, seed=args.seed)
    return "RIDGE_DONE"


def cmd_score(args) -> str:
    from dof.deep.evaluate.scoring import score_test
    cfg = _cfg(args)
    score_test(cfg, cfg.split, args.results, workers=args.workers, seed=args.seed)
    return "SCORE_DONE"


def cmd_measure(args) -> None:
    from dof.deep.train.measure import measure
    cfg = _cfg(args)
    measure(cfg, cfg.split, args.fit, steps=args.steps, valid_stocks=args.valid_stocks, valid_symbols=args.valid_symbols,
            loader_workers=args.loader_workers, seed=args.seed, note=args.note, out_file=args.out_file)
    return None


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m dof.deep", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {name: sub.add_parser(name) for name in ("build", "summary", "fit", "ridge", "score", "measure")}
    for p in cmds.values():
        p.add_argument("--config", required=True, type=Path)
        p.add_argument("--local", type=Path, default=None)
    b = cmds["build"]
    b.add_argument("--parts", nargs="+", choices=("train", "valid", "test"), default=["train", "valid", "test"])
    b.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 4), help="default: cores minus 4")
    b.add_argument("--mem-gb", type=float, default=12.0, help="sum of the memory estimates of the running tasks")
    f = cmds["fit"]
    f.add_argument("--fit", required=True)
    f.add_argument("--seed", type=int, required=True)
    f.add_argument("--data-seed", type=int, default=None, help="seed of the batch order (default: --seed)")
    f.add_argument("--out", required=True, type=Path, help="the fit's folder")
    f.add_argument("--placebo", action="store_true", help="shuffled training labels (a leak check)")
    for name, out in (("ridge", "--out"), ("score", "--results")):
        p = cmds[name]
        p.add_argument(out, required=True, type=Path)
        p.add_argument("--workers", type=int, default=6)
        p.add_argument("--seed", type=int, default=0)
    m = cmds["measure"]
    m.add_argument("--fit", required=True)
    m.add_argument("--steps", type=int, default=1500)
    m.add_argument("--valid-stocks", type=int, default=6)
    m.add_argument("--valid-symbols", nargs="+", default=None)
    m.add_argument("--loader-workers", type=int, default=None)
    m.add_argument("--seed", type=int, default=None, help="seed of the measured batch order (default: the clock)")
    m.add_argument("--note", default="")
    m.add_argument("--out-file", required=True, type=Path)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run = {"build": cmd_build, "summary": cmd_summary, "fit": cmd_fit, "ridge": cmd_ridge, "score": cmd_score,
           "measure": cmd_measure}[args.cmd]
    try:
        verdict = run(args)
    except FAILURES as exc:
        print(f"{args.cmd} stopped: {exc}", flush=True)
        print(f"{args.cmd.upper()}_FAILED: {type(exc).__name__}: {exc}", flush=True)
        return 1
    if verdict:
        print(verdict, flush=True)
    return 0
