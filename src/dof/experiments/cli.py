"""Command line of the experiment framework.

  python -m dof.experiments export --results-root DIR --out DIR [--spec NAME ...] [--root NAME=DIR ...]
  python -m dof.experiments totals --results DIR [--expect FILE]
  python -m dof.experiments report --results DIR (--spec NAME ... | --all) --out-dir DIR
  python -m dof.experiments run    --spec NAME --config FILE --out DIR [--local FILE] [--input NAME=DIR ...]
                                   [--workers N]

The last line is the verdict: ``EXPORT_DONE``, ``TOTALS <n> keys``, ``TOTALS EQUAL <n> keys``, ``REPORT_DONE``,
``RUN_DONE``, or ``<CMD>_FAILED`` (``TOTALS DIFFER`` lines before it). ``run`` takes the run configuration from
``--config`` and the machine settings from ``--local`` (else ``DOF_LOCAL_CONFIG``); ``--input NAME=DIR`` names the
run folder of an experiment whose seed 0 the spec reuses; ``--root NAME=DIR`` gives the results root of one spec
whose records are not under ``--results-root``. ``report`` writes ``<out-dir>/<name>.md`` for each spec and
replaces an existing file: a report is a function of the tables and the spec. Exit status: 0 done; 1 a refusal or
a failed check; 2 a bad argument or an output folder that exists.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from dof.deep import errors as deep_errors
from dof.experiments import errors

FAILURES = (errors.SpecError, errors.ExportError, errors.AnalysisError, errors.ReportError, deep_errors.ConfigError,
            AssertionError, RuntimeError)
ORDER = ("alldays", "seeds", "adjacent", "decay", "earlier")


def _specs(names) -> list:
    from dof.experiments.specs import SPECS
    return [SPECS[n] for n in (names or ORDER)]


def cmd_export(args) -> int:
    from dof.experiments.export import export
    out = Path(args.out)
    for p in (out, out.with_name(out.name + ".partial")):
        if p.exists():
            print(f"{p.name} exists: an export is written once; remove it first")
            print("EXPORT_FAILED")
            return 2
    roots = {}
    for item in args.root or []:
        name, sep, folder = item.partition("=")
        if not sep or not name or not folder:
            print(f"--root {item!r}: expected NAME=DIR")
            print("EXPORT_FAILED")
            return 2
        roots[name] = Path(folder)
    export(Path(args.results_root), out, _specs(args.spec), roots=roots)
    return 0


def cmd_totals(args) -> int:
    from dof.experiments.cells import read_tables, totals
    ours = totals(read_tables(Path(args.results)))
    print(json.dumps(ours, indent=1, sort_keys=True))
    if args.expect is None:
        print(f"TOTALS {len(ours)} keys")
        return 0
    theirs = json.loads(Path(args.expect).read_text(encoding="utf-8"))["totals"]
    differ = [k for k in sorted(set(ours) | set(theirs)) if ours.get(k) != theirs.get(k)]
    for k in differ:
        print(f"TOTALS DIFFER {k}: {ours.get(k)} != {theirs.get(k)}")
    if differ:
        return 1
    print(f"TOTALS EQUAL {len(ours)} keys")
    return 0


def cmd_report(args) -> int:
    from dof.experiments.cells import read_tables
    from dof.experiments.render import render
    rt = read_tables(Path(args.results))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for spec in _specs(None if args.all else args.spec):
        path = out / f"{spec.name}.md"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(render(rt, spec), encoding="utf-8")
        os.replace(tmp, path)
        print(f"REPORT {spec.name}")
    print("REPORT_DONE")
    return 0


def cmd_run(args) -> int:
    from dof.deep.config import load
    from dof.experiments.runner import run
    inputs = {}
    for item in args.input or []:
        name, sep, folder = item.partition("=")
        if not sep or not name or not folder:
            print(f"--input {item!r}: expected NAME=DIR")
            print("RUN_FAILED")
            return 2
        inputs[name] = Path(folder)
    spec = _specs([args.spec])[0]
    cfg = load(args.config, local=args.local)
    run(spec, out=Path(args.out), cfg=cfg, inputs=inputs or None, workers=args.workers)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    from dof.experiments.specs import SPECS
    ap = argparse.ArgumentParser(prog="python -m dof.experiments", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export", help="stored records into the result tables")
    p.add_argument("--results-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--spec", action="append", choices=sorted(SPECS))
    p.add_argument("--root", action="append", metavar="NAME=DIR")
    p = sub.add_parser("totals", help="cell counts and exact sums per experiment and record kind")
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--expect", type=Path)
    p = sub.add_parser("report", help="the Markdown reports of the specs, from the result tables")
    p.add_argument("--results", required=True, type=Path)
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--spec", action="append", choices=sorted(SPECS))
    which.add_argument("--all", action="store_true")
    p.add_argument("--out-dir", required=True, type=Path)
    p = sub.add_parser("run", help="fit, gate, score once and report a spec whose steps this package runs")
    p.add_argument("--spec", required=True, choices=sorted(SPECS))
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--local", type=Path)
    p.add_argument("--input", action="append", metavar="NAME=DIR")
    p.add_argument("--workers", type=int, default=0)
    args = ap.parse_args(argv)
    fn = {"export": cmd_export, "totals": cmd_totals, "report": cmd_report, "run": cmd_run}[args.cmd]
    try:
        return fn(args)
    except FAILURES as exc:
        print(f"{type(exc).__name__}: {exc}")
        print(f"{args.cmd.upper()}_FAILED")
        return 1
