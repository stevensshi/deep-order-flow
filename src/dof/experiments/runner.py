"""``run(spec)``: fit, gate, score once, decide (seed rule), collect and report, for the specs whose steps the training
package provides (the main-run kind and the seeds kind). A spec whose data steps this package does not run (the
adjacent-days cut, the decay bases) is refused before anything is written.

Layout of a run folder ``out``: ``seed<s>/<fit>/fit_seed<s>.json`` and ``test_seed<s>.json`` (``RUN_LAYOUT``), the
ledgers ``experiment.json``, ``events.jsonl``, ``decision.json``, the result tables in ``tables/`` and ``report.md``.
Every step reuses what exists: a fit record of the same identity is kept, a test file is never scored twice.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from dof.experiments import ledger
from dof.experiments.cells import Collected, collect, env_versions, read_tables, write_tables
from dof.experiments.errors import ExportError, SpecError
from dof.experiments.gate import RoundGate, audit_order
from dof.experiments.spec import ExperimentSpec, RecordSource, pair_round, spec_identity_sha256

RUN_LAYOUT: tuple[RecordSource, ...] = (
    RecordSource("fit_valid", "seed{seed}/{model}/fit_seed{seed}.json", ("main",), "finished", None),
    RecordSource("test", "seed{seed}/{model}/test_seed{seed}.json", ("main", "corporate_actions"), "needs_finished_fit",
                 None),
)
TABLE_DIRS = ("tables/", "tables.partial/")


@dataclass(frozen=True)
class RunResult:
    out: Path
    fitted: dict[tuple[str, int], str]        # (fit or linear model, seed) -> status
    scored: tuple[tuple[str, int], ...]       # (model, seed) scored in this call
    decision: dict | None
    tables: Path                              # out/tables
    report: Path                              # out/report.md


def _only(c: Collected, keep) -> Collected:
    """The rows of the records whose id passes ``keep``."""
    out = Collected(c.experiment, skipped=dict(c.skipped))
    for name in ("cells", "stock_days", "records", "fits", "checks", "check_losses"):
        setattr(out, name, [r for r in getattr(c, name) if keep(r["record_id"])])
    out.times = [t for t in c.times if t[2] == 0]
    return out


def _tables(spec: ExperimentSpec, ident: str, out: Path, inputs: Mapping[str, Path] | None) -> Path:
    """Collect the run folder (and seed 0 of ``seed0_from``) into ``out/tables``; the folder of an earlier call of the
    same spec is replaced."""
    own = collect(out, replace(spec, root=".", exclude=TABLE_DIRS), RUN_LAYOUT)
    parts = [own]
    if spec.seed0_from is not None:
        if not inputs or spec.seed0_from not in inputs:
            raise SpecError(f"{spec.name}: seed 0 comes from {spec.seed0_from}: give its run folder as an input")
        other = collect(Path(inputs[spec.seed0_from]), replace(spec, name=spec.seed0_from, root=".", exclude=TABLE_DIRS),
                        RUN_LAYOUT)
        fits = set(spec.fits)
        ids = {r["record_id"] for r in other.records if r["seed"] == 0 and r["model"] in fits}
        parts.append(_only(other, ids.__contains__))
    audit = {spec.name: audit_order(spec, own.times)}
    partial, final = out / "tables.partial", out / "tables"
    write_tables(parts, partial, {"audit": audit, "specs": {spec.name: ident}, "env": env_versions()})
    if final.exists():
        manifest = final / "MANIFEST.json"
        old = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
        if old.get("specs", {}).get(spec.name) != ident:
            raise ExportError(f"{final.name} exists and was not written by this spec; move it away")
        shutil.rmtree(final)                    # only out/tables, a folder this function wrote for the same spec
    os.replace(partial, final)
    return final


def run(spec: ExperimentSpec, *, out: Path, cfg, inputs: Mapping[str, Path] | None = None, draw=None,
        workers: int = 0, log=print) -> RunResult:
    """Run ``spec`` with the run configuration ``cfg`` (``dof.deep.config.RunConfig``) into the folder ``out``."""
    spec.check()
    missing = [s for s in ("fit", "score") if s not in spec.steps]
    if missing:
        raise SpecError(f"{spec.name}: this package does not run its {missing[0]} step; its stored results are reported "
                        "with 'report'")
    unknown = [n for n in spec.fits if n not in cfg.fits] + [n for n in spec.linear if n not in cfg.linear]
    if unknown:
        raise SpecError(f"{spec.name}: {unknown} are not fits or linear references of the configuration")
    from dof.deep.config import Seeds, protocol_sha256
    from dof.deep.evaluate.scoring import score_test
    from dof.deep.linear.ridge import fit_ridge
    from dof.deep.train.fit import fit
    out = Path(out)
    cfg0 = replace(cfg, linear={n: cfg.linear[n] for n in spec.linear}, train=replace(cfg.train, fit_order=spec.fits))
    split = cfg0.split
    out.mkdir(parents=True, exist_ok=True)
    ledger.begin(out, spec, protocol_sha256(cfg0))
    ident = spec_identity_sha256(spec)
    rounds = list(spec.rounds)
    fitted: dict[tuple[str, int], str] = {}
    scored: list[tuple[str, int]] = []
    decision = None
    if spec.linear:
        for name, rec in fit_ridge(cfg0, split, out / "seed0", workers=workers, log=log).items():
            fitted[(name, 0)] = rec.status
            ledger.event(out, "fit_ended", fit=name, seed=0, status=rec.status)
    i = 0
    while i < len(rounds):
        for f, s in rounds[i]:
            rec = fit(cfg0, split, f, Seeds.same(s), out / f"seed{s}" / f, log=log)
            fitted[(f, s)] = rec.status
            ledger.event(out, "fit_ended", fit=f, seed=s, status=rec.status)
        RoundGate(spec, out).check(i, rounds)
        for s in dict.fromkeys(seed for _f, seed in rounds[i]):
            cfg_s = replace(cfg0, linear=cfg0.linear if s == 0 else {},
                            train=replace(cfg0.train, fit_order=tuple(f for f, seed in rounds[i] if seed == s)))
            for name in score_test(cfg_s, split, out / f"seed{s}", workers=workers, seed=s, log=log):
                scored.append((name, s))
                ledger.event(out, "scored", model=name, seed=s)
        if i == 0 and spec.seed_policy is not None and "decide" in spec.steps:
            decision = ledger.decide(out, spec, read_tables(_tables(spec, ident, out, inputs)), draw)
            drawn = tuple(decision["drawn"])
            for a, b in zip(drawn[::2], drawn[1::2]):
                rounds.append(pair_round(spec.fits, a, b))
            spec = replace(spec, seeds=spec.seeds + tuple(s for s in drawn if s not in spec.seeds), rounds=tuple(rounds),
                           seed_policy=replace(spec.seed_policy, drawn=drawn))
        i += 1
    tables = _tables(spec, ident, out, inputs)
    from dof.experiments.render import render
    text = render(read_tables(tables), spec)
    tmp = out / "report.md.tmp"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, out / "report.md")
    log("RUN_DONE")
    return RunResult(out, fitted, tuple(scored), decision, tables, out / "report.md")
