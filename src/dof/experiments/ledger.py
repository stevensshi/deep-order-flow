"""The ledgers of a run folder: what the run is (``experiment.json``), what happened (``events.jsonl``) and the
decision of the seed rule (``decision.json``).

``begin`` refuses a folder that holds another experiment or another protocol document. ``decide`` takes the
decision once: a later call reads it and refuses when the weights it was taken from have changed, so the seeds a
rule drew are never drawn again.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path

import dof
from dof.deep.errors import RecordConflict
from dof.deep.runtime.io import now, write_json_atomic
from dof.experiments.spec import ExperimentSpec, spec_identity_sha256


def begin(out: Path, spec: ExperimentSpec, protocol: str | None) -> dict:
    out = Path(out)
    doc = {"format": 1, "experiment": spec.name, "spec_identity_sha256": spec_identity_sha256(spec),
           "protocol_sha256": protocol, "package_version": dof.__version__}
    path = out / "experiment.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        for key in ("experiment", "spec_identity_sha256", "protocol_sha256"):
            if old.get(key) != doc[key]:
                raise RecordConflict(f"{path.name}: the folder holds {key} {old.get(key)}, this run has {doc[key]}; "
                                     "use another folder")
        return old
    write_json_atomic(path, doc, indent=1)
    return doc


def event(out: Path, kind: str, **fields) -> None:
    with open(Path(out) / "events.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"time": now(), "event": kind, **fields}, sort_keys=True) + "\n")


def draw_seeds(n: int, taken: set[int]) -> list[int]:
    """``secrets.randbits(32)``; a value is drawn again only when it is 0 or a seed already used or drawn."""
    out: list[int] = []
    while len(out) < n:
        v = secrets.randbits(32)
        if v != 0 and v not in taken and v not in out:
            out.append(v)
    return out


def _inputs(spec: ExperimentSpec, tables) -> dict[str, str | None]:
    """The weights sha256 of every fit the rule reads: the older and recent fits of the first seeds."""
    pol, out = spec.seed_policy, {}
    for s in pol.first:
        for fit in (pol.older, pol.recent):
            rid = tables.rid(experiment=spec.experiment_of(s), record="fit_valid", model=fit, seed=s)
            out[f"{fit}/seed={s}"] = None if rid is None else tables.record(rid)["weights_sha256"]
    return out


def decide(out: Path, spec: ExperimentSpec, tables, draw=None) -> dict:
    """The rule for more seeds on validation numbers only, taken once and kept in ``decision.json``."""
    from dof.experiments.analyses.seeds import rule_values
    path = Path(out) / "decision.json"
    inputs = _inputs(spec, tables)
    if path.exists():
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc["inputs"] != inputs:
            raise RecordConflict(f"{path.name} was taken from other weights than the fits here now; it is not followed")
        return doc
    pol = spec.seed_policy
    v = rule_values(tables, spec)
    drawn = (draw or draw_seeds)(pol.more, set(spec.seeds)) if v["more"] or pol.force_more else []
    doc = {**v, "differences": {str(s): d for s, d in v["differences"].items()},
           "stocks": {str(s): n for s, n in v["stocks"].items()}, "first": list(pol.first), "inputs": inputs,
           "drawn": [int(s) for s in drawn]}
    write_json_atomic(path, doc, indent=1)
    return doc
