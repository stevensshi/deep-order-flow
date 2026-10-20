"""Stored records of several experiments into one folder of result tables.

``export`` refuses when the output folder (or its ``.partial`` sibling) exists, when the root of a spec is not a
folder, and when a spec yields no record (a wrong root). A spec's records are under ``results_root / spec.root``, or under ``roots[spec.name] / spec.root`` when
``roots`` names the spec (the earlier stage's records sit in another lab folder). Every record is checked against its
spec (protocol hashes), record ids must be unique across the specs, and the order of work is audited; only then is
the ``.partial`` folder written and renamed. No path of a results root is printed or written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from dof.experiments.cells import collect, env_versions, write_tables
from dof.experiments.errors import ExportError
from dof.experiments.gate import audit_order
from dof.experiments.spec import ExperimentSpec, spec_identity_sha256


def export(results_root: Path, out: Path, specs: Sequence[ExperimentSpec], *, roots: Mapping[str, Path] | None = None,
           log=print) -> dict:
    """The result tables of ``specs`` (each under ``results_root / spec.root``, or ``roots[name] / spec.root``)
    written to ``out``; returns the manifest."""
    results_root, out, roots = Path(results_root), Path(out), dict(roots or {})
    unknown = sorted(set(roots) - {s.name for s in specs})
    if unknown:
        raise ExportError(f"roots name specs that are not exported: {unknown}")
    partial = out.with_name(out.name + ".partial")
    for p in (out, partial):
        if p.exists():
            raise ExportError(f"{p.name} exists: an export is written once; remove it first")
    collected = []
    for spec in specs:
        spec.check()
        root = Path(roots.get(spec.name, results_root)) / spec.root
        if not root.is_dir():
            raise ExportError(f"{spec.name}: its root {spec.root!r} is not a folder under the given results root")
        c = collect(root, spec)
        if not c.records:
            raise ExportError(f"{spec.name}: no record under its root {spec.root!r}")
        log(f"EXPORT {spec.name}: {len(c.records)} records, {len(c.cells)} cells")
        for reason, n in sorted(c.skipped.items()):
            log(f"SKIPPED {spec.name} {reason} {n}")
        collected.append(c)
    seen: dict[str, str] = {}
    for c in collected:
        for r in c.records:
            if r["record_id"] in seen:
                raise ExportError(f"record id {r['record_id']} is given twice (experiments {seen[r['record_id']]} and "
                                  f"{c.experiment})")
            seen[r["record_id"]] = c.experiment
    audit = {}
    for spec, c in zip(specs, collected):
        audit[spec.name] = a = audit_order(spec, c.times)
        log(f"AUDIT {spec.name} held_out_after_fits={a['held_out_after_fits']} rounds={a['rounds']} "
            f"checked={a['checked']} unchecked={a['unchecked']}")
    extra = {"audit": audit, "specs": {s.name: spec_identity_sha256(s) for s in specs}, "env": env_versions()}
    manifest = write_tables(collected, partial, extra)
    partial.rename(out)
    log("EXPORT_DONE")
    return manifest
