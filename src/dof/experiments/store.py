"""Where an experiment's stored records are, and which of them count.

``discover`` walks a result root and matches each JSON file against the spec's record sources; it reads no file
content. A file matches the first source whose pattern fits it; a file that fits sources with ``part`` set matches
every one of them (one stored file, one record per part). Run-state files, ledgers and excluded folders are never matched. The record rules (``rule_skip``) are applied
when ``cells.collect`` reads a found file: a fit that has not finished, a seed fit whose saved weights are missing or
whose checkpoint is still there, a score file whose fit has not finished, and an ensemble that was not formed are
skipped with a counted reason.
"""

from __future__ import annotations

import fnmatch
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from dof.experiments.errors import SpecError
from dof.experiments.spec import ExperimentSpec, RecordSource

PLACEHOLDERS = {"model": r"[^/]+", "seed": r"\d+", "basis": r"[^/]+", "date": r"\d{4}-\d{2}-\d{2}",
                "symbol": r"[^/]+", "part": r"valid|test"}
RUN_STATE = ("progress.json", "current.json", "begun_seed*.json", "*.tmp", "tmp_*")
LEDGERS = ("measure.json", "extension.json", "experiment.json", "gate_refusals.json", "decision.json")
_FIELD = re.compile(r"\{([^{}]*)\}")


def compile_pattern(pattern: str) -> re.Pattern:
    """The regular expression of a record template: literal parts escaped, the first use of a placeholder a named
    group, a later use of the same placeholder a back-reference; anchored at both ends."""
    out, seen, pos = ["^"], set(), 0
    for m in _FIELD.finditer(pattern):
        name = m.group(1)
        if name not in PLACEHOLDERS:
            raise SpecError(f"pattern {pattern!r}: {{{name}}} is not one of {tuple(PLACEHOLDERS)}")
        out.append(re.escape(pattern[pos:m.start()]))
        out.append(f"(?P={name})" if name in seen else f"(?P<{name}>{PLACEHOLDERS[name]})")
        seen.add(name)
        pos = m.end()
    out += [re.escape(pattern[pos:]), "$"]
    return re.compile("".join(out))


@dataclass(frozen=True)
class Found:
    rel: str                                 # path relative to the experiment root (never written out)
    source: RecordSource
    fields: dict[str, str] = field(hash=False)

    @property
    def seed(self) -> int | None:
        return int(self.fields["seed"]) if "seed" in self.fields else self.source.seed


def _run_state(name: str) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in RUN_STATE)


def discover(root: Path, spec: ExperimentSpec, sources: tuple[RecordSource, ...] | None = None
             ) -> tuple[list[Found], dict[str, int]]:
    """The record files under ``root`` in sorted relative-path order, each with the first source it matches, and the
    count of the files skipped per reason."""
    root = Path(root)
    compiled = [(src, compile_pattern(src.pattern)) for src in (spec.sources if sources is None else sources)]
    found: list[Found] = []
    skipped: Counter = Counter()
    if not root.is_dir():
        return found, {}
    for rel in sorted(p.relative_to(root).as_posix() for p in root.rglob("*.json")):
        name = rel.rsplit("/", 1)[-1]
        if any(rel.startswith(x) for x in spec.exclude):
            skipped["excluded folder"] += 1
        elif _run_state(name):
            skipped["run state"] += 1
        elif name in LEDGERS:
            skipped["ledger"] += 1
        else:
            hits = 0
            for src, rx in compiled:
                if hits and src.part is None:
                    continue
                m = rx.match(rel)
                if m:
                    found.append(Found(rel, src, m.groupdict()))
                    hits += 1
                    if src.part is None:
                        break
            if not hits:
                skipped["unknown kind"] += 1
    return found, dict(skipped)


def _finished(path: Path) -> bool:
    try:
        return json.loads(path.read_bytes()).get("status") == "finished"
    except (OSError, ValueError):
        return False


def rule_skip(root: Path, found: Found, doc: dict) -> tuple[str | None, str | None]:
    """(the record kind to export, None) or (None, the reason it is skipped). A decay file becomes record "decay"
    or "left_out" by its status."""
    src, record = found.source, found.source.record
    folder = (Path(root) / found.rel).parent
    status = doc.get("status")
    if record == "decay":
        kind = {"scored": "decay", "left_out": "left_out"}.get(status)
        return (kind, None) if kind else (None, f"unknown kind (status {status!r})")
    if record == "ensemble":
        if status == "scored":
            return "ensemble", None
        return None, "ensemble not formed" if status == "not formed" else f"unknown kind (status {status!r})"
    if src.fit_rule in ("finished", "finished_saved"):
        if status != "finished":
            return None, f"fit not finished (status {status!r})"
        if src.fit_rule == "finished_saved":
            seed = found.seed
            if not (folder / f"best_weights_seed{seed}.pt").exists() or (folder / f"checkpoint_seed{seed}.pt").exists():
                return None, "fit not finished (weights not saved or a checkpoint is left)"
    elif src.fit_rule == "needs_finished_fit":
        if not _finished(folder / f"fit_seed{found.seed}.json"):
            return None, "its fit has not finished"
    elif src.fit_rule != "none":
        raise SpecError(f"record source {src.pattern!r}: unknown fit_rule {src.fit_rule!r}")
    return record, None


def source_rows(src: RecordSource, record: str, doc: dict) -> list[dict]:
    """The score rows a record takes from its stored file: with ``src.part``, the rows of ``scores`` of that part whose
    fields equal ``src.select`` (a file that keeps only R² gives rows without sse and sst); else the validation rows of a
    fit, none for a left-out unit, and ``scores`` for every other kind."""
    if src.part is not None:
        return [r for r in doc.get("scores", []) if r.get("part") == src.part and all(r.get(k) == v for k, v in src.select)]
    return doc.get("valid_scores", []) if record == "fit_valid" else [] if record == "left_out" else doc.get("scores", [])


def record_part(src: RecordSource, record: str, fields: dict) -> str:
    """The part a record belongs to: the source's ``part``, else the part of its record kind."""
    return src.part or {"fit_valid": "valid", "decay": "decay", "left_out": "decay",
                        "ensemble": fields.get("part")}.get(record, "test")


def sum_hex(values) -> str | None:
    """The exact sum (``math.fsum``) of the sse or sst values of some cells, as float hex. A missing value (None) is
    left out; a NaN stays in the sum and shows. None when there are cells and none of them carries a value (the
    earlier stage's ridge rows); "0x0.0p+0" when there is no cell."""
    values = list(values)
    present = [v for v in values if v is not None]
    return None if values and not present else float(math.fsum(present)).hex()
