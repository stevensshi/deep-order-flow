"""The order of work: no held-out record is read before every fit of its round has ended.

``RoundGate`` stops ``run`` before the test scoring of a round until every fit of that round (and, for the first
round, every linear reference) has its record. ``audit_order`` checks the same rule afterwards on the times the stored
records carry (``finished`` of a fit record, ``scored`` of a score file); ISO times of one form compare in time order
as strings.
"""

from __future__ import annotations

from pathlib import Path

from dof.deep.errors import TestDayRefused
from dof.experiments.errors import ExportError
from dof.experiments.spec import ExperimentSpec


class RoundGate:
    """The gate of a run folder ``out`` (fit records at ``out/seed<s>/<fit>/fit_seed<s>.json``). It reads no
    held-out array and no record content: a record exists only when its fit has ended (finished or diverged)."""

    def __init__(self, spec: ExperimentSpec, out: Path) -> None:
        self.spec, self.out = spec, Path(out)

    def check(self, round_index: int, rounds) -> None:
        need = [(fit, seed) for fit, seed in rounds[round_index]]
        if round_index == 0:
            need += [(name, 0) for name in self.spec.linear]
        missing = [f"{fit} seed {seed}" for fit, seed in need
                   if not (self.out / f"seed{seed}" / fit / f"fit_seed{seed}.json").is_file()]
        if missing:
            raise TestDayRefused(f"{self.spec.name}: round {round_index + 1}: no test day is read before every fit of the "
                                 f"round has ended; not ended: {', '.join(missing)}")


def held_out(spec: ExperimentSpec, record: str, part: str | None) -> bool:
    """Whether a record kind of a part is read only after the fits: the parts that list the kind decide; when more
    than one does (a seed ensemble has a validation and a test file), the part of the record's own name decides."""
    parts = [p for p in spec.parts if record in p.records]
    if len(parts) > 1:
        parts = [p for p in parts if p.name == part]
    return any(p.held_out for p in parts)


def audit_order(spec: ExperimentSpec, times) -> dict:
    """``times``: (record, model, seed, "finished"|"scored", time or None, part) of every exported record of the spec.
    Raises ExportError when a held-out record has no time or was scored before the last fit of its round ended."""
    rounds = {(f, s): i for i, rnd in enumerate(spec.rounds) for f, s in rnd}
    fits = [t for t in times if t[0] == "fit_valid"]
    if not fits:
        return {"rounds": len(spec.rounds), "held_out_after_fits": None, "checked": 0, "unchecked": 0}
    limit: dict[int, str] = {}
    for _rec, model, seed, _kind, when, _part in fits:
        if model in spec.linear:
            i = 0
        elif (model, seed) in rounds:
            i = rounds[(model, seed)]
        else:
            continue
        if when is not None and (i not in limit or when > limit[i]):
            limit[i] = when
    checked, late, missing = 0, [], []
    for rec, model, seed, kind, when, part in times:
        if kind != "scored" or not held_out(spec, rec, part):
            continue
        name = f"{rec} {model} seed {seed} ({part})"
        if when is None:
            missing.append(name)
            continue
        i = rounds.get((model, seed), 0)
        checked += 1
        if i in limit and when < limit[i]:
            late.append(f"{name} scored {when}, before {limit[i]}")
    if missing:
        raise ExportError(f"{spec.name}: held-out records without a scored time: {missing}")
    if late:
        raise ExportError(f"{spec.name}: held-out records scored before every fit of their round had ended: {late}")
    return {"rounds": len(spec.rounds), "held_out_after_fits": True, "checked": checked, "unchecked": 0}
