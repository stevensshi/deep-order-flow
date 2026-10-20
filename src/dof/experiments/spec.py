"""An experiment as data: ``ExperimentSpec`` and its parts.

A spec names the fits, seeds and rounds of an experiment, its parts (the held-out ones by day and time of day),
where its stored records are and which protocol hashes they must name, and the tables of its report. ``check``
refuses a spec that is not consistent. ``spec_identity_sha256`` hashes what defines a result and leaves out the
wording and the layout of the report.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import re
from dataclasses import dataclass, field

from dof.experiments.errors import SpecError

PLACEHOLDERS = ("model", "seed", "basis", "date", "symbol", "part")
STEPS = ("fit", "score", "decide", "collect", "report")
_SHA = re.compile(r"[0-9a-f]{64}")
_NAME = re.compile(r"[a-z]+")
_FIELD = re.compile(r"\{([^{}]*)\}")


@dataclass(frozen=True)
class Protocol:
    main_sha256: str | None              # the protocol document every experiment extends
    extension_sha256: str | None         # the experiment's own protocol document (seeds, adjacent, decay)
    corporate_actions_sha256: str | None  # the corporate-action file the held-out parts were scored with
    main_document: str | None = None     # its file name under configs/protocol/ (None until the copy is added)
    extension_document: str | None = None


@dataclass(frozen=True)
class Part:
    name: str                            # "valid" | "test" | "afternoon" | "far" | "decay"
    days: tuple[str, ...]
    records: tuple[str, ...]             # record kinds that hold this part's cells
    held_out: bool                       # True: read only after every fit of its round has ended
    start: str | None = None             # "HH:MM", None = session start
    end: str | None = None               # "HH:MM", None = session end


# Fields added after the first four specs were exported. ``spec_identity_sha256`` leaves such a field out while it
# holds its default, so the identity of a spec that does not use it is the one its exported records carry.
_LATER = {"later": True}


@dataclass(frozen=True)
class RecordSource:
    record: str                          # "fit_valid" | "test" | "far" | "reference" | "ensemble" | "decay" | "valid"
    pattern: str                         # file template relative to the experiment root
    pins: tuple[str, ...]                # subset of ("main", "extension", "corporate_actions")
    fit_rule: str = "none"               # "none" | "finished" | "finished_saved" | "needs_finished_fit"
    seed: int | None = 0                 # the seed when the pattern has no {seed}; None for ensembles
    # One file, several records: with ``part`` set, the record holds only the rows of the file's ``scores`` list
    # whose "part" is ``part``, and a file may match several sources that all set ``part``.
    part: str | None = field(default=None, metadata=_LATER)
    model: str | None = field(default=None, metadata=_LATER)          # the model when the pattern has no {model}
    select: tuple[tuple[str, object], ...] = field(default=(), metadata=_LATER)   # keep rows with these values


@dataclass(frozen=True)
class SeedPolicy:
    first: tuple[int, ...]               # the seeds of the rule and of the ensemble, in protocol order
    drawn: tuple[int, ...]               # seeds drawn by the rule, in drawn order (empty until known)
    older: str                           # fit whose difference the rule reads (minuend)
    recent: str                          # subtrahend
    half: float                          # more seeds when s.d. > half * |mean|
    more: int                            # how many seeds the rule draws
    force_more: bool = False             # tests only: draw whatever the rule says


@dataclass(frozen=True)
class AdjacentPlan:
    model: str                           # the network of the experiment
    ridge: str                           # its ridge reference
    train_days: tuple[str, ...]
    cut_day: str
    cut: str                             # "HH:MM"
    far_days: tuple[str, ...]
    reference_fits: tuple[str, ...]      # main-run fits scored on the afternoon (selection-bias reference)


@dataclass(frozen=True)
class DecayDay:
    date: str
    bases: tuple[str, ...]               # ("frozen",) or ("frozen", "refreshed")
    note: str                            # fixed text printed in the table's note column
    role: str                            # fixed text: the day's role in the other experiments


@dataclass(frozen=True)
class DecayPlan:
    models: tuple[str, ...]              # decay model keys, in report order
    days: tuple[DecayDay, ...]           # in date order
    distance_from: str                   # the last training day of the models
    month_days: float
    ratio_lo: float
    ratio_hi: float


@dataclass(frozen=True)
class TableSpec:
    analysis: str                        # a key of analyses.ANALYSES
    id: str
    title: str
    args: tuple[tuple[str, object], ...]  # literal keyword arguments


@dataclass(frozen=True)
class HeadlineSpec:
    template: str
    refs: tuple[tuple[str, str, tuple[str, ...], str], ...]   # (placeholder, table id, row key, column key)


@dataclass(frozen=True)
class SectionSpec:
    id: str
    title: str
    paragraphs: tuple[str, ...] = ()     # may hold {main}, {extension}, {ca}: filled from the Protocol
    headlines: tuple[HeadlineSpec, ...] = ()
    tables: tuple[TableSpec, ...] = ()
    subsections: tuple[SectionSpec, ...] = ()


@dataclass(frozen=True)
class ExperimentSpec:
    name: str                            # "alldays" | "seeds" | "adjacent" | "decay"; the cells' experiment value
    title: str
    intro: tuple[str, ...]
    root: str                            # result root relative to the results root given to export
    config: str | None                   # run configuration file name under configs/, or None
    steps: tuple[str, ...]               # subset of STEPS
    fits: tuple[str, ...]                # fits in protocol order
    linear: tuple[str, ...]              # linear references in protocol order
    models: tuple[str, ...]              # scored models in protocol order
    report_order: tuple[str, ...]        # models in the order of the "by model" sections
    seeds: tuple[int, ...]               # every seed, protocol order
    rounds: tuple[tuple[tuple[str, int], ...], ...]   # (fit, seed) per round, in protocol order
    parts: tuple[Part, ...]
    sources: tuple[RecordSource, ...]
    exclude: tuple[str, ...]             # path prefixes under the root that are never read
    protocol: Protocol
    pairs: tuple[tuple[str, str], ...] = ()     # "does the older data help" pairs
    seed0_from: str | None = None        # experiment whose records give seed 0
    uses: tuple[str, ...] = ()           # other experiments whose records the report reads
    seed_policy: SeedPolicy | None = None
    adjacent: AdjacentPlan | None = None
    decay: DecayPlan | None = None
    sections: tuple[SectionSpec, ...] = ()

    def experiment_of(self, seed: int) -> str:
        """The experiment whose records hold this seed: seed 0 of a spec that reuses another run's seed 0."""
        return self.seed0_from if seed == 0 and self.seed0_from is not None else self.name

    def check(self) -> None:
        """Raise SpecError, naming the field, when the spec is not consistent."""
        from dof.experiments.analyses import ANALYSES
        if not _NAME.fullmatch(self.name):
            raise SpecError(f"name: {self.name!r} is not lower-case letters only")
        bad = [s for s in self.steps if s not in STEPS]
        if bad:
            raise SpecError(f"{self.name}: steps: {bad} are not among {STEPS}")
        if "decide" in self.steps and self.seed_policy is None:
            raise SpecError(f"{self.name}: steps: 'decide' needs a seed_policy")
        for i, rnd in enumerate(self.rounds):
            for fit, seed in rnd:
                if fit not in self.fits:
                    raise SpecError(f"{self.name}: rounds[{i}]: {fit!r} is not a fit of the spec")
                if seed not in self.seeds:
                    raise SpecError(f"{self.name}: rounds[{i}]: seed {seed} is not a seed of the spec")
        for field in ("main_sha256", "extension_sha256", "corporate_actions_sha256"):
            v = getattr(self.protocol, field)
            if v is not None and not _SHA.fullmatch(v):
                raise SpecError(f"{self.name}: protocol.{field}: {v!r} is not 64 lower-case hex digits")
        names = [p.name for p in self.parts]
        if len(names) != len(set(names)):
            raise SpecError(f"{self.name}: parts: a part name repeats ({names})")
        for src in self.sources:
            for ph in _FIELD.findall(src.pattern):
                if ph not in PLACEHOLDERS:
                    raise SpecError(f"{self.name}: sources: pattern {src.pattern!r} holds {{{ph}}}, not one of {PLACEHOLDERS}")
            if "{model}" not in src.pattern and src.model is None:
                raise SpecError(f"{self.name}: sources: pattern {src.pattern!r} has no {{model}} and names no model")
            if src.part is not None and src.part not in names:
                raise SpecError(f"{self.name}: sources: part {src.part!r} of {src.pattern!r} is not a part of the spec")
        if self.seed_policy is not None:
            if self.seed0_from is not None and (not self.seed_policy.first or self.seed_policy.first[0] != 0):
                raise SpecError(f"{self.name}: seed_policy.first must start with 0 when seed0_from is set")
            if self.seed_policy.more % 2:
                raise SpecError(f"{self.name}: seed_policy.more: {self.seed_policy.more} is odd (seeds are fitted in pairs)")
        ids: list[str] = []
        heads: list[tuple[str, str]] = []
        for sec in _walk(self.sections):
            for t in sec.tables:
                ids.append(t.id)
                fn = ANALYSES.get(t.analysis)
                if fn is None:
                    raise SpecError(f"{self.name}: table {t.id}: analysis {t.analysis!r} is not in ANALYSES")
                params = inspect.signature(fn).parameters
                allowed = {n for n, p in params.items() if p.kind is inspect.Parameter.KEYWORD_ONLY} - {"id", "title"}
                for arg, _value in t.args:
                    if arg not in allowed:
                        raise SpecError(f"{self.name}: table {t.id}: argument {arg!r} is not a keyword of {t.analysis}")
            for h in sec.headlines:
                heads += [(sec.id, ref[1]) for ref in h.refs]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise SpecError(f"{self.name}: sections: table ids repeat: {dup}")
        missing = [f"{sid}: {tid}" for sid, tid in heads if tid not in ids]
        if missing:
            raise SpecError(f"{self.name}: headlines: no section defines the tables {missing}")


def _walk(sections):
    for s in sections:
        yield s
        yield from _walk(s.subsections)


def ts(analysis: str, id: str, title: str, **args) -> TableSpec:
    """A table of a report section: the analysis, its id and title, and its literal keyword arguments."""
    return TableSpec(analysis, id, title, tuple(sorted(args.items())))


def pair_round(fits: tuple[str, ...], a: int, b: int) -> tuple[tuple[str, int], ...]:
    """The fits of two seeds in protocol order: the first two fits of seed a, the same two of seed b, then every
    further fit of a and of b."""
    f0, f1, *rest = fits
    out = [(f0, a), (f1, a), (f0, b), (f1, b)]
    for f in rest:
        out += [(f, a), (f, b)]
    return tuple(out)


def spec_identity_sha256(spec: ExperimentSpec) -> str:
    """sha256 of what defines the results of a spec. Titles, prose, sections, the report order and the steps are
    left out: changing the wording, or adding a step this package runs, does not change what a result is."""
    keep = ("name", "root", "config", "fits", "linear", "models", "seeds", "rounds", "parts", "sources", "exclude",
            "protocol", "pairs", "seed0_from", "seed_policy", "adjacent", "decay")
    doc = {k: _plain(getattr(spec, k)) for k in keep}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _plain(o):
    if dataclasses.is_dataclass(o):
        return {f.name: _plain(getattr(o, f.name)) for f in dataclasses.fields(o)
                if not (f.metadata.get("later") and getattr(o, f.name) == f.default)}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    return o
