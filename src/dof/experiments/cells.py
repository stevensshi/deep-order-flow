"""The result tables: their schemas, the neutral record id, the collection of stored records, totals, writing and
reading.

Six tables, one row per score cell (``cells``), per stock-day a held-out or decay record names (``stock_days``), per
exported record (``records``), per fit (``fits``), per check (``checks``) and per check and stock (``check_losses``).
A record id is ``experiment/record/model/seed=<s>|members=<m>/part`` (plus ``basis/date/symbol`` for decay records):
it holds no path. The ``row`` column keeps a cell's position in its source list, so an analysis can replay the
original order bit for bit. Sums of a record and the totals leave out a missing sse or sst (``store.sum_hex``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from dof.deep.config import canonical_json
from dof.experiments.errors import AnalysisError, ExportError
from dof.experiments.spec import ExperimentSpec, RecordSource
from dof.experiments.store import discover, record_part, rule_skip, source_rows, sum_hex

S, I, F, B = "string", "int64", "float64", "bool"
CELLS = (("record_id", S), ("experiment", S), ("record", S), ("stage", S), ("model", S), ("seed", I), ("members", S),
         ("basis", S), ("part", S), ("date", S), ("symbol", S), ("k", I), ("h_ms", F), ("n_rows", I), ("n_train", I),
         ("r2_os", F), ("sse", F), ("sst", F), ("late_windows", I), ("row", I), ("weights_sha256", S),
         ("protocol_sha256", S), ("extension_protocol_sha256", S))
STOCK_DAYS = (("record_id", S), ("experiment", S), ("record", S), ("model", S), ("seed", I), ("members", S),
              ("basis", S), ("date", S), ("symbol", S), ("status", S), ("price_ratio", F), ("ratio_lo", F),
              ("ratio_hi", F), ("ratio_inside", B), ("event", S), ("effective_date", S), ("reason", S),
              ("rows_scored", I), ("ref_day", S))
RECORDS = (("record_id", S), ("experiment", S), ("record", S), ("stage", S), ("model", S), ("seed", I), ("members", S),
           ("part", S), ("basis", S), ("date", S), ("symbol", S), ("status", S), ("cells", I), ("fsum_sse", S),
           ("fsum_sst", S), ("source_sha256", S), ("source_bytes", I), ("protocol_sha256", S),
           ("extension_protocol_sha256", S), ("corporate_actions_sha256", S), ("weights_sha256", S))
FITS = (("experiment", S), ("record_id", S), ("model", S), ("fit", S), ("seed", I), ("stage", S), ("status", S),
        ("arch", S), ("optimizer", S), ("lr", F), ("eps", F), ("days", S), ("n_params_pytorch", I),
        ("n_params_keras", I), ("train_samples", I), ("train_stock_days", I), ("train_stocks", I),
        ("steps_per_pass", I), ("best_check", I), ("best_valid_loss", F), ("stopped_by", S), ("steps", I),
        ("passes", F), ("n_checks", I), ("resumes", I), ("restore_abs_diff", F), ("weights_sha256", S),
        ("protocol_sha256", S), ("torch", S), ("lambda", F), ("cond", F), ("fallback", S), ("placebo", B),
        ("config_sha256", S), ("train_days", I), ("train_first", S), ("train_last", S))
CHECKS = (("experiment", S), ("record_id", S), ("model", S), ("seed", I), ("check", I), ("steps", I),
          ("passes_completed", I), ("passes", F), ("train_loss", F), ("rows_seen", I), ("valid_loss", F),
          ("valid_rows", I), ("new_best", B), ("median_abs_dw", F), ("share_weights_unchanged", F),
          ("median_abs_w", F), ("eps_pytorch", F))
CHECK_LOSSES = (("experiment", S), ("record_id", S), ("model", S), ("seed", I), ("check", I), ("symbol", S),
                ("valid_loss", F))
TABLES = {"cells": CELLS, "stock_days": STOCK_DAYS, "records": RECORDS, "fits": FITS, "checks": CHECKS,
          "check_losses": CHECK_LOSSES}
SORT: dict[str, tuple[str, ...]] = {
    "cells": ("experiment", "record", "model", "seed", "members", "basis", "part", "date", "symbol", "k", "record_id",
              "row"),
    "stock_days": ("experiment", "record", "model", "seed", "members", "basis", "date", "symbol", "status", "record_id"),
    "records": ("record_id",),
    "fits": ("experiment", "model", "seed", "record_id"),
    "checks": ("experiment", "model", "seed", "check", "record_id"),
    "check_losses": ("experiment", "model", "seed", "check", "symbol", "record_id"),
}
# The reference implementation names an experiment's own protocol document in the key: prereg_<experiment>_sha256.
EXTENSION_KEY = re.compile(r"prereg_[a-z]+_sha256")


def record_id(experiment, record, model, seed, members, part, basis=None, date=None, symbol=None) -> str:
    parts = [experiment, record, model, f"members={members}" if members is not None else f"seed={seed}", part]
    if record in ("decay", "left_out"):
        parts += [basis, date, symbol]
    return "/".join(str(p) for p in parts)


@dataclass
class Collected:
    """One experiment's rows: lists of plain dicts per table."""
    experiment: str
    cells: list[dict] = field(default_factory=list)
    stock_days: list[dict] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)
    fits: list[dict] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    check_losses: list[dict] = field(default_factory=list)
    times: list[tuple] = field(default_factory=list)    # (record, model, seed, "finished"|"scored", time or None, part)
    skipped: dict[str, int] = field(default_factory=dict)


def _num(v, cast):
    return None if v is None else cast(v)


def pins_of(doc: dict) -> dict:
    """The protocol hashes a stored record names (the reference implementation and this package use other keys)."""
    ext = {doc[k] for k in doc if EXTENSION_KEY.fullmatch(k) and doc[k] is not None}
    if len(ext) > 1:
        raise ExportError(f"a record names {len(ext)} different experiment protocol hashes")
    return {"main": doc.get("protocol_sha256", doc.get("prereg_sha256")), "extension": next(iter(ext), None),
            "corporate_actions": (doc.get("corporate_actions") or {}).get("sha256") or doc.get("corporate_actions_sha256")}


def collect(root: Path, spec: ExperimentSpec, sources: tuple[RecordSource, ...] | None = None) -> Collected:
    """Every record of one experiment under ``root``: each found file read once, the record rule applied, the
    protocol pins checked (ExportError on a mismatch), its rows appended."""
    root = Path(root)
    found, skipped = discover(root, spec, sources)
    skipped = Counter(skipped)
    c = Collected(spec.name)
    for f in found:
        try:
            raw = (root / f.rel).read_bytes()
        except FileNotFoundError:
            skipped["vanished"] += 1
            continue
        doc = json.loads(raw)
        record, why = rule_skip(root, f, doc)
        if record is None:
            skipped[why] += 1
            continue
        _add(c, spec, f, record, doc, raw)
    c.skipped = dict(skipped)
    return c


def _add(c: Collected, spec: ExperimentSpec, f, record: str, doc: dict, raw: bytes) -> None:
    exp, model, seed = spec.name, f.fields.get("model", f.source.model), f.seed
    members = "+".join(str(s) for s in doc["members"]) if record == "ensemble" else None
    part = record_part(f.source, record, f.fields)
    basis, date, symbol = (f.fields.get(k) for k in ("basis", "date", "symbol"))
    rid = record_id(exp, record, model, seed, members, part, basis, date, symbol)
    pins = pins_of(doc)
    want = {"main": spec.protocol.main_sha256, "extension": spec.protocol.extension_sha256,
            "corporate_actions": spec.protocol.corporate_actions_sha256}
    for pin in f.source.pins:
        if pins[pin] != want[pin]:
            raise ExportError(f"{rid}: {pin} is {pins[pin]}, the spec pins {want[pin]}")
    weights = ((doc.get("weights") or {}).get("sha256") if record == "fit_valid"
               else None if record == "ensemble" else doc.get("weights_sha256"))
    rows = source_rows(f.source, record, doc)
    stage = doc.get("stage")
    cells = []
    for i, r in enumerate(rows):
        cells.append({"record_id": rid, "experiment": exp, "record": record, "stage": stage,
                      "model": r.get("model", doc.get("model")),
                      "seed": None if members is not None else _num(r.get("seed", doc.get("seed", seed)), int),
                      "members": members, "basis": r.get("basis", doc.get("basis")), "part": r.get("part"),
                      "date": r.get("date"), "symbol": r.get("symbol"), "k": _num(r.get("k"), int),
                      "h_ms": _num(r.get("h_ms"), float), "n_rows": _num(r.get("n_rows"), int),
                      "n_train": _num(r.get("n_train"), int), "r2_os": _num(r.get("r2_os"), float),
                      "sse": _num(r.get("sse"), float), "sst": _num(r.get("sst"), float),
                      "late_windows": _num(r.get("late_windows"), int), "row": i, "weights_sha256": weights,
                      "protocol_sha256": pins["main"], "extension_protocol_sha256": pins["extension"]})
    c.cells += cells
    c.records.append({"record_id": rid, "experiment": exp, "record": record, "stage": stage, "model": model, "seed": seed,
                      "members": members, "part": part, "basis": basis, "date": date, "symbol": symbol,
                      "status": doc.get("status", "scored"), "cells": len(cells),
                      "fsum_sse": sum_hex(x["sse"] for x in cells), "fsum_sst": sum_hex(x["sst"] for x in cells),
                      "source_sha256": hashlib.sha256(raw).hexdigest(), "source_bytes": len(raw),
                      "protocol_sha256": pins["main"], "extension_protocol_sha256": pins["extension"],
                      "corporate_actions_sha256": pins["corporate_actions"], "weights_sha256": weights})
    base = {"record_id": rid, "experiment": exp, "record": record, "model": model, "seed": seed, "members": members,
            "basis": basis, "date": date, "symbol": symbol, "status": None, "price_ratio": None, "ratio_lo": None,
            "ratio_hi": None, "ratio_inside": None, "event": None, "effective_date": None, "reason": None,
            "rows_scored": None, "ref_day": None}
    if record in ("decay", "left_out"):
        c.stock_days.append({**base, "status": doc.get("status"), "price_ratio": _num(doc.get("ratio"), float),
                             "ratio_inside": doc.get("ratio_inside"), "reason": doc.get("reason"),
                             "rows_scored": _num(doc.get("rows_scored"), int), "ref_day": doc.get("ref_day")})
    elif isinstance(doc.get("price_ratio"), dict):
        lo, hi = (doc.get("ratio_range") or [None, None])[:2]
        for key, v in doc["price_ratio"].items():
            sym, day = key.split(" ")
            c.stock_days.append({**base, "symbol": sym, "date": day, "status": "scored", "price_ratio": float(v),
                                 "ratio_lo": _num(lo, float), "ratio_hi": _num(hi, float)})
        for x in (doc.get("corporate_actions") or {}).get("left_out", []):
            c.stock_days.append({**base, "symbol": x.get("symbol"), "date": x.get("date"), "status": "left_out",
                                 "event": x.get("event"), "effective_date": x.get("effective_date")})
    if record == "fit_valid":
        _fit_rows(c, rid, doc, pins["main"])
        c.times.append((record, model, seed, "finished", doc.get("finished"), part))
    elif record in ("test", "far", "reference", "ensemble"):
        c.times.append((record, model, seed, "scored", doc.get("scored"), part))


def _fit_rows(c: Collected, rid: str, doc: dict, protocol: str | None) -> None:
    spec, tr, npar = doc.get("spec") or {}, doc.get("train") or {}, doc.get("n_params") or {}
    seed, model, dates = _num(doc.get("seed"), int), doc.get("model"), tr.get("dates")
    c.fits.append({
        "experiment": c.experiment, "record_id": rid, "model": model, "fit": doc.get("fit", model), "seed": seed,
        "stage": doc.get("stage"), "status": doc.get("status"), "arch": spec.get("arch"),
        "optimizer": spec.get("optimizer"), "lr": _num(spec.get("lr"), float), "eps": _num(spec.get("eps"), float),
        "days": spec.get("days", tr.get("days")), "n_params_pytorch": _num(npar.get("pytorch"), int),
        "n_params_keras": _num(npar.get("keras_equivalent"), int), "train_samples": _num(tr.get("samples"), int),
        "train_stock_days": _num(tr.get("stock_days"), int), "train_stocks": _num(tr.get("stocks"), int),
        "steps_per_pass": _num(tr.get("steps_per_pass"), int), "best_check": _num(doc.get("best_check"), int),
        "best_valid_loss": _num(doc.get("best_valid_loss"), float), "stopped_by": doc.get("stopped_by"),
        "steps": _num(doc.get("steps"), int), "passes": _num(doc.get("passes"), float),
        "n_checks": len(doc["checks"]) if "checks" in doc else None,
        "resumes": len(doc["resumes"]) if "resumes" in doc else None,
        "restore_abs_diff": _num((doc.get("restore_check") or {}).get("abs_diff"), float),
        "weights_sha256": (doc.get("weights") or {}).get("sha256"), "protocol_sha256": protocol,
        "torch": doc.get("torch"), "lambda": _num(doc.get("lambda"), float), "cond": _num(doc.get("cond"), float),
        "fallback": doc.get("fallback"), "placebo": doc.get("placebo_shuffled_labels"),
        "config_sha256": hashlib.sha256(canonical_json(doc["config"]).encode()).hexdigest() if "config" in doc else None,
        "train_days": len(dates) if dates else None, "train_first": dates[0] if dates else None,
        "train_last": dates[-1] if dates else None})
    for ck in doc.get("checks", []):
        n = _num(ck.get("check"), int)
        c.checks.append({
            "experiment": c.experiment, "record_id": rid, "model": model, "seed": seed, "check": n,
            "steps": _num(ck.get("steps"), int), "passes_completed": _num(ck.get("passes_completed"), int),
            "passes": _num(ck.get("passes"), float), "train_loss": _num(ck.get("train_loss"), float),
            "rows_seen": _num(ck.get("rows_seen"), int), "valid_loss": _num(ck.get("valid_loss"), float),
            "valid_rows": _num(ck.get("valid_rows"), int), "new_best": ck.get("new_best"),
            "median_abs_dw": _num(ck.get("median_abs_dw"), float),
            "share_weights_unchanged": _num(ck.get("share_weights_unchanged"), float),
            "median_abs_w": _num(ck.get("median_abs_w"), float), "eps_pytorch": _num(ck.get("eps_pytorch"), float)})
        for sym, v in (ck.get("valid_loss_by_stock") or {}).items():
            c.check_losses.append({"experiment": c.experiment, "record_id": rid, "model": model, "seed": seed,
                                   "check": n, "symbol": sym, "valid_loss": _num(v, float)})


def totals(src) -> dict[str, dict]:
    """Per "<experiment>/<record>" (sorted): the cell count, the source count and the exact sums of sse and sst as
    float hex. ``src``: a list of Collected, or ResultTables (read through ``rows``, which keeps None apart from NaN)."""
    if isinstance(src, ResultTables):
        recs = list(zip(*(src.records[k].tolist() for k in ("experiment", "record", "record_id"))))
        cells = [(e, r, a, b) for e, r, rid in recs for a, b in zip(src.rows(rid).sse, src.rows(rid).sst)]
    else:
        recs = [(r["experiment"], r["record"]) for c in src for r in c.records]
        cells = [(x["experiment"], x["record"], x["sse"], x["sst"]) for c in src for x in c.cells]
    sources = Counter(f"{e}/{r}" for e, r, *_ in recs)
    sse: dict = defaultdict(list)
    sst: dict = defaultdict(list)
    for e, r, a, b in cells:
        sse[f"{e}/{r}"].append(a)
        sst[f"{e}/{r}"].append(b)
    return {k: {"cells": len(sse[k]), "sources": sources[k], "fsum_sse": sum_hex(sse[k]), "fsum_sst": sum_hex(sst[k])}
            for k in sorted(sources)}


def _sort_key(fields):
    def key(row):
        return tuple((row[f] is None, 0 if row[f] is None else row[f]) for f in fields)
    return key


def env_versions() -> dict:
    import platform

    import numpy
    import pandas
    import pyarrow
    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
            "pyarrow": pyarrow.__version__}


def write_tables(collected: Sequence[Collected], out: Path, manifest_extra: dict) -> dict:
    """The six tables as Parquet (zstd) in ``out``, each through a ``.tmp`` file; then ``MANIFEST.json``. Returns the
    manifest."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    import dof
    types = {S: pa.string(), I: pa.int64(), F: pa.float64(), B: pa.bool_()}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, schema in TABLES.items():
        rows = sorted((r for c in collected for r in getattr(c, name)), key=_sort_key(SORT[name]))
        t = pa.Table.from_pylist(rows, schema=pa.schema([(f, types[ty]) for f, ty in schema]))
        path = out / f"{name}.parquet"
        tmp = path.with_name(path.name + ".tmp")
        pq.write_table(t, tmp, compression="zstd")
        os.replace(tmp, path)
        outputs[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": t.num_rows}
    manifest = {"format": 1, "package_version": dof.__version__, "outputs": outputs, "totals": totals(collected),
                **manifest_extra}
    tmp = out / "MANIFEST.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out / "MANIFEST.json")
    return manifest


# -- reading ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rows:
    """One record's cells in source order."""
    symbol: tuple[str, ...] = ()
    date: tuple[str, ...] = ()
    k: tuple[int, ...] = ()
    r2: tuple[float, ...] = ()
    sse: tuple[float, ...] = ()
    sst: tuple[float, ...] = ()

    def __add__(self, other: Rows) -> Rows:
        return Rows(self.symbol + other.symbol, self.date + other.date, self.k + other.k, self.r2 + other.r2,
                    self.sse + other.sse, self.sst + other.sst)

    def __len__(self) -> int:
        return len(self.symbol)

    def take(self, keep: Sequence[bool]) -> Rows:
        def sel(col):
            return tuple(v for v, ok in zip(col, keep) if ok)
        return Rows(sel(self.symbol), sel(self.date), sel(self.k), sel(self.r2), sel(self.sse), sel(self.sst))

    @staticmethod
    def concat(parts) -> Rows:
        cols: list[list] = [[], [], [], [], [], []]
        for p in parts:
            for dst, src in zip(cols, (p.symbol, p.date, p.k, p.r2, p.sse, p.sst)):
                dst.extend(src)
        return Rows(*(tuple(c) for c in cols))


def plain(v):
    """A table value as a Python value: missing (None, pandas NA) -> None, numpy scalars -> Python scalars."""
    import pandas as pd
    if v is None or v is pd.NA:
        return None
    return v.item() if hasattr(v, "item") else v


@dataclass(frozen=True)
class ResultTables:
    cells: object                 # pandas DataFrames, one per table
    stock_days: object
    records: object
    fits: object
    checks: object
    check_losses: object
    index: dict = field(repr=False, compare=False, default_factory=dict)

    def rid(self, *, experiment, record, model, seed=None, members=None, part=None) -> str | None:
        hits = [r for r in self.index["records"].get((experiment, record, model), [])
                if (seed is None or r["seed"] == seed) and (members is None or r["members"] == members)
                and (part is None or r["part"] == part)]
        if len(hits) > 1:
            raise AnalysisError(f"{len(hits)} records of {experiment}/{record}/{model} seed={seed} members={members} "
                                f"part={part}: the record is ambiguous")
        return hits[0]["record_id"] if hits else None

    def record(self, record_id: str) -> dict:
        return self.index["by_id"][record_id]

    def rows(self, record_id: str) -> Rows:
        return self.index["rows"].get(record_id, Rows())

    def stock_days_of(self, record_id: str):
        sd = self.stock_days
        return sd[sd["record_id"] == record_id].sort_values(["symbol", "date", "status"], kind="stable")

    def fit(self, record_id: str) -> dict:
        return self.index["fits"][record_id]

    def checks_of(self, record_id: str):
        ck = self.checks
        return ck[ck["record_id"] == record_id].sort_values("check", kind="stable")

    def decay_units(self, model: str, date: str, basis: str) -> list[dict]:
        return self.index["units"].get((model, date, basis), [])


def read_tables(folder: Path) -> ResultTables:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    folder = Path(folder)
    mapper = {pa.int64(): pd.Int64Dtype(), pa.bool_(): pd.BooleanDtype()}.get
    arrow = {name: pq.read_table(folder / f"{name}.parquet") for name in TABLES}
    frames = {name: t.to_pandas(types_mapper=mapper) for name, t in arrow.items()}
    groups: dict = defaultdict(list)
    cols = ("record_id", "row", "symbol", "date", "k", "r2_os", "sse", "sst")
    for rid, row, *vals in zip(*(arrow["cells"].column(c).to_pylist() for c in cols)):
        groups[rid].append((row, *vals))
    rows = {}
    for rid, items in groups.items():
        items.sort(key=lambda x: x[0])
        rows[rid] = Rows(*(tuple(x[i] for x in items) for i in range(1, 7)))
    recs = arrow["records"].to_pylist()
    by_key: dict = defaultdict(list)
    for r in recs:
        by_key[(r["experiment"], r["record"], r["model"])].append(r)
    units: dict = defaultdict(list)
    for u in arrow["stock_days"].to_pylist():
        if u["experiment"] == "decay" and u["record"] in ("decay", "left_out"):
            units[(u["model"], u["date"], u["basis"])].append(
                {"record_id": u["record_id"], "symbol": u["symbol"], "status": u["status"], "ratio": u["price_ratio"],
                 "ratio_inside": u["ratio_inside"], "reason": u["reason"]})
    for v in units.values():
        v.sort(key=lambda u: u["symbol"] + ".json")        # the file-name order of the stored unit files
    index = {"rows": rows, "records": dict(by_key), "by_id": {r["record_id"]: r for r in recs},
             "fits": {f["record_id"]: f for f in arrow["fits"].to_pylist()}, "units": dict(units)}
    return ResultTables(**frames, index=index)
