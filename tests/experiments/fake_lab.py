"""A small stored lab in the five layouts of the stored experiments, for the hermetic tests (not collected).

Real model names, days and seeds; three symbols AAA, BBB, CCC; ten horizons. Cell i (a running number over the whole
lab) has r2 = 0.001 * (i % 7) - 0.002, sst = 1 + i, sse = sst * (1 - r2). One cell has no R² (sst = 0). CCC is left out
of 2026-06-12 by a corporate action; BBB on 2026-05-18 has a price ratio outside [0.6, 1.6]. Fake protocol pins.
The earlier stage's layout (``pooled115/``) holds one file per network with its validation and test rows, a file
the spec excludes, and a linear file whose rows keep R² only and mix the pooled Ridge-OF with
rows the spec leaves out.
``FakeLab.totals`` holds the totals computed with ``math.fsum`` from what was written.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from dof.experiments.spec import Protocol
from dof.experiments.specs import FITS, LINEAR, SPECS, TEST_DAYS, VALID_DAYS

SYMBOLS = ("AAA", "BBB", "CCC")
PINS = {"main": "a" * 64, "seeds": "b" * 64, "adjacent": "c" * 64, "decay": "d" * 64, "ca": "e" * 64}
SEEDS2 = (3089492870, 771810125)
ADJ_SEEDS = (3089492870, 771810125, 18531247)
MEMBERS = [0, 3089492870, 771810125]
LEFT_OUT = ("CCC", "2026-06-12")
OUTSIDE = ("BBB", "2026-05-18")
DECAY_MODELS = ("CNN-LSTM-OF-lr3e-5-eps1e-7", "LSTM-MLP-OF")
DECAY_DAYS = {"2019-10-18": ("frozen",), "2019-10-30": ("frozen",), "2019-12-30": ("frozen",),
              "2021-08-13": ("frozen", "refreshed"), "2025-12-12": ("frozen", "refreshed")}
DECAY_LEFT_OUT = ("CCC", "2025-12-12", "refreshed")
EARLIER_NETWORKS = ("LSTM-OF", "LSTM-MLP-OF", "CNN-LSTM-OF-lr3e-5-eps1e-7")
EARLIER_VALID = ("2019-08-30",)
EARLIER_TEST = ("2019-10-18", "2019-10-30", "2019-12-30")


@dataclass
class FakeLab:
    root: Path
    pins: dict
    totals: dict                    # "<experiment>/<record>" -> {cells, sources, fsum_sse, fsum_sst}
    exported: dict                  # experiment -> {relative path: record kind}


class _Lab:
    def __init__(self, root: Path) -> None:
        self.root, self.i, self.minute = Path(root), 0, 0
        self.groups: dict = defaultdict(lambda: {"cells": 0, "sources": 0, "sse": [], "sst": []})
        self.exported: dict = defaultdict(dict)

    def time(self, day: str) -> str:
        self.minute += 1
        return f"{day}T00:{self.minute // 60:02d}:{self.minute % 60:02d}"

    def cells(self, model, seed, part, days, symbols=SYMBOLS, skip=(), basis=None, nan_first=False) -> list[dict]:
        rows = []
        for d in days:
            for s in symbols:
                if (s, d) in skip:
                    continue
                for k in range(1, 11):
                    i = self.i
                    self.i += 1
                    r2 = 0.001 * (i % 7) - 0.002
                    sst = 1.0 + i
                    row = {"part": part, "date": d, "k": k, "h_ms": 100.0 * k, "n_rows": 1000 + i, "r2_os": r2,
                           "sse": sst * (1 - r2), "sst": sst, "late_windows": 0, "symbol": s, "model": model,
                           "seed": seed, "n_train": 5000}
                    if basis is not None:
                        row["basis"] = basis
                    rows.append(row)
        if nan_first:
            rows[0].update(r2_os=float("nan"), sst=0.0, sse=0.0)
        return rows

    def write(self, rel: str, doc: dict, experiment: str | None = None, kind: str | None = None, cells=()) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc), encoding="utf-8")
        if experiment is not None:
            g = self.groups[f"{experiment}/{kind}"]
            g["sources"] += 1
            g["cells"] += len(cells)
            g["sse"] += [c["sse"] for c in cells if "sse" in c]
            g["sst"] += [c["sst"] for c in cells if "sst" in c]
            root = {"alldays": "results/", "seeds": "results_seeds/", "adjacent": "adj/results/",
                    "decay": "decay/scores/", "earlier": "pooled115/"}[experiment]
            self.exported[experiment][rel[len(root):]] = kind

    def touch(self, rel: str) -> None:
        (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.root / rel).write_bytes(b"")


def _sha(model, seed) -> str:
    return hashlib.sha256(f"{model} {seed}".encode()).hexdigest()


def _ext_key(kind: str) -> str:
    """The key under which a stored record names its experiment's own protocol hash."""
    return f"prereg_{kind}_sha256"


def _fit_doc(lab: _Lab, model: str, seed: int, rows: list[dict], day: str, linear: bool, recent: bool, pin: str) -> dict:
    dates = ["2025-11-28", "2025-12-08", "2025-12-09", "2025-12-10"] if recent else ["2019-01-30", "2025-12-10"]
    doc = {"model": model, "fit": model, "stage": "real", "seed": seed, "status": "finished", "prereg_sha256": pin,
           "valid_scores": rows, "config": {"seed": seed, "fit": model}, "finished": lab.time(day),
           "train": {"days": "recent" if recent else "all", "dates": dates, "samples": 10000, "stocks": 3,
                     "stock_days": 3 * len(dates), "steps_per_pass": 40},
           "weights": {"sha256": _sha(model, seed)}}
    if linear:
        doc.update(spec={"days": doc["train"]["days"]}, **{"lambda": 0.1}, cond=12.5, fallback=None)
        return doc
    checks = [{"check": n, "steps": 100 * n, "passes_completed": n // 2, "passes": 0.5 * n, "train_loss": 1.0 - 0.01 * n,
               "rows_seen": 25600 * n, "valid_loss": 1.1 - 0.001 * n * (3 - n), "valid_rows": 3000,
               "new_best": n != 3, "median_abs_dw": 1.5e-6 * n, "share_weights_unchanged": 0.0, "median_abs_w": 0.05,
               "eps_pytorch": 1e-7, "valid_loss_by_stock": {s: 1.0 + 0.01 * n for s in SYMBOLS}} for n in (1, 2, 3)]
    doc.update(spec={"arch": model.rsplit("-", 1)[0], "optimizer": "adam", "lr": 1e-5, "eps": 1e-7,
                     "days": doc["train"]["days"]}, n_params={"pytorch": 84938, "keras_equivalent": 84458},
               checks=checks, best_check=2, best_valid_loss=checks[1]["valid_loss"], stopped_by="patience", steps=300,
               passes=7.5, resumes=[], restore_check={"abs_diff": 0.0}, torch="2.0")
    return doc


def _scores_doc(model, seed, rows, scored, pins: dict, ratios=True, ca=True) -> dict:
    doc = {"model": model, "stage": "real", "seed": seed, "scores": rows, "scored": scored,
           "weights_sha256": _sha(model, seed), **pins}
    if ratios:
        days = sorted({(r["symbol"], r["date"]) for r in rows})
        doc["price_ratio"] = {f"{s} {d}": (2.0 if (s, d) == OUTSIDE else 1.0 + 0.01 * j) for j, (s, d) in enumerate(days)}
        doc["ratio_range"] = [0.6, 1.6]
    if ca:
        doc["corporate_actions"] = {"file": "ca.csv", "sha256": PINS["ca"], "left_out": [
            {"symbol": LEFT_OUT[0], "date": LEFT_OUT[1], "event": "forward_split", "effective_date": LEFT_OUT[1]}]}
    return doc


def _alldays(lab: _Lab) -> None:
    main = PINS["main"]
    for m in FITS + LINEAR:
        rows = lab.cells(m, 0, "valid", VALID_DAYS)
        lab.write(f"results/{m}/fit_seed0.json", _fit_doc(lab, m, 0, rows, "2026-01-01", m in LINEAR, "recent" in m, main),
                  "alldays", "fit_valid", rows)
    for m in FITS + LINEAR:
        rows = lab.cells(m, 0, "test", TEST_DAYS, skip=(LEFT_OUT,), nan_first=(m == "LSTM-MLP-OF-recent"))
        lab.write(f"results/{m}/test_seed0.json", _scores_doc(m, 0, rows, lab.time("2026-01-02"), {"prereg_sha256": main}),
                  "alldays", "test", rows)
    lab.write("results/smoke/LSTM-MLP-OF-recent/fit_seed0.json", {"status": "finished"})
    lab.write("results/progress.json", {"phase": "done"})
    lab.write("results/measure.json", [{"fit": "x"}])


def _seeds(lab: _Lab) -> None:
    main, ext = PINS["main"], PINS["seeds"]
    for s in SEEDS2:
        for m in FITS:
            rows = lab.cells(m, s, "valid", VALID_DAYS)
            folder = f"results_seeds/seed{s}/{m}"
            lab.write(f"{folder}/fit_seed{s}.json", _fit_doc(lab, m, s, rows, "2026-02-01", False, "recent" in m, main),
                      "seeds", "fit_valid", rows)
            lab.touch(f"{folder}/best_weights_seed{s}.pt")
            lab.write(f"{folder}/begun_seed{s}.json", {"seed": s})
        lab.write(f"results_seeds/seed{s}/progress.json", {"phase": "done"})
    extra = "results_seeds/seed18531247/LSTM-MLP-OF-recent"
    lab.write(f"{extra}/fit_seed18531247.json", _fit_doc(lab, "LSTM-MLP-OF-recent", 18531247, [], "2026-02-01", False,
                                                         True, main))
    lab.touch(f"{extra}/best_weights_seed18531247.pt")
    lab.touch(f"{extra}/checkpoint_seed18531247.pt")
    for s in SEEDS2:
        for m in FITS:
            rows = lab.cells(m, s, "test", TEST_DAYS, skip=(LEFT_OUT,))
            doc = _scores_doc(m, s, rows, lab.time("2026-02-02"), {"prereg_sha256": main, _ext_key("seeds"): ext})
            lab.write(f"results_seeds/seed{s}/{m}/test_seed{s}.json", doc, "seeds", "test", rows)
    label = "+".join(str(s) for s in MEMBERS)
    for m in FITS:
        for part, days in (("valid", VALID_DAYS), ("test", TEST_DAYS)):
            rows = lab.cells(m, label, part, days, skip=(LEFT_OUT,) if part == "test" else ())
            doc = _scores_doc(m, None, rows, lab.time("2026-02-03"), {"prereg_sha256": main, _ext_key("seeds"): ext},
                              ratios=part == "test", ca=part == "test")
            doc.update(status="scored", members=MEMBERS, part=part)
            del doc["seed"], doc["weights_sha256"]
            lab.write(f"results_seeds/ensemble/{m}/{part}_first3seeds.json", doc, "seeds", "ensemble", rows)
    for name, doc in (("decision", {"seeds_drawn": []}), ("extension", {"x": 1}), ("current", {"fit": None})):
        lab.write(f"results_seeds/{name}.json", doc)


def _adjacent(lab: _Lab) -> None:
    main, ext = PINS["main"], PINS["adjacent"]
    pins = {"prereg_sha256": main, _ext_key("adj"): ext}
    net, ridge = "LSTM-MLP-OF-recent", "Ridge-OF-recent"
    for folder, model, seed, linear in [*((f"seed{s}/{net}", net, s, False) for s in ADJ_SEEDS),
                                        (f"ridge/{ridge}", ridge, 0, True)]:
        rows = lab.cells(model, seed, "valid", ("2025-12-12",))
        lab.write(f"adj/results/{folder}/fit_seed{seed}.json",
                  _fit_doc(lab, model, seed, rows, "2026-03-01", linear, True, main), "adjacent", "fit_valid", rows)
    for folder, model, seed in [*((f"seed{s}/{net}", net, s) for s in ADJ_SEEDS), (f"ridge/{ridge}", ridge, 0)]:
        rows = lab.cells(model, seed, "test", ("2025-12-12",))
        doc = _scores_doc(model, seed, rows, lab.time("2026-03-02"), pins)
        doc.update(status="scored")
        doc["corporate_actions"]["left_out"] = []
        lab.write(f"adj/results/{folder}/test_seed{seed}.json", doc, "adjacent", "test", rows)
        rows = lab.cells(model, seed, "test", TEST_DAYS, skip=(LEFT_OUT,))
        doc = _scores_doc(model, seed, rows, lab.time("2026-03-02"), pins)
        doc.update(status="scored")
        lab.write(f"adj/results/{folder}/far_seed{seed}.json", doc, "adjacent", "far", rows)
    for n in FITS:
        rows = lab.cells(n, 0, "test", ("2025-12-12",))
        doc = _scores_doc(n, 0, rows, lab.time("2026-03-03"), pins, ratios=False, ca=False)
        doc.update(status="scored", skipped=[])
        lab.write(f"adj/results/reference/{n}/test_seed0.json", doc, "adjacent", "reference", rows)
    for name in ("experiment", "gate_refusals", "current"):
        lab.write(f"adj/results/{name}.json", {"x": 1})
    for s in ADJ_SEEDS:
        lab.write(f"adj/results/seed{s}/progress.json", {"phase": "done"})


def _decay(lab: _Lab) -> None:
    for model in DECAY_MODELS:
        for date, bases in DECAY_DAYS.items():
            for basis in bases:
                for j, sym in enumerate(SYMBOLS):
                    rel = f"decay/scores/{model}/{basis}/{date}/{sym}.json"
                    base = {"date": date, "symbol": sym, "basis": basis, "model": model,
                            _ext_key("decay"): PINS["decay"], "weights_sha256": "f" * 64}
                    if (sym, date, basis) == DECAY_LEFT_OUT:
                        lab.write(rel, {**base, "status": "left_out", "reason": "reused ticker (rule 1)", "scores": []},
                                  "decay", "left_out")
                        continue
                    ratio = 0.5 if (sym, date) == ("BBB", "2021-08-13") else 1.7 if (sym, date) == ("AAA", "2025-12-12") \
                        else 1.0 + 0.05 * j
                    rows = lab.cells(model, 0, "decay", (date,), symbols=(sym,), basis=basis)
                    lab.write(rel, {**base, "status": "scored", "ratio": ratio, "ratio_inside": 0.6 <= ratio <= 1.6,
                                    "rows_scored": 1000 + j, "ref_day": "2019-07-30", "scores": rows}, "decay", "decay", rows)


def _earlier(lab: _Lab) -> None:
    for m in EARLIER_NETWORKS:
        valid = lab.cells(m, 0, "valid", EARLIER_VALID)
        test = lab.cells(m, 0, "test", EARLIER_TEST)
        doc = {"model": m, "seed": 0, "finished": lab.time("2019-11-01"), "fit": {"epochs_run": 3}, "scores": valid + test}
        lab.write(f"pooled115/{m}/pooled_seed0.json", doc)
        for kind, cells in (("valid", valid), ("test", test)):
            g = lab.groups[f"earlier/{kind}"]
            g["sources"] += 1
            g["cells"] += len(cells)
            g["sse"] += [c["sse"] for c in cells]
            g["sst"] += [c["sst"] for c in cells]
            lab.exported["earlier"][f"{m}/pooled_seed0.json:{kind}"] = kind
    lab.write("pooled115/CNN-LSTM-OF-lr0.3-eps1/pooled_seed0.json", {"model": "x", "scores": []})

    def r2_only(rows, training, model):
        return [{"symbol": r["symbol"], "training": training, "model": model, "lam": 0.1, "part": r["part"],
                 "date": r["date"], "k": r["k"], "n_rows": r["n_rows"], "r2_os": r["r2_os"]} for r in rows]
    rows = []
    for kind, days in (("valid", EARLIER_VALID), ("test", EARLIER_TEST)):
        mine = r2_only(lab.cells("Ridge-OF", 0, kind, days), "pooled", "Ridge-OF")
        rows += mine
        rows += r2_only(lab.cells("Ridge-OF", 0, kind, days), "per-stock", "Ridge-OF")
        rows += r2_only(lab.cells("ARX-OF", 0, kind, days), "pooled", "ARX-OF")
        g = lab.groups[f"earlier/{kind}"]
        g["sources"] += 1
        g["cells"] += len(mine)
        lab.exported["earlier"][f"linear/linear.json:{kind}"] = kind
    lab.write("pooled115/linear/linear.json", {"symbols": list(SYMBOLS), "lambda_pooled": 0.1, "scores": rows})


def write_fake_lab(root: Path) -> FakeLab:
    lab = _Lab(root)
    _alldays(lab)
    _seeds(lab)
    _adjacent(lab)
    _decay(lab)
    _earlier(lab)
    totals = {k: {"cells": g["cells"], "sources": g["sources"], "fsum_sse": float(math.fsum(g["sse"])).hex(),
                  "fsum_sst": float(math.fsum(g["sst"])).hex()} for k, g in sorted(lab.groups.items())}
    return FakeLab(Path(root), dict(PINS), totals, {k: dict(v) for k, v in lab.exported.items()})


def fake_specs(lab: FakeLab) -> dict:
    """The five real specs with the fake protocol pins (the earlier stage pins none)."""
    p = lab.pins
    proto = {"alldays": (p["main"], None, p["ca"]), "seeds": (p["main"], p["seeds"], p["ca"]),
             "adjacent": (p["main"], p["adjacent"], p["ca"]), "decay": (p["main"], p["decay"], None),
             "earlier": (None, None, None)}
    out = {}
    for name, spec in SPECS.items():
        main, ext, ca = proto[name]
        out[name] = replace(spec, protocol=Protocol(main, ext, ca, spec.protocol.main_document,
                                                    spec.protocol.extension_document))
    return out


ORDER = ("alldays", "seeds", "adjacent", "decay", "earlier")


def exported_tables(root: Path):
    """A fake lab exported with the five fake-pinned specs: (lab, specs, ResultTables)."""
    from dof.experiments.cells import read_tables
    from dof.experiments.export import export
    lab = write_fake_lab(Path(root) / "lab")
    specs = fake_specs(lab)
    export(lab.root, Path(root) / "tables", [specs[n] for n in ORDER], log=lambda msg: None)
    return lab, specs, read_tables(Path(root) / "tables")
