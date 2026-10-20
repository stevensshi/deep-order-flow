"""Training golden recipes: flat values from cache files, training batches, initial weights, fit records and
score records.

numpy only: tensors arrive as numpy arrays. The recorder of the reference implementation and the golden tests
of the training package call these functions on their own outputs, through their own adapters.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import hashing as H

FIT_FIELDS = ("status", "best_check", "best_valid_loss", "stopped_by", "steps", "passes", "patience_check",
              "restore_check", "optimizer", "n_params", "train")
CHECK_FIELDS = ("check", "steps", "passes_completed", "train_loss", "rows_seen", "valid_loss", "valid_loss_by_stock",
                "valid_rows", "new_best", "median_abs_dw", "share_weights_unchanged", "median_abs_w", "eps_pytorch")
CELL_FLOATS = ("r2_os", "sse", "sst", "h_ms")
CELL_INTS = ("n_rows", "late_windows")


def under(prefix: str, values: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in values.items()}


def cache_values(train_dir: Path, eval_dir: Path) -> dict:
    """Every file of a feature cache: a .npy by its array hash, each key of a .npz as "path#key", a .json by the
    sha256 of its canonical JSON without the volatile keys (times, durations, code hashes, sizes). Keys are
    "train/<relative path>" and "eval/<relative path>". Any other file is listed under "<part>/other_files"."""
    out: dict = {}
    for part, root in (("train", Path(train_dir)), ("eval", Path(eval_dir))):
        other = []
        for p in sorted((q for q in root.rglob("*") if q.is_file()), key=lambda q: q.relative_to(root).as_posix()):
            rel = p.relative_to(root).as_posix()
            if p.suffix == ".npy":
                out[f"{part}/{rel}"] = H.array_sha256(np.load(p))
            elif p.suffix == ".npz":
                with np.load(p) as z:
                    for k in sorted(z.files):
                        out[f"{part}/{rel}#{k}"] = H.array_sha256(z[k])
            elif p.suffix == ".json":
                doc = json.loads(p.read_text(encoding="utf-8"))
                out[f"{part}/{rel}"] = hashlib.sha256(H.canonical_json(H.scrub(doc, H.VOLATILE)).encode()).hexdigest()
            else:
                other.append(rel)
        if other:
            out[f"{part}/other_files"] = H.canonical_json(other)
    return out


def days_values(rows, n_train: int, steps_per_pass: int) -> dict:
    """``rows``: (symbol, date, group, n_rows, n_samples, sample rows) per training stock-day, in stream order."""
    table = [[s, d, int(g), int(nr), int(ns), H.array_sha256(r)] for s, d, g, nr, ns, r in rows]
    return {"list": H.canonical_json(table), "n_train": int(n_train), "steps_per_pass": int(steps_per_pass)}


def batch_values(batches) -> dict:
    """``batches``: dicts with x, zy (arrays), late, broken, not_finite (counts), in stream order."""
    out = {}
    for i, bt in enumerate(batches):
        x = np.asarray(bt["x"])
        p = f"b{i:03d}"
        out[f"{p}/x"] = H.array_sha256(x)
        out[f"{p}/zy"] = H.array_sha256(bt["zy"])
        out[f"{p}/late"] = int(bt["late"])
        out[f"{p}/broken"] = int(bt["broken"])
        out[f"{p}/not_finite"] = int(bt["not_finite"])
        out[f"{p}/rows"] = int(len(x))
    return out


def pass_sha256(batches: np.ndarray, order) -> str:
    """The batch table of one pass and every stock-day's sample order, as one hash."""
    return hashlib.sha256(batches.tobytes() + b"".join(o.tobytes() for o in order)).hexdigest()


def init_values(named: dict, flat) -> dict:
    """``named``: parameter name -> array, in parameter order; ``flat``: all parameters concatenated."""
    out = {"flat": H.array_sha256(flat), "n_params": int(sum(np.asarray(a).size for a in named.values())),
           "shapes": H.canonical_json({k: list(np.asarray(a).shape) for k, a in named.items()})}
    for k, a in named.items():
        out[f"param/{k}"] = H.array_sha256(a)
    return out


def cells_values(prefix: str, rows) -> dict:
    """Score cells as "<prefix>/<symbol>/<date>/k<k>/<field>"; floats as hex."""
    out: dict = {}
    for r in rows:
        base = f"{prefix}/{r['symbol']}/{r['date']}/k{r['k']}"
        if f"{base}/n_rows" in out:
            raise ValueError(f"duplicate score cell {base}")
        for f in CELL_FLOATS:
            if f in r:
                out[f"{base}/{f}"] = H.fhex(r[f])
        for f in CELL_INTS:
            if f in r:
                out[f"{base}/{f}"] = int(r[f])
    return out


def fit_values(record: dict, weights_flat_sha: str) -> dict:
    """A network's fit record: outcome, optimizer facts, every check, the validation cells, the configuration
    hash, the weights file hash and the hash of the saved weights rebuilt into the network."""
    out = H.flatten({k: record.get(k) for k in FIT_FIELDS})
    for c in record.get("checks", []):
        out.update(H.flatten({f: c.get(f) for f in CHECK_FIELDS}, f"check/{c['check']}"))
    out.update(cells_values("valid_scores", record.get("valid_scores", [])))
    out["config"] = hashlib.sha256(H.canonical_json(record.get("config")).encode()).hexdigest()
    out["weights_pt_sha256"] = record.get("weights", {}).get("sha256")
    out["weights_flat"] = weights_flat_sha
    return out


def ridge_values(record: dict, npz_path: Path) -> dict:
    """A linear reference's fit record and its saved weights."""
    out = H.flatten({"lambda": float(record["lambda"]), "valid_loss_by_lambda": record["valid_loss_by_lambda"],
                     "cond": record["cond"], "fallback": record["fallback"], "n_train": record["train"]["samples"]})
    with np.load(npz_path) as z:
        out["W"] = H.array_sha256(z["W"])
        out["b"] = H.array_sha256(z["b"])
    out["npz_sha256"] = H.sha256_file(Path(npz_path))
    out.update(cells_values("valid_scores", record["valid_scores"]))
    return out


def score_doc_values(doc: dict) -> dict:
    """A test-day score file: its cells, the stock-days the corporate-action file left out, the price ratios."""
    out = cells_values("cells", doc["scores"])
    out["corporate_actions/left_out"] = H.json_value(doc["corporate_actions"]["left_out"])
    out.update(H.flatten({"price_ratio": doc["price_ratio"]}))
    return out
