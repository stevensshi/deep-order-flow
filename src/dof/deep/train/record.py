"""The fit record: the JSON document a fit writes last, so its existence means the fit has ended and its weights
are saved. A record that exists is reused; one made under another identity or another protocol document stops
the run (it is never overwritten). No torch import: the test gate and the ridge read records in torch-free
processes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dof.deep.errors import RecordConflict
from dof.deep.runtime.io import write_json_atomic


@dataclass(frozen=True)
class FitRecord:
    doc: dict
    folder: Path            # where the record lives; not written into it

    @classmethod
    def load(cls, path: Path) -> FitRecord:
        path = Path(path)
        return cls(json.loads(path.read_text(encoding="utf-8")), path.parent)

    @property
    def status(self) -> str:
        return self.doc["status"]

    @property
    def model(self) -> str:
        return self.doc["model"]

    @property
    def seed(self) -> int:
        return self.doc["seed"] if "seed" in self.doc else self.doc["seeds"]["init"]

    @property
    def weights_path(self) -> Path:
        return self.folder / self.doc["weights"]["file"]

    @property
    def weights_sha256(self) -> str | None:
        return self.doc.get("weights", {}).get("sha256")


def existing(path: Path, ident_sha: str, protocol_sha: str | None) -> FitRecord | None:
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("identity_sha256") != ident_sha or doc.get("protocol_sha256") != protocol_sha:
        raise RecordConflict(f"{path} was made under another identity or another protocol document; move it away "
                             "before running")
    return FitRecord(doc, path.parent)


def base_doc(*, out_name: str, fit: str, split_name: str, config: dict, protocol: str | None, spec: dict,
             placebo: bool) -> dict:
    """The keys every fit record starts with, finished or diverged."""
    return {"model": out_name, "fit": fit, "stage": split_name, "config": config, "protocol_sha256": protocol,
            "spec": spec, "placebo_shuffled_labels": placebo}


def diverged_doc(base: dict, *, error: str, checks: list, steps: int, resumes: list, started: str, finished: str,
                 added: dict) -> dict:
    """A fit whose loss is not finite: recorded, never scored on the test days. ``added``: identity, identity_sha256,
    seeds, device_settings."""
    return {"model": base["model"], "fit": base["fit"], "status": "diverged", "error": error, "stage": base["stage"],
            "config": base["config"], "protocol_sha256": base["protocol_sha256"], "spec": base["spec"],
            "placebo_shuffled_labels": base["placebo_shuffled_labels"], "checks": checks, "steps": steps,
            "resumes": resumes, "test_scored": False, "started": started, "finished": finished, **added}


def finished_doc(base: dict, *, seed: int, run: dict, train: dict, train_store, optimizer: dict, n_params: dict,
                 checks: list, best_check: int, best: float, stop: str | None, steps: int, per_pass: int, resumes: list,
                 patience: int, max_checks: int, expected: int, again: float, weights: dict, rows: list,
                 timing: dict, memory: dict, added: dict) -> dict:
    """``run``: provenance, torch, device. ``timing``: sec_fit, wall_seconds, started, finished. ``memory``:
    peak_rss_kb, gpu_peak_allocated_bytes, gpu_peak_reserved_bytes, gpu_memory_limit_bytes."""
    return {"model": base["model"], "fit": base["fit"], "status": "finished", "stage": base["stage"], "seed": seed,
            "config": base["config"], "protocol_sha256": base["protocol_sha256"], "spec": base["spec"],
            "placebo_shuffled_labels": base["placebo_shuffled_labels"], **run, "train": train, "train_store": train_store,
            "optimizer": optimizer, "n_params": n_params, "checks": checks, "best_check": best_check,
            "best_valid_loss": best, "stopped_by": stop, "steps": steps, "passes": steps / per_pass, "resumes": resumes,
            "patience_check": {"patience": patience, "max_checks": max_checks, "checks_expected": expected,
                               "ok": len(checks) == expected},
            "restore_check": {"valid_loss_of_saved_weights": again, "best_valid_loss": best, "abs_diff": abs(again - best)},
            "weights": weights, "valid_scores": rows, "test_scored": False,
            "sec_train_per_check": float(np.mean([c["sec_train"] for c in checks])),
            "sec_valid_per_check": float(np.mean([c["sec_valid"] for c in checks])),
            "sec_fit": timing["sec_fit"], "wall_seconds": timing["wall_seconds"], **memory,
            "started": timing["started"], "finished": timing["finished"], **added}


def progress(path: Path, doc: dict) -> None:
    write_json_atomic(path, doc)
