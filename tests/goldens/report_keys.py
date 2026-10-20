"""Golden keys of the report tables: the one place that names each compared number of the five experiment reports.

The report golden test (``test_dof_golden_reports.py``) and the recorder of the reference implementation both build
their keys with this file, so the two sides name every value the same way. It imports only the standard library
and lies outside ``recipes/``; every report golden records its sha256 (``sha256_self``), and the golden test refuses
a golden that names another one.

A key is ``report/table/<row key parts>/column``; a float value also has the key ``<key>.hex`` with its
``float.hex``. A key exists exactly when the printed string is not empty, with two exceptions: the three
cells-without-R² tables of the main run have one key per model, "0" included, and the decay list of price ratios
outside the range has only the two keys ``decay/outside/count/value`` and ``decay/outside/lines/value``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

SCORES = ("stocks", "r2", "se")
OLDER = ("stocks", "diff", "se", "higher")
FITS = ("status", "checks", "best_check", "stopped_by", "steps", "passes", "best_valid_loss", "interruptions", "weights")
CHECKS = ("steps", "passes", "train_loss", "valid_loss", "new_best")
PART = ("stocks", "seed=*", "mean", "sd", "se")
SEED_SCORES = ("stocks", "seed=*", "mean", "sd", "min", "max", "seeds", "ensemble", "ensemble_stocks")
SEED_OLDER = ("stocks", "seed=*", "mean", "sd", "min", "max", "seeds", "ensembles")
DECAY_MODELS = ("CNN-LSTM-OF-lr3e-5-eps1e-7", "LSTM-MLP-OF")

# report -> table id -> compared column keys; "seed=*" stands for one column per seed of the report
TABLES: dict[str, dict[str, tuple[str, ...]]] = {
    "alldays": {
        "valid_scores": SCORES, "valid_older": OLDER,
        "test_primary_scores": SCORES, "test_primary_older": OLDER,
        "test_secondary_scores": SCORES, "test_secondary_older": OLDER,
        "checks": (*CHECKS, "median_dw"),
        "no_r2_valid": ("cells",), "no_r2_test_primary": ("cells",), "no_r2_test_secondary": ("cells",),
    },
    "adjacent": {
        "fits": FITS, "checks": CHECKS,
        "valid": PART, "afternoon": PART, "far": PART,
        "pair": ("value",),
        "reference": SCORES,
    },
    "decay": {
        "table": ("days_since", "months", "i_stock_days", "i_stocks", "i_r2", "i_se", "ii_stock_days", "ii_stocks",
                  "ii_r2", "ii_se", "diff_stocks", "diff", "diff_se"),
        "days": ("held", "scored", "inside", "below", "above", "reused", "group", "sample",
                 *(f"no_r2:{m}" for m in DECAY_MODELS)),
        "outside": ("value",),
    },
    "earlier": {
        "test_scores": SCORES, "valid_scores": SCORES,
        "rescore": ("stocks", "r2", "again_stocks", "again_r2", "diff", "cells", "equal"),
    },
    "seeds": {
        "seed_valid": SEED_SCORES, "seed_test_primary": SEED_SCORES, "seed_test_secondary": SEED_SCORES,
        "seed_older_valid": SEED_OLDER, "seed_older_test_primary": SEED_OLDER, "seed_older_test_secondary": SEED_OLDER,
        "fits": FITS, "checks": CHECKS,
        "finite": ("files", "cells", "finite", "zero_variance", "not_finite", "other"),
        "rule": ("value",),
    },
}


def key(report: str, table: str, row: tuple[str, ...], column: str) -> str:
    return "/".join((report, table, *row, column))


def days_label(dates) -> str:
    return ",".join(sorted(dates))


def horizon_label(ks) -> str:
    return "k=1..10" if ks is None else "k=" + ",".join(str(k) for k in ks)


def seed_label(s) -> str:
    return f"seed={s}"


def expand(columns: tuple[str, ...], seeds) -> tuple[str, ...]:
    """The compared columns with "seed=*" replaced by one column per seed, in the order of ``seeds``."""
    out: list[str] = []
    for c in columns:
        out += [seed_label(s) for s in seeds] if c == "seed=*" else [c]
    return tuple(out)


def sha256_self() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
