"""Inputs of a build: the download manifest and the list of tickers."""

from __future__ import annotations

import csv
from pathlib import Path


def read_manifest(manifest_csv: Path) -> dict[str, dict[str, str]]:
    """File name -> row of MANIFEST.csv, keeping the LAST ``gzip_test == "ok"`` row.

    The download script appends a row per attempt, so a failed attempt may be
    followed by a good one. A file with no ok row is absent.
    """
    rows: dict[str, dict[str, str]] = {}
    with open(manifest_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("gzip_test") == "ok":
                rows[row["file"]] = row
    return rows


def load_symbols_file(path: Path) -> list[str]:
    """Tickers, in file order, from a CSV whose header has a ``ticker`` column."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "ticker" not in reader.fieldnames:
            raise ValueError(f"{path}: no 'ticker' column in header")
        return [row["ticker"].strip() for row in reader if row["ticker"].strip()]
