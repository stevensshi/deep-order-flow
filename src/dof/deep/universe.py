"""Which stocks a run reads: the universe file, reused tickers, and the corporate-action file of the test days.

A stock is left out on a test day when the corporate-action file has, for its ticker, a confirmed event of an
excluding type (a split, a ticker change, a reused ticker) with an effective date on or before that day. The
effective date is the first trading day on the new scale. The file is read only when the test days are scored,
and only when its sha256 is the one the configuration pins: another file would change the test set.
"""

from __future__ import annotations

import csv
import datetime

from dof.deep.config import RunConfig, Split
from dof.deep.errors import ConfigError, TestDayRefused
from dof.lob.provenance import sha256_file
from dof.lob.tables import load_symbols_file


def universe(cfg: RunConfig, split: Split) -> list[str]:
    """The tickers of the universe file, in its order (only the split's own list, if it has one)."""
    try:
        tickers = load_symbols_file(cfg.universe)
    except FileNotFoundError:
        raise ConfigError(f"data.universe: {cfg.universe} does not exist") from None
    except ValueError as exc:
        raise ConfigError(f"data.universe: {exc}") from None
    if split.symbols is not None:
        missing = [s for s in split.symbols if s not in tickers]
        if missing:
            raise ConfigError(f"split {split.name}: {missing} are not in {cfg.universe.name}")
        tickers = [s for s in tickers if s in split.symbols]
    return tickers


def reused(split: Split, symbol: str, date: str) -> bool:
    """The ticker belongs to another issuer on ``date``: it is left out from that date on."""
    since = dict(split.reused_from).get(symbol)
    return since is not None and date >= since


def corporate_actions(cfg: RunConfig) -> dict:
    """The configuration's corporate-action file: its sha256 and its rows. Refuses when the file is missing or is
    not the pinned file; a missing column, or a confirmed excluding event without a readable effective date (such
    a row can neither be applied nor be ignored), is a configuration error."""
    p = cfg.protocol
    path = p.corporate_actions
    if not path.exists():
        raise TestDayRefused(f"{path.name} missing: the test days are not scored without the corporate-action file")
    sha = sha256_file(path)
    if sha != p.corporate_actions_sha256:
        raise TestDayRefused(f"{path.name} has the sha256 {sha}, not {p.corporate_actions_sha256}: it is not the "
                             "corporate-action file of the configuration")
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows = [{k: (v or "").strip() for k, v in r.items() if k is not None} for r in reader]
    missing = [c for c in p.ca_columns if c not in header]
    if missing:
        raise ConfigError(f"{path.name}: columns {missing} are missing (the file has {header}); set "
                          "protocol.ca_columns to the file's names")
    ticker, event, effective, status = p.ca_columns
    for r in rows:
        r["applies"] = r[status].lower() == p.ca_confirmed and r[event].lower() in p.ca_excluding
        if r["applies"]:
            try:
                datetime.date.fromisoformat(r[effective])
            except ValueError:
                raise ConfigError(f"{path.name}: the confirmed {r[event]} of {r[ticker]} has no date of the form "
                                  f"YYYY-MM-DD ({r[effective]!r})") from None
    return {"file": path.name, "sha256": sha, "columns": header, "rows": rows}


def ca_excluded(cfg: RunConfig, split: Split, ca: dict) -> dict[tuple[str, str], dict]:
    """(ticker, test day) -> the event that leaves the stock out on that day: a confirmed event of an excluding
    type whose effective date is on or before the test day (the first such event, by date)."""
    ticker, _event, effective, _status = cfg.protocol.ca_columns
    out: dict[tuple[str, str], dict] = {}
    for r in sorted((r for r in ca["rows"] if r["applies"]), key=lambda r: r[effective]):
        for d in split.test_days:
            if r[effective] <= d:
                out.setdefault((r[ticker], d), {k: r[k] for k in cfg.protocol.ca_columns})
    return out
