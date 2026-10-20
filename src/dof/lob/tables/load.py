"""Loading of one stock-day: the state table joined with the event columns it needs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dof.lob.tables.schema import symbol_stem


class InvalidStockDay(Exception):
    """The stock-day is not ``valid`` in build_log.json; the feature layer never processes it."""

    def __init__(self, symbol: str, reason: str) -> None:
        super().__init__(f"{symbol}: {reason}")
        self.symbol = symbol
        self.reason = reason


def load_states(day_dir: Path, symbol: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """State table + event ``msg_type``/``trading_state`` (joined on seq), sorted by seq.

    ``columns`` (optional) limits the state-table columns read (``seq`` is always
    read); ``None`` reads them all. Callers of the flow branch pass ``FLOW_COLUMNS``,
    so a 9.5M-row stock-day does not carry the 20 unused count columns.

    Raises ``InvalidStockDay`` unless the day's build is complete, the stock is
    valid in it, and both of the stock's tables exist.
    """
    day_dir = Path(day_dir)
    log = json.loads((day_dir / "build_log.json").read_text(encoding="utf-8"))
    if log.get("status") != "complete":
        raise InvalidStockDay(symbol, f"build status={log.get('status')}")
    entry = log.get("symbols", {}).get(symbol)
    if entry is None:
        raise InvalidStockDay(symbol, "missing" if symbol in log.get("missing_symbols", []) else "not in build_log")
    if entry.get("status") != "valid":
        raise InvalidStockDay(symbol, f"status={entry.get('status')} first_error={entry.get('first_error')}")
    stem = symbol_stem(symbol)
    if not ((day_dir / f"{stem}.states.parquet").is_file() and (day_dir / f"{stem}.events.parquet").is_file()):
        raise InvalidStockDay(symbol, "files missing")
    # Memory: a to_pandas() + merge() holds the arrow table, the pandas frame and a merged
    # copy at once (about 2.7x the frame; one 9.5M-row stock-day passed 9 GB). So the
    # state columns are read one at a time and the two event columns are attached by seq
    # lookup (no merge copy).
    want = None
    if columns is not None:
        want = list(dict.fromkeys(["seq", *[c for c in columns if c not in ("msg_type", "trading_state")]]))
    # One column at a time, releasing arrow's pool after each: peak ~ frame + one column
    # (a whole-table read kept the decoded arrow buffers cached next to the pandas copy).
    pf = pq.ParquetFile(day_dir / f"{stem}.states.parquet")
    names = want if want is not None else pf.schema_arrow.names
    pool = pa.default_memory_pool()
    data = {}
    for c in names:
        col = pf.read(columns=[c]).column(0)
        # Own the memory: to_numpy() can be a zero-copy view into arrow's pool, which then
        # keeps the whole frame's pages after the frame is freed.
        arr = np.array(col.to_numpy(), copy=True)
        del col
        # Size columns are int64 in the schema but always far below 2^31 shares; holding
        # them as int32 is lossless (checked per column) and saves 80 B per row.
        if "_sz_" in c and arr.dtype == np.int64 and len(arr) and arr.max() < 2**31 and arr.min() >= 0:
            arr = arr.astype(np.int32)
        data[c] = arr
        pool.release_unused()
    states = pd.DataFrame(data, copy=False)
    del data
    if len(states) > 1 and not (np.diff(states["seq"].to_numpy()) > 0).all():
        states = states.sort_values("seq", kind="stable").reset_index(drop=True)
    events = pq.read_table(day_dir / f"{stem}.events.parquet", columns=["seq", "msg_type", "trading_state"])
    pool.release_unused()
    ev_seq = events.column("seq").to_numpy()
    st_seq = states["seq"].to_numpy()
    idx = np.searchsorted(ev_seq, st_seq)
    if len(st_seq) and (idx.max() >= len(ev_seq) or not (ev_seq[idx] == st_seq).all()):
        raise ValueError(f"{symbol}: state rows without a matching event seq in {day_dir}")
    take = pa.array(idx)
    for col in ("msg_type", "trading_state"):   # categorical: one byte-sized code per row, not one object
        if columns is not None and col not in columns:
            continue
        states[col] = pd.Categorical(events.column(col).take(take).to_pandas())
    del events, take
    pool.release_unused()
    return states
