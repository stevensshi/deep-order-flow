"""Column buffers and Parquet writers of one stock's two tables; the rows written per message."""

from __future__ import annotations

import datetime
import json
import os
import time
from array import array
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dof.lob import SCHEMA_VERSION
from dof.lob.book import SENTINEL_ASK_PX, SENTINEL_BID_PX, Book
from dof.lob.tables.schema import _TYPECODE, EVENTS_SCHEMA, STATES_SCHEMA, symbol_stem
from dof.lob.units import PRICE_SCALE


class _TableWriter:
    """Column buffers for one table of one stock, flushed as a row group every ``flush_rows``."""

    def __init__(self, path: Path, schema: pa.Schema, metadata: dict[str, str], flush_rows: int,
                 symbol: str | None = None) -> None:
        self.path = path
        self.schema = schema.with_metadata(metadata)
        self.flush_rows = flush_rows
        self.symbol = symbol
        self.rows = 0
        self.pending = 0
        self.write_s = 0.0
        self.cols: list[array | list[str] | None] = []
        for f in schema:
            if pa.types.is_dictionary(f.type):
                self.cols.append(None)          # the symbol column: constant per file
            elif pa.types.is_string(f.type):
                self.cols.append([])
            else:
                self.cols.append(array(_TYPECODE[f.type]))
        self.writer = pq.ParquetWriter(str(path), self.schema, compression="zstd")

    def flush(self) -> None:
        if not self.pending:
            return
        t = time.perf_counter()
        n = self.pending
        arrays = []
        for f, col in zip(self.schema, self.cols):
            if col is None:
                arrays.append(pa.DictionaryArray.from_arrays(
                    pa.Array.from_buffers(pa.int32(), n, [None, pa.py_buffer(array("i", bytes(4 * n)))]),
                    pa.array([self.symbol], pa.string())))
            elif isinstance(col, list):
                arrays.append(pa.array(col, pa.string()))
            else:
                # Zero-copy view of the array.array buffer (copied into Parquet on write).
                arrays.append(pa.Array.from_buffers(f.type, n, [None, pa.py_buffer(col)]))
        self.writer.write_table(pa.Table.from_arrays(arrays, schema=self.schema))
        for i, col in enumerate(self.cols):
            if isinstance(col, list):
                self.cols[i] = []
            elif col is not None:
                self.cols[i] = array(col.typecode)
        self.pending = 0
        self.write_s += time.perf_counter() - t

    def row_done(self) -> None:
        self.rows += 1
        self.pending += 1
        if self.pending >= self.flush_rows:
            self.flush()

    def close(self) -> None:
        self.flush()
        t = time.perf_counter()
        self.writer.close()
        self.write_s += time.perf_counter() - t


class _SymbolSink:
    """The open writers and book for one tracked stock."""

    def __init__(self, symbol: str, locate: int, day_dir: Path, date: datetime.date, flush_rows: int) -> None:
        self.symbol = symbol
        self.book = Book(symbol, locate)
        stem = symbol_stem(symbol)
        self.final = {"events": day_dir / f"{stem}.events.parquet", "states": day_dir / f"{stem}.states.parquet"}
        meta = {"schema_version": SCHEMA_VERSION, "price_scale": str(PRICE_SCALE),
                "sentinel_bid_px": str(SENTINEL_BID_PX), "sentinel_ask_px": str(SENTINEL_ASK_PX),
                "symbol": symbol, "date": date.isoformat()}
        self.events = _TableWriter(Path(str(self.final["events"]) + ".tmp"), EVENTS_SCHEMA,
                                   {**meta, "table": "events"}, flush_rows, symbol)
        self.states = _TableWriter(Path(str(self.final["states"]) + ".tmp"), STATES_SCHEMA,
                                   {**meta, "table": "states"}, flush_rows)
        self.day = (date - datetime.date(1970, 1, 1)).days

    def close(self) -> None:
        self.events.close()
        self.states.close()

    @property
    def write_s(self) -> float:
        return self.events.write_s + self.states.write_s


def _event_row(sink: _SymbolSink, msg, target: list | None, state: str) -> None:
    """Append one event row; ``target`` is the resting order as it was BEFORE apply."""
    c = sink.events.cols
    t = msg.msg_type
    side, price, exec_price, shares, exec_shares = "", 0, 0, 0, 0
    if t == "A" or t == "F":
        side, price, shares = msg.side, msg.price, msg.shares
    elif t == "U":
        side = ("B" if target[0] else "S") if target is not None else ""
        price, shares = msg.price, msg.shares
    elif t in ("E", "C", "X", "D"):
        if target is not None:
            side, price = ("B" if target[0] else "S"), target[1]
        if t == "E":
            exec_price, exec_shares = price, msg.exec_shares
        elif t == "C":
            exec_price, exec_shares = msg.exec_price, msg.exec_shares
        elif t == "X":
            shares = msg.shares
        else:  # D: remaining shares come from the book
            shares = target[2] if target is not None else 0
    elif t == "P":
        side, price, exec_price, shares, exec_shares = msg.side, msg.price, msg.price, msg.shares, msg.shares
    elif t == "Q":
        price, exec_price, shares, exec_shares = msg.price, msg.price, msg.shares, msg.shares
    c[0].append(sink.day)
    c[2].append(msg.seq)
    c[3].append(msg.ts_ns)
    c[4].append(t)
    c[5].append(msg.order_ref)
    c[6].append(msg.new_order_ref)
    c[7].append(side)
    c[8].append(price)
    c[9].append(exec_price)
    c[10].append(shares)
    c[11].append(exec_shares)
    c[12].append(msg.printable)
    c[13].append(msg.match_number)
    c[14].append(msg.cross_type)
    c[15].append(state)
    c[16].append(msg.broken_match)
    sink.events.row_done()


def _state_row(sink: _SymbolSink, snap) -> None:
    c = sink.states.cols
    c[0].append(snap.seq)
    c[1].append(snap.ts_ns)
    i = 2
    for values in (snap.bid_px, snap.bid_sz, snap.bid_ct, snap.ask_px, snap.ask_sz, snap.ask_ct):
        for v in values:
            c[i].append(v)
            i += 1
    c[i].append(snap.flags)
    sink.states.row_done()


def _write_json_atomic(path: Path, obj: object) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
