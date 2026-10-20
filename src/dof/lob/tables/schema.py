"""Column layout of the per-stock event and state tables (versioned by ``SCHEMA_VERSION``)."""

from __future__ import annotations

import pyarrow as pa

from dof.lob.book import LEVELS

EMIT_RULE = "row when top-10 (px,sz,ct,flags) changes after A/F/E/C/X/D/U; always after P/Q/H"
_REFS = ("A", "F", "E", "C", "X", "D", "U")   # types that act on a resting order

EVENTS_SCHEMA = pa.schema([
    ("date", pa.date32()),
    ("symbol", pa.dictionary(pa.int32(), pa.string())),
    ("seq", pa.int64()),
    ("ts_ns", pa.int64()),
    ("msg_type", pa.string()),
    ("order_ref", pa.uint64()),
    ("new_order_ref", pa.uint64()),
    ("side", pa.string()),
    ("price", pa.int64()),
    ("exec_price", pa.int64()),
    ("shares", pa.int64()),
    ("exec_shares", pa.int64()),
    ("printable", pa.string()),
    ("match_number", pa.uint64()),
    ("cross_type", pa.string()),
    ("trading_state", pa.string()),
    ("broken_match", pa.uint64()),
])


def _states_fields() -> list[tuple[str, pa.DataType]]:
    fields: list[tuple[str, pa.DataType]] = [("seq", pa.int64()), ("ts_ns", pa.int64())]
    for side in ("bid", "ask"):
        fields += [(f"{side}_px_{k}", pa.int64()) for k in range(1, LEVELS + 1)]
        fields += [(f"{side}_sz_{k}", pa.int64()) for k in range(1, LEVELS + 1)]
        fields += [(f"{side}_ct_{k}", pa.int32()) for k in range(1, LEVELS + 1)]
    fields.append(("flags", pa.int8()))
    return fields


STATES_SCHEMA = pa.schema(_states_fields())
assert len(STATES_SCHEMA) == 63

# array typecode per arrow type for the column buffers; strings use list[str].
_TYPECODE = {pa.int64(): "q", pa.uint64(): "Q", pa.int32(): "i", pa.int8(): "b", pa.date32(): "i"}


def symbol_stem(symbol: str) -> str:
    """The file-name stem of a ticker's two tables (spaces and slashes become underscores)."""
    return symbol.replace(" ", "_").replace("/", "_")
