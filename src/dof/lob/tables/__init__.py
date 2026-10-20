"""Export one ITCH day to per-stock Parquet: an event table and a 10-level state table.

The two tables are linked by ``seq`` (the message's ordinal in the source
file). Columns, dtypes and the ``build_log.json`` layout are fixed by the ``schema`` module
and versioned by ``SCHEMA_VERSION``.

Failure policy:
- The day directory lock is taken OUTSIDE the try block. If another process
  holds it we raise without writing or deleting anything — the lock is theirs.
- Rejections (already built, input not verified complete in MANIFEST.csv,
  sha256 mismatch) raise before any stream is read: no log, no deletions.
- Once the stream is being read, any exception (a truncated download raises
  ``ItchFormatError``) removes every partial ``*.parquet.tmp``, leaves no final
  Parquet from this run, writes ``build_log.json`` with ``status="failed"`` and
  re-raises. A ``complete`` log is never overwritten by a ``failed`` one
  (with ``overwrite=True`` the old log is deleted only after the checks pass).
- A stream read to its end is ``complete`` only if no byte limit stopped it and
  the end-of-messages event (S "C") was seen; otherwise the log says
  ``partial`` and ``load_states`` refuses the day.

This module never seeds or reduces book levels from outside data: the book is
built from the ITCH messages alone.
"""

from dof.lob.tables.build import BuildResult, build_day
from dof.lob.tables.load import InvalidStockDay, load_states
from dof.lob.tables.manifest import load_symbols_file, read_manifest
from dof.lob.tables.schema import EMIT_RULE, EVENTS_SCHEMA, STATES_SCHEMA

__all__ = ["BuildResult", "EMIT_RULE", "EVENTS_SCHEMA", "InvalidStockDay", "STATES_SCHEMA", "build_day",
           "load_states", "load_symbols_file", "read_manifest"]
