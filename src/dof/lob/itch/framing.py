"""Framing of the decompressed ITCH byte stream into messages, with per-type counts."""

from __future__ import annotations

import gzip
import os
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import BinaryIO, Iterator

from dof.lob.itch.spec import _CHR, _LEN_BY_BYTE


class ItchFormatError(Exception):
    """The byte stream is not well-formed ITCH 5.0 (length mismatch, truncation, bad gzip)."""


@dataclass
class ParseStats:
    messages_total: int = 0
    bytes_decompressed: int = 0
    counts_by_type: Counter = field(default_factory=Counter)    # every framed message, key = 1-char str
    tracked_by_type: Counter = field(default_factory=Counter)   # yielded messages
    skipped_untracked: int = 0          # per-stock messages whose locate is not tracked
    unknown_type_counts: Counter = field(default_factory=Counter)  # types not in MSG_LEN
    system_events: list[tuple[int, int, str]] = field(default_factory=list)   # (seq, ts_ns, code)
    stopped_by_limit: bool = False
    first_ts_ns: int | None = None
    last_ts_ns: int | None = None


def _open(path: str | os.PathLike[str]) -> BinaryIO:
    if os.fspath(path).endswith(".gz"):
        return gzip.open(path, "rb")  # type: ignore[return-value]
    return open(path, "rb")


def _read(fh: BinaryIO, size: int) -> bytes:
    try:
        return fh.read(size)
    except EOFError as exc:  # compressed stream ended early: the download is truncated
        raise ItchFormatError(f"truncated gzip: {exc}") from exc
    except (gzip.BadGzipFile, zlib.error) as exc:
        raise ItchFormatError(f"bad gzip: {exc}") from exc


def _frames(
    fh: BinaryIO, limit_bytes: int | None, chunk_size: int, stats: ParseStats
) -> Iterator[tuple[int, bytes, int, int]]:
    """Split the decompressed stream into messages; yield (seq, buf, start, length).

    ``buf[start:start+length]`` is the payload of a message whose type is in
    ``MSG_LEN`` and whose length matches it. Unknown types are counted and
    skipped here so that ``iter_messages`` and ``scan_counts`` share one
    framing path. A message incomplete at a chunk end is carried into the next
    chunk (``buf = tail + chunk``).

    Per-type counts are kept in flat local arrays and folded into ``stats``
    when the generator finishes, raises or is closed: a Counter increment per
    message costs about 55 s over the ~368M messages of one day.
    ``bytes_decompressed`` (bytes of fully framed messages) is likewise
    updated per chunk and at the end. Kept as one function on purpose: it is the
    innermost loop of every read.
    """
    counts = [0] * 256
    lens = _LEN_BY_BYTE
    seq = 0
    base = 0          # decompressed offset of buf[0]
    pos = 0           # offset in buf of the first byte not yet consumed
    tail = b""
    try:
        while True:
            chunk = _read(fh, chunk_size)
            if not chunk:
                break
            buf = tail + chunk if tail else chunk
            n = len(buf)
            pos = 0
            while pos < n:
                if limit_bytes is not None and base + pos + 2 > limit_bytes:
                    stats.stopped_by_limit = True
                    return
                if pos + 2 > n:
                    break
                length = (buf[pos] << 8) | buf[pos + 1]
                end = pos + 2 + length
                if limit_bytes is not None and base + end > limit_bytes:
                    stats.stopped_by_limit = True
                    return
                if end > n:
                    break
                start = pos + 2
                if length == 0:
                    raise ItchFormatError(f"empty message at seq={seq}")
                t = buf[start]
                counts[t] += 1
                want = lens[t]
                if want == 0:
                    pass  # type outside the spec: skip by its length prefix
                elif want != length:
                    raise ItchFormatError(
                        f"length mismatch type={_CHR[t]} seq={seq} got={length} want={want}"
                    )
                else:
                    yield seq, buf, start, length
                seq += 1
                pos = end
            tail = buf[pos:]
            base += pos
            pos = 0
            stats.bytes_decompressed = base
        if tail:
            raise ItchFormatError(f"truncated stream at seq={seq}: {len(tail)} trailing bytes")
    finally:
        stats.bytes_decompressed = base + pos
        stats.messages_total += seq
        for i, c in enumerate(counts):
            if c:
                stats.counts_by_type[_CHR[i]] += c
                if lens[i] == 0:
                    stats.unknown_type_counts[_CHR[i]] += c
