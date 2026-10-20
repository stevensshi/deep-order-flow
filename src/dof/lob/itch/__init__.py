"""Nasdaq TotalView-ITCH 5.0 parser: framing, per-stock locate filtering, decoding.

Spec: ``SPEC_URL`` (revision of April 28, 2023). Every offset used here is a payload
offset (the 2-byte length prefix excluded) taken from that spec; the R-message ETP
flag is read at @33, where the spec puts it.

Failure policy: a known type with a wrong length means the framing has slipped,
so everything after it is untrustworthy — ``ItchFormatError`` (fail closed).
A type the spec does not know (a future addition) is skipped by its length
prefix and counted, never an error. A truncated download (gzip ``EOFError``, a
partial trailing message) is an ``ItchFormatError``; messages yielded before it
stay yielded, and the caller decides the build failed from the exception.
"""

from dof.lob.itch.decode import LocateFilter, Message, iter_messages, scan_counts
from dof.lob.itch.filenames import parse_itch_date
from dof.lob.itch.framing import ItchFormatError, ParseStats
from dof.lob.itch.spec import BOOK_TYPES, DEFAULT_CHUNK, EVENT_TYPES, MSG_LEN, SPEC_URL

__all__ = ["SPEC_URL", "MSG_LEN", "BOOK_TYPES", "EVENT_TYPES", "DEFAULT_CHUNK", "ItchFormatError", "Message",
           "LocateFilter", "ParseStats", "iter_messages", "scan_counts", "parse_itch_date"]
