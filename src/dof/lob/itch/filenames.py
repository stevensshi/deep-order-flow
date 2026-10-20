"""Trading dates from ITCH file names."""

from __future__ import annotations

import datetime
import os
import re
from pathlib import Path


_DATE_EMI = re.compile(r"^(\d{2})(\d{2})(\d{4})\.NASDAQ_ITCH50(\.gz)?$")
_DATE_S = re.compile(r"^S(\d{2})(\d{2})(\d{2})-v50\.txt(\.gz)?$")


def parse_itch_date(path: str | os.PathLike[str]) -> datetime.date:
    """Trading date from an ITCH file name (EMI ``MMDDYYYY.NASDAQ_ITCH50`` or ``SMMDDYY-v50.txt``)."""
    name = Path(path).name
    m = _DATE_EMI.match(name)
    if m:
        return datetime.date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
    m = _DATE_S.match(name)
    if m:
        return datetime.date(2000 + int(m.group(3)), int(m.group(1)), int(m.group(2)))
    raise ValueError(f"not an ITCH 5.0 file name: {name!r}")
