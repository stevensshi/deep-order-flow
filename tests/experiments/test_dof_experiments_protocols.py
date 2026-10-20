"""The edited protocol copies in protocols/: each names the sha256 its experiment spec pins for the original, and the
sha256 of its own text below the marker line; the index protocols/README.md lists the same two hashes. The byte copy of
the corporate-action file is the file the specs and configs/alldays.toml pin."""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path

from dof.experiments.specs import CORPORATE_ACTIONS, PROTOCOL_ADJACENT, PROTOCOL_DECAY, PROTOCOL_MAIN, PROTOCOL_SEEDS

ROOT = Path(__file__).resolve().parents[2]
PROTOCOLS = ROOT / "protocols"
MARKER = b"<!-- document text below -->\n"
CORPORATE_ACTION_FILE = ROOT / "configs" / "protocol" / "corporate_actions_20251210_20260612.csv"
COPIES = {"preregistration.md": PROTOCOL_MAIN, "preregistration_seeds.md": PROTOCOL_SEEDS,
          "preregistration_adjacent.md": PROTOCOL_ADJACENT, "preregistration_decay.md": PROTOCOL_DECAY}
ORIGINAL = re.compile(r"^> - sha256 of the original: `([0-9a-f]{64})`$", re.MULTILINE)
BODY = re.compile(r"^> - sha256 of the text below the marker line: `([0-9a-f]{64})`$", re.MULTILINE)
ROW = re.compile(r"^\| `([a-z_]+\.md)` \| [^|]+ \| `([0-9a-f]{64})` \| `([0-9a-f]{64})` \|$", re.MULTILINE)


def _split(name: str) -> tuple[str, bytes]:
    raw = (PROTOCOLS / name).read_bytes()
    assert raw.count(MARKER) == 1, f"{name}: the marker line must occur once"
    note, body = raw.split(MARKER)
    return note.decode("utf-8"), body


def test_protocol_copies_name_their_pins() -> None:
    for name, pin in COPIES.items():
        note, body = _split(name)
        assert ORIGINAL.findall(note) == [pin], name
        assert BODY.findall(note) == [hashlib.sha256(body).hexdigest()], name


def test_protocol_index_matches_copies() -> None:
    rows = {m[0]: (m[1], m[2]) for m in ROW.findall((PROTOCOLS / "README.md").read_text(encoding="utf-8"))}
    assert set(rows) == set(COPIES)
    for name, pin in COPIES.items():
        assert rows[name] == (pin, hashlib.sha256(_split(name)[1]).hexdigest()), name


def test_protocol_folder_holds_only_the_copies() -> None:
    assert sorted(p.name for p in PROTOCOLS.iterdir()) == sorted([*COPIES, "README.md"])


def test_corporate_action_file_is_the_pinned_file() -> None:
    with open(ROOT / "configs" / "alldays.toml", "rb") as fh:
        pin = tomllib.load(fh)["protocol"]["corporate_actions_sha256"]
    got = hashlib.sha256(CORPORATE_ACTION_FILE.read_bytes()).hexdigest()
    assert got == pin == CORPORATE_ACTIONS
