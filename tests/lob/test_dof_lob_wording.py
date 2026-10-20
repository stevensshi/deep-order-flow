"""Scans over every file of the repository: no machine path anywhere; optionally, no word of a pattern list that
is kept outside the repository (``DOF_PRIVATE_PATTERNS_FILE``, one regular expression per line). The pattern scan
skips one file, the byte copy of the corporate-action file, and only while its sha256 equals the pin in
``configs/alldays.toml``: its SEC quotes use legal terms that are on the pattern list."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve()
# A drive letter is a single letter: "differences:\n" in source code is not a path.
MACHINE_PATHS = (re.compile(r"/(home|mnt|Users)/"), re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\"))
PATTERNS_ENV = "DOF_PRIVATE_PATTERNS_FILE"
SKIP_DIRS = frozenset({".git", "__pycache__", ".venv", ".pytest_cache", "build", "dist"})
PINNED_EXEMPT = "configs/protocol/corporate_actions_20251210_20260612.csv"   # skipped by the pattern scan while pinned


def _files() -> list[Path]:
    """The files git would track (tracked or untracked, ignored ones left out); every file outside the skipped
    folders when the tree is not a git work tree."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-co", "--exclude-standard", "-z"],
                             capture_output=True, check=True, timeout=60, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
        return [ROOT / n for n in out.stdout.decode("utf-8").split("\0") if n]
    except (OSError, subprocess.SubprocessError):
        return [p for p in ROOT.rglob("*") if p.is_file() and not set(p.relative_to(ROOT).parts) & SKIP_DIRS]


def _hits(patterns, skip: frozenset = frozenset()) -> list[str]:
    hits = []
    for p in _files():
        if not p.is_file() or p.resolve() == SELF or p.resolve() in skip:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            hits += [f"{p.relative_to(ROOT)}:{i}: /{rx.pattern}/" for rx in patterns if rx.search(line)]
    return hits


def _pinned_exempt() -> frozenset:
    """The corporate-action file while its sha256 equals ``protocol.corporate_actions_sha256`` of
    ``configs/alldays.toml``; empty otherwise, so a changed file is scanned like any other."""
    path = ROOT / PINNED_EXEMPT
    with open(ROOT / "configs" / "alldays.toml", "rb") as fh:
        pin = tomllib.load(fh)["protocol"]["corporate_actions_sha256"]
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == pin:
        return frozenset({path.resolve()})
    return frozenset()


def test_no_machine_paths() -> None:
    hits = _hits(MACHINE_PATHS)
    assert not hits, "\n".join(hits[:50])


@pytest.mark.local
def test_private_patterns() -> None:
    path = os.environ.get(PATTERNS_ENV)
    if not path:
        pytest.skip(f"{PATTERNS_ENV} is not set")
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    hits = _hits([re.compile(ln, re.IGNORECASE) for ln in lines if ln.strip()], skip=_pinned_exempt())
    assert not hits, "\n".join(hits[:50])
