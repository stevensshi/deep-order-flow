"""Code identity of an artifact: package version, source hashes, git state and library versions.

Keys never hold an absolute path: package files are keyed by their path inside the package's parent folder
("dof/lob/book.py"), extra files by the label the caller gives them. Git failures give None values, never raise.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path
from typing import Mapping

import dof


def sha256_file(path: Path, bufsize: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(bufsize)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True,
                             timeout=30, check=True, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _blas() -> dict:
    import numpy as np
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    except (KeyError, TypeError, ValueError):
        return {"name": None, "version": None}
    return {"name": blas.get("name"), "version": blas.get("version")}


def package_sources(package_dir: Path | None = None) -> dict[str, str]:
    """sha256 of every .py file of the package, keyed by its path relative to the package's parent folder."""
    pkg = Path(package_dir) if package_dir is not None else Path(dof.__file__).resolve().parent
    files = sorted(p for p in pkg.rglob("*.py") if "__pycache__" not in p.relative_to(pkg).parts)
    return {p.relative_to(pkg.parent).as_posix(): sha256_file(p) for p in files}


def code_identity(repo_root: Path | None = None, extra_files: Mapping[str, Path] | None = None) -> dict[str, object]:
    """``repo_root``: where git is asked (default: the package's own folder, so an installed copy outside a
    work tree has None git fields). ``extra_files``: label -> file, hashed under its label (scripts outside the
    package that made the artifact)."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent
    head = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    status_lines = None if status is None else len([ln for ln in status.splitlines() if ln.strip()])
    sources = package_sources()
    for label, path in (extra_files or {}).items():
        sources[str(label)] = sha256_file(Path(path))
    return {
        "version": dof.__version__,
        "git_head": head.strip() if head else None,
        "git_dirty": None if status_lines is None else status_lines > 0,
        "git_status_lines": status_lines,
        "source_sha256": sources,
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
        "pyarrow": _version("pyarrow"),
        "scipy": _version("scipy"),
        "blas": _blas(),
    }
