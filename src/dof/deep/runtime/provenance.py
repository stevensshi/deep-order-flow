"""The code behind an artifact: the library's code identity plus the versions of the training dependencies."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path
from typing import Mapping

from dof.lob.provenance import code_identity


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def provenance(extra: Mapping[str, Path] | None = None) -> dict:
    """``code_identity`` (package version, source hashes, git state, library versions) with the torch and
    zstandard versions; the CUDA and cuDNN versions only when torch is already imported (it is never imported
    here)."""
    out = code_identity(extra_files=extra)
    out["torch"] = _version("torch")
    out["zstandard"] = _version("zstandard")
    torch = sys.modules.get("torch")
    if torch is not None:
        out["cuda"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version() or None
    return out
