from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import dof

ROOT = Path(__file__).resolve().parents[2]


def test_dof_is_regular_package_from_src() -> None:
    assert Path(dof.__file__).as_posix().endswith("src/dof/__init__.py")


_IMPORT_CHECK = r"""
import sys, sysconfig
from pathlib import Path
import dof.lob, dof.lob.features, dof.lob.tables, dof.lob.stats, dof.lob.validation
import dof
paths = sysconfig.get_paths()
roots = [Path(paths[k]).resolve() for k in ("stdlib", "platstdlib", "purelib", "platlib") if k in paths]
roots.append(Path(list(dof.__path__)[0]).resolve().parent)
outside = sorted(n for n, m in list(sys.modules.items())
                 if getattr(m, "__file__", None) and not any(Path(m.__file__).resolve().is_relative_to(r) for r in roots))
assert not outside, outside
"""


def test_import_dof_loads_only_stdlib_deps_and_dof() -> None:
    """Importing the library loads nothing but the standard library, installed packages and dof itself."""
    proc = subprocess.run([sys.executable, "-c", _IMPORT_CHECK], env=dict(os.environ), capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_features_do_not_import_pyarrow() -> None:
    """The feature layer works on frames in memory; only the table layer reads Parquet. pandas itself imports
    pyarrow when it is installed, so the check blocks pyarrow and requires the feature layer to import without it."""
    code = "import sys; sys.modules['pyarrow'] = None; import dof.lob.features"
    proc = subprocess.run([sys.executable, "-c", code], env=dict(os.environ), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_version_matches_pyproject() -> None:
    with open(ROOT / "pyproject.toml", "rb") as fh:
        assert tomllib.load(fh)["project"]["version"] == dof.__version__
