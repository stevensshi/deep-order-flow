"""Test configuration.

- Tests marked ``golden`` run only when ``DOF_GOLDENS=1``; they need one BLAS thread and ``PYTHONHASHSEED=0``.
- ``load_golden`` reads a golden file and fails fast, naming the field, when the recipe or a library version
  differs from the recording.
- ``env_path`` reads a path from a ``DOF_*`` environment variable (None when unset).
- At the end of the session, every loaded module outside ``tests/`` must come from the standard library, the
  installed packages, or the source tree of ``dof``; anything else (a stray import from another checkout on
  the path) fails the session.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import sysconfig
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
GOLDEN_SETTINGS = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
ENV_FIELDS = ("python", "numpy", "pandas", "pyarrow", "scipy", "blas", "torch")
TEST_TOOLS = frozenset({"pytest", "_pytest", "pluggy", "iniconfig", "py"})


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


@pytest.fixture
def env_path():
    return _env_path


@pytest.fixture(scope="session")
def synth_run(tmp_path_factory):
    """A synthetic training run built once per session (``tests/deep_synth.py``); read only. A test that changes
    the cache copies it into its own ``tmp_path`` first."""
    import deep_synth
    return deep_synth.build_synth_run(tmp_path_factory.mktemp("synth_run"))


def pytest_collection_modifyitems(config, items):
    if os.environ.get("DOF_GOLDENS") == "1":
        return
    skip = pytest.mark.skip(reason="golden tests run only with DOF_GOLDENS=1")
    for item in items:
        if item.get_closest_marker("golden") is not None:
            item.add_marker(skip)


def pytest_runtest_setup(item):
    if item.get_closest_marker("golden") is None or os.environ.get("DOF_GOLDENS") != "1":
        return
    bad = [f"{k}={os.environ.get(k)!r} (needs {v!r})" for k, v in GOLDEN_SETTINGS.items() if os.environ.get(k) != v]
    if bad:
        pytest.fail("golden tests need one BLAS thread and a fixed hash seed: " + ", ".join(bad), pytrace=False)


def _running_env() -> dict:
    from recipes import hashing as H
    env = H.env_block()
    try:
        env["torch"] = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        env["torch"] = None
    return env


def read_golden(path: Path) -> dict:
    """A golden document, after checking that it can be compared in this environment."""
    from recipes import hashing as H
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("recipe_sha256") != H.recipe_sha256():
        pytest.fail(f"{path.name}: recipe changed since recording", pytrace=False)
    running = _running_env()
    for field in ENV_FIELDS:
        if field in doc.get("env", {}) and doc["env"][field] != running.get(field):
            pytest.fail(f"{path.name}: env.{field} is {doc['env'][field]!r} in the golden, {running.get(field)!r} here",
                        pytrace=False)
    return doc


@pytest.fixture
def load_golden():
    return read_golden


def outside_modules(skip_under=()) -> list[str]:
    """Loaded modules (with a file) that lie outside the standard library, the installed packages and the source
    tree that holds ``dof``."""
    paths = sysconfig.get_paths()
    roots = [Path(paths[k]).resolve() for k in ("stdlib", "platstdlib", "purelib", "platlib") if k in paths]
    dof = sys.modules.get("dof")
    if dof is not None:
        roots.append(Path(list(dof.__path__)[0]).resolve().parent)
    roots += [Path(p).resolve() for p in skip_under]
    bad = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        # A relative __file__ names no file that was imported: torch sets the placeholders "_ops.py" and
        # "_classes.py" on its torch.ops and torch.classes module objects.
        if f is None or not os.path.isabs(f) or name.split(".")[0] in TEST_TOOLS:
            continue
        p = Path(f).resolve()
        if not any(p.is_relative_to(r) for r in roots):
            bad.append(name)
    return sorted(bad)


def pytest_sessionfinish(session, exitstatus):
    bad = outside_modules(skip_under=(TESTS,))
    if bad:
        print(f"\nmodules loaded from outside the standard library, the installed packages and the dof source tree: {bad}")
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
