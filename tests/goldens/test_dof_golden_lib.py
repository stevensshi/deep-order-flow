"""Golden tests of the order-book library: the recipes run on ``dof`` through ``dof_backend`` and must reproduce
the recorded goldens bit for bit (see README.md in this folder)."""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from recipes import compare as C
from recipes import hashing as H
from recipes import lib_chain, lobster

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
COMMITTED = ("lib/synth.json", "lib/synth_halfday.json", "results_totals.json")
DATA_DIR_ENV = "DOF_GOLDEN_DATA_DIR"        # the real-data goldens, kept on the machine that holds the data
FULL_DIR_ENV = "DOF_GOLDEN_FULL_DIR"        # diagnostics only: the full-form goldens with clear values
BUILD_DAY = "2019-01-30"


def _backend():
    from dof_backend import dof_backend
    return dof_backend()


def _golden_files(env_path) -> list[Path]:
    files = sorted(DATA.rglob("*.json"))
    data_dir = env_path(DATA_DIR_ENV)
    if data_dir is not None:
        files += sorted(data_dir.rglob("*.json"))
    return files


def _real_golden(load_golden, env_path, root_env: str, name: str):
    data_dir, root = env_path(DATA_DIR_ENV), env_path(root_env)
    if data_dir is None or root is None:
        pytest.skip(f"needs {DATA_DIR_ENV} and {root_env}")
    return load_golden(data_dir / "lib" / f"{name}.json"), root


def _compare(golden: dict, values: dict, env_path) -> None:
    got = H.committed_form(values, golden["real_data"]) if golden["form"] == "committed" else values
    d = C.diff(golden["values"], got)
    full_dir = env_path(FULL_DIR_ENV)
    if d and full_dir is not None:
        # The full form holds clear values; keys that only the reference implementation produces are dropped.
        full = json.loads((full_dir / "lib" / f"{golden['golden']}.json").read_text(encoding="utf-8"))

        def keep(vals: dict) -> dict:
            return {k: v for k, v in vals.items() if not k.startswith(H.DROPPED_PREFIXES)}
        print("full-form differences:\n" + "\n".join(C.diff(keep(full["values"]), keep(values))[:50]))
    assert not d, f"{len(d)} differences:\n" + "\n".join(d[:50])


@pytest.mark.golden
def test_synth(tmp_path, load_golden, env_path) -> None:
    golden = load_golden(DATA / "lib" / "synth.json")
    values, _logs = lib_chain.run_synth(_backend(), tmp_path)
    _compare(golden, values, env_path)


@pytest.mark.golden
def test_synth_halfday(tmp_path, load_golden, env_path) -> None:
    golden = load_golden(DATA / "lib" / "synth_halfday.json")
    _compare(golden, lib_chain.run_halfday(_backend(), tmp_path), env_path)


@pytest.mark.golden
@pytest.mark.local
def test_real_smoke(load_golden, env_path) -> None:
    golden, root = _real_golden(load_golden, env_path, "DOF_GOLDEN_BOOK_ROOT", "real_smoke")
    _compare(golden, lib_chain.run_real(_backend(), root), env_path)


@pytest.mark.golden
@pytest.mark.local
def test_lobster_samples(load_golden, env_path) -> None:
    golden, root = _real_golden(load_golden, env_path, "DOF_GOLDEN_LOBSTER_DIR", "lobster_samples")
    values, _reports = lobster.run(_backend(), root)
    _compare(golden, values, env_path)


@pytest.mark.golden
@pytest.mark.local
@pytest.mark.slow
def test_real_build_outputs(tmp_path, load_golden, env_path) -> None:
    """Rebuild the two smoke stocks of one real day from the raw ITCH file; their tables must have the hashes
    the stored build log recorded."""
    golden, itch_dir = _real_golden(load_golden, env_path, "DOF_GOLDEN_ITCH_DIR", "real_build_outputs")
    b = _backend()
    res = b.build_day(itch_dir / "01302019.NASDAQ_ITCH50.gz", list(lib_chain.REAL_SYMBOLS), tmp_path,
                      datetime.date.fromisoformat(BUILD_DAY))
    log = json.loads(Path(res.log_path).read_text(encoding="utf-8"))
    values = lib_chain.build_log_values(log, BUILD_DAY, symbols=lib_chain.REAL_SYMBOLS)
    want = {k: v for k, v in golden["values"].items() if k in values}
    d = C.diff(want, H.committed_form(values, golden["real_data"]))
    assert not d, "\n".join(d[:50])


@pytest.mark.golden
def test_goldens_have_no_machine_paths(env_path) -> None:
    for path in _golden_files(env_path):
        H.assert_no_machine_paths(json.loads(path.read_text(encoding="utf-8")))


def test_recipe_hash_matches_goldens(env_path) -> None:
    present = sorted(p.relative_to(DATA).as_posix() for p in DATA.rglob("*.json"))
    assert present == sorted(COMMITTED), f"tests/goldens/data holds {present}, expected {sorted(COMMITTED)}"
    r = H.recipe_sha256()
    for path in _golden_files(env_path):
        got = json.loads(path.read_text(encoding="utf-8")).get("recipe_sha256")
        assert got == r, f"{path.name}: recipe changed since recording"


_IMPORT_CHECK = r"""
import sys, sysconfig
from pathlib import Path
import recipes.backends, recipes.compare, recipes.hashing, recipes.lib_chain, recipes.lobster
import recipes.synth_stream, recipes.train_values
here = Path(sys.argv[1]).resolve()
paths = sysconfig.get_paths()
roots = [Path(paths[k]).resolve() for k in ("stdlib", "platstdlib", "purelib", "platlib") if k in paths] + [here]
own = sorted(m for m in sys.modules if m == "dof" or m.startswith("dof."))
outside = sorted(n for n, m in list(sys.modules.items())
                 if getattr(m, "__file__", None) and not any(Path(m.__file__).resolve().is_relative_to(r) for r in roots))
assert not own and not outside, (own, outside)
"""


def test_recipes_import_nothing_of_dof() -> None:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(HERE)
    proc = subprocess.run([sys.executable, "-P", "-c", _IMPORT_CHECK, str(HERE)], env=env, capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
