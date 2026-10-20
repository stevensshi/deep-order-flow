"""The shape of the training package: what importing it loads, no module patching, no SystemExit in library code,
no machine fact in the configuration files."""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "dof" / "deep"
CONFIGS = ROOT / "configs"
TORCH_FREE = ("dof.deep", "dof.deep.config", "dof.deep.cache", "dof.deep.linear.ridge", "dof.deep.data.stream",
              "dof.deep.data.eval", "dof.deep.evaluate.gate")
# The only assignments to attributes of imported modules: the device's cuDNN and TF32 switches (the settings of
# torch itself), and the BLAS thread variables of the ridge's worker processes.
ALLOWED_TARGETS = {"runtime/device.py": {"torch.backends.cudnn.benchmark", "torch.backends.cudnn.allow_tf32",
                                         "torch.backends.cuda.matmul.allow_tf32"},
                   "linear/ridge.py": {"os.environ[]"}}
MACHINE_PATH = re.compile(r"/(home|mnt|Users)/")
DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], env=dict(os.environ), capture_output=True, text=True, timeout=120)


def test_import_dof_deep_loads_no_torch() -> None:
    code = f"import sys\nimport {', '.join(TORCH_FREE)}\nassert 'torch' not in sys.modules, 'torch was imported'"
    proc = _run(code)
    assert proc.returncode == 0, proc.stdout + proc.stderr


_IMPORT_CHECK = r"""
import importlib, os, sys, sysconfig
from pathlib import Path
for name in sys.argv[1:]:
    importlib.import_module(name)
import dof
paths = sysconfig.get_paths()
roots = [Path(paths[k]).resolve() for k in ("stdlib", "platstdlib", "purelib", "platlib") if k in paths]
roots.append(Path(list(dof.__path__)[0]).resolve().parent)
# a relative __file__ names no imported file (torch's torch.ops and torch.classes carry such placeholders)
outside = sorted(n for n, m in list(sys.modules.items())
                 if getattr(m, "__file__", None) and os.path.isabs(m.__file__)
                 and not any(Path(m.__file__).resolve().is_relative_to(r) for r in roots))
assert not outside, outside
"""


def test_import_dof_deep_loads_only_stdlib_deps_and_dof() -> None:
    modules = list(TORCH_FREE)
    if importlib.util.find_spec("torch") is not None:
        modules += ["dof.deep.models", "dof.deep.train.fit"]
    proc = subprocess.run([sys.executable, "-c", _IMPORT_CHECK, *modules], env=dict(os.environ), capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _imported_names(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.asname or a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names |= {a.asname or a.name for a in node.names}
    return names


def _chain(node) -> str | None:
    """"a.b.c" for an attribute chain rooted at a name; "a.b[]" for a subscript of one; None otherwise."""
    if isinstance(node, ast.Subscript):
        base = _chain(node.value)
        return None if base is None else base + "[]"
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join([node.id, *reversed(parts)])


def _targets(node) -> list:
    if isinstance(node, ast.Assign):
        out = []
        for t in node.targets:
            out += list(t.elts) if isinstance(t, (ast.Tuple, ast.List)) else [t]
        return out
    if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        return [node.target]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("setattr", "delattr")
            and node.args):
        return [node.args[0]]
    return []


def test_no_module_patching() -> None:
    """No assignment to an attribute (or an item of an attribute) of an imported name, and no setattr or delattr on
    one, outside the allow-list."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = _imported_names(tree)
        for node in ast.walk(tree):
            for t in _targets(node):
                is_call = isinstance(node, ast.Call)
                if not (is_call or isinstance(t, (ast.Attribute, ast.Subscript))):
                    continue
                dotted = _chain(t)
                if dotted is None or dotted.split(".")[0].split("[")[0] not in imported:
                    continue
                if not is_call and dotted in ALLOWED_TARGETS.get(rel, set()):
                    continue
                found.append(f"{rel}:{node.lineno}: {dotted}")
    assert not found, "\n".join(found)


def test_no_systemexit_in_library() -> None:
    """Library code raises the exceptions of dof.deep.errors; only the command lines decide exit codes."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name in ("cli.py", "__main__.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found += [f"{path.relative_to(SRC)}:{n.lineno}" for n in ast.walk(tree)
                  if isinstance(n, ast.Name) and n.id == "SystemExit"]
    assert not found, found


def _strings(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)
    elif isinstance(o, str):
        yield o


def test_configs_hold_no_machine_fact() -> None:
    files = sorted(CONFIGS.glob("*.toml"))
    assert {p.name for p in files} >= {"models.toml", "smoke.toml", "alldays.toml", "local.example.toml"}
    for path in files:
        with open(path, "rb") as fh:
            doc = tomllib.load(fh)
        if path.name != "local.example.toml":
            assert "machine" not in doc, f"{path.name} holds a [machine] table"
        for s in _strings(doc):
            assert not s.startswith("/") and not MACHINE_PATH.search(s) and not DRIVE.search(s), (path.name, s)
    with open(CONFIGS / "local.example.toml", "rb") as fh:
        machine = tomllib.load(fh)["machine"]
    paths = [machine[k] for k in ("book_root", "disk_guard_path") if k in machine]
    paths += [v for c in machine.get("cache", {}).values() for v in c.values()]
    assert paths and all(p.startswith("path/to/") for p in paths), paths
