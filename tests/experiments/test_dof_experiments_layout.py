"""The shape of the experiment framework: importing it loads no torch and nothing outside the standard library, the
installed packages and ``dof``; no module patching; no SystemExit in library code; no file over 400 lines."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "dof" / "experiments"
SKIP = ("dof.experiments.__main__",)          # running it starts the command line


def _run(code: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code, *args], env=dict(os.environ), capture_output=True, text=True,
                          timeout=120)


_WALK = r"""
import importlib, pkgutil, sys
import dof.experiments
names = [m.name for m in pkgutil.walk_packages(dof.experiments.__path__, "dof.experiments.")]
for name in names:
    if name not in sys.argv[1:]:
        importlib.import_module(name)
assert "torch" not in sys.modules, "torch was imported"
print(len(names))
"""


def test_import_loads_no_torch() -> None:
    proc = _run(_WALK, *SKIP)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert int(proc.stdout.split()[-1]) > 10


_IMPORT_CHECK = r"""
import importlib, os, pkgutil, sys, sysconfig
from pathlib import Path
import dof, dof.experiments
for m in pkgutil.walk_packages(dof.experiments.__path__, "dof.experiments."):
    if m.name not in sys.argv[1:]:
        importlib.import_module(m.name)
paths = sysconfig.get_paths()
roots = [Path(paths[k]).resolve() for k in ("stdlib", "platstdlib", "purelib", "platlib") if k in paths]
roots.append(Path(list(dof.__path__)[0]).resolve().parent)
outside = sorted(n for n, m in list(sys.modules.items())
                 if getattr(m, "__file__", None) and os.path.isabs(m.__file__)
                 and not any(Path(m.__file__).resolve().is_relative_to(r) for r in roots))
assert not outside, outside
"""


def test_import_loads_only_stdlib_deps_and_dof() -> None:
    proc = _run(_IMPORT_CHECK, *SKIP)
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
    one. The allow-list is empty."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = _imported_names(tree)
        for node in ast.walk(tree):
            for t in _targets(node):
                if not (isinstance(node, ast.Call) or isinstance(t, (ast.Attribute, ast.Subscript))):
                    continue
                dotted = _chain(t)
                if dotted is not None and dotted.split(".")[0].split("[")[0] in imported:
                    found.append(f"{path.relative_to(SRC)}:{node.lineno}: {dotted}")
    assert not found, "\n".join(found)


def test_no_systemexit_in_library() -> None:
    found = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name in ("cli.py", "__main__.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found += [f"{path.relative_to(SRC)}:{n.lineno}" for n in ast.walk(tree)
                  if isinstance(n, ast.Name) and n.id == "SystemExit"]
    assert not found, found


def test_files_at_most_400_lines() -> None:
    long = {str(p.relative_to(SRC)): n for p in sorted(SRC.rglob("*.py"))
            if (n := len(p.read_text(encoding="utf-8").splitlines())) > 400}
    assert not long, long
