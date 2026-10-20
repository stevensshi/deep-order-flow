from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import dof
from dof.lob.provenance import code_identity, package_sources


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_provenance_extra_files(tmp_path: Path) -> None:
    f = tmp_path / "script.py"
    f.write_text("print('x')\n")
    ident = code_identity(extra_files={"script.py": f})
    assert ident["source_sha256"]["script.py"] == _sha(f)
    assert "dof/lob/tables/build.py" in ident["source_sha256"]
    # git unavailable -> None values, no exception
    bare = code_identity(repo_root=tmp_path)
    assert bare["git_head"] is None and bare["git_dirty"] is None and bare["git_status_lines"] is None


def test_package_sources_independent_of_location(tmp_path: Path) -> None:
    src = Path(dof.__file__).resolve().parent
    copy = tmp_path / "site" / "dof"
    shutil.copytree(src, copy, ignore=shutil.ignore_patterns("__pycache__"))
    assert package_sources(copy) == package_sources()


def test_code_identity_fields() -> None:
    ident = code_identity()
    assert ident["version"] == dof.__version__
    assert ident["scipy"] and ident["blas"]["name"] and ident["blas"]["version"]
    assert not [k for k in ident["source_sha256"] if k.startswith("/") or ":" in k]


def test_git_fields_none_outside_work_tree(tmp_path: Path) -> None:
    ident = code_identity(repo_root=tmp_path)
    assert (ident["git_head"], ident["git_dirty"], ident["git_status_lines"]) == (None, None, None)
