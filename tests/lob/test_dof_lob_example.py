from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_example_runs_end_to_end(tmp_path: Path) -> None:
    """The synthetic example prints a table of 3 stocks x 10 horizons with finite R² values."""
    script = ROOT / "examples" / "synthetic_end_to_end.py"
    proc = subprocess.run([sys.executable, str(script), "--out", str(tmp_path)], capture_output=True, text=True,
                          timeout=600, env=dict(os.environ))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = [ln.split() for ln in proc.stdout.splitlines() if ln.split()[:1] and ln.split()[0] in ("AAA", "BBB", "CCC")]
    assert [r[0] for r in rows] == ["AAA", "BBB", "CCC"], proc.stdout
    for r in rows:
        values = [float(v) for v in r[1:]]
        assert len(values) == 10 and all(math.isfinite(v) for v in values), r
