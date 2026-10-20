"""``run`` on the synthetic training run: fit, gate, score once, collect and report; a second run scores nothing; the
round gate refuses before a round has ended; another identity in the same folder is refused; stored-only specs are
refused before anything is written."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from dof.deep.errors import RecordConflict, TestDayRefused  # noqa: E402
from dof.experiments.errors import SpecError  # noqa: E402
from dof.experiments.gate import RoundGate  # noqa: E402
from dof.experiments.runner import run  # noqa: E402
from dof.experiments.specs import SPECS  # noqa: E402
from synth_specs import FITS, LINEAR, alldays_like  # noqa: E402

pytestmark = pytest.mark.slow


def _quiet(*a):
    return None


@pytest.fixture(scope="module")
def first_run(synth_run, tmp_path_factory):
    out = tmp_path_factory.mktemp("run") / "out"
    return out, run(alldays_like(synth_run.cfg), out=out, cfg=synth_run.cfg, log=_quiet)


def test_run_alldays_like_end_to_end(first_run) -> None:
    out, res = first_run
    for m in FITS + LINEAR:
        assert (out / "seed0" / m / "fit_seed0.json").is_file() and (out / "seed0" / m / "test_seed0.json").is_file()
    assert (out / "tables" / "MANIFEST.json").is_file() and (out / "report.md").is_file()
    assert set(res.scored) == {(m, 0) for m in FITS + LINEAR}
    assert res.fitted == {(m, 0): "finished" for m in FITS + LINEAR}
    manifest = json.loads((out / "tables" / "MANIFEST.json").read_text())
    assert manifest["audit"]["alldays"]["held_out_after_fits"] is True and manifest["audit"]["alldays"]["checked"] == 4
    assert manifest["totals"]["alldays/test"]["sources"] == 4
    assert "#### Test day, TinyMLP-all" in (out / "report.md").read_text()
    events = [json.loads(x)["event"] for x in (out / "events.jsonl").read_text().splitlines()]
    assert events.count("fit_ended") == 4 and events.count("scored") == 4


def test_second_run_scores_nothing(synth_run, first_run) -> None:
    out, _res = first_run
    mtimes = {m: (out / "seed0" / m / "test_seed0.json").stat().st_mtime_ns for m in FITS + LINEAR}
    res = run(alldays_like(synth_run.cfg), out=out, cfg=synth_run.cfg, log=_quiet)
    assert res.scored == ()
    assert mtimes == {m: (out / "seed0" / m / "test_seed0.json").stat().st_mtime_ns for m in FITS + LINEAR}


def test_round_gate_refuses_before_round_ended(synth_run, tmp_path) -> None:
    spec = alldays_like(synth_run.cfg)
    for name in (FITS[0], *LINEAR):
        (tmp_path / "seed0" / name).mkdir(parents=True)
        (tmp_path / "seed0" / name / "fit_seed0.json").write_text("{}")
    with pytest.raises(TestDayRefused, match=f"not ended: {FITS[1]} seed 0"):
        RoundGate(spec, tmp_path).check(0, spec.rounds)
    assert not list(tmp_path.rglob("test_seed*.json"))
    (tmp_path / "seed0" / FITS[1]).mkdir()
    (tmp_path / "seed0" / FITS[1] / "fit_seed0.json").write_text("{}")
    RoundGate(spec, tmp_path).check(0, spec.rounds)


def test_changed_identity_refused(synth_run, first_run) -> None:
    out, _res = first_run
    spec = alldays_like(synth_run.cfg)
    other = replace(spec, rounds=(tuple((f, 0) for f in reversed(FITS)),))
    with pytest.raises(RecordConflict, match="spec_identity_sha256"):
        run(other, out=out, cfg=synth_run.cfg, log=_quiet)
    res = run(replace(spec, title="Another title"), out=out, cfg=synth_run.cfg, log=_quiet)
    assert res.scored == ()


def test_stored_only_specs_refuse_run(synth_run, tmp_path) -> None:
    for name in ("adjacent", "decay"):
        with pytest.raises(SpecError, match="does not run its fit step"):
            run(SPECS[name], out=tmp_path / name, cfg=synth_run.cfg, log=_quiet)
        assert not (tmp_path / name).exists()
