"""The seed rule inside ``run``: decided once after round 1, the drawn seeds fitted and scored in round 2, the
decision read again (never drawn twice), and refused when the weights it was taken from have changed."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from dof.deep.errors import RecordConflict  # noqa: E402
from dof.experiments.runner import run  # noqa: E402
from synth_specs import FITS, alldays_like, seeds_like  # noqa: E402

pytestmark = pytest.mark.slow


def _quiet(*a):
    return None


def _no_draw(n, taken):
    raise AssertionError("the rule drew seeds a second time")


@pytest.fixture(scope="module")
def seeds_run(synth_run, tmp_path_factory):
    root = tmp_path_factory.mktemp("seeds")
    run(alldays_like(synth_run.cfg), out=root / "alldays", cfg=synth_run.cfg, log=_quiet)
    res = run(seeds_like(synth_run.cfg), out=root / "seeds", cfg=synth_run.cfg, inputs={"alldays": root / "alldays"},
              draw=lambda n, taken: [11, 12], log=_quiet)
    return root, res


def test_seed_policy_decides_once_and_fits_round_two(synth_run, seeds_run) -> None:
    root, res = seeds_run
    out = root / "seeds"
    decision = json.loads((out / "decision.json").read_text())
    assert decision["drawn"] == [11, 12] and decision["first"] == [0, 1, 2]
    assert set(decision["inputs"]) == {f"{f}/seed={s}" for f in FITS for s in (0, 1, 2)}
    for s in (1, 2, 11, 12):
        for f in FITS:
            assert (out / f"seed{s}" / f / f"fit_seed{s}.json").is_file()
            assert (out / f"seed{s}" / f / f"test_seed{s}.json").is_file()
    assert set(res.scored) == {(f, s) for f in FITS for s in (1, 2, 11, 12)}
    manifest = json.loads((out / "tables" / "MANIFEST.json").read_text())
    assert manifest["totals"]["alldays/fit_valid"]["sources"] == 2 and manifest["totals"]["seeds/test"]["sources"] == 8
    assert "seed 12" in (out / "report.md").read_text()
    again = run(seeds_like(synth_run.cfg), out=out, cfg=synth_run.cfg, inputs={"alldays": root / "alldays"},
                draw=_no_draw, log=_quiet)
    assert again.scored == () and again.decision["drawn"] == [11, 12]


def test_decision_inputs_changed_refused(synth_run, seeds_run) -> None:
    root, _res = seeds_run
    path = root / "seeds" / "decision.json"
    doc = json.loads(path.read_text())
    key = next(iter(doc["inputs"]))
    doc["inputs"][key] = "0" * 64
    path.write_text(json.dumps(doc))
    with pytest.raises(RecordConflict, match="decision.json was taken from other weights"):
        run(seeds_like(synth_run.cfg), out=root / "seeds", cfg=synth_run.cfg, inputs={"alldays": root / "alldays"},
            draw=_no_draw, log=_quiet)
