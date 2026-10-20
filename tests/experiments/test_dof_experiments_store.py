"""Finding the stored records: every layout matches its record kind, and the record rules skip what does not count."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import dof.experiments.cells as cells_mod
from dof.experiments.cells import collect
from dof.experiments.store import discover
from fake_lab import fake_specs, write_fake_lab


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    return write_fake_lab(tmp_path_factory.mktemp("lab"))


@pytest.fixture
def own_lab(tmp_path):
    """A lab of its own that a test may change."""
    return write_fake_lab(tmp_path / "lab")


def test_sources_match_their_layouts(lab) -> None:
    specs = fake_specs(lab)
    for name, spec in specs.items():
        found, skipped = discover(lab.root / spec.root, spec)
        assert skipped.get("unknown kind", 0) == 0, (name, skipped)
        c = collect(lab.root / spec.root, spec)
        assert len(c.records) == len(lab.exported[name]), (name, c.skipped)
        kinds = {}
        for r in c.records:
            kinds[r["record"]] = kinds.get(r["record"], 0) + 1
        want = {}
        for kind in lab.exported[name].values():
            want[kind] = want.get(kind, 0) + 1
        assert kinds == want, name


def test_unfinished_fit_skipped(own_lab) -> None:
    spec = fake_specs(own_lab)["alldays"]
    path = own_lab.root / "results" / "LSTM-MLP-OF-recent" / "fit_seed0.json"
    doc = json.loads(path.read_text())
    doc["status"] = "diverged"
    path.write_text(json.dumps(doc))
    c = collect(own_lab.root / spec.root, spec)
    assert c.skipped.get("fit not finished (status 'diverged')") == 1
    assert not any(r["model"] == "LSTM-MLP-OF-recent" and r["record"] == "fit_valid" for r in c.records)


def test_seed_fit_with_checkpoint_skipped(lab) -> None:
    spec = fake_specs(lab)["seeds"]
    c = collect(lab.root / spec.root, spec)
    assert c.skipped.get("fit not finished (weights not saved or a checkpoint is left)") == 1
    assert not any(r["seed"] == 18531247 for r in c.records)


def test_adjacent_test_needs_finished_fit(own_lab) -> None:
    spec = fake_specs(own_lab)["adjacent"]
    (own_lab.root / "adj/results/seed771810125/LSTM-MLP-OF-recent/fit_seed771810125.json").unlink()
    c = collect(own_lab.root / spec.root, spec)
    assert c.skipped.get("its fit has not finished") == 2
    assert not any(r["seed"] == 771810125 for r in c.records)


def test_decay_status_mapping(own_lab) -> None:
    spec = fake_specs(own_lab)["decay"]
    path = own_lab.root / "decay/scores/LSTM-MLP-OF/frozen/2019-10-18/AAA.json"
    doc = json.loads(path.read_text())
    doc["status"] = "other"
    path.write_text(json.dumps(doc))
    c = collect(own_lab.root / spec.root, spec)
    assert c.skipped == {"unknown kind (status 'other')": 1}
    kinds = {r["record"] for r in c.records}
    assert kinds == {"decay", "left_out"}
    assert sum(r["record"] == "left_out" for r in c.records) == 2


def test_run_state_ledgers_and_smoke_ignored(lab, monkeypatch) -> None:
    read: list[str] = []
    original = Path.read_bytes

    def spy(self):
        read.append(self.as_posix())
        return original(self)
    monkeypatch.setattr(Path, "read_bytes", spy)
    for spec in fake_specs(lab).values():
        collect(lab.root / spec.root, spec)
    names = [p.rsplit("/", 1)[-1] for p in read]
    assert read and not any("/smoke/" in p for p in read)
    for bad in ("progress.json", "current.json", "measure.json", "extension.json", "experiment.json",
                "gate_refusals.json", "decision.json"):
        assert bad not in names, bad
    assert not any(n.startswith("begun_seed") for n in names)


def test_vanished_file_skipped(own_lab, monkeypatch) -> None:
    spec = fake_specs(own_lab)["alldays"]
    real = cells_mod.discover

    def vanish(root, spec, sources=None):
        found, skipped = real(root, spec, sources)
        (Path(root) / found[0].rel).unlink()
        return found, skipped
    monkeypatch.setattr(cells_mod, "discover", vanish)
    c = collect(own_lab.root / spec.root, spec)
    assert c.skipped.get("vanished") == 1


def test_one_file_matches_every_part_source(lab) -> None:
    spec = fake_specs(lab)["earlier"]
    found, skipped = discover(lab.root / spec.root, spec)
    by_file: dict[str, list[str]] = {}
    for f in found:
        by_file.setdefault(f.rel, []).append(f.source.part)
    assert by_file["linear/linear.json"] == ["valid", "test"]
    assert by_file["LSTM-OF/pooled_seed0.json"] == ["valid", "test"]
    assert skipped == {"excluded folder": 1}


def test_earlier_records_keep_their_part_and_selected_rows(lab) -> None:
    spec = fake_specs(lab)["earlier"]
    c = collect(lab.root / spec.root, spec)
    ids = sorted(r["record_id"] for r in c.records)
    assert ids == sorted(f"earlier/{p}/{m}/seed=0/{p}" for p in ("valid", "test")
                         for m in ("LSTM-OF", "LSTM-MLP-OF", "CNN-LSTM-OF-lr3e-5-eps1e-7", "Ridge-OF"))
    ridge = [x for x in c.cells if x["model"] == "Ridge-OF"]
    assert ridge and all(x["sse"] is None and x["sst"] is None and x["seed"] == 0 for x in ridge)
    assert {x["part"] for x in c.cells if x["record"] == "valid"} == {"valid"}
    assert {x["part"] for x in c.cells if x["record"] == "test"} == {"test"}
    assert len(ridge) == 4 * 3 * 10                      # pooled Ridge-OF only: 4 days x 3 symbols x 10 horizons
    sums = {(r["model"], r["record"]): (r["fsum_sse"], r["fsum_sst"]) for r in c.records}
    assert sums[("Ridge-OF", "test")] == (None, None) and sums[("Ridge-OF", "valid")] == (None, None)
    assert all(v is not None for m, kind in sums if m != "Ridge-OF" for v in sums[(m, kind)])
