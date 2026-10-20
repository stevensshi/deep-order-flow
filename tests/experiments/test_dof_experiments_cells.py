"""The result tables: schemas, neutral record ids, source order, exact totals, stock-days, protocol pins, the audit of
the order of work, and reading the tables back."""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace

import pandas as pd
import pyarrow.parquet as pq
import pytest

from dof.experiments.cells import TABLES, read_tables, totals
from dof.experiments.errors import ExportError
from dof.experiments.export import export
from dof.experiments.spec import RecordSource
from fake_lab import fake_specs, write_fake_lab

MACHINE = re.compile(r"/(home|mnt|Users)/")
ORDER = ("alldays", "seeds", "adjacent", "decay", "earlier")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("export")
    lab = write_fake_lab(root / "lab")
    specs = fake_specs(lab)
    manifest = export(lab.root, root / "tables", [specs[n] for n in ORDER], log=lambda m: None)
    return lab, root / "tables", manifest


def _quiet(*a):
    return None


def test_export_schemas_exact(exported) -> None:
    _lab, folder, _m = exported
    types = {"string": "string", "int64": "int64", "float64": "double", "bool": "bool"}
    for name, schema in TABLES.items():
        got = pq.read_schema(folder / f"{name}.parquet")
        assert [(f.name, str(f.type)) for f in got] == [(c, types[t]) for c, t in schema], name


def test_record_ids_unique_and_neutral(exported) -> None:
    _lab, folder, _m = exported
    ids = pd.read_parquet(folder / "records.parquet")["record_id"].tolist()
    assert len(ids) == len(set(ids))
    for rid in ids:
        assert ".json" not in rid and "results" not in rid and not rid.startswith("/") and " " not in rid, rid


def test_row_ordinal_is_source_order(exported) -> None:
    lab, folder, _m = exported
    rt = read_tables(folder)
    doc = json.loads((lab.root / "results" / "CNN-LSTM-OF-alldays" / "test_seed0.json").read_text())
    rows = rt.rows(rt.rid(experiment="alldays", record="test", model="CNN-LSTM-OF-alldays", seed=0))
    assert list(rows.symbol) == [r["symbol"] for r in doc["scores"]]
    assert list(rows.date) == [r["date"] for r in doc["scores"]]
    assert list(rows.k) == [r["k"] for r in doc["scores"]]
    assert [x.hex() for x in rows.sse] == [float(r["sse"]).hex() for r in doc["scores"]]


def test_totals_equal_direct_sums(exported) -> None:
    lab, folder, manifest = exported
    assert totals(read_tables(folder)) == lab.totals
    assert manifest["totals"] == lab.totals


def test_ensemble_cells_have_members_no_seed(exported) -> None:
    _lab, folder, _m = exported
    rt = read_tables(folder)
    ens = rt.cells[rt.cells["record"] == "ensemble"]
    assert len(ens) and ens["seed"].isna().all() and (ens["members"] == "0+3089492870+771810125").all()
    assert all("/members=0+3089492870+771810125/" in r for r in ens["record_id"])


def test_stock_days_from_test_files(exported) -> None:
    lab, folder, _m = exported
    rt = read_tables(folder)
    rid = rt.rid(experiment="alldays", record="test", model="LSTM-MLP-OF-recent", seed=0)
    sd = rt.stock_days_of(rid)
    doc = json.loads((lab.root / "results" / "LSTM-MLP-OF-recent" / "test_seed0.json").read_text())
    scored = sd[sd["status"] == "scored"]
    assert len(scored) == len(doc["price_ratio"])
    for sym, day, ratio, lo, hi in zip(scored["symbol"], scored["date"], scored["price_ratio"], scored["ratio_lo"],
                                       scored["ratio_hi"]):
        assert ratio == doc["price_ratio"][f"{sym} {day}"] and (lo, hi) == (0.6, 1.6)
    gone = sd[sd["status"] == "left_out"]
    assert gone[["symbol", "date", "event", "effective_date"]].values.tolist() == [
        ["CCC", "2026-06-12", "forward_split", "2026-06-12"]]


def test_decay_stock_days(exported) -> None:
    _lab, folder, _m = exported
    rt = read_tables(folder)
    units = rt.decay_units("LSTM-MLP-OF", "2025-12-12", "refreshed")
    assert [u["symbol"] for u in units] == ["AAA", "BBB", "CCC"]
    assert [u["status"] for u in units] == ["scored", "scored", "left_out"]
    assert units[2]["reason"] == "reused ticker (rule 1)"
    frozen = rt.decay_units("LSTM-MLP-OF", "2025-12-12", "frozen")
    assert frozen[0]["ratio"] == 1.7 and frozen[0]["ratio_inside"] is False
    sd = rt.stock_days
    row = sd[(sd["record_id"] == frozen[1]["record_id"])].iloc[0]
    assert row["rows_scored"] == 1001 and row["ref_day"] == "2019-07-30"


def _export_one(tmp_path, name, change):
    lab = write_fake_lab(tmp_path / "lab")
    spec = change(fake_specs(lab)[name])
    export(lab.root, tmp_path / "out", [spec], log=_quiet)


def _pin_mismatch(tmp_path, name, field, pin) -> None:
    def change(spec):
        return replace(spec, protocol=replace(spec.protocol, **{field: "0" * 64}))
    with pytest.raises(ExportError, match=rf"^{name}/[^ ]+: {pin} is [0-9a-f]{{64}}, the spec pins 0{{64}}$"):
        _export_one(tmp_path, name, change)
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.partial").exists()


def test_main_pin_mismatch_refused(tmp_path) -> None:
    _pin_mismatch(tmp_path, "alldays", "main_sha256", "main")


def test_extension_pin_mismatch_refused(tmp_path) -> None:
    _pin_mismatch(tmp_path, "adjacent", "extension_sha256", "extension")


def test_ca_pin_mismatch_refused(tmp_path) -> None:
    _pin_mismatch(tmp_path, "seeds", "corporate_actions_sha256", "corporate_actions")


def test_pin_none_requires_none(tmp_path) -> None:
    def change(spec):                   # the alldays fit records name no extension document: pin it as None ...
        srcs = tuple(replace(s, pins=s.pins + ("extension",)) for s in spec.sources)
        return replace(spec, sources=srcs)
    _export_one(tmp_path / "a", "alldays", change)

    def change2(spec):                  # ... and a spec value of None refuses a record that names one
        return replace(spec, protocol=replace(spec.protocol, extension_sha256=None))
    with pytest.raises(ExportError, match="extension is c{64}, the spec pins None"):
        _export_one(tmp_path / "b", "adjacent", change2)


def test_duplicate_record_id_refused(tmp_path) -> None:
    """Within one spec the first matching source wins, so a source listed twice gives each record once; a second
    spec of the same name over the same root gives its test records a second time."""
    lab = write_fake_lab(tmp_path / "lab")
    spec = fake_specs(lab)["alldays"]
    twice = replace(spec, sources=spec.sources + (spec.sources[1],))
    export(lab.root, tmp_path / "ok", [twice], log=_quiet)
    other = replace(spec, sources=(RecordSource("test", "{model}/test_seed0.json", ("main", "corporate_actions")),))
    with pytest.raises(ExportError, match="is given twice"):
        export(lab.root, tmp_path / "out", [spec, other], log=_quiet)
    assert not (tmp_path / "out.partial").exists()


def test_audit_refuses_early_held_out_read(tmp_path, exported) -> None:
    lab = write_fake_lab(tmp_path / "lab")
    path = lab.root / "results" / "Ridge-OF-recent" / "test_seed0.json"
    doc = json.loads(path.read_text())
    doc["scored"] = "2025-12-31T23:59:59"
    path.write_text(json.dumps(doc))
    with pytest.raises(ExportError, match="scored before every fit of their round had ended"):
        export(lab.root, tmp_path / "out", [fake_specs(lab)["alldays"]], log=_quiet)
    audit = exported[2]["audit"]
    assert audit["alldays"] == {"rounds": 1, "held_out_after_fits": True, "checked": 5, "unchecked": 0}
    assert audit["adjacent"]["held_out_after_fits"] is True and audit["adjacent"]["checked"] == 11
    assert audit["seeds"]["checked"] == 6 + 3
    assert audit["decay"] == {"rounds": 0, "held_out_after_fits": None, "checked": 0, "unchecked": 0}


def test_audit_refuses_held_out_without_time(tmp_path) -> None:
    lab = write_fake_lab(tmp_path / "lab")
    path = lab.root / "adj/results/seed3089492870/LSTM-MLP-OF-recent/far_seed3089492870.json"
    doc = json.loads(path.read_text())
    del doc["scored"]
    path.write_text(json.dumps(doc))
    with pytest.raises(ExportError, match="without a scored time"):
        export(lab.root, tmp_path / "out", [fake_specs(lab)["adjacent"]], log=_quiet)


def test_output_must_not_exist(tmp_path) -> None:
    lab = write_fake_lab(tmp_path / "lab")
    (tmp_path / "out").mkdir()
    with pytest.raises(ExportError, match="exists"):
        export(lab.root, tmp_path / "out", [fake_specs(lab)["alldays"]], log=_quiet)
    assert list((tmp_path / "out").iterdir()) == []


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


def test_outputs_hold_no_path(exported) -> None:
    _lab, folder, _m = exported
    for name in TABLES:
        df = pd.read_parquet(folder / f"{name}.parquet")
        for col in df.columns:
            for v in df[col].dropna():
                if isinstance(v, str):              # a bool column with nulls also reads back as object
                    assert not v.startswith("/") and not MACHINE.search(v), (name, col, v)
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    for s in _strings(manifest):
        assert not s.startswith("/") and not MACHINE.search(s), s
    keys = [k for k in _strings({k: None for k in manifest})]
    assert not any(w in k for k in keys for w in ("time", "date", "created", "written")), keys


def test_read_tables_types(exported) -> None:
    _lab, folder, _m = exported
    rt = read_tables(folder)
    assert str(rt.cells["seed"].dtype) == "Int64" and rt.cells["seed"].isna().any()
    assert str(rt.cells["r2_os"].dtype) == "float64" and rt.cells["r2_os"].isna().sum() == 1
    assert math.isnan(rt.rows(rt.rid(experiment="alldays", record="test", model="LSTM-MLP-OF-recent", seed=0)).r2[0])
    assert isinstance(rt.rows(rt.rid(experiment="alldays", record="test", model="LSTM-MLP-OF-recent", seed=0)).k[0], int)


def test_export_roots_name_another_folder(tmp_path) -> None:
    lab = write_fake_lab(tmp_path / "lab")
    specs = fake_specs(lab)
    moved = tmp_path / "stage1"
    moved.mkdir()
    (lab.root / "pooled115").rename(moved / "pooled115")
    with pytest.raises(ExportError, match="earlier: its root"):
        export(lab.root, tmp_path / "t1", [specs[n] for n in ORDER], log=_quiet)
    with pytest.raises(ExportError, match="not exported"):
        export(lab.root, tmp_path / "t2", [specs["alldays"]], roots={"earlier": moved}, log=_quiet)
    empty = tmp_path / "empty"
    (empty / "pooled115").mkdir(parents=True)
    with pytest.raises(ExportError, match="earlier: no record"):
        export(lab.root, tmp_path / "t4", [specs[n] for n in ORDER], roots={"earlier": empty}, log=_quiet)
    m = export(lab.root, tmp_path / "t3", [specs[n] for n in ORDER], roots={"earlier": moved}, log=_quiet)
    assert m["totals"] == lab.totals
    assert "stage1" not in json.dumps(m)


def test_sum_hex_rules() -> None:
    from dof.experiments.store import sum_hex
    assert sum_hex([1.0, None, 2.0]) == (3.0).hex()
    assert sum_hex([None, None]) is None
    assert sum_hex([]) == (0.0).hex()
    assert sum_hex([1.0, float("nan")]) == "nan"
