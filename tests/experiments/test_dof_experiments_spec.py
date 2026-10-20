"""The experiment specs: the five stored experiments validate; ``check`` refuses an inconsistent spec, naming the field;
the identity hash ignores wording."""

from __future__ import annotations

import dataclasses
import re
from dataclasses import replace
from pathlib import Path

import pytest

from dof.experiments.errors import SpecError
from dof.experiments.spec import SectionSpec, pair_round, spec_identity_sha256, ts
from dof.experiments.specs import SPECS

SPEC_DIR = Path(__file__).resolve().parents[2] / "src" / "dof" / "experiments" / "specs"
MACHINE = re.compile(r"/(home|mnt|Users)/")


def _strings(o):
    if dataclasses.is_dataclass(o):
        for f in dataclasses.fields(o):
            yield from _strings(getattr(o, f.name))
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _strings(v)
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(o, str):
        yield o


def test_real_specs_validate() -> None:
    assert set(SPECS) == {"alldays", "seeds", "adjacent", "decay", "earlier"}
    for name, spec in SPECS.items():
        assert spec.name == name
        spec.check()


def test_spec_identity_ignores_wording() -> None:
    spec = SPECS["alldays"]
    same = replace(spec, title="Another title", intro=("Other words.",), sections=(), steps=("collect", "report"))
    assert spec_identity_sha256(same) == spec_identity_sha256(spec)
    other = replace(spec, rounds=(tuple((f, 0) for f in reversed(spec.fits)),))
    assert spec_identity_sha256(other) != spec_identity_sha256(spec)


def _with_table(spec, table):
    return replace(spec, sections=(SectionSpec("x", "X", tables=(table,)),))


def test_unknown_analysis_refused() -> None:
    with pytest.raises(SpecError, match="analysis 'no_such_analysis'"):
        _with_table(SPECS["alldays"], ts("no_such_analysis", "t1", "T")).check()


def test_unknown_argument_refused() -> None:
    bad = ts("score_rows", "t1", "T", record="test", model="LSTM-MLP-OF-recent", bogus=1)
    with pytest.raises(SpecError, match="argument 'bogus'"):
        _with_table(SPECS["alldays"], bad).check()


def test_round_fit_unknown_refused() -> None:
    with pytest.raises(SpecError, match=r"rounds\[0\]"):
        replace(SPECS["alldays"], rounds=((("No-Such-Fit", 0),),)).check()
    with pytest.raises(SpecError, match=r"rounds\[0\]: seed 7"):
        replace(SPECS["alldays"], rounds=((("LSTM-MLP-OF-recent", 7),),)).check()


def test_bad_pin_refused() -> None:
    spec = SPECS["alldays"]
    with pytest.raises(SpecError, match="protocol.main_sha256"):
        replace(spec, protocol=replace(spec.protocol, main_sha256="ABC")).check()


def test_odd_more_refused() -> None:
    spec = SPECS["seeds"]
    with pytest.raises(SpecError, match="seed_policy.more"):
        replace(spec, seed_policy=replace(spec.seed_policy, more=3)).check()


def test_duplicate_table_id_refused() -> None:
    spec = SPECS["alldays"]
    with pytest.raises(SpecError, match="table ids repeat"):
        replace(spec, sections=spec.sections + spec.sections[:3]).check()


def test_pair_round_order() -> None:
    assert pair_round(("a", "b", "c"), 1, 2) == (("a", 1), ("b", 1), ("a", 2), ("b", 2), ("c", 1), ("c", 2))


def test_seeds_round_one_is_the_protocol_order() -> None:
    assert SPECS["seeds"].rounds[0] == (
        ("LSTM-MLP-OF-recent", 3089492870), ("LSTM-MLP-OF-alldays", 3089492870),
        ("LSTM-MLP-OF-recent", 771810125), ("LSTM-MLP-OF-alldays", 771810125),
        ("CNN-LSTM-OF-alldays", 3089492870), ("CNN-LSTM-OF-alldays", 771810125))


def test_specs_hold_no_machine_fact() -> None:
    for spec in SPECS.values():
        for s in _strings(spec):
            assert not s.startswith("/") and not MACHINE.search(s), (spec.name, s)


def test_spec_modules_are_short() -> None:
    for name in SPECS:
        n = len((SPEC_DIR / f"{name}.py").read_text(encoding="utf-8").splitlines())
        assert n <= 200, f"specs/{name}.py has {n} lines"


def test_pattern_compiles_backreference() -> None:
    from dof.experiments.store import compile_pattern
    rx = compile_pattern("seed{seed}/{model}/fit_seed{seed}.json")
    assert rx.match("seed7/M/fit_seed7.json").groupdict() == {"seed": "7", "model": "M"}
    assert rx.match("seed7/M/fit_seed8.json") is None and rx.match("seed7/M/fit_seed7.jsonx") is None
    with pytest.raises(SpecError, match=r"\{nope\}"):
        compile_pattern("x/{nope}.json")


def test_source_without_model_placeholder_needs_model() -> None:
    from dof.experiments.spec import RecordSource
    spec = SPECS["earlier"]
    bad = replace(spec, sources=(RecordSource("test", "linear/linear.json", (), part="test"),))
    with pytest.raises(SpecError, match="names no model"):
        bad.check()


def test_source_part_must_be_a_part_of_the_spec() -> None:
    from dof.experiments.spec import RecordSource
    spec = SPECS["earlier"]
    bad = replace(spec, sources=(RecordSource("test", "{model}/pooled_seed0.json", (), part="far"),))
    with pytest.raises(SpecError, match="part 'far'"):
        bad.check()


def test_later_source_fields_leave_identity_unchanged() -> None:
    from dof.experiments.spec import RecordSource
    spec = SPECS["alldays"]
    plain = replace(spec, sources=tuple(RecordSource(s.record, s.pattern, s.pins, s.fit_rule, s.seed)
                                        for s in spec.sources))
    assert spec_identity_sha256(plain) == spec_identity_sha256(spec)
    split = replace(spec, sources=tuple(replace(s, part="test") for s in spec.sources))
    assert spec_identity_sha256(split) != spec_identity_sha256(spec)
