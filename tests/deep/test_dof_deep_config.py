"""Run configurations: loading, validation, includes, the compatibility dict and the result identity."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import replace
from pathlib import Path

import pytest

from dof.deep.config import (
    FitSpec,
    MachineConfig,
    Seeds,
    canonical_json,
    identity,
    identity_sha256,
    legacy_config_dict,
    load,
    load_machine,
)
from dof.deep.errors import ConfigError

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
SMOKE = CONFIGS / "smoke.toml"
ALLDAYS = CONFIGS / "alldays.toml"
GPU_CAP = 9663676416            # the reference implementation's allocator cap, held by every stored record
# sha256 of the canonical JSON of the stored records' "config" dict: the smoke split with two checks of 12 steps
# per seed, and the pre-registered split with seed 0.
SMOKE_HASHES = {
    0: "91bafb2338d4297d3db474ccc570d4ded583c18d115267f5e12e2f7552ed6143",
    3089492870: "0d66f72ea401310426047d190d82090ee6109084b6f41b5a2a6744bf685e5b0a",
    771810125: "4e9ec7aa7c47e20179d522475fd0726add02e69ddf7b343150bc3c4e59b17ca7",
}
REAL_HASH = "100e4cce9e4e5ed1d69e9f95b77b7a2f159c6b1555b6ad23304d7ff9fe72f6f7"


@pytest.fixture(autouse=True)
def _no_local_file(monkeypatch):
    monkeypatch.delenv("DOF_LOCAL_CONFIG", raising=False)


def _sha(o) -> str:
    return hashlib.sha256(canonical_json(o).encode()).hexdigest()


def _variant(tmp_path: Path, *edits: tuple[str, str], models_edits=(), extra: dict | None = None) -> Path:
    """smoke.toml and models.toml copied into ``tmp_path`` with ``old -> new`` replacements (first match only)."""
    text = SMOKE.read_text(encoding="utf-8")
    for old, new in edits:
        assert old in text, old
        text = text.replace(old, new, 1)
    models = (CONFIGS / "models.toml").read_text(encoding="utf-8")
    for old, new in models_edits:
        assert old in models, old
        models = models.replace(old, new, 1)
    (tmp_path / "models.toml").write_text(models, encoding="utf-8")
    for name, body in (extra or {}).items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    path = tmp_path / "smoke.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_smoke_config_fields() -> None:
    cfg = load(SMOKE)
    sp = cfg.split
    assert (sp.name, sp.train_days, sp.recent_days, sp.valid_days, sp.test_days) == (
        "smoke", ("2019-01-30", "2019-03-27", "2019-07-30"), ("2019-07-30",), ("2019-08-30",), ("2019-10-18",))
    assert (sp.symbols, sp.reused_from, sp.price_rule_parts) == (("UAL", "FAST"), (), ("valid",))
    assert cfg.half_days == {"2025-11-28": "13:00"}
    assert cfg.universe == CONFIGS / "universe_115.csv"
    p = cfg.protocol
    assert p.corporate_actions == CONFIGS / "protocol" / "corporate_actions_smoke_fixture.csv"
    assert p.corporate_actions_sha256 == "189defa893f01d740c2da2a241d5831f7d55a5f54abce9cdb5f724d5cf402686"
    assert p.ca_columns == ("ticker", "event", "effective_date", "status")
    assert p.ca_excluding == ("forward_split", "reverse_split", "ticker_change", "ticker_reused")
    assert (p.ca_confirmed, p.document, p.document_sha256, p.test_days_need_document) == ("confirmed", None, None, False)
    f = cfg.features
    assert (f.convention, f.dedupe, f.history, f.latency_ns, f.horizons, f.gap_days, f.ratio_lo, f.ratio_hi) == (
        "cks", True, 100, 10_000_000, tuple(range(1, 11)), 300, 0.6, 1.6)
    t = cfg.train
    assert (t.batch_size, t.check_steps, t.max_checks, t.patience, t.adam_betas, t.eval_batch, t.restore_best) == (
        256, 200, 2, 5, (0.9, 0.999), 512, True)
    assert t.fit_order == ("LSTM-MLP-OF-recent", "LSTM-MLP-OF-alldays", "CNN-LSTM-OF-alldays") and t.eval_row_cap is None
    assert (cfg.ridge.lambdas, cfg.ridge.chunk_rows) == ((1e-3, 1e-2, 1e-1, 1.0, 10.0), 16384)
    # The catalogue's models after Kolm, Turiel and Westray (2023), in their order; a new family adds its own entry
    # without touching this test.
    paper_models = ["LSTM-OF", "CNN-LSTM-OF", "LSTM-MLP-OF", "LSTM3-OF", "MLP-OF", "LSTM-OFI"]
    assert [m for m in cfg.models if m in paper_models] == paper_models
    cnn = cfg.models["CNN-LSTM-OF"]
    assert (cnn.family, cnn.input, cnn.input_dim, cnn.init, cnn.precision, cnn.n_out, cnn.history) == (
        "cnn_lstm", "of", 20, "keras", "float32", 10, 100)
    assert cnn.params == {"filters": 32, "inception_filters": 64, "lstm_units": 64, "leaky_slope": 0.01}
    assert cfg.models["MLP-OF"].params == {"hidden": (500, 500, 100)}
    assert cfg.models["LSTM-MLP-OF"].params == {"units": 120, "hidden": 128}
    assert cfg.models["LSTM3-OF"].precision == "amp" and cfg.models["LSTM3-OF"].params == {"units": 150, "layers": 3}
    assert (cfg.models["LSTM-OFI"].input, cfg.models["LSTM-OFI"].input_dim) == ("ofi", 10)
    assert cfg.fits["CNN-LSTM-OF-alldays"] == FitSpec("CNN-LSTM-OF-alldays", "CNN-LSTM-OF", "keras_adam", 3e-5, 1e-7,
                                                      "all", "CNN-LSTM-OF-lr3e-5-eps1e-7")
    assert cfg.fits["LSTM-MLP-OF-recent"] == FitSpec("LSTM-MLP-OF-recent", "LSTM-MLP-OF", "adam", 1e-5, 1e-7, "recent",
                                                     None)
    assert {n: s.days for n, s in cfg.linear.items()} == {"Ridge-OF-alldays": "all", "Ridge-OF-recent": "recent"}
    assert cfg.scored_models == (*t.fit_order, "Ridge-OF-alldays", "Ridge-OF-recent")
    assert cfg.machine == MachineConfig()


def test_unknown_key_names_the_key(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"train\.foo"):
        load(_variant(tmp_path, ("[train]\n", "[train]\nfoo = 1\n")))


@pytest.mark.parametrize("edit, models_edit", [
    (('optimizer = "adam"', 'optimizer = "sgd"'), None),
    (('days = "recent"', 'days = "weekly"'), None),
    (None, ('input = "of"', 'input = "lob"')),
    (None, ('init = "keras"', 'init = "xavier"')),
    (None, ('precision = "float32"', 'precision = "half"')),
    (('arch = "LSTM-MLP-OF"', 'arch = "NO-SUCH-MODEL"'), None),
    (('"CNN-LSTM-OF-alldays"]', '"NO-SUCH-FIT"]'), None),
    (("restore_best = true", "restore_best = false"), None),
    (("test_days_need_document = false", "test_days_need_document = true"), None),
    (("ratio_range = [0.6, 1.6]", "ratio_range = [1.6, 0.6]"), None),
    (("lambdas = [0.001, 0.01, 0.1, 1.0, 10.0]", "lambdas = []"), None),
])
def test_bad_choices_raise(tmp_path: Path, edit, models_edit) -> None:
    path = _variant(tmp_path, *([edit] if edit else []), models_edits=[models_edit] if models_edit else [])
    with pytest.raises(ConfigError):
        load(path)


def test_include_merges_and_refuses_duplicates(tmp_path: Path) -> None:
    dup = '[models.MLP-OF]\nfamily = "mlp"\n'
    path = _variant(tmp_path, ('include = ["models.toml"]', 'include = ["models.toml", "more.toml"]'),
                    extra={"more.toml": dup})
    with pytest.raises(ConfigError, match=r"models\.MLP-OF\.family.*models\.toml.*more\.toml"):
        load(path)
    ok = _variant(tmp_path, ('include = ["models.toml"]', 'include = ["models.toml", "more.toml"]'),
                  extra={"more.toml": '[models.Extra]\nfamily = "mlp"\ninput = "of"\ninput_dim = 20\nhidden = [8]\n'
                                      'init = "torch"\nprecision = "float32"\n'})
    cfg = load(ok)
    assert "Extra" in cfg.models and "MLP-OF" in cfg.models
    cycle = _variant(tmp_path, ('include = ["models.toml"]', 'include = ["models.toml", "loop.toml"]'),
                     extra={"loop.toml": 'include = ["smoke.toml"]\n'})
    with pytest.raises(ConfigError, match="include cycle"):
        load(cycle)
    absolute = _variant(tmp_path, ('include = ["models.toml"]', f'include = ["{(tmp_path / "models.toml").as_posix()}"]'))
    with pytest.raises(ConfigError, match="absolute"):
        load(absolute)


@pytest.mark.parametrize("edit", [
    ('train_days = ["2019-01-30", "2019-03-27", "2019-07-30"]', 'train_days = ["2019-03-27", "2019-01-30", "2019-07-30"]'),
    ('valid_days = ["2019-08-30"]', 'valid_days = ["2019-07-30"]'),
    ('recent_days = ["2019-07-30"]', 'recent_days = ["2019-08-30"]'),
])
def test_split_check(tmp_path: Path, edit) -> None:
    with pytest.raises(ConfigError, match="split smoke"):
        load(_variant(tmp_path, edit))


def test_legacy_dict_matches_recorded_smoke_hashes() -> None:
    cfg = load(SMOKE)
    cfg = replace(cfg, train=replace(cfg.train, check_steps=12, max_checks=2),
                  machine=MachineConfig(gpu_memory_limit_bytes=GPU_CAP))
    for seed, want in SMOKE_HASHES.items():
        assert _sha(legacy_config_dict(cfg, cfg.split, Seeds.same(seed))) == want


def test_legacy_dict_matches_stored_real_records_hash() -> None:
    cfg = replace(load(ALLDAYS), machine=MachineConfig(gpu_memory_limit_bytes=GPU_CAP))
    assert _sha(legacy_config_dict(cfg, cfg.split, Seeds.same(0))) == REAL_HASH


def _strings(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)
    elif isinstance(o, str):
        yield o


def test_identity_excludes_machine_settings() -> None:
    cfg = load(SMOKE)
    name, seeds = "LSTM-MLP-OF-recent", Seeds.same(0)
    base = identity_sha256(cfg, cfg.split, seeds, name)
    for machine in (MachineConfig(loader_workers=8), MachineConfig(gpu_memory_limit_bytes=GPU_CAP),
                    MachineConfig(book_root=Path("books")), MachineConfig(device="cuda"),
                    MachineConfig(deterministic=True)):
        assert identity_sha256(replace(cfg, machine=machine), cfg.split, seeds, name) == base
    arch = cfg.models["LSTM-MLP-OF"]
    changed = [
        replace(cfg, train=replace(cfg.train, check_steps=201)),
        replace(cfg, models={**cfg.models, "LSTM-MLP-OF": replace(arch, params={**arch.params, "units": 121})}),
        replace(cfg, protocol=replace(cfg.protocol, corporate_actions_sha256="0" * 64)),
    ]
    for c in changed:
        assert identity_sha256(c, cfg.split, seeds, name) != base
    assert identity_sha256(cfg, cfg.split, Seeds(0, 1), name) != base
    doc = identity(cfg, cfg.split, seeds, name)
    assert not [s for s in _strings(doc) if s.startswith("/")]
    for linear in cfg.linear:
        assert identity_sha256(cfg, cfg.split, seeds, linear) != base


def test_identity_ignores_other_fits() -> None:
    cfg = load(SMOKE)
    more = replace(cfg, fits={**cfg.fits, "MLP-OF-all": FitSpec("MLP-OF-all", "MLP-OF", "adam", 1e-3, 1e-7, "all", None)})
    for name in cfg.fits:
        assert identity_sha256(more, more.split, Seeds.same(0), name) == identity_sha256(cfg, cfg.split, Seeds.same(0), name)


def test_seeds_tag_and_legacy_seed_field() -> None:
    assert Seeds.same(5).tag == "seed5" and Seeds(5, 6).tag == "seed5-6"
    cfg = load(SMOKE)
    assert legacy_config_dict(cfg, cfg.split, Seeds.same(5))["seed"] == 5
    assert legacy_config_dict(cfg, cfg.split, Seeds(5, 6))["seed"] == {"init": 5, "data": 6}


def test_machine_paths_relative_to_local_file(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "machine"
    folder.mkdir()
    local = folder / "local.toml"
    absolute_eval = (tmp_path / "elsewhere" / "eval").as_posix()
    local.write_text('[machine]\nbook_root = "books"\nloader_workers = 3\n\n[machine.cache.smoke]\ntrain = "c/train"\n'
                     f'eval = "{absolute_eval}"\n', encoding="utf-8")
    m = load_machine(local)
    assert m.book_root == folder.resolve() / "books" and m.loader_workers == 3
    c = m.cache_for("smoke")
    assert c.train == folder.resolve() / "c" / "train" and c.eval == Path(absolute_eval)
    assert c.compact_root == folder.resolve() / "c" / "compact"
    monkeypatch.setenv("DOF_LOCAL_CONFIG", str(local))
    assert load(SMOKE).machine == m
    assert load(SMOKE, local=local).machine == m


def test_missing_cache_entry_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"machine\.cache\.smoke is not set"):
        MachineConfig().cache_for("smoke")
    bad = tmp_path / "local.toml"
    bad.write_text("[machine]\nloader_workers = 1\nspeed = 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"machine\.speed"):
        load_machine(bad)


def test_run_config_pickles() -> None:
    cfg = load(SMOKE)
    assert pickle.loads(pickle.dumps(cfg)) == cfg
