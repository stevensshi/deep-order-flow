"""Golden tests of the training package. Each test mirrors one part of the recorder of the reference
implementation, with ``dof.deep`` calls in place of the reference calls; ``recipes/train_values.py`` turns the
outputs into flat values, which must equal the recorded training goldens bit for bit.

The goldens are local only: they are read from ``DOF_GOLDEN_DATA_DIR``. The stored smoke cache comes from
``DOF_GOLDEN_SMOKE_CACHE`` (read only; a fixture checks that no file of it changes) and the order-book tables from
``DOF_GOLDEN_BOOK_ROOT``. Everything runs on the CPU with two torch threads, as recorded. Every ``dof.deep`` module
except ``config`` is imported inside the fixture or test that uses it, so a module that cannot be imported fails
the tests that need it, not the collection of this file.

The batch test forks loader processes, so it comes first: it forks before any torch arithmetic of the process.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from dof.deep.config import CachePaths, MachineConfig, Seeds, legacy_config_dict, load
from recipes import compare as C
from recipes import hashing as H

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR_ENV = "DOF_GOLDEN_DATA_DIR"
SMOKE_ENV = "DOF_GOLDEN_SMOKE_CACHE"
BOOK_ENV = "DOF_GOLDEN_BOOK_ROOT"
FULL_DIR_ENV = "DOF_GOLDEN_FULL_DIR"        # diagnostics only: the full-form goldens with clear values

NET_KEYS = ("LSTM-OF", "CNN-LSTM-OF", "LSTM-MLP-OF", "LSTM3-OF", "MLP-OF", "LSTM-OFI")
FIRST_SEEDS = (0, 3089492870, 771810125)
EXTRA_SEEDS = (1, 2)
N_BATCHES = 64
REPEAT = ("LSTM-MLP-OF-recent", 0)
RESTARTS = (("LSTM-MLP-OF-recent", 3089492870), ("CNN-LSTM-OF-alldays", 0))   # the second covers the Keras step count
CACHE_FILES = {".npy": 36, ".npz": 6, ".json": 15}
TORCH_THREADS = 2
GPU_CAP = 9663676416        # the reference's allocator cap; only copied into the compatibility dict

pytestmark = [pytest.mark.golden, pytest.mark.local]


def _env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


@pytest.fixture(scope="module", autouse=True)
def torch_cpu():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.fail("the training goldens are compared on the CPU: hide the GPU (CUDA_VISIBLE_DEVICES=)", pytrace=False)
    torch.set_num_threads(TORCH_THREADS)
    return torch


@pytest.fixture(scope="module")
def smoke_cfg():
    machine = MachineConfig(book_root=_env(BOOK_ENV), device="cpu", torch_threads=TORCH_THREADS, loader_workers=0,
                            loader_prefetch=4, gpu_memory_limit_bytes=GPU_CAP)
    return replace(load(ROOT / "configs" / "smoke.toml"), machine=machine)


@pytest.fixture(scope="module")
def dry_cfg(smoke_cfg):
    """Two checks of 12 steps; every validation day cut to its first 1,024 samples."""
    return replace(smoke_cfg, train=replace(smoke_cfg.train, check_steps=12, max_checks=2, eval_row_cap=1024))


def _snapshot(cache: CachePaths) -> dict:
    return {p.as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for root in (cache.train, cache.eval) for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def stored():
    """The stored smoke cache, read only."""
    root = _env(SMOKE_ENV)
    if root is None:
        pytest.skip(f"needs {SMOKE_ENV}")
    cache = CachePaths(train=root / "train", eval=root / "eval")
    before = _snapshot(cache)
    yield cache
    assert _snapshot(cache) == before, "a file of the stored smoke cache changed"


@pytest.fixture(scope="module")
def tmp(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("train")


def _golden(load_golden, name: str, *roots: str) -> dict:
    missing = [r for r in (DATA_DIR_ENV, *roots) if _env(r) is None]
    if missing:
        pytest.skip("needs " + " and ".join(missing))
    golden = load_golden(_env(DATA_DIR_ENV) / "train" / f"{name}.json")
    assert golden["env"]["torch_intraop_threads"] == TORCH_THREADS
    return golden


def _compare(golden: dict, values: dict, prefix: str | None = None) -> None:
    got = H.committed_form(values, True)
    want = golden["values"] if prefix is None else {k: v for k, v in golden["values"].items() if k.startswith(prefix)}
    d = C.diff(want, got)
    full_dir = _env(FULL_DIR_ENV)
    if d and full_dir is not None:
        # The full form holds clear values; keys that only the reference implementation produces are dropped.
        full = json.loads((full_dir / "train" / f"{golden['golden']}.json").read_text(encoding="utf-8"))

        def keep(vals: dict) -> dict:
            return {k: v for k, v in vals.items()
                    if not k.startswith(H.DROPPED_PREFIXES) and (prefix is None or k.startswith(prefix))}
        print("full-form differences:\n" + "\n".join(C.diff(keep(full["values"]), keep(values))[:50]))
    assert not d, f"{len(d)} differences:\n" + "\n".join(d[:50])


def test_batches(load_golden, dry_cfg, stored, torch_cpu) -> None:
    """T2: the training days, the batches of the seed-0 pass (loader processes, in-process, from batch 10), and the
    pass hashes of five seeds."""
    golden = _golden(load_golden, "batches", SMOKE_ENV)
    torch = torch_cpu
    from dof.deep.data.stream import close_days, draw_pass, make_loader, steps_per_pass, train_days
    from dof_train_adapter import day_rows, take_batches
    from recipes import train_values as TV
    split, bs, hist = dry_cfg.split, dry_cfg.train.batch_size, dry_cfg.features.history
    out: dict = {}
    days = {w: train_days(dry_cfg, split, w, stored, compact=False) for w in ("recent", "all")}
    try:
        for w, ds in days.items():
            out.update(TV.under(f"days/{w}", TV.days_values(day_rows(ds), sum(len(d.rows) for d in ds),
                                                             steps_per_pass(ds, bs))))
        all_days = days["all"]
        plan = draw_pass(all_days, torch.Generator().manual_seed(0), batch_size=bs)
        w4 = TV.batch_values(take_batches(make_loader(all_days, plan, history=hist, workers=4, prefetch=4), N_BATCHES))
        w0_list = take_batches(make_loader(all_days, plan, history=hist, workers=0, prefetch=4), N_BATCHES)
        w0 = TV.batch_values(w0_list)
        s10 = TV.batch_values(take_batches(make_loader(all_days, plan, history=hist, workers=0, prefetch=4, start=10), 5))
        out.update(TV.under("batches/all/seed=0", w0))
        out["batches/all/seed=0/workers4_equal"] = w4 == w0
        out["batches/all/seed=0/start10_equal"] = s10 == TV.batch_values(w0_list[10:15])
        del w0_list
        for s in (*FIRST_SEEDS, *EXTRA_SEEDS):
            for w, ds in days.items():
                gen = torch.Generator().manual_seed(s)
                p1 = draw_pass(ds, gen, batch_size=bs)
                p2 = draw_pass(ds, gen, batch_size=bs)
                key = f"passes/{w}/seed={s}"
                out[f"{key}/pass1"] = TV.pass_sha256(p1.batches, p1.order)
                out[f"{key}/pass2"] = TV.pass_sha256(p2.batches, p2.order)
                out[f"{key}/n_batches"] = int(len(p1.batches))
    finally:
        for ds in days.values():
            close_days(ds)
    _compare(golden, out)


def test_batches_compact(load_golden, dry_cfg, stored, tmp, torch_cpu) -> None:
    """T2 with the compact backend: the stored record files converted into a store in ``tmp``; the 64 first batches
    of the seed-0 pass read from it, in this process and with loader processes."""
    golden = _golden(load_golden, "batches", SMOKE_ENV)
    pytest.importorskip("zstandard")
    torch = torch_cpu
    from dof.deep.data import compact
    from dof.deep.data.store import load_store
    from dof.deep.data.stream import close_days, draw_pass, make_loader, train_days
    from dof_train_adapter import take_batches
    from recipes import train_values as TV
    args = (stored.train, tmp / "compact", "*/train_*.rec.npy", ["plan.json", "summary.json"])
    compact.convert(*args, rate=0, pause=0)
    compact.finalize(*args, rate=0, pause=0)
    cache = CachePaths(stored.train, stored.eval, compact=tmp / "compact")
    cfg = replace(dry_cfg, machine=replace(dry_cfg.machine, compact_store=True, compact_min_available_after=512 * 2**20,
                                           compact_min_cgroup_room_after=256 * 2**20))
    days = train_days(cfg, cfg.split, "all", cache, compact=True)
    try:
        load_store(days)
        plan = draw_pass(days, torch.Generator().manual_seed(0), batch_size=cfg.train.batch_size)
        for workers in (0, 4):
            loader = make_loader(days, plan, history=cfg.features.history, workers=workers, prefetch=4)
            values = TV.under("batches/all/seed=0", TV.batch_values(take_batches(loader, N_BATCHES)))
            _compare(golden, values, prefix="batches/all/seed=0/b")
    finally:
        close_days(days)


def test_legacy_config_hash_matches_fits_golden(load_golden, dry_cfg) -> None:
    golden = _golden(load_golden, "fits")
    for name in dry_cfg.train.fit_order:
        for s in FIRST_SEEDS:
            doc = legacy_config_dict(dry_cfg, dry_cfg.split, Seeds.same(s))
            got = hashlib.sha256(H.canonical_json(doc).encode()).hexdigest()
            assert got == golden["values"][f"fit/{name}/seed={s}/config"], (name, s)


def test_cache(load_golden, smoke_cfg, tmp) -> None:
    """T1: the smoke cache rebuilt from the order-book tables has the stored cache's hashes."""
    golden = _golden(load_golden, "cache", BOOK_ENV)
    from dof.deep.cache import build
    from recipes import train_values as TV
    root = tmp / "cache_rebuild"
    cache = CachePaths(root / "train", root / "eval")
    build(smoke_cfg, smoke_cfg.split, cache, workers=0)
    files = [p for p in root.rglob("*") if p.is_file()]
    counts = {suf: sum(p.suffix == suf for p in files) for suf in CACHE_FILES}
    assert counts == CACHE_FILES and len(files) == sum(CACHE_FILES.values()), counts
    values = TV.cache_values(cache.train, cache.eval)
    # The golden's own key "stored_equals_rebuilt" is the reference's check stored == rebuilt; its other keys are
    # the stored hashes, so equality with them is the same check for this rebuild.
    assert golden["values"]["stored_equals_rebuilt"] is True
    _compare({**golden, "values": {k: v for k, v in golden["values"].items() if k != "stored_equals_rebuilt"}}, values)


def test_init(load_golden, smoke_cfg, torch_cpu) -> None:
    """T3: the initial weights of the six catalogue models for the first three seeds."""
    golden = _golden(load_golden, "init")
    torch = torch_cpu
    from dof.deep.models import build
    from dof_train_adapter import flat, named_parameters
    from recipes import train_values as TV
    out: dict = {}
    for key in NET_KEYS:
        for s in FIRST_SEEDS:
            torch.manual_seed(s)
            m = build(smoke_cfg.models[key])
            out.update(TV.under(f"init/{key}/seed={s}", TV.init_values(named_parameters(m), flat(m))))
    _compare(golden, out)


class _Interrupted(Exception):
    """Stands for a fit that is stopped between two checks."""


def _say(prefix: str):
    return lambda msg: print(f"  [{prefix}] {msg}", flush=True)


def _fit(cfg, name: str, seed: int, out: Path, cache: CachePaths, **kw):
    from dof.deep.train.fit import fit
    rec = fit(cfg, cfg.split, name, Seeds.same(seed), out, cache=cache, log=_say(f"{name} seed {seed}"), **kw)
    assert rec.status == "finished", (name, seed, rec.status)
    return rec


def _weights_flat(cfg, name: str, rec) -> str:
    """The saved weights loaded into a fresh model of the fit's architecture, as one array hash."""
    import torch

    from dof.deep.models import build, flat_weights
    m = build(cfg.models[cfg.fits[name].arch])
    m.load_state_dict(torch.load(rec.weights_path, map_location="cpu", weights_only=True))
    return H.array_sha256(flat_weights(m))


def test_fits(load_golden, dry_cfg, stored, tmp, torch_cpu) -> None:
    """T4: the dry fits of three seeds; the same fit again (C4); with deterministic algorithms (C6); stopped at its
    second check and continued (C5)."""
    golden = _golden(load_golden, "fits", SMOKE_ENV)
    torch = torch_cpu
    from dof.deep.train.fit import FitHooks, fit
    from recipes import train_values as TV
    out: dict = {}
    records = {}
    for name in dry_cfg.train.fit_order:
        for s in FIRST_SEEDS:
            rec = _fit(dry_cfg, name, s, tmp / "fits" / f"seed{s}" / name, stored)
            flat = _weights_flat(dry_cfg, name, rec)
            records[(name, s)] = flat
            out.update(TV.under(f"fit/{name}/seed={s}", TV.fit_values(rec.doc, flat)))

    # C4: the same fit again gives the same weights.
    name, s = REPEAT
    again = _fit(dry_cfg, name, s, tmp / "fits_repeat" / f"seed{s}" / name, stored)
    out[f"repeat/{name}/seed={s}/equal"] = _weights_flat(dry_cfg, name, again) == records[(name, s)]

    # C6: the same fit with torch's deterministic algorithms on.
    det = f"det_flag/{name}/seed={s}"
    det_cfg = replace(dry_cfg, machine=replace(dry_cfg.machine, deterministic=True))
    try:
        rec = _fit(det_cfg, name, s, tmp / "fits_det" / f"seed{s}" / name, stored)
        out[f"{det}/equal_to_default"] = _weights_flat(det_cfg, name, rec) == records[(name, s)]
    except RuntimeError as exc:
        out[f"{det}/error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        torch.use_deterministic_algorithms(False)

    # C5: a fit stopped at its second check and continued gives the uninterrupted fit's weights.
    def stop_at_two(n: int) -> None:
        if n == 2:
            raise _Interrupted(f"stopped at the validation reading of check {n}")
    for name, s in RESTARTS:
        res = tmp / "restart" / f"seed{s}" / name
        with pytest.raises(_Interrupted):
            fit(dry_cfg, dry_cfg.split, name, Seeds.same(s), res, cache=stored, hooks=FitHooks(before_validation=stop_at_two),
                log=_say(f"{name} seed {s} (stopped)"))
        assert (res / f"checkpoint_seed{s}.pt").exists() and not (res / f"fit_seed{s}.json").exists()
        rec = _fit(dry_cfg, name, s, res, stored)
        out[f"restart/{name}/seed={s}/equal"] = _weights_flat(dry_cfg, name, rec) == records[(name, s)]
        out[f"restart/{name}/seed={s}/resumes"] = H.json_value([r["after_check"] for r in rec.doc.get("resumes", [])])
    _compare(golden, out)


def test_ridge(load_golden, dry_cfg, stored, tmp) -> None:
    """T5: the two linear references, with one worker process."""
    golden = _golden(load_golden, "ridge", SMOKE_ENV)
    from dof.deep.linear.ridge import fit_ridge
    from recipes import train_values as TV
    root = tmp / "ridge_seed0"
    recs = fit_ridge(dry_cfg, dry_cfg.split, root, cache=stored, workers=1)
    out: dict = {}
    for name in dry_cfg.linear:
        out.update(TV.under(f"ridge/{name}", TV.ridge_values(recs[name].doc, root / name / "weights_seed0.npz")))
    _compare(golden, out)


@pytest.fixture(scope="module")
def seed0_fits(dry_cfg, stored, tmp) -> dict:
    """The three dry seed-0 fits, made once, into ``tmp/score_seed0/<name>``; the test-day and validation scores
    use them."""
    root = tmp / "score_seed0"
    return {name: _fit(dry_cfg, name, 0, root / name, stored) for name in dry_cfg.train.fit_order}


def test_scores_test(load_golden, dry_cfg, stored, tmp, seed0_fits) -> None:
    """T6: the test days scored once for the three seed-0 fits and the two linear references."""
    golden = _golden(load_golden, "scores_test", SMOKE_ENV)
    from dof.deep.evaluate.scoring import score_test
    from dof.deep.linear.ridge import fit_ridge
    from recipes import train_values as TV
    root = tmp / "score_seed0"
    fit_ridge(dry_cfg, dry_cfg.split, root, cache=stored, workers=1)
    score_test(dry_cfg, dry_cfg.split, root, cache=stored, workers=1)       # the row cap is never applied here
    out: dict = {}
    for name in dry_cfg.scored_models:
        doc = json.loads((root / name / "test_seed0.json").read_text(encoding="utf-8"))
        out.update(TV.under(f"test/{name}", TV.score_doc_values(doc)))
    _compare(golden, out)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["LSTM-MLP-OF-recent", "LSTM-MLP-OF-alldays", "CNN-LSTM-OF-alldays"],
                         ids=["lstm_mlp_recent", "lstm_mlp_alldays", "cnn_lstm_alldays"])
def test_scores_valid(load_golden, smoke_cfg, stored, seed0_fits, name, torch_cpu) -> None:
    """T7: the validation cells and per-stock losses of a seed-0 fit's saved weights, on the whole validation days."""
    golden = _golden(load_golden, "scores_valid", SMOKE_ENV)
    torch = torch_cpu
    from dof.deep.evaluate.parts import score_part
    from dof.deep.models import build
    from recipes import train_values as TV
    rec = seed0_fits[name]
    model = build(smoke_cfg.models[smoke_cfg.fits[name].arch])
    model.load_state_dict(torch.load(rec.weights_path, map_location="cpu", weights_only=True))
    cpu = torch.device("cpu")
    offs = torch.arange(-smoke_cfg.features.history + 1, 1, device=cpu)
    rows, per = score_part(model, smoke_cfg, smoke_cfg.split, stored, "valid", cpu, offs, name=name,
                           n_train=rec.doc["train"]["samples"], seed=0, amp=False)
    out = TV.under(f"valid/{name}", TV.cells_values("cells", rows))
    out.update(H.flatten({"loss_by_stock": per}, f"valid/{name}"))
    _compare(golden, out, prefix=f"valid/{name}/")
