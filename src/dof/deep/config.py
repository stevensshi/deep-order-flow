"""Run configurations: frozen dataclasses loaded from TOML.

A run configuration (``load``) holds everything that defines a result: the split, the calendar, the universe,
the protocol files, the features, the training and ridge settings, the model catalogue, the fits and the linear
references. Machine settings (device, threads, loader processes, cache folders, memory and disk guards) come from
a separate local file (``load_machine``) and never enter a result's identity.

Two dicts describe one fit or ridge result:
- ``identity(cfg, split, seeds, model)``: what defines that one result, and nothing else (no machine setting, no
  absolute path, no other fit). Records are reused and checkpoints resumed only under an equal identity.
- ``legacy_config_dict(cfg, split, seeds)``: the configuration dict of the reference implementation, key for
  key, kept in every record for compatibility with the stored results. It holds one machine setting (the GPU
  memory cap) because the stored records hold it; it is not the identity.

``load`` never imports torch and opens no data file; ``identity`` reads the universe file to hash it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dof.deep import config_schema as S
from dof.deep.errors import ConfigError, TestDayRefused
from dof.lob.provenance import sha256_file

LOCAL_ENV = "DOF_LOCAL_CONFIG"
PATH_KEYS = ("data.universe", "protocol.corporate_actions", "protocol.document")


@dataclass(frozen=True)
class Split:
    """One chronological split of days (ISO dates)."""
    name: str
    train_days: tuple[str, ...]
    recent_days: tuple[str, ...]
    valid_days: tuple[str, ...]
    test_days: tuple[str, ...]
    symbols: tuple[str, ...] | None         # None: every ticker of the universe file
    reused_from: tuple[tuple[str, str], ...]  # ticker -> first date on which it belongs to another issuer
    price_rule_parts: tuple[str, ...]       # parts whose stock-days the price-ratio rule leaves out at build time

    def days(self, which: str) -> tuple[str, ...]:
        return {"all": self.train_days, "recent": self.recent_days, "valid": self.valid_days, "test": self.test_days}[which]

    def check(self) -> None:
        """The split is chronological and its parts do not overlap."""
        tr, va, te = self.train_days, self.valid_days, self.test_days
        if list(tr) != sorted(set(tr)) or list(va) != sorted(set(va)) or list(te) != sorted(set(te)):
            raise ConfigError(f"split {self.name}: a part is not in date order or names a day twice")
        if not (tr[-1] < va[0] and va[-1] < te[0]):
            raise ConfigError(f"split {self.name}: the split is not chronological")
        if not set(self.recent_days) <= set(tr):
            raise ConfigError(f"split {self.name}: a recent day is not a training day")


@dataclass(frozen=True)
class FeatureConfig:
    convention: str
    dedupe: bool
    history: int
    latency_ns: int
    horizons: tuple[int, ...]
    gap_days: int
    ratio_lo: float
    ratio_hi: float


@dataclass(frozen=True)
class ProtocolConfig:
    corporate_actions: Path
    corporate_actions_sha256: str
    ca_columns: tuple[str, ...]
    ca_excluding: tuple[str, ...]
    ca_confirmed: str
    document: Path | None
    document_sha256: str | None
    test_days_need_document: bool


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    check_steps: int
    max_checks: int
    patience: int
    adam_betas: tuple[float, float]
    eval_batch: int
    restore_best: bool
    fit_order: tuple[str, ...]
    eval_row_cap: int | None = None         # the dry run's cut of every validation day to its first rows
    restore_rtol: float = 1e-6              # the saved weights must give back the best check's loss to this tolerance


@dataclass(frozen=True)
class RidgeConfig:
    lambdas: tuple[float, ...]
    chunk_rows: int


@dataclass(frozen=True)
class ModelConfig:
    name: str
    family: str
    input: str
    input_dim: int
    init: str
    precision: str
    n_out: int
    history: int
    params: dict

    def param(self, key: str):
        try:
            return self.params[key]
        except KeyError:
            raise ConfigError(f"models.{self.name}.{key}: the key is required by family {self.family!r}") from None


@dataclass(frozen=True)
class FitSpec:
    name: str
    arch: str
    optimizer: str
    lr: float
    eps: float
    days: str
    setting_source: str | None


@dataclass(frozen=True)
class LinearSpec:
    name: str
    days: str


@dataclass(frozen=True)
class CachePaths:
    train: Path
    eval: Path
    compact: Path | None = None

    @property
    def compact_root(self) -> Path:
        """The compact store of the training records: beside the training cache, never inside it."""
        return self.compact if self.compact is not None else self.train.parent / "compact"


@dataclass(frozen=True)
class MachineConfig:
    book_root: Path | None = None
    device: str = "cpu"
    torch_threads: int = 0                  # 0: torch's own default
    loader_workers: int = 0
    loader_prefetch: int = 4
    progress_every: int = 500
    gpu_memory_limit_bytes: int | None = None
    cudnn_benchmark: bool = True
    cudnn_allow_tf32: bool = True
    matmul_allow_tf32: bool = False
    deterministic: bool = False
    compact_store: bool = False
    compact_min_available_after: int = 4 * 2**30
    compact_min_cgroup_room_after: int = 4 * 2**30
    disk_guard_path: Path | None = None
    disk_floor_gb: int = 150
    build_floor_gb: int = 160
    cache: dict = field(default_factory=dict)       # split name -> CachePaths

    def cache_for(self, split_name: str) -> CachePaths:
        try:
            return self.cache[split_name]
        except KeyError:
            raise ConfigError(f"machine.cache.{split_name} is not set") from None


@dataclass(frozen=True)
class RunConfig:
    path: Path
    split: Split
    half_days: dict
    universe: Path
    protocol: ProtocolConfig
    features: FeatureConfig
    train: TrainConfig
    ridge: RidgeConfig
    models: dict            # name -> ModelConfig
    fits: dict              # name -> FitSpec
    linear: dict            # name -> LinearSpec
    machine: MachineConfig

    @property
    def scored_models(self) -> tuple[str, ...]:
        """The models scored on the test days, and no other: the fits of ``fit_order`` and the linear references."""
        return self.train.fit_order + tuple(self.linear)


@dataclass(frozen=True)
class Seeds:
    """``init`` seeds the global generator before the network is built; ``data`` seeds the fit's own generator
    (pass order, placebo targets, loader)."""
    init: int
    data: int

    @classmethod
    def same(cls, seed: int) -> Seeds:
        return cls(seed, seed)

    @property
    def tag(self) -> str:
        return f"seed{self.init}" if self.init == self.data else f"seed{self.init}-{self.data}"


def load(config: Path, *, local: Path | None = None) -> RunConfig:
    """The run configuration in ``config`` (with its includes) and the machine settings of ``local`` (else the
    file named by ``DOF_LOCAL_CONFIG``, else the defaults)."""
    path = Path(config).resolve()
    tree: dict = {}
    origin: dict = {}
    S.read_tree(path, (), tree, origin)
    unknown = [k for k in tree if k not in S.TOP_LEVEL]
    if unknown:
        raise ConfigError(f"{unknown[0]}: unknown table or key")
    at = {k: origin[k].parent for k in PATH_KEYS if k in origin}

    sp = S.parse_table("split", tree.get("split", {}), S.SPLIT)
    split = Split(sp["name"], sp["train_days"], sp["recent_days"], sp["valid_days"], sp["test_days"], sp["symbols"],
                  tuple(sp["reused_from"].items()), sp["price_rule_parts"])
    split.check()
    cal = S.parse_table("calendar", tree.get("calendar", {}), S.CALENDAR)
    data = S.parse_table("data", tree.get("data", {}), S.DATA)
    pr = S.parse_table("protocol", tree.get("protocol", {}), S.PROTOCOL)
    if pr["test_days_need_document"] and pr["document"] is None:
        raise ConfigError("protocol.document: required when protocol.test_days_need_document is true")
    protocol = ProtocolConfig(
        S.resolve(at["protocol.corporate_actions"], pr["corporate_actions"]), pr["corporate_actions_sha256"],
        pr["ca_columns"], pr["ca_excluding"], pr["ca_confirmed"],
        None if pr["document"] is None else S.resolve(at["protocol.document"], pr["document"]),
        pr["document_sha256"], pr["test_days_need_document"])
    fe = S.parse_table("features", tree.get("features", {}), S.FEATURES)
    lo, hi = fe["ratio_range"]
    if not lo < hi:
        raise ConfigError(f"features.ratio_range: needs lo < hi, got {[lo, hi]}")
    features = FeatureConfig(fe["convention"], fe["dedupe"], fe["history"], fe["latency_ns"], fe["horizons"],
                             fe["gap_days"], lo, hi)
    tr = S.parse_table("train", tree.get("train", {}), S.TRAIN)
    if not tr["restore_best"]:
        raise ConfigError("train.restore_best: only true is supported (the best check's weights are scored)")
    train = TrainConfig(tr["batch_size"], tr["check_steps"], tr["max_checks"], tr["patience"], tr["adam_betas"],
                        tr["eval_batch"], tr["restore_best"], tr["fit_order"], tr["eval_row_cap"], tr["restore_rtol"])
    ri = S.parse_table("ridge", tree.get("ridge", {}), S.RIDGE)
    ridge = RidgeConfig(ri["lambdas"], ri["chunk_rows"])

    models = {}
    for name, raw in S.named_tables(tree, "models").items():
        m = S.parse_table(f"models.{name}", raw, S.MODEL, rest="params")
        models[name] = ModelConfig(name, m["family"], m["input"], m["input_dim"], m["init"], m["precision"],
                                   len(features.horizons), features.history,
                                   {k: S.tuples(v) for k, v in m["params"].items()})
    fits = {}
    for name, raw in S.named_tables(tree, "fits").items():
        f = S.parse_table(f"fits.{name}", raw, S.FIT)
        if f["arch"] not in models:
            raise ConfigError(f"fits.{name}.arch: {f['arch']!r} is not a model of the configuration")
        fits[name] = FitSpec(name, f["arch"], f["optimizer"], f["lr"], f["eps"], f["days"], f["setting_source"])
    linear = {name: LinearSpec(name, S.parse_table(f"linear.{name}", raw, S.LINEAR)["days"])
              for name, raw in S.named_tables(tree, "linear").items()}
    missing = [n for n in train.fit_order if n not in fits]
    if missing:
        raise ConfigError(f"train.fit_order: {missing} are not fits of the configuration")
    return RunConfig(path, split, cal["half_days"], S.resolve(at["data.universe"], data["universe"]), protocol,
                     features, train, ridge, models, fits, linear, load_machine(local))


def load_machine(local: Path | None) -> MachineConfig:
    """The ``[machine]`` table of the local file; paths in it are relative to the file's folder or absolute.
    ``local`` None: the file named by ``DOF_LOCAL_CONFIG``; neither: the defaults."""
    if local is None:
        env = os.environ.get(LOCAL_ENV)
        if not env:
            return MachineConfig()
        local = Path(env)
    path = Path(local).resolve()
    raw = S.read_toml(path)
    extra = [k for k in raw if k != "machine"]
    if extra:
        raise ConfigError(f"{path.name}: {extra[0]}: only a [machine] table is allowed in a local file")
    m = S.parse_table("machine", raw.get("machine", {}), S.MACHINE, rest="_rest")
    rest = m.pop("_rest")
    cache_raw = rest.pop("cache", {})
    if rest:
        raise ConfigError(f"machine.{next(iter(rest))}: unknown key")
    base = path.parent
    cache = {}
    for name, entry in S.named_tables({"cache": cache_raw}, "cache").items():
        c = S.parse_table(f"machine.cache.{name}", entry, S.MACHINE_CACHE)
        cache[name] = CachePaths(S.resolve(base, c["train"]), S.resolve(base, c["eval"]),
                                 None if c["compact"] is None else S.resolve(base, c["compact"]))
    for key in ("book_root", "disk_guard_path"):
        if m[key] is not None:
            m[key] = S.resolve(base, m[key])
    return MachineConfig(**m, cache=cache)


def canonical_json(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


def legacy_fit_dict(spec: FitSpec) -> dict:
    out = {"arch": spec.arch, "optimizer": spec.optimizer, "lr": spec.lr, "eps": spec.eps, "days": spec.days}
    if spec.setting_source is not None:
        out["p0_key"] = spec.setting_source
    return out


def legacy_config_dict(cfg: RunConfig, split: Split, seeds: Seeds) -> dict:
    """The reference implementation's configuration dict, byte-compatible with the stored records."""
    return {"stage": split.name, "train_days": list(split.train_days), "recent_days": list(split.recent_days),
            "valid_days": list(split.valid_days), "test_days": list(split.test_days),
            "symbols": None if split.symbols is None else list(split.symbols), "gap_days": cfg.features.gap_days,
            "ratio_range": [cfg.features.ratio_lo, cfg.features.ratio_hi], "reused_from": dict(split.reused_from),
            "half_days": dict(cfg.half_days), "price_rule_parts": list(split.price_rule_parts),
            "corporate_actions": cfg.protocol.corporate_actions.name,
            "corporate_actions_sha256": cfg.protocol.corporate_actions_sha256,
            "ca_excluding": list(cfg.protocol.ca_excluding), "ca_confirmed": cfg.protocol.ca_confirmed,
            "seed": seeds.init if seeds.init == seeds.data else {"init": seeds.init, "data": seeds.data},
            "history": cfg.features.history, "batch_size": cfg.train.batch_size, "check_steps": cfg.train.check_steps,
            "max_checks": cfg.train.max_checks, "patience": cfg.train.patience,
            "fits": {name: legacy_fit_dict(f) for name, f in cfg.fits.items()},
            "linear": {name: {"days": spec.days} for name, spec in cfg.linear.items()},
            "convention": cfg.features.convention, "dedupe": cfg.features.dedupe,
            "adam_betas": list(cfg.train.adam_betas), "eval_batch": cfg.train.eval_batch,
            "keras_init": all(cfg.models[f.arch].init == "keras" for f in cfg.fits.values()),
            "restore_best": cfg.train.restore_best, "gpu_memory_limit_bytes": cfg.machine.gpu_memory_limit_bytes}


def identity(cfg: RunConfig, split: Split, seeds: Seeds, model: str) -> dict:
    """What defines the result of one fit or linear reference ``model``: never a machine setting, never a path."""
    if model in cfg.fits:
        spec = cfg.fits[model]
        m = cfg.models[spec.arch]
        entry = {**legacy_fit_dict(spec),
                 "model_config": {"family": m.family, "input": m.input, "input_dim": m.input_dim, "init": m.init,
                                  "precision": m.precision, "n_out": m.n_out, "history": m.history,
                                  "params": m.params}}
    elif model in cfg.linear:
        entry = {"days": cfg.linear[model].days,
                 "ridge": {"lambdas": list(cfg.ridge.lambdas), "chunk_rows": cfg.ridge.chunk_rows}}
    else:
        raise ConfigError(f"{model!r} is neither a fit nor a linear reference of the configuration")
    try:
        universe_sha = sha256_file(cfg.universe)
    except FileNotFoundError:
        raise ConfigError(f"data.universe: {cfg.universe} does not exist") from None
    f, p, t = cfg.features, cfg.protocol, cfg.train
    train = {k: getattr(t, k) for k in t.__dataclass_fields__ if k != "fit_order"}
    # Through JSON and back: tuples become lists, so the dict equals the one a record file gives back.
    return json.loads(json.dumps({
        "format": 1,
        "split": {"name": split.name, "train_days": split.train_days, "recent_days": split.recent_days,
                  "valid_days": split.valid_days, "test_days": split.test_days, "symbols": split.symbols,
                  "reused_from": dict(split.reused_from), "price_rule_parts": split.price_rule_parts},
        "half_days": dict(cfg.half_days),
        "features": {"convention": f.convention, "dedupe": f.dedupe, "history": f.history, "latency_ns": f.latency_ns,
                     "horizons": f.horizons, "gap_days": f.gap_days, "ratio_range": [f.ratio_lo, f.ratio_hi]},
        "protocol": {"corporate_actions": p.corporate_actions.name, "corporate_actions_sha256": p.corporate_actions_sha256,
                     "ca_columns": p.ca_columns, "ca_excluding": p.ca_excluding, "ca_confirmed": p.ca_confirmed,
                     "document_sha256": p.document_sha256},
        "universe_sha256": universe_sha,
        "train": train,
        "model": entry,
        "seeds": {"init": seeds.init, "data": seeds.data},
    }))


def identity_sha256(cfg: RunConfig, split: Split, seeds: Seeds, model: str) -> str:
    return hashlib.sha256(canonical_json(identity(cfg, split, seeds, model)).encode()).hexdigest()


def protocol_sha256(cfg: RunConfig) -> str | None:
    """sha256 of the protocol document; None when the configuration names none. A missing document, or one that
    is not the pinned file, refuses the test days."""
    doc = cfg.protocol.document
    if doc is None:
        return None
    if not doc.is_file():
        raise TestDayRefused(f"the protocol document {doc.name} is missing")
    sha = sha256_file(doc)
    if cfg.protocol.document_sha256 is not None and sha != cfg.protocol.document_sha256:
        raise TestDayRefused(f"the protocol document {doc.name} has the sha256 {sha}, not the pinned "
                             f"{cfg.protocol.document_sha256}")
    return sha
