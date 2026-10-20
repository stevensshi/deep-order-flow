"""The keys of a run configuration and of a local machine file: their types and defaults, and the helpers that
check one TOML table against them.

A schema maps each allowed key to ``(check, default)``; ``check(dotted_key, value)`` returns the value in its
Python form (lists become tuples) or raises ``ConfigError`` naming the dotted key. ``REQUIRED`` marks a key
without a default. Path values stay strings here; ``config.load`` resolves them against the folder of the file
that sets them.
"""

from __future__ import annotations

import datetime
import re
import tomllib
from pathlib import Path
from typing import Callable

from dof.deep.errors import ConfigError

REQUIRED = object()
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HHMM = re.compile(r"([01][0-9]|2[0-3]):[0-5][0-9]")

Check = Callable[[str, object], object]


def _fail(key: str, what: str, value) -> ConfigError:
    return ConfigError(f"{key}: {what}, got {value!r}")


def string(key: str, v):
    if not isinstance(v, str):
        raise _fail(key, "a string is needed", v)
    return v


def boolean(key: str, v):
    if not isinstance(v, bool):
        raise _fail(key, "true or false is needed", v)
    return v


def integer(key: str, v):
    if isinstance(v, bool) or not isinstance(v, int):
        raise _fail(key, "an integer is needed", v)
    return v


def number(key: str, v):
    """A float; an integer is taken as the float of the same value."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise _fail(key, "a number is needed", v)
    return float(v)


def iso_date(key: str, v):
    try:
        datetime.date.fromisoformat(string(key, v))
    except ValueError:
        raise _fail(key, "an ISO date (YYYY-MM-DD) is needed", v) from None
    if len(v) != 10:
        raise _fail(key, "an ISO date (YYYY-MM-DD) is needed", v)
    return v


def sha256_hex(key: str, v):
    if not isinstance(v, str) or not _HEX64.fullmatch(v):
        raise _fail(key, "a sha256 in lower-case hex is needed", v)
    return v


def one_of(*choices: str) -> Check:
    def check(key: str, v):
        if v not in choices:
            raise _fail(key, f"one of {list(choices)} is needed", v)
        return v
    return check


def list_of(item: Check, *, non_empty: bool = False, length: int | None = None) -> Check:
    def check(key: str, v):
        if not isinstance(v, list):
            raise _fail(key, "a list is needed", v)
        if non_empty and not v:
            raise _fail(key, "a non-empty list is needed", v)
        if length is not None and len(v) != length:
            raise _fail(key, f"a list of {length} items is needed", v)
        return tuple(item(f"{key}[{i}]", x) for i, x in enumerate(v))
    return check


def table_of(key_check: Check, value_check: Check) -> Check:
    """A TOML table of free keys -> a plain dict in the file's order."""
    def check(key: str, v):
        if not isinstance(v, dict):
            raise _fail(key, "a table is needed", v)
        return {key_check(f"{key} key", k): value_check(f"{key}.{k}", x) for k, x in v.items()}
    return check


def hh_mm(key: str, v):
    if not isinstance(v, str) or not _HHMM.fullmatch(v):
        raise _fail(key, "a time HH:MM is needed", v)
    return v


path = string        # resolved later against the folder of the file that sets it

TOP_LEVEL = ("include", "split", "calendar", "data", "protocol", "features", "train", "ridge", "models", "fits",
             "linear")

SPLIT = {
    "name": (string, REQUIRED),
    "train_days": (list_of(iso_date, non_empty=True), REQUIRED),
    "recent_days": (list_of(iso_date), REQUIRED),
    "valid_days": (list_of(iso_date, non_empty=True), REQUIRED),
    "test_days": (list_of(iso_date, non_empty=True), REQUIRED),
    "symbols": (list_of(string, non_empty=True), None),
    "reused_from": (table_of(string, iso_date), {}),
    "price_rule_parts": (list_of(one_of("valid", "test")), ("valid",)),
}
CALENDAR = {"half_days": (table_of(iso_date, hh_mm), {})}
DATA = {"universe": (path, REQUIRED)}
PROTOCOL = {
    "corporate_actions": (path, REQUIRED),
    "corporate_actions_sha256": (sha256_hex, REQUIRED),
    "ca_columns": (list_of(string, length=4), ("ticker", "event", "effective_date", "status")),
    "ca_excluding": (list_of(string), ("forward_split", "reverse_split", "ticker_change", "ticker_reused")),
    "ca_confirmed": (string, "confirmed"),
    "document": (path, None),
    "document_sha256": (sha256_hex, None),
    "test_days_need_document": (boolean, True),
}
FEATURES = {
    "convention": (one_of("cks", "current_size"), REQUIRED),
    "dedupe": (boolean, REQUIRED),
    "history": (integer, REQUIRED),
    "latency_ns": (integer, REQUIRED),
    "horizons": (list_of(integer, non_empty=True), REQUIRED),
    "gap_days": (integer, REQUIRED),
    "ratio_range": (list_of(number, length=2), REQUIRED),
}
TRAIN = {
    "batch_size": (integer, REQUIRED),
    "check_steps": (integer, REQUIRED),
    "max_checks": (integer, REQUIRED),
    "patience": (integer, REQUIRED),
    "adam_betas": (list_of(number, length=2), REQUIRED),
    "eval_batch": (integer, REQUIRED),
    "restore_best": (boolean, REQUIRED),
    "fit_order": (list_of(string), REQUIRED),
    "eval_row_cap": (integer, None),
    "restore_rtol": (number, 1e-6),
}
RIDGE = {"lambdas": (list_of(number, non_empty=True), REQUIRED), "chunk_rows": (integer, REQUIRED)}
MODEL = {
    "family": (string, REQUIRED),
    "input": (one_of("of", "ofi"), REQUIRED),
    "input_dim": (integer, REQUIRED),
    "init": (one_of("keras", "torch"), REQUIRED),
    "precision": (one_of("float32", "amp"), REQUIRED),
}
FIT = {
    "arch": (string, REQUIRED),
    "optimizer": (one_of("adam", "keras_adam"), REQUIRED),
    "lr": (number, REQUIRED),
    "eps": (number, REQUIRED),
    "days": (one_of("all", "recent"), REQUIRED),
    "setting_source": (string, None),
}
LINEAR = {"days": (one_of("all", "recent"), REQUIRED)}

MACHINE = {
    "book_root": (path, None),
    "device": (one_of("cpu", "cuda"), "cpu"),
    "torch_threads": (integer, 0),
    "loader_workers": (integer, 0),
    "loader_prefetch": (integer, 4),
    "progress_every": (integer, 500),
    "gpu_memory_limit_bytes": (integer, None),
    "cudnn_benchmark": (boolean, True),
    "cudnn_allow_tf32": (boolean, True),
    "matmul_allow_tf32": (boolean, False),
    "deterministic": (boolean, False),
    "compact_store": (boolean, False),
    # Free memory that must remain after the compact store is loaded (bytes). An integer only: the local file
    # cannot switch the two rules off.
    "compact_min_available_after": (integer, 4 * 2**30),
    "compact_min_cgroup_room_after": (integer, 4 * 2**30),
    "disk_guard_path": (path, None),
    "disk_floor_gb": (integer, 150),
    "build_floor_gb": (integer, 160),
}
MACHINE_CACHE = {"train": (path, REQUIRED), "eval": (path, REQUIRED), "compact": (path, None)}


def parse_table(name: str, raw, schema: dict, *, rest: str | None = None) -> dict:
    """The keys of ``schema`` from the TOML table ``raw`` (defaults filled in). An unknown key is a
    ``ConfigError`` naming ``name.key``, unless ``rest`` names the entry that collects the other keys."""
    if not isinstance(raw, dict):
        raise _fail(name, "a table is needed", raw)
    out: dict = {}
    for key, (check, default) in schema.items():
        if key in raw:
            out[key] = check(f"{name}.{key}", raw[key])
        elif default is REQUIRED:
            raise ConfigError(f"{name}.{key}: the key is required")
        else:
            out[key] = default
    extra = {k: v for k, v in raw.items() if k not in schema}
    if rest is not None:
        out[rest] = extra
    elif extra:
        raise ConfigError(f"{name}.{next(iter(extra))}: unknown key")
    return out


def resolve(base: Path, value: str) -> Path:
    """``value`` relative to the folder ``base``; an absolute ``value`` is kept."""
    p = Path(value)
    return p if p.is_absolute() else Path(base) / p


def read_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"{path}: no such configuration file") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from None


def _merge(dst: dict, src: dict, file: Path, origin: dict, prefix: str = "") -> None:
    """Recursive over tables; a leaf key or array set in two files is an error naming both files."""
    for k, v in src.items():
        dotted = f"{prefix}{k}"
        if isinstance(v, dict):
            if k in dst and not isinstance(dst[k], dict):
                raise ConfigError(f"{dotted} is set in both {origin.get(dotted, 'another file')} and {file}")
            _merge(dst.setdefault(k, {}), v, file, origin, dotted + ".")
        else:
            if k in dst:
                where = origin.get(dotted, "another file")
                raise ConfigError(f"{dotted} is set in both {where} and {file}")
            dst[k] = v
            origin[dotted] = file


def read_tree(path: Path, stack: tuple[Path, ...], tree: dict, origin: dict) -> None:
    """Depth first, in list order: the files a file includes, then the file's own keys."""
    if path in stack:
        raise ConfigError("include cycle: " + " -> ".join(str(p) for p in (*stack, path)))
    raw = read_toml(path)
    include = list_of(string)(f"{path.name}: include", raw.pop("include", []))
    for item in include:
        if Path(item).is_absolute():
            raise ConfigError(f"{path}: include {item!r} is an absolute path")
        read_tree((path.parent / item).resolve(), (*stack, path), tree, origin)
    _merge(tree, raw, path, origin)


def named_tables(tree: dict, table: str) -> dict:
    raw = tree.get(table, {})
    if not isinstance(raw, dict) or any(not isinstance(v, dict) for v in raw.values()):
        raise ConfigError(f"{table}: one sub-table per name is needed ([{table}.<name>])")
    return raw


def tuples(v):
    if isinstance(v, list):
        return tuple(tuples(x) for x in v)
    return v
