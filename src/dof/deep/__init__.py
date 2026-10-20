"""Deep order-flow forecasting models on the features of ``dof.lob``.

- ``config``: run configurations from TOML (``load``), machine settings from a local file, result identities;
- ``cache``: the feature cache of a split (one record file per training stock-day, whole arrays per validation
  and test stock-day);
- ``data``: the training stream, the evaluation days, the optional compact backend;
- ``models``: the model registry, one file per family;
- ``train``: the fit loop (early stopping, checkpoints, restart);
- ``evaluate``: validation losses, score cells and the test-day gate;
- ``linear``: the pooled ridge references.

Importing this package does not import torch: the names that need it are loaded on first use.
"""

from __future__ import annotations

import importlib

from dof.deep.config import (
    CachePaths,
    MachineConfig,
    RunConfig,
    Seeds,
    Split,
    identity,
    legacy_config_dict,
    load,
    load_machine,
)
from dof.deep.errors import (
    CacheError,
    CheckpointConflict,
    ConfigError,
    DeviceError,
    DiskFloor,
    Diverged,
    RecordConflict,
    RestoreMismatch,
    TestDayRefused,
)

# name -> (module, attribute), imported on first access (PEP 562)
_LAZY = {
    "register": ("dof.deep.models", "register"),
    "build_model": ("dof.deep.models", "build"),
    "fit": ("dof.deep.train.fit", "fit"),
    "FitRecord": ("dof.deep.train.record", "FitRecord"),
    "FitHooks": ("dof.deep.train.fit", "FitHooks"),
    "score_part": ("dof.deep.evaluate.parts", "score_part"),
    "score_test": ("dof.deep.evaluate.scoring", "score_test"),
    "TestGate": ("dof.deep.evaluate.gate", "TestGate"),
    "fit_ridge": ("dof.deep.linear.ridge", "fit_ridge"),
    "build_cache": ("dof.deep.cache", "build"),
}

__all__ = ["CacheError", "CachePaths", "CheckpointConflict", "ConfigError", "DeviceError", "DiskFloor", "Diverged",
           "MachineConfig", "RecordConflict", "RestoreMismatch", "RunConfig", "Seeds", "Split", "TestDayRefused",
           "identity", "legacy_config_dict", "load", "load_machine", *_LAZY]


def __getattr__(name: str):
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'dof.deep' has no attribute {name!r}") from None
    return getattr(importlib.import_module(module), attr)
