"""The model registry: one file per family, each registering itself with ``@register("<family>")``.

On import, this package imports every module of its own folder (sorted by name; names starting with "_" and
``init`` excepted), so a new family is one new file plus one ``[models.<NAME>]`` entry of a configuration; no other
file changes.

``build(model_cfg)`` constructs the family's module, whose default PyTorch initialisation draws from the global
generator in construction order; with ``init = "keras"`` the Keras initialisation then draws again, in
``module.modules()`` order. A seed set with ``torch.manual_seed`` before ``build`` therefore fixes the weights.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Callable

import torch
from torch import nn

from dof.deep.config import ModelConfig
from dof.deep.errors import ConfigError
from dof.deep.models.init import keras_init

FAMILIES: dict[str, Callable[[ModelConfig], nn.Module]] = {}


def register(name: str) -> Callable:
    """Decorator: ``@register("lstm")`` on a function that takes a ``ModelConfig`` and returns the module."""
    def deco(fn: Callable[[ModelConfig], nn.Module]) -> Callable[[ModelConfig], nn.Module]:
        if name in FAMILIES:
            raise ValueError(f"model family {name!r} is registered twice")
        FAMILIES[name] = fn
        return fn
    return deco


def build(model_cfg: ModelConfig) -> nn.Module:
    try:
        make = FAMILIES[model_cfg.family]
    except KeyError:
        raise ConfigError(f"models.{model_cfg.name}.family: unknown model family {model_cfg.family!r} "
                          f"(registered: {sorted(FAMILIES)})") from None
    m = make(model_cfg)
    if model_cfg.init == "keras":
        keras_init(m)
    return m


def n_params(m: nn.Module) -> dict[str, int]:
    """PyTorch count, and the Keras-equivalent count (without nn.LSTM's second bias vector)."""
    total = sum(p.numel() for p in m.parameters())
    extra = sum(p.numel() for mod in m.modules() if isinstance(mod, nn.LSTM)
                for name, p in mod.named_parameters() if name.startswith("bias_hh"))
    return {"pytorch": total, "keras_equivalent": total - extra}


def flat_weights(m: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in m.parameters()])


def _discover() -> None:
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if not info.name.startswith("_") and info.name != "init":
            importlib.import_module(f"{__name__}.{info.name}")


_discover()
