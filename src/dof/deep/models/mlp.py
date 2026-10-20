"""MLP (Kolm, Turiel and Westray 2023, eqs. 7-8): the history x d window flattened, ReLU hidden layers, a linear output per
horizon."""

from __future__ import annotations

import torch
from torch import nn

from dof.deep.config import ModelConfig
from dof.deep.models import register


class Mlp(nn.Module):
    def __init__(self, d: int, hidden: tuple[int, ...], *, history: int, n_out: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        n = history * d
        for h in hidden:
            layers += [nn.Linear(n, h), nn.ReLU()]
            n = h
        layers.append(nn.Linear(n, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


@register("mlp")
def build_mlp(cfg: ModelConfig) -> nn.Module:
    return Mlp(cfg.input_dim, tuple(cfg.param("hidden")), history=cfg.history, n_out=cfg.n_out)
