"""LSTM (one layer or stacked) and LSTM-MLP (Kolm, Turiel and Westray 2023, Table 2): the last hidden state to the
outputs."""

from __future__ import annotations

import torch
from torch import nn

from dof.deep.config import ModelConfig
from dof.deep.models import register


class Lstm(nn.Module):
    """``hidden`` > 0 gives the LSTM-MLP: the last hidden state goes through one ReLU layer first."""

    def __init__(self, d: int, units: int, layers: int = 1, hidden: int = 0, *, n_out: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(d, units, num_layers=layers, batch_first=True)
        if hidden:
            self.head: nn.Module = nn.Sequential(nn.Linear(units, hidden), nn.ReLU(), nn.Linear(hidden, n_out))
        else:
            self.head = nn.Linear(units, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


@register("lstm")
def build_lstm(cfg: ModelConfig) -> nn.Module:
    return Lstm(cfg.input_dim, cfg.param("units"), cfg.param("layers"), hidden=0, n_out=cfg.n_out)


@register("lstm_mlp")
def build_lstm_mlp(cfg: ModelConfig) -> nn.Module:
    return Lstm(cfg.input_dim, cfg.param("units"), 1, cfg.param("hidden"), n_out=cfg.n_out)
