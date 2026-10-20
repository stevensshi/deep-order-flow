"""A small Transformer encoder on the order-flow window: rows projected to ``d_model``, a learned position
embedding added, pre-norm encoder layers, the last row's state to the horizons.

No attention mask: attention looks both ways inside the window, and the window ends at row t, so no row after t is
read. Dropout and its rate come from the configuration entry; the module draws no random number at import.
"""

from __future__ import annotations

import torch
from torch import nn

from dof.deep.config import ModelConfig
from dof.deep.models import register


class TransformerOF(nn.Module):
    def __init__(self, d: int, *, history: int, n_out: int, d_model: int, heads: int, layers: int, ff: int,
                 dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(d, d_model)
        self.pos = nn.Parameter(torch.zeros(1, history, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, history, d) float32, oldest row first -> (B, n_out)."""
        return self.head(self.encoder(self.proj(x) + self.pos)[:, -1])


@register("transformer")
def build_transformer(cfg: ModelConfig) -> nn.Module:
    return TransformerOF(cfg.input_dim, history=cfg.history, n_out=cfg.n_out, d_model=cfg.param("d_model"),
                         heads=cfg.param("heads"), layers=cfg.param("layers"), ff=cfg.param("ff"),
                         dropout=cfg.param("dropout"))
