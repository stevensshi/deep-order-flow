"""CNN-LSTM (Kolm, Turiel and Westray 2023, Table 1, blocks 2-5, the order-flow variant, p. 12). Tensor layout
(batch, channels, time, levels).

Block 2: (1 x 2) stride (1 x 2) across the two sides of a level, then two (4 x 1) temporal convolutions with
padding. Block 3: (1 x 10) across the ten levels. Block 4: an inception module of three sub-blocks. Block 5: an
LSTM to the horizons. Every convolution is followed by a leaky ReLU. The temporal convolutions pad inside the
window only; nothing after row t is read.
"""

from __future__ import annotations

import torch
from torch import nn

from dof.deep.config import ModelConfig
from dof.deep.models import register

# Block 2 convolves across the sides of each level, so the bid and ask flow of one level must be adjacent columns.
# The order-flow matrix is [bOF_1..10, aOF_1..10]; this reorders it to [bOF_1, aOF_1, bOF_2, aOF_2, ...].
INTERLEAVE = [i // 2 + 10 * (i % 2) for i in range(20)]


class CnnLstm(nn.Module):
    def __init__(self, d: int, *, filters: int, inception_filters: int, lstm_units: int, leaky_slope: float,
                 n_out: int) -> None:
        super().__init__()
        if d != 20:
            raise ValueError("the CNN-LSTM is built for the 20-column OF input")
        f, g, a = filters, inception_filters, leaky_slope
        self.register_buffer("perm", torch.tensor(INTERLEAVE), persistent=False)

        def conv(cin: int, cout: int, kernel: tuple[int, int], stride: tuple[int, int] = (1, 1)) -> list[nn.Module]:
            pad = 0 if stride != (1, 1) or kernel[0] == 1 else "same"
            return [nn.Conv2d(cin, cout, kernel, stride=stride, padding=pad), nn.LeakyReLU(a)]
        self.block2 = nn.Sequential(*conv(1, f, (1, 2), (1, 2)), *conv(f, f, (4, 1)), *conv(f, f, (4, 1)))
        self.block3 = nn.Sequential(*conv(f, f, (1, 10)))
        self.inc1 = nn.Sequential(*conv(f, g, (1, 1)), *conv(g, g, (3, 1)))
        self.inc2 = nn.Sequential(*conv(f, g, (1, 1)), *conv(g, g, (5, 1)))
        self.inc3 = nn.Sequential(nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)), *conv(f, g, (1, 1)))
        self.lstm = nn.LSTM(3 * g, lstm_units, batch_first=True)
        self.head = nn.Linear(lstm_units, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :, self.perm].unsqueeze(1)                 # (B, 1, 100, 20), sides of a level adjacent
        x = self.block3(self.block2(x))                     # (B, 32, 100, 1)
        x = torch.cat([self.inc1(x), self.inc2(x), self.inc3(x)], dim=1)   # (B, 192, 100, 1)
        out, _ = self.lstm(x.squeeze(3).transpose(1, 2))    # (B, 100, 192)
        return self.head(out[:, -1])


@register("cnn_lstm")
def build_cnn_lstm(cfg: ModelConfig) -> nn.Module:
    return CnnLstm(cfg.input_dim, filters=cfg.param("filters"), inception_filters=cfg.param("inception_filters"),
                   lstm_units=cfg.param("lstm_units"), leaky_slope=cfg.param("leaky_slope"), n_out=cfg.n_out)
