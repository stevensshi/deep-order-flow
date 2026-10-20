"""The Keras default initialisation (the models of Kolm, Turiel and Westray 2023 were built in Keras): Glorot-uniform
kernels, orthogonal recurrent kernels, zero biases, and an LSTM forget-gate bias of 1 (their p. 9)."""

from __future__ import annotations

import torch
from torch import nn


def keras_init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, p in m.named_parameters():
                if name.startswith("weight_ih"):
                    nn.init.xavier_uniform_(p)
                elif name.startswith("weight_hh"):
                    # Keras: one orthogonal (4*units, units) recurrent kernel
                    nn.init.orthogonal_(p)
                else:
                    nn.init.zeros_(p)
                    if name.startswith("bias_ih"):
                        # PyTorch gate order i, f, g, o: the forget gate is the second quarter. Only bias_ih gets
                        # the 1; bias_hh stays 0, so the effective forget bias is 1.
                        h = p.shape[0] // 4
                        with torch.no_grad():
                            p[h:2 * h] = 1.0
