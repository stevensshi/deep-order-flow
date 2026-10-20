"""Optimizers: PyTorch's Adam, and Adam in the Keras form (``KerasAdam``). Neither replaces anything in
``torch.optim``.

Keras:    theta -= lr * sqrt(1 - b2^t) / (1 - b1^t) * m / (sqrt(v) + eps)
PyTorch:  theta -= lr / (1 - b1^t) * m / (sqrt(v) / sqrt(1 - b2^t) + eps_pt)
Multiply PyTorch's fraction by sqrt(1 - b2^t) above and below: the two are equal when
eps_pt = eps / sqrt(1 - b2^t). ``KerasAdam.step`` sets that value in every parameter group, then lets Adam do the
update, so the moments and the update are PyTorch's own code.
"""

from __future__ import annotations

import torch

from dof.deep.config import FitSpec


class KerasAdam(torch.optim.Adam):
    keras_form = True

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-7, **kwargs) -> None:
        if kwargs.get("amsgrad"):
            raise ValueError("KerasAdam: amsgrad is not emulated")
        super().__init__(params, lr=lr, betas=betas, eps=eps, **kwargs)
        self.keras_eps = float(eps)
        self.keras_t = 0          # Keras' ``iterations``: one count for all variables

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("KerasAdam: a closure is not supported (the gradients must exist before the step)")
        t = self.keras_t + 1
        for group in self.param_groups:
            # Keras counts steps once for all variables. PyTorch counts per parameter and skips a parameter without
            # a gradient; then the two bias corrections would no longer agree.
            if any(p.grad is None for p in group["params"]):
                raise RuntimeError("KerasAdam: a parameter has no gradient, so the step counts would diverge")
            group["eps"] = self.keras_eps / (1.0 - group["betas"][1] ** t) ** 0.5
        out = super().step()
        self.keras_t = t
        first = self.param_groups[0]["params"][0]
        if int(self.state[first]["step"]) != t:
            raise RuntimeError("KerasAdam: PyTorch's step count differs from the Keras step count")
        return out


def make_optimizer(model: torch.nn.Module, spec: FitSpec, betas: tuple[float, float]) -> torch.optim.Optimizer:
    if getattr(torch.optim.Adam, "keras_form", False):
        raise RuntimeError("torch.optim.Adam has been replaced in this process: stopped")
    if spec.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=spec.lr, betas=betas, eps=spec.eps)
    if spec.optimizer == "keras_adam":
        return KerasAdam(model.parameters(), lr=spec.lr, betas=betas, eps=spec.eps)
    raise ValueError(spec.optimizer)


def optimizer_facts(opt: torch.optim.Optimizer) -> dict:
    """Read from the optimizer object, not from the configuration."""
    g = opt.param_groups[0]
    keras = bool(getattr(opt, "keras_form", False))
    return {"class": type(opt).__name__, "bases": [c.__name__ for c in type(opt).__mro__[1:3]], "keras_form": keras,
            "lr": float(g["lr"]), "eps": float(opt.keras_eps) if keras else float(g["eps"]), "eps_pytorch_last_step": float(g["eps"]),
            "betas": [float(b) for b in g["betas"]], "steps": int(opt.keras_t) if keras else None}
