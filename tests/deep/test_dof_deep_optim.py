"""KerasAdam against the hand-written Keras update, and the optimizer factory."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dof.deep.config import FitSpec  # noqa: E402
from dof.deep.optim import KerasAdam, make_optimizer, optimizer_facts  # noqa: E402

LR_PAPER = 1e-3         # Kolm, Turiel and Westray (2023), Table 2: the CNN-LSTM's learning rate
KERAS_EPS = 1.0         # an epsilon large enough to matter
SEED = 0
BETAS = (0.9, 0.999)


def keras_reference(theta0: np.ndarray, grads: np.ndarray, lr: float, b1: float, b2: float, eps: float,
                    dtype=np.float64) -> np.ndarray:
    """Hand-written Keras Adam: the lines of ``ApplyAdamKernel`` (TensorFlow v2.3.1, training_ops_gpu.cu.cc lines
    42-43, 56-57 and 62), computed in ``dtype``. Returns the parameter after every step, as float64."""
    theta = theta0.astype(dtype).copy()
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    path = []
    for t, g in enumerate(grads.astype(dtype), start=1):
        mul_factor = dtype(lr * math.sqrt(1.0 - b2 ** t) / (1.0 - b1 ** t))
        m += dtype(1.0 - b1) * (g - m)
        v += dtype(1.0 - b2) * (g * g - v)
        theta -= mul_factor * m / (dtype(eps) + np.sqrt(v))
        path.append(theta.astype(np.float64))
    return np.stack(path)


@pytest.mark.parametrize("dtype, kw, tol", [(torch.float64, {"foreach": False}, 1e-12),
                                            (torch.float64, {"foreach": True}, 1e-12),
                                            (torch.float32, {"foreach": False}, 1e-5)])
def test_keras_adam_matches_hand_written_formula(dtype, kw, tol) -> None:
    steps, shape = 200, (8, 16)
    rng = np.random.default_rng(SEED)
    theta0 = rng.standard_normal(shape)
    scale = 10.0 ** rng.uniform(-4.0, 1.0, size=shape)       # gradient sizes from 1e-4 to 10: around epsilon = 1
    grads = rng.standard_normal((steps, *shape)) * scale
    cast = np.float32 if dtype == torch.float32 else np.float64
    ref = keras_reference(theta0.astype(cast), grads.astype(cast), LR_PAPER, *BETAS, KERAS_EPS)
    p = torch.nn.Parameter(torch.tensor(theta0, dtype=dtype))
    opt = KerasAdam([p], lr=LR_PAPER, betas=BETAS, eps=KERAS_EPS, **kw)
    path = []
    for g in grads:
        p.grad = torch.tensor(g, dtype=dtype)
        opt.step()
        path.append(p.detach().double().numpy().copy())
    assert float(np.abs(np.stack(path) - ref).max()) <= tol


def test_keras_adam_refuses_missing_gradient() -> None:
    a, b = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
    opt = KerasAdam([a, b])
    a.grad = torch.ones(2)
    with pytest.raises(RuntimeError, match="no gradient"):
        opt.step()
    b.grad = torch.ones(2)
    with pytest.raises(ValueError, match="closure"):
        opt.step(lambda: 0.0)


def _spec(optimizer: str) -> FitSpec:
    return FitSpec("f", "m", optimizer, 3e-5, 1e-7, "all", None)


def test_optimizer_facts() -> None:
    model = torch.nn.Linear(3, 2)
    adam, keras = make_optimizer(model, _spec("adam"), BETAS), make_optimizer(model, _spec("keras_adam"), BETAS)
    fa, fk = optimizer_facts(adam), optimizer_facts(keras)
    assert (fa["class"], fa["bases"], fa["keras_form"], fa["eps"], fa["steps"]) == ("Adam", ["Optimizer", "object"], False, 1e-7, None)
    assert (fk["class"], fk["bases"], fk["keras_form"], fk["eps"], fk["steps"]) == ("KerasAdam", ["Adam", "Optimizer"], True, 1e-7, 0)
    assert fa["lr"] == fk["lr"] == 3e-5 and fa["betas"] == fk["betas"] == [0.9, 0.999]
    model(torch.ones(1, 3)).sum().backward()
    keras.step()
    fk = optimizer_facts(keras)
    assert fk["steps"] == 1 and fk["eps_pytorch_last_step"] == 1e-7 / (1.0 - 0.999) ** 0.5


def test_make_optimizer_leaves_torch_optim_alone() -> None:
    before = torch.optim.Adam
    model = torch.nn.Linear(3, 2)
    for kind in ("adam", "keras_adam"):
        make_optimizer(model, _spec(kind), BETAS)
    assert torch.optim.Adam is before and not hasattr(torch.optim.Adam, "keras_form")
