"""The model registry and the six catalogue models."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dof.deep.config import load  # noqa: E402
from dof.deep.models import FAMILIES, build, flat_weights, n_params, register  # noqa: E402
from dof.deep.models.cnn_lstm import INTERLEAVE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def models():
    return load(ROOT / "configs" / "smoke.toml").models


def test_families_discovered() -> None:
    assert {"mlp", "lstm", "lstm_mlp", "cnn_lstm"} <= set(FAMILIES)


def test_register_twice_raises(monkeypatch) -> None:
    monkeypatch.setitem(FAMILIES, "test-family", lambda cfg: None)
    with pytest.raises(ValueError, match="registered twice"):
        register("test-family")(lambda cfg: None)


def test_keras_parameter_counts(models) -> None:
    """The sizes of Kolm, Turiel and Westray (2023), Table 2 (Keras counts: one LSTM bias vector)."""
    want = {"LSTM-OF": 104_110, "LSTM3-OF": 465_310, "LSTM-MLP-OF": 84_458, "MLP-OF": 1_302_110}
    for name, n in want.items():
        assert n_params(build(models[name]))["keras_equivalent"] == n, name


def test_output_shapes(models) -> None:
    for name, cfg in models.items():
        out = build(cfg)(torch.zeros(4, 100, cfg.input_dim))
        assert tuple(out.shape) == (4, 10), name


def test_keras_init_forget_bias(models) -> None:
    lstm = build(models["LSTM-OF"]).lstm
    h = lstm.bias_ih_l0.shape[0] // 4
    assert torch.equal(lstm.bias_ih_l0[h:2 * h], torch.ones(h))
    assert not lstm.bias_ih_l0[:h].any() and not lstm.bias_ih_l0[2 * h:].any()
    assert not lstm.bias_hh_l0.any()


def test_cnn_interleave() -> None:
    assert INTERLEAVE == [x for i in range(10) for x in (i, i + 10)]


def test_same_seed_same_weights(models) -> None:
    def weights(seed: int):
        torch.manual_seed(seed)
        return flat_weights(build(models["CNN-LSTM-OF"]))
    assert torch.equal(weights(7), weights(7))
    assert not torch.equal(weights(7), weights(8))
