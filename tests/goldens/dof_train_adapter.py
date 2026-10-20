"""The inputs of the training recipes (``recipes/train_values.py``), taken from ``dof.deep`` outputs.

The adapter renames and re-routes calls and converts argument shapes only; it never computes, casts, rounds, sorts
or filters a returned value. Tensors are handed over as numpy arrays, as the recipes expect.
"""

from __future__ import annotations


def day_rows(days) -> list[tuple]:
    """(symbol, date, group, n_rows, n_samples, sample rows) per training stock-day, in stream order."""
    return [(d.symbol, d.date, d.group, d.n_rows, len(d.rows), d.rows) for d in days]


def take_batches(loader, n: int) -> list[dict]:
    """The first ``n`` batches of a loader; each tensor as its numpy array, every other value as it is."""
    out = []
    it = iter(loader)
    try:
        for _ in range(n):
            bt = next(it)
            out.append({k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in bt.items()})
    finally:
        del it
    return out


def named_parameters(model) -> dict:
    """Parameter name -> its array, in parameter order."""
    return {n: p.detach().numpy() for n, p in model.named_parameters()}


def flat(model):
    """All parameters concatenated (``dof.deep.models.flat_weights``), as an array."""
    from dof.deep.models import flat_weights
    return flat_weights(model).numpy()
