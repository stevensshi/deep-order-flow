"""A network on a whole part: the pooled validation loss, and the score cells of every stock of a part."""

from __future__ import annotations

import numpy as np

from dof.deep.config import CachePaths, RunConfig, Split
from dof.deep.data.eval import LOSS_ARRAYS, SCORE_ARRAYS, iter_eval, to_device
from dof.deep.errors import CacheError
from dof.deep.evaluate.predict import score_rows, valid_loss


def validation_loss(model, cfg: RunConfig, split: Split, cache: CachePaths, device, offs, *, amp: bool, symbols=None,
                    row_cap: int | None = None) -> tuple[float, dict[str, float], int]:
    """The equal-weight mean over stocks of each stock's MSE on z-scored labels over the sample rows of its
    validation days; one stock on the device at a time. Returns (loss, loss per stock, rows read)."""
    per, n = {}, 0
    n_out = len(cfg.features.horizons)
    for stock in iter_eval(cfg, split, cache, "valid", LOSS_ARRAYS, symbols=symbols, row_cap=row_cap):
        g = to_device(stock, device)
        per[stock.symbol] = valid_loss(model, g, offs, eval_batch=cfg.train.eval_batch, n_out=n_out, amp=amp)
        n += stock.n()
        del g
    if not per:
        raise CacheError("no stock on the validation days")
    return float(np.mean(list(per.values()))), per, n


def score_part(model, cfg: RunConfig, split: Split, cache: CachePaths, part: str, device, offs, *, name: str,
               n_train: int, seed: int, amp: bool, gate=None, tokens=None, skip=(), row_cap: int | None = None
               ) -> tuple[list[dict], dict[str, float]]:
    """The score cells of every stock of ``part``; for the validation days also each stock's validation loss.
    A test part needs the gate and a token for ``name``. ``skip``: (stock, day) pairs left out by a rule applied at
    scoring (the corporate actions on the test days)."""
    rows, per = [], {}
    n_out = len(cfg.features.horizons)
    for stock in iter_eval(cfg, split, cache, part, SCORE_ARRAYS, gate=gate, tokens=tokens, models=(name,), skip=skip,
                           row_cap=row_cap):
        g = to_device(stock, device)
        if part == "valid":
            per[stock.symbol] = valid_loss(model, g, offs, eval_batch=cfg.train.eval_batch, n_out=n_out, amp=amp)
        r = score_rows(model, stock, g, offs, eval_batch=cfg.train.eval_batch, n_out=n_out, amp=amp)
        for x in r:
            x.update(symbol=stock.symbol, model=name, seed=seed, n_train=n_train)
        rows += r
        del g
    return rows, per
