"""Predictions for every sample row of a day, the validation loss, and the score cells of a stock.

The window of a sample row is ``window_index`` on the device (``gather``): rows t-history+1..t through the offsets
``offs``. Every prediction counts the windows whose last row is a later event than the label's start row; at shift
0 the count must be 0.
"""

from __future__ import annotations

import numpy as np
import torch

from dof.deep.data.eval import DeviceDay, EvalStock
from dof.lob.features import inverse_target
from dof.lob.stats import r2_os


def gather(dd: DeviceDay, rows_b: torch.Tensor, offs: torch.Tensor, shift: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Windows (B, history, d) of sample rows ``rows_b``, and the number of windows whose last row is a LATER event
    than the label's start (must be 0 at shift 0)."""
    idx = rows_b[:, None] + offs + shift
    late = (dd.seq[idx[:, -1]] > dd.labseq[rows_b]).sum()
    return dd.z[idx], late


@torch.no_grad()
def predict(model, dd: DeviceDay, offs, *, eval_batch: int, n_out: int, amp: bool, shift: int = 0) -> tuple[np.ndarray, int]:
    """Predictions (m, n_out) in z units for ALL sample rows of the day, float64; and the late-window count."""
    model.eval()
    out = torch.empty((len(dd.rows), n_out), dtype=torch.float32, device=dd.rows.device)
    late = torch.zeros((), dtype=torch.int64, device=dd.rows.device)
    for a in range(0, len(dd.rows), eval_batch):
        x, lt = gather(dd, dd.rows[a:a + eval_batch], offs, shift)
        late += lt
        with torch.autocast(device_type=dd.z.device.type, enabled=amp):
            out[a:a + eval_batch] = model(x).float()
    return out.double().cpu().numpy(), int(late)


def valid_loss(model, ddays: list[DeviceDay], offs, *, eval_batch: int, n_out: int, amp: bool, shift: int = 0) -> float:
    """MSE on the z-scored labels over all sample rows of the days: the early-stopping quantity of one stock."""
    se, n = 0.0, 0
    for dd in ddays:
        p, late = predict(model, dd, offs, eval_batch=eval_batch, n_out=n_out, amp=amp, shift=shift)
        if shift == 0 and late:
            raise AssertionError(f"{dd.day.tag}: {late} validation windows end after their label starts")
        zy = dd.zy[dd.rows].double().cpu().numpy()
        se += float(((p - zy) ** 2).sum())
        n += p.size
    return se / n


def score_rows(model, stock: EvalStock, ddays: list[DeviceDay], offs, *, eval_batch: int, n_out: int, amp: bool,
               shift: int = 0) -> list[dict]:
    """Per day and horizon: R²_OS in dollars (benchmark: that day's mean label), with SSE and SST."""
    rows = []
    for dd in ddays:
        p, late = predict(model, dd, offs, eval_batch=eval_batch, n_out=n_out, amp=amp, shift=shift)
        if shift == 0 and late:
            raise AssertionError(f"{dd.day.tag}: {late} windows end after their label starts")
        yh = inverse_target(p, stock.scalers["y"])
        y = np.asarray(dd.day.arr("y"))[dd.rows.cpu().numpy()]
        for j, k in enumerate(stock.meta["k_valid"]):
            rows.append({"part": dd.day.part, "date": dd.day.date, "k": int(k), "h_ms": stock.meta["h_ms"][str(k)],
                         "n_rows": int(len(y)), "r2_os": r2_os(y[:, j], yh[:, j]),
                         "sse": float(((y[:, j] - yh[:, j]) ** 2).sum()),
                         "sst": float(((y[:, j] - y[:, j].mean()) ** 2).sum()), "late_windows": late})
    return rows
