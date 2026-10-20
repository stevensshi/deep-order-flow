"""One training step, and the size of the weight change over the step before a check."""

from __future__ import annotations

import torch

from dof.deep.models import flat_weights


def train_step(model, batch: dict, opt, scaler, device, amp: bool) -> torch.Tensor:
    """MSE on the z-scored labels of all horizons; one optimizer step. Returns the batch loss."""
    x, zy = batch["x"].to(device), batch["zy"].to(device)
    with torch.autocast(device_type=device.type, enabled=amp):
        pred = model(x)
    loss = torch.nn.functional.mse_loss(pred.float(), zy)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return loss


def check_weights(model, w0: torch.Tensor) -> dict:
    """The weight change of the step before a check, against the weights ``w0`` before it."""
    dw = (flat_weights(model) - w0).abs()
    return {"median_abs_dw": float(dw.median()), "share_weights_unchanged": float((dw == 0).float().mean()),
            "median_abs_w": float(w0.abs().median())}
