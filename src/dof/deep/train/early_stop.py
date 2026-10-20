"""Early stopping over checks: a check improves only with a strictly lower validation loss; the fit stops after
``patience`` checks without improvement, or after ``max_checks`` checks. The weights of the best check are kept."""

from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int, max_checks: int) -> None:
        self.patience, self.max_checks = patience, max_checks
        self.best = float("inf")
        self.best_state: dict | None = None
        self.wait = 0

    def update(self, loss: float, model) -> bool:
        improved = loss < self.best
        if improved:
            self.best, self.wait = loss, 0
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
        return improved

    def stop_reason(self, n_checks: int) -> str | None:
        if self.wait >= self.patience:
            return "patience"
        if n_checks >= self.max_checks:
            return "max checks"
        return None

    def state_dict(self) -> dict:
        state = None if self.best_state is None else {k: v.detach().cpu() for k, v in self.best_state.items()}
        return {"best": self.best, "wait": self.wait, "best_state": state}

    def load_state_dict(self, d: dict, device) -> None:
        self.best_state = None if d["best_state"] is None else {k: v.to(device) for k, v in d["best_state"].items()}
        self.best, self.wait = d["best"], d["wait"]
