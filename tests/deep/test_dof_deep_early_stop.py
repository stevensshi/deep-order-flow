"""Early stopping over checks."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dof.deep.train.early_stop import EarlyStopping  # noqa: E402


def _model(value: float) -> torch.nn.Module:
    m = torch.nn.Linear(1, 1)
    with torch.no_grad():
        m.weight.fill_(value)
        m.bias.fill_(value)
    return m


def test_equal_loss_is_not_an_improvement() -> None:
    es = EarlyStopping(patience=5, max_checks=50)
    assert es.update(1.0, _model(1.0))
    assert not es.update(1.0, _model(2.0))
    assert es.wait == 1 and es.best == 1.0 and float(es.best_state["weight"]) == 1.0


def test_patience_stops() -> None:
    es = EarlyStopping(patience=2, max_checks=50)
    for i, loss in enumerate((1.0, 2.0, 3.0), start=1):
        es.update(loss, _model(loss))
        assert es.stop_reason(i) == (None if i < 3 else "patience")


def test_max_checks_stops() -> None:
    es = EarlyStopping(patience=5, max_checks=2)
    es.update(1.0, _model(1.0))
    assert es.stop_reason(1) is None
    es.update(0.5, _model(0.5))
    assert es.stop_reason(2) == "max checks"


def test_expected_check_count() -> None:
    """The fit stops at best check + patience, or at max_checks, whichever comes first."""
    for losses, patience, max_checks, want in (([3.0, 2.0, 1.0, 1.5, 1.6, 1.7, 0.1], 3, 50, 6),
                                               ([3.0, 2.0, 1.0, 0.5], 3, 4, 4)):
        es = EarlyStopping(patience, max_checks)
        n, best_check = 0, 0
        for n, loss in enumerate(losses, start=1):
            if es.update(loss, _model(loss)):
                best_check = n
            reason = es.stop_reason(n)
            if reason:
                break
        assert n == want == (best_check + patience if reason == "patience" else max_checks)


def test_state_roundtrip() -> None:
    es = EarlyStopping(patience=5, max_checks=50)
    es.update(2.0, _model(2.0))
    es.update(3.0, _model(3.0))
    again = EarlyStopping(patience=5, max_checks=50)
    again.load_state_dict(es.state_dict(), torch.device("cpu"))
    assert (again.best, again.wait) == (2.0, 1)
    assert all(torch.equal(again.best_state[k], es.best_state[k]) for k in es.best_state)
