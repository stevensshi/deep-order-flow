"""The call shapes a library implementation offers to the golden recipes.

A backend is a frozen set of callables. It routes each call to an implementation and converts argument shapes
only (for example a ``(start_ns, end_ns)`` tuple into the implementation's own session object); it never
computes, casts, rounds, sorts or filters a returned value. The recipes read results by attribute name.

Call shapes:
- ``build_day(path, symbols, out_dir, date)``: one ITCH file into per-stock tables, no manifest check; returns an
  object with ``status`` and ``log_path``;
- ``load_states(day_dir, symbol, columns=None)``; raises ``InvalidStockDay``; ``FLOW_COLUMNS``;
- ``flow_pipeline(frame, *, convention, dedupe, session=None)``: ``session`` is ``(start_ns, end_ns)`` of the
  session, or None for the default full-day session (09:40 to 15:50);
- ``count_mid_changes(frame, window=None)``: ``window`` is ``(open_ns, close_ns)`` of regular trading, or None for
  09:30 to 16:00;
- ``estimate_delta_t(frames, dates, windows=None)``: ``windows`` is one ``(open_ns, close_ns)`` per day, or None for
  full days;
- ``horizon_labels(kf, ks, tw)``, ``history_valid(kf, history)``, ``k_valid_from(labels)``,
  ``sample_mask(labels, k_valid, hist_ok)``;
- ``fit_scaler(x)``, ``fit_scaler_columnwise(get_column, ncols)``, ``apply_scaler(x, scaler)``,
  ``inverse_target(z, scaler)``, ``lagged_design(z, rows, history)``;
- ``suff_stats_init(p, q)``, ``suff_stats_update(s, X, Y)``, ``solve_ridge(s, lam)``, ``r2_os(y, yhat)``,
  ``r2_zero(y, yhat)``, ``block_t_test(loss_diff, block_id)``;
- ``change_events(frame, convention)``, ``minute_buckets(frame, convention)``, ``fit_pc1(ofi)``, ``integrated_ofi(ofi, pw)``;
- ``iter_messages(path, locates, *, stats=None, limit_bytes=None, chunk_size=...)``, ``LocateFilter(symbols)``,
  ``ParseStats()``, ``scan_counts(path, *, limit_bytes=None, chunk_size=...)``;
- ``set_chunk_rows(n)``: a context manager that sets the order-flow chunk size read by ``flow_pipeline``;
- ``read_lobster(message_csv, orderbook_csv)``, ``reconcile_continuous(day)``, ``reconcile_transition(day)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class LibBackend:
    name: str
    build_day: Callable[..., Any]
    load_states: Callable[..., Any]
    InvalidStockDay: type
    FLOW_COLUMNS: Sequence[str]
    flow_pipeline: Callable[..., Any]
    count_mid_changes: Callable[..., Any]
    estimate_delta_t: Callable[..., Any]
    horizon_labels: Callable[..., Any]
    history_valid: Callable[..., Any]
    k_valid_from: Callable[..., Any]
    sample_mask: Callable[..., Any]
    fit_scaler: Callable[..., Any]
    fit_scaler_columnwise: Callable[..., Any]
    apply_scaler: Callable[..., Any]
    inverse_target: Callable[..., Any]
    lagged_design: Callable[..., Any]
    suff_stats_init: Callable[..., Any]
    suff_stats_update: Callable[..., Any]
    solve_ridge: Callable[..., Any]
    r2_os: Callable[..., Any]
    r2_zero: Callable[..., Any]
    block_t_test: Callable[..., Any]
    change_events: Callable[..., Any]
    minute_buckets: Callable[..., Any]
    fit_pc1: Callable[..., Any]
    integrated_ofi: Callable[..., Any]
    iter_messages: Callable[..., Any]
    LocateFilter: Callable[..., Any]
    ParseStats: Callable[..., Any]
    scan_counts: Callable[..., Any]
    set_chunk_rows: Callable[..., Any]
    read_lobster: Callable[..., Any]
    reconcile_continuous: Callable[..., Any]
    reconcile_transition: Callable[..., Any]
