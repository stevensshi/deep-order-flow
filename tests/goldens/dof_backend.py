"""The golden recipes' call shapes, mapped onto ``dof``.

The backend renames and re-routes calls and converts argument shapes only (for example a ``(start_ns, end_ns)``
tuple into a session); it never computes, casts, rounds, sorts or filters a returned value.
"""

from __future__ import annotations

import contextlib
import sys


def dof_backend():
    from dof.lob import features as F
    from dof.lob import itch as I
    from dof.lob import stats as S
    from dof.lob import tables as T
    from dof.lob import validation as V
    from recipes.backends import LibBackend

    def build_day(path, symbols, out_dir, date):
        return T.build_day(path, list(symbols), out_dir, manifest_csv=None, date=date)

    def trimmed_session(start_end):
        """(start_ns, end_ns) of the trimmed session -> the Session with those ends and the default margin."""
        if start_end is None:
            return F.Session.full_day()
        margin = F.Session().margin_ns
        return F.Session(open_ns=start_end[0] - margin, close_ns=start_end[1] + margin)

    def regular_session(window):
        """(open_ns, close_ns) of regular trading -> Session."""
        return F.Session.full_day() if window is None else F.Session(open_ns=window[0], close_ns=window[1])

    def flow_pipeline(frame, *, convention, dedupe, session=None):
        return F.flow_pipeline(frame, trimmed_session(session), F.FlowConfig(convention=convention, dedupe=dedupe))

    def count_mid_changes(frame, window=None):
        return F.count_mid_changes(frame, regular_session(window))

    def estimate_delta_t(frames, dates, windows=None):
        sessions = [regular_session(None)] * len(frames) if windows is None else [regular_session(w) for w in windows]
        return F.estimate_delta_t(frames, dates, sessions)

    def horizon_labels(kf, ks, tw):
        return F.horizon_labels(kf, k=ks, train_window=tw)

    def history_valid(kf, history):
        return F.history_valid(kf, history)

    def lagged_design(z, rows, history):
        return F.lagged_design(z, rows, history=history)

    def block_t_test(loss_diff, block_id):
        return S.block_t_test(loss_diff, block_id)

    def change_events(frame, convention):
        return F.change_events(frame, F.MinuteConfig(convention=convention))

    def minute_buckets(frame, convention):
        return F.minute_buckets(frame, config=F.MinuteConfig(convention=convention))

    @contextlib.contextmanager
    def set_chunk_rows(n):
        mod = sys.modules[F.flow_pipeline.__module__]      # the module whose global flow_pipeline reads
        old = mod.OF_CHUNK_ROWS
        mod.OF_CHUNK_ROWS = n
        try:
            yield
        finally:
            mod.OF_CHUNK_ROWS = old

    return LibBackend(
        name="dof", build_day=build_day, load_states=T.load_states, InvalidStockDay=T.InvalidStockDay,
        FLOW_COLUMNS=F.FLOW_COLUMNS, flow_pipeline=flow_pipeline, count_mid_changes=count_mid_changes,
        estimate_delta_t=estimate_delta_t, horizon_labels=horizon_labels, history_valid=history_valid,
        k_valid_from=F.k_valid_from, sample_mask=F.sample_mask, fit_scaler=F.fit_scaler,
        fit_scaler_columnwise=F.fit_scaler_columnwise, apply_scaler=F.apply_scaler, inverse_target=F.inverse_target,
        lagged_design=lagged_design, suff_stats_init=S.suff_stats_init, suff_stats_update=S.suff_stats_update,
        solve_ridge=S.solve_ridge, r2_os=S.r2_os, r2_zero=S.r2_zero, block_t_test=block_t_test,
        change_events=change_events, minute_buckets=minute_buckets, fit_pc1=F.fit_pc1, integrated_ofi=F.integrated_ofi,
        iter_messages=I.iter_messages, LocateFilter=I.LocateFilter, ParseStats=I.ParseStats,
        scan_counts=I.scan_counts, set_chunk_rows=set_chunk_rows, read_lobster=V.read_lobster,
        reconcile_continuous=V.reconcile_continuous, reconcile_transition=V.reconcile_transition)
