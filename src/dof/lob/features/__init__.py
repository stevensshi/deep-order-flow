"""Feature layer: order flow, the preprocessing of Kolm, Turiel and Westray (2023) and of Cont, Cucuringu and
Zhang (2023), labels, scaling, PCA.

Inputs are the per-stock tables of ``dof.lob.tables`` (``dof.lob.tables.schema`` is
their source of truth): the 10-level state table joined on ``seq`` with the event table's
``msg_type`` and ``trading_state``. All functions are pure (inputs are not
modified). Times are nanoseconds since midnight; ITCH time is already US
Eastern local time, so no time-zone conversion is needed.

Order-flow sign convention: the default ``current_size`` follows Cont, Cucuringu and
Zhang (2023), §2.1, and Kolm, Turiel and Westray (2023), eqs. (2)-(3): a bid price drop
contributes -q_n (the CURRENT size), an ask price rise -q_n. ``cks`` (Cont, Kukanov and
Stoikov 2014) switches those two cases to -q_{n-1} (the previous size). OFI = bOF - aOF.

Two branches with deliberately different event sets:
- the flow branch (``flow_series``, after Kolm, Turiel and Westray): session 09:40-15:50,
  FIXED order: dedupe by timestamp keeping the last row -> drop Q/H rows (no segment
  break: they do not change the book) -> drop crossed / halted / missing-level rows
  (segment break) -> 20-dim OF.
- the minute branch (``minute_ofi``, after Cont, Cucuringu and Zhang): rows whose 10-level
  (px, sz) vector changed, from 09:30, no dedupe, 1-minute buckets,
  Q = sum(q^b + q^a) / (2 M dN) shared by all levels (their eq. 3), PCA weights fitted on
  the estimation window only.
"""
from dof.lob.features.minute_ofi import (
    MinuteConfig,
    ChangeEvents,
    PcaWeights,
    minute_buckets,
    change_events,
    window_ofi,
    fit_pc1,
    integrated_ofi,
)
from dof.lob.features.cleaning import (
    ASK_PX,
    ASK_SZ,
    BID_PX,
    BID_SZ,
    OPEN_END_NS,
    InvalidIntervals,
    PriceSeries,
    asof_index,
    breaks_between,
    in_interval,
    invalid_intervals,
    RowMasks,
    flow_clean,
    mid_at,
    mid_px,
    price_series,
    row_masks,
)
from dof.lob.features.flow_series import (
    DELTA_T_NUMERATOR_MS,
    FLOW_COLUMNS,
    FlowConfig,
    FlowFrame,
    TrainWindow,
    count_mid_changes,
    estimate_delta_t,
    history_valid,
    flow_pipeline,
)
from dof.lob.features.labels import LabelSet, k_valid_from, horizon_labels, sample_mask
from dof.lob.features.order_flow import SignConvention, order_flow
from dof.lob.features.session import Session, session_for
from dof.lob.features.scaling import (
    Scaler,
    apply_scaler,
    fit_scaler,
    fit_scaler_columnwise,
    inverse_target,
    lagged_design,
)
from dof.lob.units import NS, PRICE_SCALE, hms

# OF_CHUNK_ROWS is read at call time from dof.lob.features.flow_series, where it is set; it is not re-exported here.
__all__ = [
    "ASK_PX", "ASK_SZ", "BID_PX", "BID_SZ", "MinuteConfig", "ChangeEvents", "DELTA_T_NUMERATOR_MS", "InvalidIntervals",
    "FLOW_COLUMNS", "FlowConfig", "FlowFrame", "LabelSet", "NS", "OPEN_END_NS", "PRICE_SCALE", "PcaWeights", "PriceSeries",
    "Scaler", "SignConvention", "TrainWindow", "apply_scaler", "asof_index", "breaks_between", "minute_buckets",
    "change_events", "window_ofi", "count_mid_changes", "estimate_delta_t", "fit_pc1", "fit_scaler",
    "fit_scaler_columnwise", "history_valid", "hms", "in_interval", "integrated_ofi", "invalid_intervals",
    "inverse_target", "k_valid_from", "flow_clean", "horizon_labels", "flow_pipeline", "lagged_design", "mid_at", "mid_px",
    "order_flow", "RowMasks", "row_masks", "Session", "session_for",
    "price_series", "sample_mask",
]
