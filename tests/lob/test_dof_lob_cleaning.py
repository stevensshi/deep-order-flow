from __future__ import annotations

import numpy as np

from dof.lob import features as F
from dof.lob.book import FLAG_CROSSED, FLAG_HALTED, SENTINEL_ASK_PX, SENTINEL_BID_PX
from dof.lob.features import hms
from hand_frames import BASE_ASK, BASE_BID, FULL, MS, T10, _frame, _mid_row, _row, _tw


def test_flow_dedupe_keeps_last() -> None:
    rows = [_mid_row(T10, 1_000_000), _mid_row(T10, 1_001_000), _mid_row(T10, 1_002_000), _mid_row(T10 + 5, 1_003_000)]
    kf = F.flow_pipeline(_frame(rows), FULL)
    assert kf.ts_ns.tolist() == [T10, T10 + 5]
    assert kf.mid.tolist() == [100.2, 100.3]
    assert kf.funnel["dedupe_removed"] == 2
    kf_all = F.flow_pipeline(_frame(rows), FULL, F.FlowConfig(dedupe=False))
    assert len(kf_all.ts_ns) == 4 and kf_all.funnel["dedupe_removed"] == 0
    assert kf_all.config.dedupe is False and kf.config.dedupe is True


def test_flow_drops_qh_without_break() -> None:
    rows = [_row(T10), _row(T10 + 1, mt="Q"), _row(T10 + 2, mt="H"), _row(T10 + 3, b1=(BASE_BID, 150))]
    kf = F.flow_pipeline(_frame(rows), FULL)
    assert kf.ts_ns.tolist() == [T10, T10 + 3]
    assert kf.segment.tolist() == [0, 0] and kf.of_valid.tolist() == [False, True]
    assert kf.funnel["qh_removed"] == 2 and len(kf.intervals.start_ts) == 0
    assert kf.ofi[1, 0] == 50


def test_flow_crossed_halted_missing_break_segments() -> None:
    missing = [(BASE_BID - 100 * k, 100) for k in range(9)] + [(SENTINEL_BID_PX, 0)]
    rows = [
        _row(T10), _row(T10 + 1),
        _row(T10 + 2, flags=FLAG_CROSSED),
        _row(T10 + 3), _row(T10 + 4),
        _row(T10 + 5, flags=FLAG_HALTED),
        _row(T10 + 6), _row(T10 + 7),
        _row(T10 + 8, bid=missing),
        _row(T10 + 9), _row(T10 + 10),
        _row(T10 + 11, flags=1),          # LOCKED is kept
    ]
    kf = F.flow_pipeline(_frame(rows), FULL)
    assert kf.intervals.start_ts.tolist() == [T10 + 2, T10 + 5, T10 + 8]
    assert kf.segment.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 3]
    assert kf.of_valid.tolist() == [False, True, False, True, False, True, False, True, True]
    for r in ("crossed_removed", "halted_removed", "missing_levels_removed"):
        assert kf.funnel[r] == 1, r
    assert kf.funnel["segment_first_rows"] == 4 and kf.funnel["rows_out"] == 9
    assert kf.funnel["rows_in"] == 12 and kf.funnel["rows_outside_session"] == 0


def test_flow_session_window() -> None:
    rows = [_row(hms(9, 39, 59)), _row(hms(9, 40)), _row(hms(15, 49, 59)), _row(hms(15, 50))]
    kf = F.flow_pipeline(_frame(rows), FULL)
    assert kf.ts_ns.tolist() == [hms(9, 40), hms(15, 49, 59)]
    assert kf.funnel["rows_outside_session"] == 2


# -- raw invalid intervals: H-only halts and same-nanosecond restores break segments --


def test_flow_h_only_halt_breaks_segment() -> None:
    rows = [_row(T10), _row(T10 + 1, b1=(BASE_BID, 110)),
            _row(T10 + 2, mt="H", flags=FLAG_HALTED), _row(T10 + 3, mt="H"),
            _row(T10 + 4, b1=(BASE_BID, 120)), _row(T10 + 5, b1=(BASE_BID, 130))]
    kf = F.flow_pipeline(_frame(rows), FULL)
    assert kf.ts_ns.tolist() == [T10, T10 + 1, T10 + 4, T10 + 5]
    assert kf.intervals.start_ts.tolist() == [T10 + 2]
    assert kf.segment.tolist() == [0, 0, 1, 1]
    assert kf.of_valid.tolist() == [False, True, False, True]
    assert kf.funnel["invalid_intervals"] == 1
    iv = kf.intervals
    assert (iv.start_seq.tolist(), iv.end_seq.tolist(), iv.close_seq.tolist(), iv.reason.tolist()) == ([2], [2], [3], ["halted"])


def test_flow_same_ns_cross_restore_breaks_segment() -> None:
    rows = [_mid_row(T10, 1_000_000), _mid_row(T10 + 1 * MS, 1_000_000),
            _mid_row(T10 + 2 * MS, 1_000_000, flags=FLAG_CROSSED), _mid_row(T10 + 2 * MS, 1_001_000),
            _mid_row(T10 + 3 * MS, 1_001_000)]
    kf = F.flow_pipeline(_frame(rows), FULL, F.FlowConfig(latency_ns=500_000))
    # dedupe keeps only the restored row of the 2 ms group, yet the break is recorded
    assert kf.ts_ns.tolist() == [T10, T10 + 1 * MS, T10 + 2 * MS, T10 + 3 * MS]
    assert kf.intervals.start_ts.tolist() == [T10 + 2 * MS]
    assert kf.segment.tolist() == [0, 0, 1, 1]
    assert F.history_valid(kf, 1).tolist() == [False, True, False, True]
    iv = kf.intervals
    assert (iv.start_seq.tolist(), iv.end_seq.tolist(), iv.close_seq.tolist()) == ([2], [2], [3])
    assert iv.start_ts.tolist() == iv.close_ts.tolist() == [T10 + 2 * MS]
    # a label from row 1 over (1 ms, 1 ms + h] spans the interval -> invalid
    lab = F.horizon_labels(kf, k=[1], train_window=_tw(10.0))   # h_1 = 2 ms, latency 0.5 ms (the frame's config)
    assert not lab.valid[1, 0]
    assert lab.valid[2, 0] or lab.funnel["label_crosses_session_end"] == 0


# -- the vectorised invalid intervals against the row loop they replaced ----------------------------


def _loop_intervals(frame) -> dict:
    """The row-by-row loop that ``invalid_intervals`` replaced, kept here as the reference."""
    fl = frame["flags"].to_numpy()
    missing = np.zeros(len(frame), dtype=bool)
    for side, sentinel in (("bid", SENTINEL_BID_PX), ("ask", SENTINEL_ASK_PX)):
        for k in range(1, 11):
            missing |= frame[f"{side}_px_{k}"].to_numpy() == sentinel
    order = ("halted", "crossed", "missing_levels")
    every = {"halted": (fl & FLAG_HALTED) != 0, "crossed": (fl & FLAG_CROSSED) != 0, "missing_levels": missing}
    bad = every["halted"] | every["crossed"] | every["missing_levels"]
    ts = frame["ts_ns"].to_numpy().astype(np.int64)
    sq = frame["seq"].to_numpy().astype(np.int64)
    n = len(frame)
    out: dict[str, list] = {k: [] for k in ("start_seq", "end_seq", "close_seq", "start_ts", "close_ts", "reason")}
    i = 0
    while i < n:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j < n and bad[j]:
            j += 1
        run = {r for r in order if every[r][i:j].any()}
        out["start_seq"].append(sq[i])
        out["end_seq"].append(sq[j - 1])
        out["close_seq"].append(sq[j] if j < n else F.OPEN_END_NS)
        out["start_ts"].append(ts[i])
        out["close_ts"].append(ts[j] if j < n else F.OPEN_END_NS)
        out["reason"].append(next(r for r in order if r in run))
        i = j
    res = {k: np.array(out[k], dtype=np.int64) for k in ("start_seq", "end_seq", "close_seq", "start_ts", "close_ts")}
    res["reason"] = np.array(out["reason"], dtype=str)
    return res


def _random_frame(rng):
    missing_bid = [(BASE_BID - 100 * k, 100) for k in range(9)] + [(SENTINEL_BID_PX, 0)]
    missing_ask = [(BASE_ASK + 100 * k, 100) for k in range(9)] + [(SENTINEL_ASK_PX, 0)]
    rows = []
    for i in range(1 + int(rng.integers(0, 40))):
        flags = int(rng.choice([0, 0, 0, 1, FLAG_CROSSED, FLAG_HALTED, FLAG_CROSSED | FLAG_HALTED]))
        u = rng.random()
        kw = {"bid": missing_bid} if u < 0.1 else ({"ask": missing_ask} if u < 0.2 else {})
        rows.append(_row(T10 + 1_000 * i, flags=flags, **kw))
    df = _frame(rows)
    df["seq"] = 3 * df["seq"] + 7          # gaps between seqs, as in a real state table
    return df


def test_invalid_intervals_matches_loop_reference() -> None:
    rng = np.random.default_rng(11)
    frames = [_random_frame(rng) for _ in range(30)]
    frames.append(frames[0].iloc[:0])      # no row at all: no run, reason dtype "<U1"
    for frame in frames:
        got, want = F.invalid_intervals(frame), _loop_intervals(frame)
        for k, w in want.items():
            g = getattr(got, k)
            assert g.dtype == w.dtype and g.shape == w.shape, (k, g.dtype, w.dtype)
            np.testing.assert_array_equal(g, w)
    assert F.invalid_intervals(frames[-1]).reason.dtype == np.dtype("<U1")

