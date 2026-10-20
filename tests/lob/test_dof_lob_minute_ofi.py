from __future__ import annotations

import numpy as np
import pytest

from dof.lob import features as F
from dof.lob.book import FLAG_CROSSED, FLAG_HALTED, SENTINEL_BID_PX
from dof.lob.features import hms
from hand_frames import BASE_ASK, BASE_BID, _frame, _row


def test_minute_event_set() -> None:
    t0 = hms(9, 35)
    rows = [
        _row(t0),                                        # event (first row)
        _row(t0 + 1, mt="P"), _row(t0 + 2, mt="Q"), _row(t0 + 3, mt="H"),
        _row(t0 + 4, ct=2),                              # only counts changed -> not an event
        _row(t0 + 5, b1=(BASE_BID, 150)),                # event
        _row(t0 + 6, mt="P", b1=(BASE_BID, 150)),
        _row(hms(9, 29), b1=(BASE_BID, 200)),            # before 09:30 (ts out of order on purpose)
    ]
    ev = F.change_events(_frame(rows))
    assert ev.ts_ns.tolist() == [t0, t0 + 5]
    assert ev.has_prev.tolist() == [False, True]
    assert ev.funnel["pqh_removed"] == 4 and ev.funnel["unchanged_removed"] == 1


def test_minute_bucket_hand_example() -> None:
    lvl2_up = [(BASE_BID, 150), (BASE_BID - 100, 120)] + [(BASE_BID - 100 * k, 100) for k in range(2, 10)]
    rows = [
        _row(hms(9, 30, 30)),                                          # E0, bucket 1
        _row(hms(9, 31, 10), b1=(BASE_BID, 150)),                      # E1: bOF_1 = +50
        _row(hms(9, 31, 20), bid=lvl2_up, a1=(BASE_ASK, 60)),          # E2: bOF_2 = +20, aOF_1 = -40
    ]
    b = F.minute_buckets(_frame(rows))
    assert len(b) == 390
    r = b.iloc[1]                                                       # bucket (09:31, 09:32]
    assert r["bucket_end_ns"] == hms(9, 32) and r["n_events"] == 2
    ofi = [r[f"OFI_{m}"] for m in range(1, 11)]
    assert ofi == [90.0, 20.0] + [0.0] * 8
    # depth: E1 = (150 + 900) + 1000 = 2050; E2 = (150 + 120 + 800) + (60 + 900) = 2030
    q = (2050 + 2030) / (2 * 10 * 2)
    assert r["Q"] == q == 102.0
    assert r["ofi_1"] == pytest.approx(90 / q) and r["ofi_2"] == pytest.approx(20 / q)
    assert r["valid"] and r["invalid_reason"] == ""
    # bucket 1 holds E0 only, which has no predecessor: OFI 0, Q from its depth
    assert b.iloc[0]["OFI_1"] == 0.0 and b.iloc[0]["Q"] == 2000 / 20
    # mid as-of and log return
    assert r["mid_end"] == pytest.approx((BASE_BID + BASE_ASK) / 2 / 1e4)
    assert r["ret"] == 0.0
    # window helper agrees, and the convention switch is wired through
    ev = F.change_events(_frame(rows))
    o, qq, dn, broken = F.window_ofi(ev, np.array([hms(9, 32)]), 60 * F.NS)
    assert o[0, :2].tolist() == [90.0, 20.0] and qq[0] == q and dn[0] == 2 and not broken[0]
    with pytest.raises(ValueError):
        F.minute_buckets(_frame(rows), config=F.MinuteConfig(bucket_ns=300 * F.NS))


def test_minute_invalid_buckets() -> None:
    zero = [(BASE_BID - 100 * k, 0) for k in range(10)]
    zero_a = [(BASE_ASK + 100 * k, 0) for k in range(10)]
    missing = [(BASE_BID - 100 * k, 100) for k in range(9)] + [(SENTINEL_BID_PX, 0)]
    m = lambda mm, s=0: hms(9, 30 + mm, s)  # noqa: E731
    rows = [
        _row(m(0, 10)), _row(m(0, 20), b1=(BASE_BID, 110)),            # bucket 1 (09:31): no as-of at 09:30
        _row(m(1, 10), b1=(BASE_BID, 120)),                            # bucket 2 ok
        # bucket 3 (09:33) empty
        _row(m(3, 10), bid=zero, ask=zero_a),                          # bucket 4: zero depth
        _row(m(4, 10), b1=(BASE_BID, 130)),                            # bucket 5: halted, restored inside
        _row(m(4, 20), b1=(BASE_BID, 140), flags=FLAG_HALTED), _row(m(4, 30), b1=(BASE_BID, 141)),
        _row(m(5, 10), b1=(BASE_BID, 150)),                            # bucket 6: crossed, restored inside
        _row(m(5, 20), b1=(BASE_BID, 160), flags=FLAG_CROSSED), _row(m(5, 30), b1=(BASE_BID, 161)),
        _row(m(6, 10), b1=(BASE_BID, 170)),                            # bucket 7: missing levels
        _row(m(6, 20), bid=missing), _row(m(6, 30), b1=(BASE_BID, 171)),
        _row(m(7, 10), b1=(BASE_BID, 180)),                            # bucket 8: halt carried by H rows only
        _row(m(7, 20), b1=(BASE_BID, 180), mt="H", flags=FLAG_HALTED), _row(m(7, 30), b1=(BASE_BID, 180), mt="H"),
        _row(m(8, 10), b1=(BASE_BID, 190)),                            # bucket 9: as-of at 09:38 is the H resume row -> ok
        _row(m(9, 10), b1=(BASE_BID, 200)),                            # bucket 10: halt starts 09:39:50 ...
        _row(m(9, 50), b1=(BASE_BID, 210), flags=FLAG_HALTED),
        # bucket 11 (09:41): no rows at all, fully halted
        _row(m(11, 10), b1=(BASE_BID, 220)),                           # bucket 12: ... restored 09:41:10
        _row(m(12, 10), b1=(BASE_BID, 230)),                           # bucket 13 ok
    ]
    b = F.minute_buckets(_frame(rows))
    reasons = b["invalid_reason"].tolist()[:13]
    assert reasons == ["no_mid", "", "empty_bucket", "zero_depth", "halted", "crossed", "missing_levels",
                       "halted", "", "halted", "halted", "halted", ""]
    assert b["valid"].tolist()[:13] == [r == "" for r in reasons]
    assert np.isnan(b.iloc[3]["ofi_1"])                    # Q == 0 -> no normalised ofi
    assert np.isnan(b.iloc[0]["ret"])                      # no as-of row at 09:30


def test_minute_halted_beats_crossed() -> None:
    rows = [_row(hms(9, 30, 10)), _row(hms(9, 30, 20), b1=(BASE_BID, 120), flags=FLAG_CROSSED),
            _row(hms(9, 30, 30), b1=(BASE_BID, 130), flags=FLAG_HALTED | FLAG_CROSSED)]
    ev = F.change_events(_frame(rows))
    # one raw run of bad rows -> one interval, reason by priority halted > crossed, never restored
    assert np.sort(ev.intervals.start_ts).tolist() == [hms(9, 30, 20)] and ev.intervals.reason.tolist() == ["halted"]
    assert ev.intervals.close_seq.tolist() == [np.iinfo(np.int64).max]
    assert ev.intervals.reason.tolist() == ["halted"]
    assert F.minute_buckets(_frame(rows)).iloc[0]["invalid_reason"] == "halted"


def test_minute_window_broken_by_h_only_halt_and_same_ns_cross() -> None:
    t = hms(9, 35)
    rows = [_row(t), _row(t + 1, b1=(BASE_BID, 110)),
            _row(t + 2, mt="H", flags=FLAG_HALTED), _row(t + 3, mt="H"),          # H-only halt
            _row(t + 4, b1=(BASE_BID, 120)),
            _row(t + 5, b1=(BASE_BID, 130), flags=FLAG_CROSSED), _row(t + 5, b1=(BASE_BID, 140)),   # same ns
            _row(t + 6, b1=(BASE_BID, 150))]
    ev = F.change_events(_frame(rows))
    assert np.sort(ev.intervals.start_ts).tolist() == [t + 2, t + 5]
    assert ev.segment.tolist() == [0, 0, 1, 2, 2]
    assert ev.has_prev.tolist() == [False, True, False, False, True]
    _, _, _, broken = F.window_ofi(ev, np.array([t + 1, t + 2, t + 4, t + 5, t + 6]), 1)
    assert broken.tolist() == [False, True, False, True, False]


def test_minute_analysis_window() -> None:
    b = F.minute_buckets(_frame([_row(hms(9, 30, 5))]))
    ia = b[b["in_analysis"]]
    assert len(ia) == 330
    assert ia["bucket_end_ns"].min() == hms(10, 1) and ia["bucket_end_ns"].max() == hms(15, 30)
    assert b["bucket_end_ns"].iloc[0] == hms(9, 31) and b["bucket_end_ns"].iloc[-1] == hms(16, 0)


def test_pc1_sign_and_l1() -> None:
    rng = np.random.default_rng(2)
    f = rng.normal(size=(200, 1))
    load = np.array([-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1])
    x = f @ load[None, :] + 0.05 * rng.normal(size=(200, 10))
    pw = F.fit_pc1(x)
    assert abs(np.abs(pw.w).sum() - 1.0) < 1e-12
    assert pw.w[0] > 0
    assert 0 < pw.explained_variance_ratio <= 1 and pw.explained_variance_ratio > 0.9
    assert pw.n_obs == 200
    # the direction is the loading direction, L1-normalised
    assert np.allclose(pw.w, -load / np.abs(load).sum(), atol=0.02)
    assert np.allclose(F.integrated_ofi(x, pw), x @ pw.w)
    with pytest.raises(ValueError):
        F.fit_pc1(x[:9])

