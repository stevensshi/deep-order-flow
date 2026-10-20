"""End to end on synthetic data, without any market data: ITCH 5.0 bytes -> per-stock order-book tables -> order
flow and labels after Kolm, Turiel and Westray (2023) -> ridge regression on two training days -> out-of-sample R²
per horizon on a third day.

  python examples/synthetic_end_to_end.py --out /tmp/dof-example

The three days are seeded random days (``dof.lob.testing.synth.random_day``); their numbers say nothing about real
markets. Prints one row per stock: R² against the evaluation day's mean label, in percent, for horizons k = 1..10.
"""

from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import numpy as np

from dof.lob import features as F
from dof.lob import stats as S
from dof.lob.tables import build_day, load_states
from dof.lob.testing.synth import write_random_day

SYMBOLS = ("AAA", "BBB", "CCC")
TRAIN = (("2019-01-30", 1), ("2019-01-31", 2))      # (date, seed)
EVAL = (("2019-02-01", 3),)
HISTORY, CHUNK, LAMBDA = 100, 16_384, 0.1


def day_arrays(book: Path, symbol: str, date: str, tw: F.TrainWindow, session: F.Session):
    """Order flow (n, 20), labels of the valid horizons (n, K), sample mask, valid horizons."""
    kf = F.flow_pipeline(load_states(book / date, symbol, columns=F.FLOW_COLUMNS), session)
    labels = F.horizon_labels(kf, train_window=tw)
    k_valid = F.k_valid_from(labels)
    mask = F.sample_mask(labels, k_valid, F.history_valid(kf, HISTORY))
    ks = [int(k) for k in labels.k]
    return kf.of, labels.y[:, [ks.index(k) for k in k_valid]], mask, k_valid


def fit_and_score(book: Path, symbol: str, session: F.Session) -> tuple[list[int], list[float]]:
    train_dates = [d for d, _ in TRAIN]
    frames = [load_states(book / d, symbol, columns=F.FLOW_COLUMNS) for d in train_dates]
    tw = F.estimate_delta_t(frames, [datetime.date.fromisoformat(d) for d in train_dates], [session] * len(frames))
    del frames
    train = [day_arrays(book, symbol, d, tw, session) for d in train_dates]
    ny = len(train[0][3])
    sc_x = F.fit_scaler_columnwise(lambda c: np.concatenate([of[m, c] for of, _, m, _ in train]), 20)
    sc_y = F.fit_scaler_columnwise(lambda c: np.concatenate([y[m, c] for _, y, m, _ in train]), ny)
    s = S.suff_stats_init(HISTORY * 20, ny)
    for of, y, m, _ in train:
        z, zy, rows = F.apply_scaler(of, sc_x), F.apply_scaler(y, sc_y), np.flatnonzero(m)
        for a in range(0, len(rows), CHUNK):
            ch = rows[a:a + CHUNK]
            S.suff_stats_update(s, F.lagged_design(z, ch, HISTORY), zy[ch])
    w, b, _info = S.solve_ridge(s, LAMBDA)
    (of, y, m, k_valid), = [day_arrays(book, symbol, d, tw, session) for d, _ in EVAL]
    z, rows = F.apply_scaler(of, sc_x), np.flatnonzero(m)
    zh = np.vstack([F.lagged_design(z, rows[a:a + CHUNK], HISTORY) @ w + b for a in range(0, len(rows), CHUNK)])
    yhat, yt = F.inverse_target(zh, sc_y), y[rows]
    return k_valid, [S.r2_os(yt[:, j], yhat[:, j]) for j in range(len(k_valid))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="a folder for the ITCH files and the tables")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    itch, book = args.out / "itch", args.out / "book"
    itch.mkdir(parents=True, exist_ok=True)
    for date, seed in TRAIN + EVAL:
        path = write_random_day(itch, date, seed)
        res = build_day(path, list(SYMBOLS), book, manifest_csv=None, date=datetime.date.fromisoformat(date),
                        overwrite=True)
        if res.status != "complete":
            raise SystemExit(f"{date}: build status {res.status}")
    session = F.Session.full_day()
    print("R² (%) on the evaluation day, against its mean label")
    header = None
    for symbol in SYMBOLS:
        k_valid, r2 = fit_and_score(book, symbol, session)
        if header is None:
            header = "symbol " + " ".join(f"{'k=' + str(k):>8}" for k in k_valid)
            print(header)
        print(f"{symbol:<6} " + " ".join(f"{100 * v:8.3f}" for v in r2))
    print(f"seconds={time.perf_counter() - t0:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
