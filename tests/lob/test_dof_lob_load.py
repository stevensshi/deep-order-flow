"""``load_states``: the stock-day must be valid; the joined event columns; pins of the reading rules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dof.lob.tables import InvalidStockDay, build_day, load_states
from dof.lob.testing import synth as ls


def test_load_states_rejects_invalid(tmp_path: Path) -> None:
    gz = ls.write_stream(tmp_path / "01302019.NASDAQ_ITCH50.gz", ls.stream(ls.SCENARIO_FULL), gz=True)
    res = build_day(gz, ["TEST", "NOPE"], tmp_path / "out", manifest_csv=None)
    with pytest.raises(InvalidStockDay) as exc:
        load_states(res.day_dir, "TEST")
    assert exc.value.symbol == "TEST" and "invalid" in exc.value.reason
    with pytest.raises(InvalidStockDay):
        load_states(res.day_dir, "NOPE")
    # A valid day (messages before the unknown ref) loads with the joined event columns.
    gz2 = ls.write_stream(tmp_path / "01312019.NASDAQ_ITCH50.gz",
                          ls.stream(ls.SCENARIO_FULL[:25] + [ls.msg_S(ls.T(20, 5), "C")]), gz=True)
    res2 = build_day(gz2, ["TEST"], tmp_path / "out", manifest_csv=None)
    df = load_states(res2.day_dir, "TEST")
    want = [r for r in ls.EXPECTED_FULL if r[0] < 25]
    assert df["seq"].tolist() == [r[0] for r in want]
    assert df["msg_type"].tolist()[:3] == ["H", "A", "A"]
    assert df.loc[df["seq"] == 23, "trading_state"].item() == "H"
    assert df["bid_px_1"].tolist() == [r[2][0] for r in want]


# -- pins: each fails if the behaviour named in its docstring changes --------------------------------


def _valid_day(tmp_path: Path) -> Path:
    """A valid stock-day: the messages before the unknown ref of SCENARIO_FULL, then the end of messages."""
    payloads = ls.SCENARIO_FULL[:25] + [ls.msg_S(ls.T(20, 5), "C")]
    gz = ls.write_stream(tmp_path / "01312019.NASDAQ_ITCH50.gz", ls.stream(payloads), gz=True)
    return build_day(gz, ["TEST"], tmp_path / "out", manifest_csv=None).day_dir


def test_load_states_refuses_partial_build(tmp_path: Path) -> None:
    gz = ls.write_stream(tmp_path / "01312019.NASDAQ_ITCH50.gz", ls.stream(ls.SCENARIO_FULL[:25]), gz=True)
    day = build_day(gz, ["TEST"], tmp_path / "out", manifest_csv=None).day_dir      # no end-of-messages event
    with pytest.raises(InvalidStockDay, match="build status=partial"):
        load_states(day, "TEST")


def test_load_states_missing_file_raises_invalid_stock_day(tmp_path: Path) -> None:
    day = _valid_day(tmp_path)
    (day / "TEST.events.parquet").unlink()
    with pytest.raises(InvalidStockDay, match="files missing"):
        load_states(day, "TEST")


def test_load_states_columns_subset(tmp_path: Path) -> None:
    """Only the requested state columns plus seq are read; an event column only when it is asked for."""
    day = _valid_day(tmp_path)
    assert list(load_states(day, "TEST", columns=["ts_ns", "bid_px_1"]).columns) == ["seq", "ts_ns", "bid_px_1"]
    assert list(load_states(day, "TEST", columns=["ts_ns", "msg_type"]).columns) == ["seq", "ts_ns", "msg_type"]


def test_load_states_size_columns_int32(tmp_path: Path) -> None:
    """Size columns are held as int32 (lossless here); price columns stay int64."""
    df = load_states(_valid_day(tmp_path), "TEST")
    sizes = [c for c in df.columns if "_sz_" in c]
    assert len(sizes) == 20 and all(df[c].dtype == np.int32 for c in sizes)
    assert all(df[c].dtype == np.int64 for c in df.columns if "_px_" in c)


def test_load_states_sorts_by_seq(tmp_path: Path) -> None:
    """A states table stored out of seq order is returned sorted by seq."""
    day = _valid_day(tmp_path)
    want = load_states(day, "TEST")
    path = day / "TEST.states.parquet"
    t = pq.read_table(path)
    pq.write_table(t.take(pa.array(np.arange(t.num_rows - 1, -1, -1))), path, compression="zstd")
    got = load_states(day, "TEST")
    assert got["seq"].tolist() == sorted(got["seq"].tolist())
    assert got.equals(want)


def test_load_states_seq_mismatch_raises(tmp_path: Path) -> None:
    """A state row whose seq has no event row is an error."""
    day = _valid_day(tmp_path)
    states = pq.read_table(day / "TEST.states.parquet", columns=["seq"]).column("seq").to_pylist()
    path = day / "TEST.events.parquet"
    ev = pq.read_table(path)
    keep = pa.array([s != states[1] for s in ev.column("seq").to_pylist()])
    pq.write_table(ev.filter(keep), path, compression="zstd")
    with pytest.raises(ValueError, match="state rows without a matching event seq"):
        load_states(day, "TEST")
