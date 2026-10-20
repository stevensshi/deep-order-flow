"""The universe file, reused tickers and the corporate-action file of the test days."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from dof.deep.config import load
from dof.deep.errors import ConfigError, TestDayRefused
from dof.deep.universe import ca_excluded, corporate_actions, reused, universe

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "configs" / "smoke.toml"
HEADER = "ticker,event,ratio,effective_date,status,source_url,quote\n"


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.delenv("DOF_LOCAL_CONFIG", raising=False)
    return load(SMOKE)


def _with_ca(cfg, path: Path, sha: str | None = None):
    sha = sha if sha is not None else hashlib.sha256(path.read_bytes()).hexdigest()
    return replace(cfg, protocol=replace(cfg.protocol, corporate_actions=path, corporate_actions_sha256=sha))


def test_universe_follows_file_order(cfg) -> None:
    assert universe(cfg, cfg.split) == ["FAST", "UAL"]          # the split lists them as UAL, FAST


def test_universe_missing_symbol_raises(cfg) -> None:
    with pytest.raises(ConfigError, match="ZZZZ"):
        universe(cfg, replace(cfg.split, symbols=("UAL", "ZZZZ")))
    with pytest.raises(ConfigError):
        universe(replace(cfg, universe=ROOT / "configs" / "no_such_universe.csv"), cfg.split)


def test_reused_from_the_date_on(cfg) -> None:
    split = replace(cfg.split, reused_from=(("FB", "2025-01-01"),))
    assert not reused(split, "FB", "2024-12-31")
    assert reused(split, "FB", "2025-01-01") and reused(split, "FB", "2025-12-10")
    assert not reused(split, "UAL", "2025-12-10")


def test_corporate_actions_checks(cfg, tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    with pytest.raises(TestDayRefused, match="missing"):
        corporate_actions(_with_ca(cfg, missing, "0" * 64))
    good = tmp_path / "good.csv"
    good.write_text(HEADER + "FAST,forward_split,2:1,2019-10-10,confirmed,,x\n", encoding="utf-8")
    with pytest.raises(TestDayRefused, match="sha256"):
        corporate_actions(_with_ca(cfg, good, "0" * 64))
    no_status = tmp_path / "no_status.csv"
    no_status.write_text("ticker,event,effective_date\nFAST,forward_split,2019-10-10\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="status"):
        corporate_actions(_with_ca(cfg, no_status))
    bad_date = tmp_path / "bad_date.csv"
    bad_date.write_text(HEADER + "FAST,forward_split,2:1,10/10/2019,confirmed,,x\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="YYYY-MM-DD"):
        corporate_actions(_with_ca(cfg, bad_date))
    ok = corporate_actions(_with_ca(cfg, good))
    assert ok["rows"][0]["applies"] and ok["file"] == "good.csv"


def test_ca_excluded_smoke_fixture(cfg) -> None:
    ca = corporate_actions(cfg)
    assert ca["sha256"] == cfg.protocol.corporate_actions_sha256 and len(ca["rows"]) == 4
    out = ca_excluded(cfg, cfg.split, ca)
    assert set(out) == {("FAST", "2019-10-18")}
    assert out[("FAST", "2019-10-18")] == {"ticker": "FAST", "event": "forward_split", "effective_date": "2019-10-10",
                                           "status": "confirmed"}
