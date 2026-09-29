"""
momentum_book.py 与相关paper_trading改动的测试（不联网）。
"""
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import src.momentum_book as mb
import src.paper_trading as pt


class TestCalendar:
    def test_last_trading_day(self):
        assert mb.is_last_trading_day(date(2026, 9, 30)) is True
        assert mb.is_last_trading_day(date(2026, 9, 29)) is False
        # 2026-10-31是周六 → 10/30(周五)是月末最后交易日
        assert mb.is_last_trading_day(date(2026, 10, 30)) is True


def _closes(n=300):
    idx = pd.bdate_range("2025-06-02", periods=n)
    t = np.arange(n)
    return pd.DataFrame({"UP": 100 * 1.003 ** t, "MID": 100 * 1.001 ** t,
                         "FLAT": np.full(n, 100.0), "DOWN": 100 * 0.998 ** t}, index=idx)


class TestRankAndPlan:
    def test_rank_skips_last_month(self):
        r = mb.rank_momentum(_closes())
        assert list(r.index) == ["UP", "MID", "FLAT", "DOWN"]
        c = _closes()
        expected = c["UP"].iloc[-1 - mb.SKIP] / c["UP"].iloc[-1 - mb.LOOKBACK] - 1
        assert r["UP"] == pytest.approx(expected)

    def test_rank_needs_history(self):
        assert mb.rank_momentum(_closes(100)).empty

    def test_plan_skips_unaffordable_and_vetoed(self):
        ranked = pd.Series([0.9, 0.8, 0.7, 0.6], index=["BIG", "VETO", "A", "B"])
        prices = {"BIG": 1069.0, "VETO": 50.0, "A": 100.0, "B": 200.0}
        plan = mb.plan_rebalance(ranked, prices, held=[], book_value=2000, risk_ok=lambda t: t != "VETO")
        # 单仓预算=2000×40%=800：BIG买不起，VETO被风控否决
        assert plan["targets"] == ["A", "B"] and plan["shares"] == {"A": 8, "B": 4}
        assert ("BIG", "买不起") in plan["skipped"] and ("VETO", "风控否决") in plan["skipped"]

    def test_plan_keeps_holdings_still_in_top(self):
        ranked = pd.Series([0.9, 0.8, 0.7], index=["A", "C", "B"])
        plan = mb.plan_rebalance(ranked, {"A": 10.0, "B": 10.0, "C": 10.0}, held=["A", "B"], book_value=2000)
        assert plan["targets"] == ["A", "C"] and plan["sell"] == ["B"] and plan["buy"] == ["C"]
        assert "A" not in plan["shares"]


class TestPaperTradingChanges:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            pt._ledger_path("typo")
        assert pt._ledger_path("momentum").endswith("momentum_trades.json")

    def test_momentum_positions_skip_time_stop(self):
        opened = datetime.now(pt.ET) - timedelta(days=25)
        pos = {"stop_loss": 80.0, "target": 10000.0, "opened_at": str(opened)}
        assert pt._check_position_alert(100.0, {**pos, "strategy": "Agent/LONG/AggressiveSwing"},
                                        datetime.now(pt.ET))[1] == "time_stop"
        assert pt._check_position_alert(100.0, {**pos, "strategy": "Momentum/Monthly"},
                                        datetime.now(pt.ET)) == (None, None)
        # 价格止损对动量仓位照常生效
        assert pt._check_position_alert(79.0, {**pos, "strategy": "Momentum/Monthly"},
                                        datetime.now(pt.ET))[1] == "stop_loss"


def test_rebalance_end_to_end_with_mocks(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "_MOM", str(tmp_path / "mom.json"))
    monkeypatch.setattr(mb, "_LOGFILE", str(tmp_path / "log.json"))
    monkeypatch.setattr(mb, "universe", lambda wl=None: ["UP", "MID", "FLAT", "DOWN"])
    closes = _closes()
    monkeypatch.setattr("yfinance.download", lambda *a, **k: pd.concat({"Close": closes}, axis=1))
    monkeypatch.setattr("src.risk_layer.risk_check", lambda t, **k: {"ok": True})

    r = mb.rebalance(force=True)
    assert r["plan"]["targets"] == ["UP", "MID"]
    book = pt.list_positions("momentum")
    assert sorted(p["ticker"] for p in book["open"]) == ["MID", "UP"]
    assert all(p["strategy"] == "Momentum/Monthly" for p in book["open"])
    assert "月度动量换仓" in mb.format_rebalance(r)

    # 非月末且不强制：不动
    monkeypatch.setattr(mb, "is_last_trading_day", lambda d: False)
    assert "skipped" in mb.rebalance()
