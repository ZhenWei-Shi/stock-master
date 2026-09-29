"""
performance_report.py 测试：资金曲线还原、逐笔统计、报告拼装（不联网）。
quantstats只在curve_metrics里用，没装时跳过对应测试。
"""
import pandas as pd
import pytest

from src.performance_report import build_equity_curve, trade_summary, build_report, format_telegram

CAL = pd.bdate_range("2026-09-14", periods=6)   # 9/14(一) ~ 9/21(一)


def _pos(ticker, shares, entry, opened, closed=None, exit_px=None, pnl=None, reason=None):
    return {"ticker": ticker, "shares": shares, "entry_price": entry, "opened_at": opened,
            "closed_at": closed, "exit_price": exit_px, "status": "closed" if closed else "open",
            "pnl": pnl, "pnl_pct": (exit_px / entry - 1) * 100 if exit_px else None, "exit_reason": reason}


class TestEquityCurve:
    def test_open_position_marked_to_close(self):
        closes = {"X": pd.Series([10, 11, 12, 13, 14, 15], index=CAL, dtype=float)}
        eq = build_equity_curve([_pos("X", 10, 10.0, "2026-09-15 15:30:00-04:00")], 1000, closes, CAL)
        # 9/14未开仓=1000；9/15开仓：现金900+10股×11=1010；9/21：900+150=1050
        assert list(eq.round(2)) == [1000, 1010, 1020, 1030, 1040, 1050]

    def test_closed_position_returns_to_cash(self):
        closes = {"X": pd.Series([10, 11, 12, 13, 14, 15], index=CAL, dtype=float)}
        p = _pos("X", 10, 10.0, "2026-09-15 09:45", "2026-09-17 10:00", 12.5, 25.0, "time")
        eq = build_equity_curve([p], 1000, closes, CAL)
        # 平仓日起按平仓价回款，不再按收盘价估值
        assert eq["2026-09-16"] == 1020 and eq["2026-09-17"] == 1025 and eq.iloc[-1] == 1025

    def test_missing_prices_fall_back_to_entry(self):
        eq = build_equity_curve([_pos("Y", 2, 50.0, "2026-09-16")], 1000, {}, CAL)
        assert eq.iloc[-1] == 1000

    def test_tz_aware_price_index(self):
        idx = CAL.tz_localize("America/New_York")
        closes = {"X": pd.Series([10.0] * 6, index=idx)}
        eq = build_equity_curve([_pos("X", 1, 10.0, "2026-09-14")], 100, closes, CAL)
        assert (eq == 100).all()


class TestTradeSummary:
    def test_regime_split_uses_close_date(self):
        ps = [
            _pos("A", 1, 100, "2026-09-01", "2026-09-05", 90, -10, "Agent自动止损"),
            # 9/16前开仓、之后被时间止损平掉 → 属于新制度
            _pos("B", 1, 100, "2026-09-10", "2026-09-21", 105, 5, "Agent超时强制平仓"),
            _pos("C", 1, 100, "2026-09-22"),
        ]
        t = trade_summary(ps)
        assert t["all"]["n"] == 2 and t["all"]["wins"] == 1
        assert t["time_stop_regime"]["n"] == 1 and t["time_stop_regime"]["wins"] == 1
        assert set(t["by_reason"]) == {"Agent自动止损", "Agent超时强制平仓"}
        assert t["open"] == ["C"]


def test_full_report_with_fake_prices():
    pytest.importorskip("quantstats")
    idx = pd.bdate_range("2026-09-01", periods=20)
    spy = pd.Series([500 + i for i in range(20)], index=idx, dtype=float)
    x = pd.Series([100 + (i % 3) for i in range(20)], index=idx, dtype=float)
    ledger = {
        "account": {"initial_value": 2000, "created_at": "2026-09-01"},
        "positions": {"1": _pos("X", 5, 100.0, "2026-09-02", "2026-09-10", 102.0, 10.0, "Agent超时强制平仓"),
                      "2": _pos("X", 5, 101.0, "2026-09-15")},
    }
    rep = build_report(ledger, fetch=lambda t: spy if t == "SPY" else x)
    m = rep["metrics"]
    assert m["days"] == len(rep["equity"]) and m["bench_ret"] > 0
    text = format_telegram(rep)
    assert "模拟盘周报" in text and "持仓中：X" in text
