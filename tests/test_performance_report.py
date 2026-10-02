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
    # 首笔9/2开仓，曲线从9/1（前一交易日）起算：多一个基准点，交易日数不算它
    assert rep["equity"].index[0] == pd.Timestamp("2026-09-01") and rep["equity"].iloc[0] == 2000
    assert m["days"] == len(rep["equity"]) - 1 and m["bench_ret"] > 0
    text = format_telegram(rep)
    assert "模拟盘周报" in text and "持仓中：X" in text


# ─────────────────────────────────────────────────────────────────────────────
# 2026-09-29：动量账本接入周报//perf
# ─────────────────────────────────────────────────────────────────────────────

import src.performance_report as pr
from src.performance_report import pool_equal_weight, format_all, load_book


def test_pool_equal_weight_averages_daily_returns():
    idx = pd.bdate_range("2026-10-01", periods=3)
    closes = pd.DataFrame({"A": [100.0, 110.0, 110.0], "B": [100.0, 90.0, 99.0]}, index=idx)
    # 第1天平均(+10%-10%)/2=0；第2天(0%+10%)/2=+5% → 累计+5%
    assert pool_equal_weight(closes, idx) == pytest.approx(5.0)


def test_load_book_missing_returns_none(tmp_path, monkeypatch):
    import src.paper_trading as pt
    monkeypatch.setattr(pt, "_MOM", str(tmp_path / "none.json"))
    assert load_book("momentum") is None


def _momentum_ledger():
    return {"account": {"initial_value": 2000, "created_at": "2026-09-30"},
            "positions": {"1": _pos("MRNA", 3, 200.0, "2026-09-30 15:40"),
                          "2": _pos("AMD", 1, 600.0, "2026-09-30 15:40")}}


def test_momentum_report_short_history_has_pool_but_no_ratios():
    pytest.importorskip("quantstats")
    idx = pd.bdate_range("2026-09-30", periods=3)
    spy = pd.Series([500.0, 505.0, 510.0], index=idx)
    px = {"MRNA": pd.Series([200.0, 204.0, 208.0], index=idx), "AMD": pd.Series([600.0, 606.0, 612.0], index=idx)}
    pool = pd.DataFrame({"X": [100.0, 101.0, 102.01]}, index=idx)
    rep = pr.build_report(_momentum_ledger(), fetch=lambda t: spy if t == "SPY" else px[t],
                          mode="momentum", pool_fetch=lambda: pool)
    assert rep["metrics"]["too_short"] and rep["pool_ret"] == pytest.approx(2.01)
    text = pr.format_telegram(rep, header=False)
    assert "月度动量账本" in text and "同池等权" in text and "暂不计算" in text
    assert "时间止损制度" not in text


def test_total_return_measured_from_initial_value():
    """2026-10-02：开仓当天收盘浮亏也要算进总收益，SPY/同池等权同样从前一天收盘算起。"""
    pytest.importorskip("quantstats")
    idx = pd.bdate_range("2026-09-29", periods=4)          # 9/29(前一天) 9/30(开仓) 10/1 10/2
    spy = pd.Series([500.0, 510.0, 510.0, 515.1], index=idx)
    # 9/30 15:40以200/600买入，当天收盘跌到190/570；10/2涨到220/660
    px = {"MRNA": pd.Series([195.0, 190.0, 200.0, 220.0], index=idx),
          "AMD": pd.Series([590.0, 570.0, 600.0, 660.0], index=idx)}
    pool = pd.DataFrame({"X": [100.0, 102.0, 102.0, 103.02]}, index=idx)
    rep = pr.build_report(_momentum_ledger(), fetch=lambda t: spy if t == "SPY" else px[t],
                          mode="momentum", pool_fetch=lambda: pool)
    m = rep["metrics"]
    # 现金2000-600-600=800，10/2持仓3×220+660=1320 → 2120，相对起始2000为+6%
    assert rep["equity"].iloc[0] == 2000 and rep["equity"].iloc[-1] == pytest.approx(2120)
    assert m["total_ret"] == pytest.approx(6.0) and m["days"] == 3
    assert m["bench_ret"] == pytest.approx(3.02)           # 500→515.1，含开仓当天
    assert rep["pool_ret"] == pytest.approx(3.02)


def test_format_all_handles_not_started_and_failures(monkeypatch):
    pytest.importorskip("quantstats")
    idx = pd.bdate_range("2026-09-01", periods=20)
    spy = pd.Series([500 + i for i in range(20)], index=idx, dtype=float)
    x = pd.Series([100 + (i % 3) for i in range(20)], index=idx, dtype=float)
    ledger = {"account": {"initial_value": 2000, "created_at": "2026-09-01"},
              "positions": {"1": _pos("X", 5, 100.0, "2026-09-02", "2026-09-10", 102.0, 10.0, "Agent超时强制平仓")}}
    paper = pr.build_report(ledger, fetch=lambda t: spy if t == "SPY" else x)
    text = format_all({"paper": paper, "momentum": None})
    assert "原模拟盘" in text and "月度动量账本</b>\n尚未开始" in text

    # 单个账本出错不影响另一个
    monkeypatch.setattr(pr, "load_book", lambda mode: (_ for _ in ()).throw(RuntimeError("boom")) if mode == "paper" else None)
    text2 = format_all()
    assert "报告生成失败：boom" in text2 and "尚未开始" in text2


def test_extra_sections_isolated():
    def boom():
        raise RuntimeError("x")
    text = format_all({"paper": None, "momentum": None},
                      extras=[("期权", lambda: "<b>期权段</b>"), ("Alpaca", boom)])
    assert "期权段" in text and "<b>Alpaca</b>\n生成失败：x" in text


def test_alpaca_section_with_fake_client():
    from types import SimpleNamespace as NS
    c = NS(get_account=lambda: NS(portfolio_value="2010.5", cash="465.1"),
           get_all_positions=lambda: [NS(symbol="MU", qty="0.7302", unrealized_pl="12.3", asset_class="us_equity"),
                                      NS(symbol="IWM261106P00265000", qty="-1", unrealized_pl="3", asset_class="us_option"),
                                      NS(symbol="IWM261106P00263000", qty="1", unrealized_pl="-2", asset_class="us_option")])
    text = pr.alpaca_section(c)
    assert "净值$2,010.50" in text and "+0.52%" in text and "MU 0.7302股 浮动$+12" in text and "期权腿：2条" in text
