"""
backtest.py 测试：出场规则、统计工具、财报日计算、信号重放（合成数据，不联网）。
vectorbt只在本地研究环境安装，组合层测试在没装时跳过。
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest import (simulate_trades, wilson_ci, trade_stats, days_to_next_earnings,
                          compute_signals, random_control_trades)


def _ohlc(opens, highs, lows, closes, start="2026-01-05"):
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes,
                         "Volume": [1e6] * len(closes)}, index=idx)


def _sig(hist, go_days, stop_frac=0.05):
    s = pd.DataFrame({"go": False, "path": "A", "score": 80, "conviction": 3, "stop_frac": stop_frac},
                     index=hist.index)
    s.iloc[go_days, 0] = True
    return s


class TestSimulateTrades:
    def test_stop_hit_intraday_fills_at_stop(self):
        h = _ohlc([100, 99, 98], [101, 100, 99], [99, 94, 97], [100, 98, 98])
        t = simulate_trades(h, _sig(h, [0]))[0]
        assert t["reason"] == "stop" and t["exit"] == 95.0 and t["ret_pct"] == -5.0

    def test_gap_below_stop_fills_at_open(self):
        h = _ohlc([100, 90], [101, 91], [99, 88], [100, 89])
        t = simulate_trades(h, _sig(h, [0]))[0]
        assert t["reason"] == "stop_gap" and t["exit"] == 90.0

    def test_time_stop_after_10_calendar_days(self):
        n = 12
        h = _ohlc([100] * n, [101] * n, [99] * n, [100] * (n - 1) + [103])
        t = simulate_trades(h, _sig(h, [0]))[0]
        # 1/5周一入场，第9个工作日是1/15（10天），1/16（11天）才超过10天
        assert t["reason"] == "time" and t["exit_date"] == pd.Timestamp("2026-01-16")

    def test_target_only_when_enabled(self):
        h = _ohlc([100, 100, 100], [101, 108, 101], [99, 99, 99], [100, 100, 100])
        assert simulate_trades(h, _sig(h, [0]))[0]["reason"] == "open"
        t = simulate_trades(h, _sig(h, [0], stop_frac=0.03), take_profit_atr=2.0)[0]
        # stop_frac=3% = 1.5ATR → ATR=2% → 目标2ATR=+4%
        assert t["reason"] == "target" and t["exit"] == pytest.approx(104.0)

    def test_no_new_entry_while_holding(self):
        n = 15
        h = _ohlc([100] * n, [101] * n, [99] * n, [100] * n)
        trades = simulate_trades(h, _sig(h, [0, 2, 4, 12]))
        assert [t["entry_date"] for t in trades] == [h.index[0], h.index[12]]

    def test_open_trade_excluded_from_stats(self):
        h = _ohlc([100, 100], [101, 101], [99, 99], [100, 100])
        df = pd.DataFrame(simulate_trades(h, _sig(h, [0])))
        assert df["ret_pct"].isna().all() and trade_stats(df) == {"n": 0}


class TestStats:
    def test_wilson_matches_known_value(self):
        # 2胜12笔（模拟盘9/28的真实样本）→ wiki记录的4.7%-44.8%
        assert wilson_ci(2, 12) == (4.7, 44.8)

    def test_trade_stats(self):
        df = pd.DataFrame({"ret_pct": [4.0, -2.0, 2.0, -4.0], "bars": [5, 5, 5, 5]})
        s = trade_stats(df)
        assert s["n"] == 4 and s["win_rate"] == 50.0 and s["avg_ret"] == 0.0 and s["payoff"] == 1.0


class TestEarnings:
    def test_days_to_next(self):
        idx = pd.DatetimeIndex(["2026-01-05", "2026-01-12", "2026-01-20"])
        d = days_to_next_earnings(idx, [date(2026, 1, 15)])
        assert list(d[:2]) == [10, 3] and np.isinf(d[2])

    def test_no_dates_means_no_filter(self):
        assert np.isinf(days_to_next_earnings(pd.DatetimeIndex(["2026-01-05"]), [])).all()


def _uptrend(n=320, seed=0):
    rng = np.random.default_rng(seed)
    close = 50 * np.exp(np.cumsum(0.004 + 0.012 * rng.standard_normal(n)))
    idx = pd.bdate_range("2025-01-01", periods=n)
    h = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                      "Volume": 1e6 * (1 + 0.2 * rng.random(n))}, index=idx)
    return h


class TestComputeSignals:
    def test_steady_uptrend_produces_go_days(self):
        h = _uptrend()
        spy = pd.Series(100.0, index=h.index)
        vix = pd.Series(15.0, index=h.index)
        sig = compute_signals(h, spy, vix, [])
        assert sig["go"].any()
        assert set(sig.loc[sig["go"], "fails"]) == {""}

    def test_earnings_blackout_blocks(self):
        h = _uptrend()
        spy = pd.Series(100.0, index=h.index)
        vix = pd.Series(15.0, index=h.index)
        # 每个交易日之后3天都有财报 → 全部落在7天禁入窗口
        earnings = sorted({(d + pd.Timedelta(days=3)).date() for d in h.index})
        sig = compute_signals(h, spy, vix, earnings)
        assert not sig["go"].any()
        assert sig["fails"].str.contains("earnings_blackout").all()

    def test_downtrend_fails_trend(self):
        h = _uptrend()
        h = h.iloc[::-1].set_axis(h.index)   # 反转成下跌
        sig = compute_signals(h, pd.Series(100.0, index=h.index), pd.Series(15.0, index=h.index), [])
        assert not sig["go"].any()


class TestRandomControl:
    def test_same_count_per_ticker(self):
        h = _uptrend(400)
        trades = pd.DataFrame({"ticker": ["X"] * 5})
        ctrl = random_control_trades({"X": h}, trades, seed=1)
        assert len(ctrl) == 5 and set(ctrl["path"]) == {"R"}


def test_portfolio_runs_with_vectorbt():
    pytest.importorskip("vectorbt")
    from src.backtest import run_portfolio
    h = _uptrend(300)
    trades = pd.DataFrame(simulate_trades(h, _sig(h, [260, 280], stop_frac=0.05), "X"))
    pf = run_portfolio({"X": h}, trades)
    assert pf.stats()["Total Trades"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 2026-09-29：ablation / SHORT
# ─────────────────────────────────────────────────────────────────────────────

from src.backtest import ablate, HARD_GATES


class TestSignalFlags:
    def test_flag_columns_and_go_consistency(self):
        h = _uptrend()
        sig = compute_signals(h, pd.Series(100.0, index=h.index), pd.Series(15.0, index=h.index), [])
        assert all(g in sig.columns for g in HARD_GATES)
        # go必须所有硬门都通过
        assert sig.loc[sig["go"], list(HARD_GATES)].all().all()

    def test_max_fails_scores_near_miss_days_but_go_unchanged(self):
        h = _uptrend()
        spy, vix = pd.Series(100.0, index=h.index), pd.Series(15.0, index=h.index)
        strict = compute_signals(h, spy, vix, [])
        loose = compute_signals(h, spy, vix, [], max_fails=1)
        assert (strict["go"] == loose["go"]).all()
        near_miss = loose[(~loose[list(HARD_GATES)]).sum(axis=1) == 1]
        assert near_miss["score"].notna().all()

    def test_ablate_turns_single_failure_into_go(self):
        idx = pd.bdate_range("2026-01-05", periods=3)
        sig = pd.DataFrame({"trend": [True, False, False], "rsi": [True, True, False],
                            "volume": True, "stop_distance": True, "earnings_blackout": True,
                            "score": [80, 80, None], "go": [True, False, False]}, index=idx)
        out = ablate(sig, ("trend",))
        assert list(out["go"]) == [True, True, False]

    def test_short_flags_on_downtrend(self):
        h = _uptrend()
        h = h.iloc[::-1].set_axis(h.index)
        sig = compute_signals(h, pd.Series(100.0, index=h.index), pd.Series(15.0, index=h.index), [],
                              direction="SHORT")
        assert sig["trend"].mean() > 0.5


class TestSimulateShort:
    def test_stop_above_entry(self):
        h = _ohlc([100, 101, 102], [101, 106, 103], [99, 100, 101], [100, 102, 102])
        t = simulate_trades(h, _sig(h, [0]), direction="SHORT")[0]
        assert t["reason"] == "stop" and t["exit"] == 105.0 and t["ret_pct"] == -5.0

    def test_gap_above_stop_fills_at_open(self):
        h = _ohlc([100, 110], [101, 111], [99, 108], [100, 109])
        t = simulate_trades(h, _sig(h, [0]), direction="SHORT")[0]
        assert t["reason"] == "stop_gap" and t["ret_pct"] == -10.0

    def test_profit_when_price_falls_then_time_stop(self):
        n = 12
        # 1/5入场，1/16（第10根，11个日历日）触发时间止损，当天收90
        h = _ohlc([100] * n, [101] * n, [89] * n, [100] * 9 + [90] * 3)
        t = simulate_trades(h, _sig(h, [0]), direction="SHORT")[0]
        assert t["reason"] == "time" and t["ret_pct"] == 10.0

    def test_random_control_short_direction(self):
        h = _uptrend(400)
        ctrl = random_control_trades({"X": h}, pd.DataFrame({"ticker": ["X"] * 3}), seed=2, direction="SHORT")
        assert len(ctrl) == 3 and (ctrl["direction"] == "SHORT").all()
