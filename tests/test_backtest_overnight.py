"""backtest_overnight.py 纯函数测试（不联网）。"""
import pandas as pd
import pytest

from src.backtest_overnight import managed_exit, entry_signals, daily_exits


def _bars(rows):
    idx = pd.date_range("2026-09-29 09:30", periods=len(rows), freq="60min", tz="America/New_York")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)


class TestManagedExit:
    def test_gap_below_stop_sells_at_open(self):
        assert managed_exit(100, _bars([(97, 99, 96, 98)])) == pytest.approx((-0.03, "gap_stop"))

    def test_first_hour_stop(self):
        r, why = managed_exit(100, _bars([(100, 101, 97.5, 99), (99, 104, 99, 103)]))
        assert why == "first_hour_stop" and r == pytest.approx(-0.02)

    def test_trailing_take_profit_uses_prior_peak(self):
        # 第1根最高103，第2根先摸到104再回落：用第2根开始前的峰值103判断，103×98.5%=101.455
        r, why = managed_exit(100, _bars([(100, 103, 99.5, 102.5), (102.5, 104, 101, 101.5)]))
        assert why == "trail" and r == pytest.approx(0.01455)

    def test_no_trail_below_entry_then_close(self):
        # 峰值只到100.5，回撤触发价低于入场价，不算止盈；也没到止损 → 收盘卖
        r, why = managed_exit(100, _bars([(100, 100.5, 99, 99.5), (99.5, 100, 98.5, 99.2)]))
        assert why == "close" and r == pytest.approx(-0.008)

    def test_no_data(self):
        assert managed_exit(100, _bars([]))[1] == "no_data"


def test_entry_signals_and_daily_exits():
    idx = pd.bdate_range("2026-01-05", periods=23)
    h = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1e6}, index=idx)
    h.iloc[-2, h.columns.get_loc("Close")] = 102.0
    h.iloc[-2, h.columns.get_loc("Volume")] = 3e6
    h.iloc[-1, h.columns.get_loc("Open")] = 103.02
    s = entry_signals(h)
    assert s["rvol"].iloc[-2] == pytest.approx(3.0) and bool(s["up"].iloc[-2])
    e = daily_exits(h)
    assert e["e0"].iloc[-2] == pytest.approx(0.01)
