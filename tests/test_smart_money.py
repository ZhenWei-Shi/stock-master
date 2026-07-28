"""
smart_money.py smart_money_flow() 回归测试

背景（2026-07-28修复）：函数固定取"最近12根5分钟K线(60分钟)"贴上
"收盘段/机构尾盘"标签，但盘中任意时段调用时这12根K线只是"到目前为止
最近60分钟"，并非真正的收盘60分钟。修复后新增 is_closing_window 字段
（15:00 ET起才算真正收盘段），非收盘时段文案改用中性的"近60分钟"表述。
本文件保护这个时间窗口判断不被之后的改动悄悄撤回。

用 monkeypatch 替换 yf.Ticker 和 datetime.now，构造确定性的5分钟K线数据，
不发真实网络请求（跟项目现有测试"直接导入生产代码测纯逻辑"的哲学一致，
smart_money_flow本身不是纯函数，用monkeypatch隔离掉它仅有的两个外部依赖）。
"""
from datetime import datetime

import pandas as pd
import pytz

import src.smart_money as sm

ET = pytz.timezone("America/New_York")


class _FakeTicker:
    def __init__(self, hist_5m, hist_daily):
        self._hist_5m = hist_5m
        self._hist_daily = hist_daily

    def history(self, period=None, interval=None):
        if interval == "5m":
            return self._hist_5m
        return self._hist_daily


def _build_hist_5m(open_price, close_price, n_open=6, n_close=12, vol=1000):
    idx = pd.date_range("2026-07-28 09:30", periods=n_open + n_close, freq="5min", tz=ET)
    closes = [open_price] * n_open + [close_price] * n_close
    return pd.DataFrame({"Close": closes, "Volume": [vol] * (n_open + n_close)}, index=idx)


def _build_daily(prev_close, today_partial):
    idx = pd.date_range("2026-07-27", periods=2, freq="D")
    return pd.DataFrame({"Close": [prev_close, today_partial]}, index=idx)


def _patch_ticker(monkeypatch, hist_5m, hist_daily):
    monkeypatch.setattr(sm.yf, "Ticker", lambda ticker: _FakeTicker(hist_5m, hist_daily))


def _freeze(monkeypatch, hour, minute):
    fixed = ET.localize(datetime(2026, 7, 28, hour, minute))

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(sm, "datetime", _FrozenDatetime)


def _bullish_data():
    # 开盘段收在prev_close下方(97 vs 100)=开盘弱；近60分钟段收在97上方(99)=近段转买
    hist_5m = _build_hist_5m(open_price=97.0, close_price=99.0)
    hist_daily = _build_daily(prev_close=100.0, today_partial=99.0)
    return hist_5m, hist_daily


class TestClosingWindowFraming:
    def test_true_closing_window_uses_institutional_framing(self, monkeypatch):
        hist_5m, hist_daily = _bullish_data()
        _patch_ticker(monkeypatch, hist_5m, hist_daily)
        _freeze(monkeypatch, 15, 30)  # 15:30 ET，真正收盘段

        r = sm.smart_money_flow("TEST")

        assert r["ok"] is True
        assert r["is_closing_window"] is True
        assert r["smf_bias"] == "bullish"
        assert "机构" in r["smf_signal"] or "收盘" in r["smf_signal"]

    def test_midday_call_uses_neutral_framing_not_institutional(self, monkeypatch):
        hist_5m, hist_daily = _bullish_data()
        _patch_ticker(monkeypatch, hist_5m, hist_daily)
        _freeze(monkeypatch, 11, 30)  # 盘中，非真正收盘段

        r = sm.smart_money_flow("TEST")

        assert r["ok"] is True
        assert r["is_closing_window"] is False
        assert r["smf_bias"] == "bullish"  # 多空方向判断本身不受时段影响
        assert "机构" not in r["smf_signal"], "盘中调用不该用'机构'措辞，会被误读成尾盘资金动作"
        assert "收盘" not in r["smf_signal"]
        assert "近60分钟" in r["smf_signal"]
        assert "非收盘时段" in r["note"]

    def test_boundary_15_00_is_closing_window(self, monkeypatch):
        hist_5m, hist_daily = _bullish_data()
        _patch_ticker(monkeypatch, hist_5m, hist_daily)
        _freeze(monkeypatch, 15, 0)

        assert sm.smart_money_flow("TEST")["is_closing_window"] is True

    def test_boundary_14_59_is_not_closing_window(self, monkeypatch):
        hist_5m, hist_daily = _bullish_data()
        _patch_ticker(monkeypatch, hist_5m, hist_daily)
        _freeze(monkeypatch, 14, 59)

        assert sm.smart_money_flow("TEST")["is_closing_window"] is False


class TestSmfBiasClassification:
    def test_bearish_open_strong_close_weak(self, monkeypatch):
        # 开盘段收在prev_close上方(103)=开盘散户追涨；近段回落到101=资金转卖
        hist_5m = _build_hist_5m(open_price=103.0, close_price=101.0)
        hist_daily = _build_daily(prev_close=100.0, today_partial=101.0)
        _patch_ticker(monkeypatch, hist_5m, hist_daily)
        _freeze(monkeypatch, 15, 30)

        r = sm.smart_money_flow("TEST")
        assert r["smf_bias"] == "bearish"

    def test_neutral_when_moves_below_threshold(self, monkeypatch):
        hist_5m = _build_hist_5m(open_price=100.0, close_price=100.1)  # 变化<0.3%阈值
        hist_daily = _build_daily(prev_close=100.0, today_partial=100.1)
        _patch_ticker(monkeypatch, hist_5m, hist_daily)
        _freeze(monkeypatch, 15, 30)

        r = sm.smart_money_flow("TEST")
        assert r["smf_bias"] == "neutral"

    def test_insufficient_bars_returns_not_ok(self, monkeypatch):
        hist_5m = _build_hist_5m(open_price=100.0, close_price=100.0, n_open=3, n_close=3)  # 共6根<15
        hist_daily = _build_daily(prev_close=100.0, today_partial=100.0)
        _patch_ticker(monkeypatch, hist_5m, hist_daily)
        _freeze(monkeypatch, 15, 30)

        r = sm.smart_money_flow("TEST")
        assert r["ok"] is False
