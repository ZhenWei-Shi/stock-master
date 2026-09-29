"""
failed_breakout.py 假突破/冲高回落识别测试（纯函数，不联网）。
"""
import numpy as np
import pandas as pd

from src.failed_breakout import find_swing_highs, descending_resistance, detect_failed_breakout


def _bars(highs, lows=None, closes=None, vols=None):
    n = len(highs)
    highs = np.array(highs, dtype=float)
    lows = np.array(lows if lows is not None else highs - 2, dtype=float)
    closes = np.array(closes if closes is not None else highs - 1, dtype=float)
    vols = np.array(vols if vols is not None else [1_000_000] * n, dtype=float)
    return pd.DataFrame({"Open": closes, "High": highs, "Low": lows, "Close": closes, "Volume": vols})


def _flat(n, level=50.0):
    return [level] * n


class TestSwingHighs:
    def test_finds_isolated_peak(self):
        h = pd.Series([1, 2, 3, 9, 3, 2, 1], dtype=float)
        assert find_swing_highs(h, 3) == [3]

    def test_tie_takes_first_only(self):
        h = pd.Series([1, 2, 3, 9, 9, 2, 1, 0], dtype=float)
        assert find_swing_highs(h, 3) == [3]

    def test_peak_needs_full_window_on_both_sides(self):
        h = pd.Series([1, 2, 3, 4, 5, 9, 1], dtype=float)
        assert find_swing_highs(h, 3) == []


class TestDescendingResistance:
    def _three_lower_peaks(self):
        h = [50.0] * 40
        for pos, val in ((5, 70), (15, 66), (25, 62)):
            h[pos] = val
        return pd.Series(h)

    def test_line_through_three_lower_highs(self):
        h = self._three_lower_peaks()
        # 三点正好共线：斜率-0.4/根，第35根外推到58
        assert abs(descending_resistance(h, 35) - 58.0) < 1e-6

    def test_ignores_swing_not_yet_confirmed(self):
        h = self._three_lower_peaks()
        # 第27根时第25根的高点右侧只有1根K线，未确认，只剩两个摆动高点
        assert descending_resistance(h, 27) is None

    def test_rising_highs_return_none(self):
        h = [50.0] * 40
        for pos, val in ((5, 60), (15, 64), (25, 68)):
            h[pos] = val
        assert descending_resistance(pd.Series(h), 35) is None


class TestDetectFailedBreakout:
    def test_spike_and_fade_triggers(self):
        highs = _flat(30) + [53.5]
        closes = _flat(30, 49.0) + [49.5]
        lows = _flat(30, 48.0) + [49.0]
        r = detect_failed_breakout(_bars(highs, lows, closes))
        assert r["triggered"] is True
        assert any("冲高回落" in s for s in r["signals"])

    def test_horizontal_pierce_and_close_back_triggers(self):
        highs = _flat(30) + [51.0]
        closes = _flat(30, 48.0) + [48.5]
        lows = _flat(30, 47.0) + [48.0]
        r = detect_failed_breakout(_bars(highs, lows, closes))
        assert r["triggered"] is True
        assert any("日高点" in s for s in r["signals"])

    def test_close_in_upper_half_not_triggered(self):
        # 刺穿后收回但收在当日区间上半部——强势，不算失败
        highs = _flat(30) + [51.0]
        closes = _flat(30, 48.0) + [49.9]
        lows = _flat(30, 47.0) + [45.0]
        r = detect_failed_breakout(_bars(highs, lows, closes))
        assert r["triggered"] is False and r["signals"]

    def test_real_breakout_close_above_level_not_triggered(self):
        highs = _flat(30) + [53.0]
        closes = _flat(30, 48.0) + [52.8]
        lows = _flat(30, 47.0) + [48.5]
        r = detect_failed_breakout(_bars(highs, lows, closes))
        assert r["triggered"] is False and r["signals"] == []

    def test_intraday_volume_ratio_is_time_scaled(self):
        highs = _flat(30) + [53.5]
        closes = _flat(30, 49.0) + [49.5]
        lows = _flat(30, 48.0) + [49.0]
        vols = [1_000_000] * 30 + [500_000]
        r = detect_failed_breakout(_bars(highs, lows, closes, vols), elapsed_frac=0.25)
        assert r["vol_ratio"] == 2.0 and "带量" in r["note"]

    def test_insufficient_data(self):
        r = detect_failed_breakout(_bars(_flat(10)))
        assert r["triggered"] is False and "数据不足" in r["note"]
