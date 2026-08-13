"""
cold_model.py 核心决策引擎回归测试

覆盖范围：所有不依赖网络请求的纯函数逻辑，直接从生产代码导入
（而非内联副本），公式/文案被改错时测试会真正失败。

  - _calc_score               九关综合评分——全系统verdict/score的唯一出口，影响面最大
  - _volume_gate               成交量门（2026-07-28修复：区分"缩量参与不足" vs "真放量下跌"）
  - _check_macd_momentum       MACD柱动能确认
  - _check_vcp_contraction     VCP波动收缩确认
  - _check_volume_price_divergence  OBV量价背离
  - _check_pullback_setup      回调企稳确认（路径B：2026-08-13新增，MA20支撑+缩量+RSI回升）
  - _calc_atr                  ATR止损距离计算（含NaN兜底路径）
  - _rsi_series                RSI序列计算

不覆盖 cold_decision()/scan_tickers() 本身——这两个是联网请求的编排层，
不是纯函数，超出"直接导入生产代码测纯逻辑"这套测试哲学的适用范围。
"""
import numpy as np
import pandas as pd
import pytest

from src.cold_model import (
    _calc_score,
    _volume_gate,
    _check_macd_momentum,
    _check_vcp_contraction,
    _check_volume_price_divergence,
    _check_pullback_setup,
    _calc_atr,
    _rsi_series,
)
import src.cold_model as cm


# ─────────────────────────────────────────────────────────────────────────────
# _calc_score —— 九关综合评分
# ─────────────────────────────────────────────────────────────────────────────

def _all_pass_gates():
    keys = ["time_window", "trend", "rsi", "volume", "stop_distance",
            "vwap", "near_high", "macro_breadth", "news_event", "sector_rotation"]
    return {k: {"pass": True} for k in keys}


class TestCalcScore:
    def test_all_pass_standard_mode_full_score(self):
        assert _calc_score(_all_pass_gates(), vix=15.0, aggressive_mode=False) == 100

    def test_all_pass_aggressive_mode_full_score(self):
        assert _calc_score(_all_pass_gates(), vix=15.0, aggressive_mode=True) == 100

    def test_trend_fail_standard_mode_deducts_20(self):
        gates = _all_pass_gates()
        gates["trend"] = {"pass": False}
        assert _calc_score(gates, vix=15.0, aggressive_mode=False) == 80

    def test_trend_fail_aggressive_mode_deducts_25(self):
        """激进模式trend权重(25)比标准模式(20)更高——两套权重表不能混用"""
        gates = _all_pass_gates()
        gates["trend"] = {"pass": False}
        assert _calc_score(gates, vix=15.0, aggressive_mode=True) == 75

    def test_warn_deducts_half_of_fail_cost(self):
        gates = _all_pass_gates()
        gates["rsi"] = {"pass": "warn"}  # 标准模式rsi权重15，warn应扣7（整数除法）
        assert _calc_score(gates, vix=15.0, aggressive_mode=False) == 93

    def test_skip_does_not_deduct(self):
        gates = _all_pass_gates()
        gates["vwap"] = {"pass": "skip"}
        assert _calc_score(gates, vix=15.0, aggressive_mode=True) == 100

    def test_missing_gate_key_defaults_to_pass_no_penalty(self):
        gates = _all_pass_gates()
        del gates["news_event"]
        assert _calc_score(gates, vix=15.0, aggressive_mode=False) == 100

    def test_vix_panic_deducts_15_regardless_of_mode(self):
        gates = _all_pass_gates()
        assert _calc_score(gates, vix=45.0, aggressive_mode=False) == 85
        assert _calc_score(gates, vix=45.0, aggressive_mode=True) == 85

    def test_vix_elevated_deducts_more_in_standard_than_aggressive(self):
        gates = _all_pass_gates()
        std_score = _calc_score(gates, vix=30.0, aggressive_mode=False)
        agg_score = _calc_score(gates, vix=30.0, aggressive_mode=True)
        assert std_score == 90   # -10
        assert agg_score == 95   # -5

    def test_sector_rotation_warn_deducts_8(self):
        gates = _all_pass_gates()
        gates["sector_rotation"] = {"pass": "warn"}
        assert _calc_score(gates, vix=15.0, aggressive_mode=False) == 92

    def test_score_floor_clamped_at_zero(self):
        gates = {k: {"pass": False} for k in _all_pass_gates()}
        score = _calc_score(gates, vix=45.0, aggressive_mode=False)
        assert score == 0

    def test_score_never_exceeds_100(self):
        gates = _all_pass_gates()
        assert _calc_score(gates, vix=5.0, aggressive_mode=False) <= 100


# ─────────────────────────────────────────────────────────────────────────────
# _volume_gate —— 2026-07-28 修复回归保护：
# 此前无论"缩量参与不足(vol_ratio<0.8)"还是"真放量下跌"，未通过时note都固定
# 显示"放量下跌，禁止做多"，会把"没量"误报成"有量在砸盘"。
# ─────────────────────────────────────────────────────────────────────────────

class TestVolumeGate:
    def test_long_healthy_volume_up_day(self):
        r = _volume_gate(vol_ratio=1.3, price_chg=2.0, direction="LONG")
        assert r["pass"] is True
        assert "放量上涨" in r["note"]

    def test_long_healthy_volume_down_day_not_selloff(self):
        r = _volume_gate(vol_ratio=1.3, price_chg=-0.5, direction="LONG")
        assert r["pass"] is True
        assert "量比正常" in r["note"]

    def test_long_low_participation_is_not_labeled_heavy_selloff(self):
        """回归核心：量比0.5x（<0.8门槛未通过），但不是真放量下跌"""
        r = _volume_gate(vol_ratio=0.5, price_chg=-1.0, direction="LONG")
        assert r["pass"] is False
        assert "量能不足" in r["note"]
        assert "放量下跌" not in r["note"], "缩量失败不应被误报为放量下跌"

    def test_long_low_participation_even_with_big_price_drop(self):
        """价格跌幅大但量比未达2x门槛，仍是"缩量"而非"放量下跌" """
        r = _volume_gate(vol_ratio=0.6, price_chg=-3.0, direction="LONG")
        assert r["pass"] is False
        assert "量能不足" in r["note"]
        assert "放量下跌" not in r["note"]

    def test_long_true_heavy_selloff_labeled_correctly(self):
        """量比>2x 且 跌幅>2% 同时成立，才是真正的放量下跌"""
        r = _volume_gate(vol_ratio=2.5, price_chg=-3.0, direction="LONG")
        assert r["pass"] is False
        assert "放量下跌" in r["note"]
        assert "较昨收" in r["note"], "note需明确价格变化的比较基准，避免被误读成两次读数间的差值"

    def test_long_boundary_vol_ratio_exactly_point_eight_passes(self):
        r = _volume_gate(vol_ratio=0.8, price_chg=0.0, direction="LONG")
        assert r["pass"] is True

    def test_short_direction_simple_threshold(self):
        assert _volume_gate(vol_ratio=0.9, price_chg=1.0, direction="SHORT")["pass"] is True
        assert _volume_gate(vol_ratio=0.5, price_chg=1.0, direction="SHORT")["pass"] is False


# ─────────────────────────────────────────────────────────────────────────────
# _check_macd_momentum
# ─────────────────────────────────────────────────────────────────────────────

class TestMacdMomentum:
    def _patch_hist(self, monkeypatch, values):
        monkeypatch.setattr(cm, "_calc_macd_hist", lambda close: pd.Series(values))

    def test_long_positive_hist_is_healthy(self, monkeypatch):
        self._patch_hist(monkeypatch, [-0.5, -0.3, -0.1, 0.05, 0.2, 0.4])
        r = _check_macd_momentum(pd.Series(range(10)), "LONG", lookback=5)
        assert r["pass"] is True
        assert "已转正" in r["note"]

    def test_long_negative_but_improving(self, monkeypatch):
        self._patch_hist(monkeypatch, [-1.0, -0.8, -0.6, -0.5, -0.4, -0.2])
        r = _check_macd_momentum(pd.Series(range(10)), "LONG", lookback=5)
        assert r["pass"] is True
        assert "改善" in r["note"]

    def test_long_negative_and_worsening_warns(self, monkeypatch):
        self._patch_hist(monkeypatch, [-0.2, -0.3, -0.4, -0.5, -0.6, -0.8])
        r = _check_macd_momentum(pd.Series(range(10)), "LONG", lookback=5)
        assert r["pass"] == "warn"
        assert "恶化" in r["note"]

    def test_short_negative_hist_is_healthy(self, monkeypatch):
        self._patch_hist(monkeypatch, [0.5, 0.3, 0.1, -0.05, -0.2, -0.4])
        r = _check_macd_momentum(pd.Series(range(10)), "SHORT", lookback=5)
        assert r["pass"] is True
        assert "已转负" in r["note"]

    def test_short_positive_and_strengthening_warns(self, monkeypatch):
        self._patch_hist(monkeypatch, [0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
        r = _check_macd_momentum(pd.Series(range(10)), "SHORT", lookback=5)
        assert r["pass"] == "warn"
        assert "走强" in r["note"]

    def test_insufficient_data_skips(self, monkeypatch):
        self._patch_hist(monkeypatch, [0.1, 0.2])  # 长度2 < lookback(5)+1
        r = _check_macd_momentum(pd.Series(range(10)), "LONG", lookback=5)
        assert r["pass"] is True
        assert "数据不足" in r["note"]


# ─────────────────────────────────────────────────────────────────────────────
# _check_vcp_contraction
# ─────────────────────────────────────────────────────────────────────────────

def _make_vcp_hist(stage_ranges_pct, stage_vols, bars_per_stage=7):
    """构造三段振幅/成交量可控的OHLCV数据，avg_price固定100简化振幅计算。"""
    rows = []
    for rng_pct, vol in zip(stage_ranges_pct, stage_vols):
        half = rng_pct / 2
        for _ in range(bars_per_stage):
            rows.append({"High": 100 + half, "Low": 100 - half, "Close": 100.0, "Volume": vol})
    return pd.DataFrame(rows)


class TestVcpContraction:
    def test_contracted_and_volume_drying_up(self):
        hist = _make_vcp_hist(stage_ranges_pct=[20.0, 12.0, 5.0], stage_vols=[1000, 600, 200])
        r = _check_vcp_contraction(hist, lookback=21)
        assert r["contracted"] is True
        assert r["vol_drying_up"] is True

    def test_not_contracted_when_last_stage_still_wide(self):
        hist = _make_vcp_hist(stage_ranges_pct=[10.0, 10.0, 9.0], stage_vols=[500, 500, 500])
        r = _check_vcp_contraction(hist, lookback=21)
        assert r["contracted"] is False

    def test_insufficient_data_returns_false(self):
        hist = _make_vcp_hist(stage_ranges_pct=[10.0], stage_vols=[500], bars_per_stage=3)
        r = _check_vcp_contraction(hist, lookback=21)
        assert r["contracted"] is False
        assert "数据不足" in r["note"]


# ─────────────────────────────────────────────────────────────────────────────
# _check_volume_price_divergence
# ─────────────────────────────────────────────────────────────────────────────

class TestVolumePriceDivergence:
    def test_long_price_up_obv_down_is_divergence_warn(self):
        close = pd.Series([10, 11, 10, 11, 12, 13])
        vol   = pd.Series([100, 10, 100, 10, 10, 10])  # 涨日缩量，跌日放量
        r = _check_volume_price_divergence(close, vol, "LONG", lookback=5)
        assert r["pass"] == "warn"
        assert "量价背离" in r["note"]

    def test_long_price_up_obv_up_is_healthy(self):
        close = pd.Series([10, 11, 10, 11, 12, 13])
        vol   = pd.Series([10, 100, 10, 100, 100, 100])  # 涨日放量
        r = _check_volume_price_divergence(close, vol, "LONG", lookback=5)
        assert r["pass"] is True
        assert "配合" in r["note"]

    def test_insufficient_data_skips(self):
        close = pd.Series([10, 11])
        vol   = pd.Series([100, 100])
        r = _check_volume_price_divergence(close, vol, "LONG", lookback=20)
        assert r["pass"] is True
        assert "数据不足" in r["note"]


# ─────────────────────────────────────────────────────────────────────────────
# _check_pullback_setup（路径B：回调企稳确认，2026-08-13新增）
# ─────────────────────────────────────────────────────────────────────────────

def _pullback_base_series():
    """基准数据：先涨后回调，缩量+MA20支撑未破+RSI企稳回升，三项全满足。"""
    close = pd.Series([88, 90, 92, 94, 96, 98, 100, 98, 97, 96, 95])
    vol   = pd.Series([800, 850, 900, 950, 1000, 1050, 600, 500, 450, 400, 350])
    rsi_s = pd.Series([60, 58, 55, 50, 45, 38, 35, 32, 35, 38, 42])
    return close, vol, rsi_s


class TestPullbackSetup:
    def test_all_conditions_met_confirms(self):
        close, vol, rsi_s = _pullback_base_series()
        r = _check_pullback_setup(close, vol, rsi_s, ma20=92, price=95, lookback=10)
        assert r["support_held"] is True
        assert r["vol_dried_up"] is True
        assert r["rsi_rebounding"] is True
        assert r["confirmed"] is True

    def test_support_broken_blocks_confirmation(self):
        close, vol, rsi_s = _pullback_base_series()
        # 现价跌破 MA20*0.98(=90.16)，即便缩量+RSI回升也不能算企稳
        r = _check_pullback_setup(close, vol, rsi_s, ma20=92, price=88, lookback=10)
        assert r["support_held"] is False
        assert r["confirmed"] is False

    def test_volume_not_drying_up_blocks_confirmation(self):
        close, _, rsi_s = _pullback_base_series()
        # 回调段反而放量（后段均量高于前段），不是健康整理
        vol = pd.Series([400, 450, 500, 550, 600, 650, 900, 950, 1000, 1050, 1100])
        r = _check_pullback_setup(close, vol, rsi_s, ma20=92, price=95, lookback=10)
        assert r["vol_dried_up"] is False
        assert r["confirmed"] is False

    def test_rsi_still_falling_blocks_confirmation(self):
        close, vol, _ = _pullback_base_series()
        # RSI一路下探到现在，没有转头向上，不算"企稳回升"
        rsi_s = pd.Series([60, 58, 55, 50, 45, 42, 40, 38, 36, 34, 32])
        r = _check_pullback_setup(close, vol, rsi_s, ma20=92, price=95, lookback=10)
        assert r["rsi_rebounding"] is False
        assert r["confirmed"] is False

    def test_insufficient_data_returns_false(self):
        close = pd.Series([95, 96, 97])
        vol   = pd.Series([500, 500, 500])
        rsi_s = pd.Series([50, 50, 50])
        r = _check_pullback_setup(close, vol, rsi_s, ma20=92, price=97, lookback=10)
        assert r["confirmed"] is False
        assert "数据不足" in r["note"]

    def test_nan_ma20_returns_false(self):
        close, vol, rsi_s = _pullback_base_series()
        r = _check_pullback_setup(close, vol, rsi_s, ma20=float("nan"), price=95, lookback=10)
        assert r["confirmed"] is False
        assert "数据不足" in r["note"]


# ─────────────────────────────────────────────────────────────────────────────
# _calc_atr
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcAtr:
    def test_constant_true_range_returns_that_constant(self):
        n = 20
        hist = pd.DataFrame({
            "High":  [101.0] * n,
            "Low":   [99.0] * n,
            "Close": [100.0] * n,
        })
        atr = _calc_atr(hist, period=14)
        assert atr == pytest.approx(2.0, abs=0.01)

    def test_all_nan_true_range_falls_back_to_1_5pct_of_price(self):
        """TR全为NaN时（如High/Low缺失）用price*1.5%兜底，避免ATR=NaN炸掉止损计算"""
        hist = pd.DataFrame({
            "High":  [np.nan],
            "Low":   [np.nan],
            "Close": [100.0],
        })
        atr = _calc_atr(hist, period=14)
        assert atr == pytest.approx(100.0 * 0.015)


# ─────────────────────────────────────────────────────────────────────────────
# _rsi_series
# ─────────────────────────────────────────────────────────────────────────────

class TestRsiSeries:
    def test_strictly_increasing_prices_approach_100(self):
        close = pd.Series(range(1, 40))
        r = _rsi_series(close, period=14).dropna()
        assert r.iloc[-1] > 99

    def test_strictly_decreasing_prices_approach_0(self):
        close = pd.Series(range(40, 1, -1))
        r = _rsi_series(close, period=14).dropna()
        assert r.iloc[-1] < 1

    def test_bounded_between_0_and_100(self):
        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(80)))
        r = _rsi_series(close, period=14).dropna()
        assert (r >= 0).all() and (r <= 100).all()
