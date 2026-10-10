"""backtest_lrs.py 纯函数测试（合成数据，不联网）。"""
import numpy as np
import pandas as pd
import pytest

from src.backtest_lrs import (TRIALS, apply_signal, deflated_sharpe, ma_signal, neighbors,
                              simulate_leveraged)

D = pd.bdate_range("2020-01-01", periods=10)


def test_simulate_leveraged_costs():
    r = pd.Series(0.01, index=D)
    rf = pd.Series(0.0001, index=D)
    lr = simulate_leveraged(r, rf, 3, spread=0.0252, expense=0.0252)
    # 3×1% − 2×(0.01%+0.01%) − 0.01%
    assert lr.iloc[0] == pytest.approx(0.03 - 2 * 0.0002 - 0.0001)


def test_ma_signal_hysteresis():
    p = pd.Series([100, 100, 100, 101.2, 103, 101, 99.5, 100], index=D[:8], dtype=float)
    s = ma_signal(p, 3, 0.01)
    assert s.iloc[:2].isna().all()
    assert s.iloc[2] == False                        # 等于均线、不过缓冲 → 维持初始空仓  # noqa: E712
    assert s.iloc[3] == False                        # 101.2 > 均线100.4，但没过上沿101.40  # noqa: E712
    assert s.iloc[4] == True                         # 103 > 上沿102.41  # noqa: E712
    assert s.iloc[5] == True                         # 回落到均线下方但没跌破下沿 → 维持在场  # noqa: E712
    assert s.iloc[6] == False                        # 99.5 < 下沿100.15  # noqa: E712


def test_apply_signal_uses_previous_day_and_costs():
    lev = pd.Series([0.0, 0.03, -0.30, 0.03, 0.03], index=D[:5])
    cash = pd.Series(0.0001, index=D[:5])
    sig = pd.Series([True, False, False, True, True], index=D[:5])
    out = apply_signal(lev, cash, sig, cost_bps=10)
    assert D[0] not in out.index                     # 第一天没有前一天的信号
    assert out[D[1]] == pytest.approx(0.03)          # 第0天收盘在场 → 吃到第1天
    assert out[D[2]] == pytest.approx(0.0001 - 0.001)   # 第1天收盘出场 → 躲开-30%、扣切换成本
    assert out[D[4]] == pytest.approx(0.03 - 0.001)     # 第3天收盘回场 → 第4天在场、扣成本


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0008, 0.01, 2500))
    one = deflated_sharpe(r, 0.0, [0.08])
    many = deflated_sharpe(r, 0.0, list(rng.normal(0.03, 0.02, 68)))
    assert one > 0.99 and many < one


def test_grid_size_and_neighbors():
    assert TRIALS == 68
    assert set(neighbors(200, 0.02)) == {(m, b) for m in (150, 200, 250) for b in (0.01, 0.02, 0.03)}
