"""backtest_regime.py 纯函数测试（合成数据，不联网）。"""
import numpy as np
import pandas as pd
import pytest

from src.backtest_regime import apply_regime, cash_from_irx, regime_signal

P = pd.period_range("2020-01", periods=24, freq="M")


def test_regime_signal_sma_and_tsmom():
    spy = pd.Series(np.r_[np.linspace(100, 130, 14), np.linspace(125, 90, 10)], index=P)
    cash = pd.Series(0.001, index=P)
    sma = regime_signal(spy, cash, "sma", 10)
    assert sma.index[0] == P[9]                     # 有10个月数据才出信号
    assert sma[P[13]] and not sma[P[-1]]            # 上涨段开、下跌段关
    ts = regime_signal(spy, cash, "tsmom", 12)
    assert ts.index[0] == P[12] and ts[P[13]] and not ts[P[-1]]


def test_apply_regime_uses_previous_month_signal_and_costs():
    rets = pd.Series(0.05, index=P[:6])
    rets[P[3]] = -0.20
    cash = pd.Series(0.001, index=P[:6])
    sig = pd.Series([True, True, False, False, True, True], index=P[:6])
    f, b, on = apply_regime(rets, sig, cash, cost_bps=10)
    assert P[0] not in f.index                       # 第一个月没有上月信号
    assert f[P[1]] == pytest.approx(0.05)            # 1月末开 → 2月在场
    assert f[P[3]] == pytest.approx(0.001 - 0.001)   # 3月末关 → 4月赚国债、切换扣10bp，躲开-20%
    assert f[P[4]] == pytest.approx(0.001)           # 4月末关 → 5月仍在国债
    assert f[P[5]] == pytest.approx(0.05 - 0.001)    # 5月末开 → 6月回到策略，再扣一次
    assert b[P[3]] == pytest.approx(-0.20) and on == pytest.approx(3 / 5)


def test_cash_from_irx_uses_prior_month_yield():
    d = pd.Series([6.0] * 40 + [3.0] * 40, index=pd.bdate_range("2020-01-01", periods=80))
    c = cash_from_irx(d)
    assert c.iloc[0] == pytest.approx(0.005)         # 6%/12
    assert c.iloc[-1] == pytest.approx(0.0025)
