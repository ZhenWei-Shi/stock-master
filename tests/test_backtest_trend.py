"""backtest_trend.py 纯函数测试（合成数据，不联网）。"""
import numpy as np
import pandas as pd
import pytest

from src.backtest_trend import trend_portfolio, judge, window_return

IDX = pd.date_range("2020-01-31", periods=36, freq="ME")


def _cash():
    return pd.Series(100 * 1.001 ** np.arange(36), index=IDX)


def test_uptrend_fully_invested_downtrend_in_cash():
    t = np.arange(36)
    m = pd.DataFrame({"UP": 100 * 1.02 ** t, "DOWN": 100 * 0.98 ** t}, index=IDX)
    s, bm, expo, _ = trend_portfolio(m, _cash(), "tsmom", 12, cost_bps=0)
    # UP一直持有、DOWN一直换成现金：各占一半
    assert expo == pytest.approx(0.5)
    assert s.iloc[-1] == pytest.approx(0.5 * 0.02 + 0.5 * 0.001)
    assert bm.iloc[-1] == pytest.approx((0.02 - 0.02) / 2)
    s2, _, expo2, _ = trend_portfolio(m, _cash(), "sma", 10, cost_bps=0)
    assert expo2 == pytest.approx(0.5) and s2.iloc[-1] == pytest.approx(s.iloc[-1])


def test_signal_uses_only_past_data():
    """第12个月末暴跌：当月收益算在信号之前的持仓里，下个月才换成现金。"""
    p = 100 * 1.01 ** np.arange(36)
    p[20:] = p[20:] * 0.5
    m = pd.DataFrame({"A": p}, index=IDX)
    s, _, _, _ = trend_portfolio(m, _cash(), "sma", 3, cost_bps=0)
    assert s[IDX[20]] == pytest.approx(p[20] / p[19] - 1)      # 跌的那个月仍持有，吃到跌幅
    assert s[IDX[21]] == pytest.approx(0.001)                   # 下个月已换成现金


def test_costs_and_late_listing():
    t = np.arange(36)
    m = pd.DataFrame({"A": 100 * 1.02 ** t, "B": 100 * 1.02 ** t}, index=IDX)
    m.loc[IDX[:20], "B"] = np.nan                               # B上市晚，有12个月历史前不参与
    s0, _, _, _ = trend_portfolio(m, _cash(), "tsmom", 12, cost_bps=0)
    s1, _, _, _ = trend_portfolio(m, _cash(), "tsmom", 12, cost_bps=50)
    assert s1.iloc[0] == pytest.approx(s0.iloc[0] - 0.005)     # 首月建仓100%换手
    assert s1.iloc[1] == pytest.approx(s0.iloc[1])             # 之后不换仓不扣成本
    first_b = IDX[33]                                          # B第20个有效价+12个月后出信号，赚下一月
    assert s0[first_b] == pytest.approx(0.02)


def test_judge_and_window():
    idx = pd.date_range("2010-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0.006, 0.045, 120), index=idx)
    bench.iloc[30:36] = -0.08                                   # 一段大跌
    strat = bench.copy()
    strat.iloc[31:36] = 0.001                                   # 策略第二个月起躲开
    j = judge(strat, bench)
    assert j["dd_ok"] and j["sharpe_ok"] and j["pass"]
    assert not judge(bench * 0.5 - 0.01, bench)["pass"]
    assert window_return(pd.Series([0.1, 0.1], index=idx[:2]), "2010-01", "2010-02") == pytest.approx(21.0)
