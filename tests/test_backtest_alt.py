"""backtest_alt.py 纯函数测试（合成数据，不联网）。"""
import numpy as np
import pandas as pd
import pytest

from src.backtest_alt import (rank_portfolio, perf_stats, excess_stats, reaction_day, event_returns,
                              adjust_for_stock_drift)


def _closes(n_days=400):
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    t = np.arange(n_days)
    # A持续上涨、B持平、C/D下跌：动量组合应一直选A
    return pd.DataFrame({"A": 100 * 1.002 ** t, "B": np.full(n_days, 100.0),
                         "C": 100 * 0.999 ** t, "D": 100 * 0.998 ** t}, index=idx)


class TestRankPortfolio:
    def test_momentum_picks_winner_and_beats_equal_weight(self):
        s, bm, to = rank_portfolio(_closes(), "ME", 252, 21, 1, cost_bps=0)
        assert len(s) > 3 and (s > bm).all() and to == pytest.approx(1 / len(s))

    def test_bottom_pick_and_costs(self):
        s0, _, _ = rank_portfolio(_closes(), "ME", 252, 21, 1, pick="bottom", cost_bps=0)
        s1, _, _ = rank_portfolio(_closes(), "ME", 252, 21, 1, pick="bottom", cost_bps=50)
        assert (s0 < 0).all() and s1.iloc[0] == pytest.approx(s0.iloc[0] - 0.01)

    def test_skips_stocks_without_history(self):
        cl = _closes()
        cl.loc[cl.index[:300], "A"] = np.nan   # A上市晚，前期不能被选
        s, bm, _ = rank_portfolio(cl, "ME", 252, 21, 1, cost_bps=0)
        assert len(s) > 0


class TestStats:
    def test_perf_and_excess(self):
        r = pd.Series([0.01, -0.02, 0.03, 0.01])
        p = perf_stats(r, 12)
        assert p["n"] == 4 and p["max_dd"] == -2.0
        e = excess_stats(r, r * 0, 12)
        assert e["ann_excess"] == pytest.approx(0.75 * 12, abs=0.01) and e["hit"] == 75.0


class TestEvents:
    IDX = pd.bdate_range("2026-09-21", periods=6)

    def test_after_close_goes_to_next_day(self):
        ts = pd.Timestamp("2026-09-22 16:00", tz="America/New_York")
        assert reaction_day(ts, self.IDX) == pd.Timestamp("2026-09-23")

    def test_before_open_same_day(self):
        ts = pd.Timestamp("2026-09-22 07:00", tz="America/New_York")
        assert reaction_day(ts, self.IDX) == pd.Timestamp("2026-09-22")

    def test_event_excess_vs_spy(self):
        cl = pd.DataFrame({"X": [100, 100, 110, 110, 121, 121]}, index=self.IDX, dtype=float)
        spy = pd.Series([100.0] * 6, index=self.IDX)
        ev = pd.DataFrame({"ticker": ["X"], "ts": [pd.Timestamp("2026-09-22 16:00", tz="America/New_York")],
                           "surprise_pct": [12.0]})
        out = event_returns(cl, spy, ev, horizons=(1, 2))
        assert out.loc[0, "car0"] == pytest.approx(0.10) and out.loc[0, "x2"] == pytest.approx(0.10)


def test_drift_adjustment_removes_steady_outperformance():
    # X每天稳定比SPY多涨1%：任何事件的"超额"都只是自身漂移，扣掉后应接近0
    idx = pd.bdate_range("2026-01-05", periods=40)
    cl = pd.DataFrame({"X": 100 * 1.01 ** np.arange(40)}, index=idx)
    spy = pd.Series(100.0, index=idx)
    ev = pd.DataFrame({"ticker": ["X"], "ts": [pd.Timestamp(idx[10]).tz_localize("America/New_York")],
                       "surprise_pct": [5.0]})
    out = adjust_for_stock_drift(event_returns(cl, spy, ev, horizons=(5,)), cl, spy, horizons=(5,))
    assert out.loc[0, "x5"] > 0.05 and abs(out.loc[0, "a5"]) < 1e-9
