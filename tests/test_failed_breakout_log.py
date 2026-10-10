"""
failed_breakout_log.py 前向样本记录器测试（全部mock，不联网）。
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import src.failed_breakout_log as fbl
from src.failed_breakout_log import gamma_context, fill_forward_returns, summarize, parse_recent_8k

# 2026-09-29 ASTS收盘时的真实GEX分布（节选）
_ASTS_GEX = {
    "gex_env": "正伽马", "total_gex_m": 10.0, "gex_king": 65.0, "flip_strike": 62.5, "pc_ratio": 1.0,
    "gex_by_strike": {"60.0": -74.03, "61.0": -14.37, "62.0": -21.05, "63.0": 27.31, "64.0": 35.19, "65.0": 149.59},
}


class TestGammaContext:
    def test_asts_0929_rejected_at_65_wall(self):
        g = gamma_context(_ASTS_GEX, high=65.47, close=61.29, prev_close=61.00)
        assert g["call_wall"] == 65.0 and g["touched_wall"] and g["rejected_at_wall"]
        assert g["close_vs_flip"] == "below"

    def test_high_short_of_wall_is_not_touch(self):
        g = gamma_context(_ASTS_GEX, high=63.5, close=62.0, prev_close=61.0)
        assert g["touched_wall"] is False and g["rejected_at_wall"] is False

    def test_within_touch_tolerance_counts(self):
        g = gamma_context(_ASTS_GEX, high=64.2, close=62.0, prev_close=61.0)  # 距65约1.2%
        assert g["touched_wall"] is True

    def test_close_above_wall_is_touch_not_rejection(self):
        g = gamma_context(_ASTS_GEX, high=66.0, close=65.5, prev_close=61.0)
        assert g["touched_wall"] is True and g["rejected_at_wall"] is False

    def test_second_largest_wall_counts(self):
        # 2026-09-29 CBOE口径：45天内$70最大、$65约为其80%，冲到$65.47被压回应记为碰墙
        gex = {**_ASTS_GEX, "gex_by_strike": {"63.0": 0.21, "65.0": 1.30, "66.0": 0.34, "70.0": 1.59}}
        g = gamma_context(gex, high=65.47, close=61.29, prev_close=61.00)
        assert g["call_walls"] == [65.0, 70.0] and g["call_wall"] == 65.0 and g["rejected_at_wall"]

    def test_small_positive_strikes_are_not_walls(self):
        gex = {**_ASTS_GEX, "gex_by_strike": {"63.0": 0.21, "70.0": 1.59}}
        g = gamma_context(gex, high=63.2, close=61.5, prev_close=61.0)
        assert g["call_walls"] == [70.0] and g["touched_wall"] is False

    def test_negative_gamma_strikes_are_not_walls(self):
        gex = {**_ASTS_GEX, "gex_by_strike": {"62.0": -21.0, "63.0": -5.0}}
        g = gamma_context(gex, high=63.0, close=61.5, prev_close=61.0)
        assert g["call_wall"] is None and g["touched_wall"] is False

    def test_error_passthrough(self):
        assert gamma_context({"error": "无期权数据"}, 1, 1, 1) == {"ok": False, "note": "无期权数据"}


def _hist(dates, closes):
    idx = pd.to_datetime(dates)
    return pd.DataFrame({"Close": closes}, index=idx)


class TestFillForwardReturns:
    def test_fills_available_horizons_only(self):
        rec = {"date": "2026-09-29", "close": 100.0, "f1": None, "f3": None, "f5": None}
        h = _hist(["2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02"], [100, 98, 97, 95])
        assert fill_forward_returns(rec, h) is True
        assert rec["f1"] == -2.0 and rec["f3"] == -5.0 and rec["f5"] is None

    def test_does_not_overwrite(self):
        rec = {"date": "2026-09-29", "close": 100.0, "f1": 1.0, "f3": None, "f5": None}
        h = _hist(["2026-09-29", "2026-09-30"], [100, 98])
        assert fill_forward_returns(rec, h) is False and rec["f1"] == 1.0

    def test_missing_date(self):
        rec = {"date": "2026-09-29", "close": 100.0, "f1": None}
        assert fill_forward_returns(rec, _hist(["2026-09-30"], [1])) is False


def _rec(f3, rejected=False, has_8k=False, env="正伽马"):
    return {"f3": f3, "below_ma200": True,
            "gamma": {"ok": True, "env": env, "touched_wall": rejected, "rejected_at_wall": rejected,
                      "close_vs_flip": "below"},
            "news": {"has_8k": has_8k}}


class TestSummarize:
    def test_groups(self):
        recs = [_rec(-4.0, rejected=True, has_8k=True), _rec(2.0), _rec(None)]
        rows = {name: (n, avg, dn) for name, n, avg, dn in summarize(recs, horizon=3)}
        assert rows["全部触发"] == (2, -1.0, 50)
        assert rows["碰到伽马墙后收回"] == (1, -4.0, 100)
        assert rows["伽马墙+8-K"] == (1, -4.0, 100)
        assert rows["负伽马环境"] == (0, None, None)


class TestRun:
    def test_records_trigger_and_is_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fbl, "_LOG_FILE", str(tmp_path / "log.json"))
        monkeypatch.setattr(fbl, "RESEARCH_UNIVERSE", [])

        class _Fri(fbl.datetime):   # 固定在交易日：周末跑时最后一根K线不是"今天"，不会记录
            @classmethod
            def now(cls, tz=None):
                return fbl.ET.localize(cls(2026, 10, 9, 15, 0))
        monkeypatch.setattr(fbl, "datetime", _Fri)
        today = fbl.datetime.now(fbl.ET).strftime("%Y-%m-%d")
        idx = pd.bdate_range(end=today, periods=30)
        highs = [50.0] * 29 + [53.5]
        closes = [49.0] * 29 + [49.5]
        lows = [48.0] * 29 + [49.0]
        hist = pd.DataFrame({"Open": closes, "High": highs, "Low": lows, "Close": closes,
                             "Volume": [1e6] * 30}, index=idx)

        class _T:
            def __init__(self, t): pass
            def history(self, **kw): return hist

        monkeypatch.setattr(fbl.yf, "Ticker", _T)
        monkeypatch.setattr("src.gex_cboe.calc_gex_cboe", lambda t: {"error": "mock"})
        monkeypatch.setattr("src.gex_scanner.calc_gex", lambda t: _ASTS_GEX)
        monkeypatch.setattr(fbl, "news_context", lambda t, d: {"has_8k": True})

        r1 = fbl.run_failed_breakout_log(["asts"])
        r2 = fbl.run_failed_breakout_log(["asts"])
        assert r1["new"] == ["ASTS"] and r2["new"] == [] and r2["total"] == 1
        rec = fbl._load()[0]
        assert rec["in_watchlist"] is True and rec["news"]["has_8k"] is True
        assert rec["gamma"]["ok"] is True and rec["f1"] is None


class TestParse8K:
    # 与EDGAR submissions API的filings.recent结构一致（列式数组）
    RECENT = {
        "form": ["8-K", "10-Q", "8-K", "8-K"],
        "filingDate": ["2026-09-28", "2026-09-28", "2026-09-20", "2026-09-29"],
        "items": ["5.02,9.01", "", "2.02", "7.01,9.01"],
        "accessionNumber": ["0001493152-26-044647", "x", "y", "0001493152-26-044700"],
        "primaryDocument": ["form8-k.htm", "q.htm", "e.htm", "pr.htm"],
    }

    def test_window_labels_and_url(self):
        out = parse_recent_8k(self.RECENT, "1780312", "2026-09-29")
        assert [f["date"] for f in out] == ["2026-09-28", "2026-09-29"]
        exec_change = out[0]
        assert exec_change["labels"] == ["高管/董事变动或薪酬安排", "财务报表与附件"]
        assert exec_change["material"] is True
        assert exec_change["url"] == ("https://www.sec.gov/Archives/edgar/data/1780312/"
                                      "000149315226044647/form8-k.htm")

    def test_routine_only_is_not_material(self):
        out = parse_recent_8k(self.RECENT, "1780312", "2026-09-29")
        assert out[1]["items"] == "7.01,9.01" and out[1]["material"] is False

    def test_unknown_item_kept_as_code(self):
        recent = {"form": ["8-K"], "filingDate": ["2026-09-29"], "items": ["6.05"],
                  "accessionNumber": ["a"], "primaryDocument": ["b.htm"]}
        out = parse_recent_8k(recent, "1", "2026-09-29")
        assert out[0]["labels"] == ["6.05"] and out[0]["material"] is True
