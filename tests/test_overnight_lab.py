"""overnight_lab.py 测试（合成数据，不联网）。"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

import src.overnight_lab as ol


def _frames(n=25, today="2026-09-29"):
    idx = pd.bdate_range(end=today, periods=n)
    cols = ["HOT", "DOWNVOL", "QUIET", "X1", "X2"]
    closes = pd.DataFrame(100.0, index=idx, columns=cols)
    vols = pd.DataFrame(1e6, index=idx, columns=cols)
    opens = pd.DataFrame(100.0, index=idx, columns=cols)
    closes.iloc[-1, 0] = 103.0; vols.iloc[-1, 0] = 3e6      # 放量上涨 → 信号
    closes.iloc[-1, 1] = 97.0; vols.iloc[-1, 1] = 3e6       # 放量下跌 → 不是信号
    return closes, vols, opens


class TestDetect:
    def test_only_high_volume_up(self):
        c, v, _ = _frames()
        sig, others = ol.detect(c, v)
        assert sig == [("HOT", 3.0, 103.0)] and "HOT" not in others and "DOWNVOL" in others

    def test_threshold_is_18(self):
        c, v, _ = _frames()
        v.iloc[-1, 0] = 1.79e6
        assert ol.detect(c, v)[0] == []

    def test_short_history(self):
        c, v, _ = _frames(n=10)
        assert ol.detect(c, v) == ([], [])


def test_fill_exits_next_open_with_cost():
    idx = pd.bdate_range("2026-09-28", periods=3)
    opens = pd.DataFrame({"HOT": [100.0, 104.0, 99.0]}, index=idx)
    trades = [{"date": "2026-09-28", "ticker": "HOT", "entry": 102.0, "exit": None},
              {"date": "2026-09-30", "ticker": "HOT", "entry": 99.0, "exit": None}]
    assert ol.fill_exits(trades, opens) == 1
    assert trades[0]["exit"] == 104.0 and trades[0]["exit_date"] == "2026-09-29"
    assert trades[0]["ret"] == pytest.approx(104 / 102 - 1 - 0.001)
    assert trades[1]["exit"] is None


def test_daily_run_records_signal_and_control_then_fills(tmp_path, monkeypatch):
    monkeypatch.setattr(ol, "_FILE", str(tmp_path / "o.json"))
    monkeypatch.setattr("src.momentum_book.universe", lambda wl=None: ["HOT", "DOWNVOL", "QUIET", "X1", "X2"])
    c, v, o = _frames()
    raw = pd.concat({"Close": c, "Volume": v, "Open": o}, axis=1)
    r = ol.run_overnight_lab(download=lambda t: raw, today=date(2026, 9, 29))
    assert r["new"] == ["HOT"] and len(r["controls"]) == 1
    assert ol.run_overnight_lab(download=lambda t: raw, today=date(2026, 9, 29))["new"] == []   # 不重复

    # 第二天：填入开盘价
    c2, v2, o2 = _frames(n=26, today="2026-09-30")
    o2.iloc[-1] = 105.0
    raw2 = pd.concat({"Close": c2, "Volume": v2, "Open": o2}, axis=1)
    r2 = ol.run_overnight_lab(download=lambda t: raw2, today=date(2026, 9, 30))
    assert r2["filled"] == 2
    st = ol._load()
    hot = next(t for t in st["trades"] if t["ticker"] == "HOT" and t["date"] == "2026-09-29")
    assert hot["ret"] == pytest.approx(105 / 103 - 1 - 0.001)
    assert "H4" in ol.summarize()


def test_stale_data_does_not_record(tmp_path, monkeypatch):
    monkeypatch.setattr(ol, "_FILE", str(tmp_path / "o.json"))
    monkeypatch.setattr("src.momentum_book.universe", lambda wl=None: ["HOT"])
    c, v, o = _frames(today="2026-09-28")
    raw = pd.concat({"Close": c, "Volume": v, "Open": o}, axis=1)
    r = ol.run_overnight_lab(download=lambda t: raw, today=date(2026, 9, 29))
    assert r["new"] == [] and "不是今天" in r["note"]
