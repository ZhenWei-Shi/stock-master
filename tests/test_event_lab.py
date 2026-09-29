"""
event_lab.py 测试：财报时间分类、交易日排程、跨式选取与报价、结果计算、每日任务编排（不联网）。
"""
from datetime import date

import pandas as pd
import pytest

import src.event_lab as el
from src.event_lab import classify_timing, schedule_for, pick_straddle, quote_pair, event_results


class TestTiming:
    def test_classify(self):
        assert classify_timing(pd.Timestamp("2026-09-30 16:00", tz="America/New_York")) == "AMC"
        assert classify_timing(pd.Timestamp("2026-10-13 08:00", tz="America/New_York")) == "BMO"
        # 15:00是yfinance未确认日期的占位时间
        assert classify_timing(pd.Timestamp("2026-11-17 15:00", tz="America/New_York")) == "unconfirmed"

    def test_amc_schedule(self):
        # MU 2026-09-30(周三)盘后：退出=9/29，财报后=10/1，入场=退出前5个交易日=9/22
        s = schedule_for(date(2026, 9, 30), "AMC")
        assert s == {"entry": date(2026, 9, 22), "exit": date(2026, 9, 29), "post": date(2026, 10, 1)}

    def test_bmo_schedule_post_is_same_day(self):
        s = schedule_for(date(2026, 10, 13), "BMO")
        assert s["exit"] == date(2026, 10, 12) and s["post"] == date(2026, 10, 13)

    def test_monday_event_exits_friday_and_skips_holiday(self):
        # 2026-09-07是劳动节：9/8(周二)盘前的退出日应是9/4(周五)
        s = schedule_for(date(2026, 9, 8), "BMO")
        assert s["exit"] == date(2026, 9, 4)


def _opt(sym, bid, ask, iv=0.8):
    return {"option": sym, "bid": bid, "ask": ask, "iv": iv}


CHAIN = {"current_price": 61.2, "iv30": 77.0, "options": [
    _opt("ASTS261002C00060000", 2.0, 2.2), _opt("ASTS261002P00060000", 1.0, 1.2),   # 早于min_expiry
    _opt("ASTS261016C00060000", 4.0, 4.4), _opt("ASTS261016P00060000", 3.0, 3.3),
    _opt("ASTS261016C00062000", 3.2, 3.5), _opt("ASTS261016P00062000", 4.0, 4.4),
    _opt("ASTS261016C00061000", 3.6, 3.9),                                          # 只有Call，不能组跨式
    _opt("ASTS261023C00061000", 5.0, 5.5), _opt("ASTS261023P00061000", 5.0, 5.5),
]}


class TestStraddle:
    def test_pick_nearest_expiry_after_event_and_atm_pair(self):
        p = pick_straddle(CHAIN, date(2026, 10, 3))
        # 现价61.2：61只有Call不能组跨式；60/62里62更近（0.8 vs 1.2）
        assert p["expiry"] == "2026-10-16" and p["strike"] == 62.0
        assert p["call"] == "ASTS261016C00062000" and p["put"] == "ASTS261016P00062000"

    def test_quote_pair(self):
        q = quote_pair(CHAIN, "ASTS261016C00060000", "ASTS261016P00060000")
        assert q["bid"] == 7.0 and q["ask"] == pytest.approx(7.7) and q["mid"] == pytest.approx(7.35)

    def test_missing(self):
        assert pick_straddle({"current_price": 0}, date(2026, 10, 3)) is None
        assert quote_pair(CHAIN, "X", "Y") is None


def _q(bid, ask, spot, iv=0.8):
    return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2, "spot": spot, "call_iv": iv, "put_iv": iv}


class TestResults:
    def test_all_three_hypotheses(self):
        ev = {"snapshots": {
            "entry": {"h1": _q(6.0, 6.5, 60)},
            "exit": {"h1": _q(7.0, 7.5, 61), "h23": _q(7.2, 7.6, 61, iv=1.2)},
            "post": {"h23": _q(4.0, 4.3, 64.05, iv=0.6)},
        }}
        r = event_results(ev)
        assert r["h1_ret"] == pytest.approx(7.0 / 6.5 - 1)               # 按ask买、bid卖
        assert r["implied_move"] == pytest.approx(7.4 / 61)
        assert r["realized_move"] == pytest.approx(0.05)
        assert r["h3_ret"] == pytest.approx((7.2 - 4.3) / 7.2)            # 按bid卖、ask买回
        assert r["iv_crush"] == pytest.approx(0.6)

    def test_partial(self):
        r = event_results({"snapshots": {"exit": {"h23": _q(7.2, 7.6, 61)}}})
        assert r["h1_ret"] is None and r["h3_ret"] is None and r["implied_move"] is not None


class TestDailyRun:
    def test_register_and_snapshot_by_stage(self, tmp_path, monkeypatch):
        monkeypatch.setattr(el, "_FILE", str(tmp_path / "lab.json"))
        monkeypatch.setattr(el, "default_universe", lambda wl=None: ["ASTS", "NVDA"])
        cal = {"ASTS": {"ts": "2026-09-30T16:00:00-04:00", "date": "2026-09-30", "timing": "AMC"},
               "NVDA": {"ts": "2026-11-17T15:00:00-05:00", "date": "2026-11-17", "timing": "unconfirmed"}}

        def fake_refresh(state, universe, force=False):
            state["calendar"] = cal
            return 2
        monkeypatch.setattr(el, "refresh_calendar", fake_refresh)
        chain = {**CHAIN, "options": [
            _opt("ASTS261002C00060000", 2.0, 2.2), _opt("ASTS261002P00060000", 1.0, 1.2)]}

        r = el.run_event_lab(fetch=lambda t: chain, today=date(2026, 9, 29))
        assert r["registered"] == ["ASTS_2026-09-30"]          # 未确认的NVDA不登记
        assert r["snapshots"] == ["ASTS:exit"]                   # 9/29是退出日；入场日9/22已过
        ev = el._load()["events"]["ASTS_2026-09-30"]
        assert ev["h23_contract"]["expiry"] == "2026-10-02" and ev["results"]["implied_move"] > 0

        r2 = el.run_event_lab(fetch=lambda t: chain, today=date(2026, 9, 29))
        assert r2["registered"] == [] and r2["snapshots"] == []   # 同一天不重复

        el.run_event_lab(fetch=lambda t: chain, today=date(2026, 10, 1))
        ev = el._load()["events"]["ASTS_2026-09-30"]
        assert "post" in ev["snapshots"] and ev["results"]["h3_ret"] is not None
        assert "事件实验室" in el.summarize()


class TestCancelMoved:
    def _state(self, cal_date, timing="AMC"):
        return {"calendar": {"MU": {"date": cal_date, "timing": timing, "ts": ""}},
                "events": {"MU_2026-10-08": {"ticker": "MU", "event_date": "2026-10-08",
                                              "snapshots": {"entry": {}}, "results": {"h1_ret": 0.1}}}}

    def test_future_event_with_new_date_is_cancelled(self):
        st = self._state("2026-10-15")
        assert el.cancel_moved_events(st, date(2026, 10, 1)) == ["MU_2026-10-08"]
        ev = st["events"]["MU_2026-10-08"]
        assert "2026-10-15" in ev["cancelled"] and ev["results"] == {}

    def test_same_date_or_unconfirmed_untouched(self):
        assert el.cancel_moved_events(self._state("2026-10-08"), date(2026, 10, 1)) == []
        assert el.cancel_moved_events(self._state("2026-10-15", "unconfirmed"), date(2026, 10, 1)) == []

    def test_past_event_kept_even_if_calendar_moved_on(self):
        # 事件已过（下季度日期出现在日历里），保留其部分结果
        assert el.cancel_moved_events(self._state("2027-01-10"), date(2026, 10, 20)) == []

    def test_cancelled_excluded_from_summary(self):
        st = self._state("2026-10-15")
        el.cancel_moved_events(st, date(2026, 10, 1))
        assert "H1 财报前买跨式：0笔" in el.summarize(st)


def test_run_in_subprocess_builds_command(monkeypatch):
    import subprocess
    seen = {}

    class R:
        returncode, stdout, stderr = 0, "noise\n日历刷新0只，新登记[]", ""

    def fake_run(args, **kw):
        seen["args"] = args
        return R()
    monkeypatch.setattr(subprocess, "run", fake_run)
    out = el.run_in_subprocess(["ASTS", "MU"])
    assert seen["args"][1:] == ["-m", "src.event_lab", "--watchlist", "ASTS,MU"]
    assert out.startswith("日历刷新")
