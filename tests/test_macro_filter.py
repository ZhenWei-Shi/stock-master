"""
macro_filter.py 宏观事件日文案误报回归测试

背景（2026-09-17发现）：get_economic_calendar() 里 FOMC/CPI/非农三个事件
各自的 action/warnings 已经按"昨日📌消化期 / 今日🚨发布 / 明日⚠️预告"三态区分好了
准确文案，但 full_macro_report()/macro_gate_check() 下游没有引用这份准确文案，
而是各自另外拼了一句固定写死"今日FOMC/CPI/非农发布，禁止开新仓"——导致 FOMC
决议公布次日（消化期，如 2026-09-16 决议后的 2026-09-17）仍被误报成"今日发布"。

修复：full_macro_report() 的 master_action 和写入快照的 calendar_warnings 字段、
macro_gate_check() 的 reason 文案，统一改成直接引用 get_economic_calendar()
返回的 warnings 原文。本文件锁定：(1) 是否触发否决(block)的逻辑本身不变；
(2) 文案必须反映真实的昨日/今日/明日状态，不能再固定写"今日"。
"""
import json
from datetime import datetime

import pytz

import src.macro_filter as mf

ET = pytz.timezone("America/New_York")


def _freeze_calendar_date(monkeypatch, year, month, day):
    fixed = ET.localize(datetime(year, month, day, 13, 0))

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(mf, "datetime", _FrozenDatetime)


class TestEconomicCalendarWording:
    def test_day_after_fomc_is_labeled_as_digestion_not_today(self, monkeypatch):
        # 2026-09-16 是真实 FOMC 决议日（_FOMC_DATES_2026），次日 09-17 应显示
        # "昨日已决议/今日消化期"，不能是"今日FOMC决议"
        _freeze_calendar_date(monkeypatch, 2026, 9, 17)
        cal = mf.get_economic_calendar()
        assert cal["high_risk_today"] is True
        joined = "；".join(cal["warnings"])
        assert "昨日FOMC已决议" in joined
        assert "今日FOMC决议" not in joined

    def test_actual_fomc_day_is_labeled_as_today(self, monkeypatch):
        _freeze_calendar_date(monkeypatch, 2026, 9, 16)
        cal = mf.get_economic_calendar()
        assert cal["high_risk_today"] is True
        joined = "；".join(cal["warnings"])
        assert "今日FOMC决议" in joined
        assert "昨日FOMC已决议" not in joined


class TestMacroGateCheckWording:
    def _write_snapshot(self, tmp_path, monkeypatch, extra):
        monkeypatch.setattr(mf, "_DATA", str(tmp_path))
        snap = {
            "generated_at": str(datetime.now(ET)),
            "top_themes": [],
            "sector_scores": {},
            "tickers_avoid": [],
            "tickers_favor": [],
            "high_risk_today": True,
            "vix_change_pct": 0,
            **extra,
        }
        with open(tmp_path / "macro_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False)

    def test_reason_uses_real_calendar_wording_not_hardcoded_today(self, tmp_path, monkeypatch):
        self._write_snapshot(tmp_path, monkeypatch, {
            "calendar_warnings": ["📌 昨日FOMC已决议（2026-09-16），今日市场消化期，谨慎建仓"],
        })
        result = mf.macro_gate_check("ASTS")
        assert result["block"] is True  # 否决逻辑本身不变
        assert "昨日FOMC已决议" in result["reason"]
        assert "今日FOMC/CPI/非农发布" not in result["reason"]

    def test_falls_back_to_generic_text_when_old_snapshot_has_no_calendar_warnings(self, tmp_path, monkeypatch):
        # 部署后到下次快照刷新前，服务器上可能还残留旧格式快照（无 calendar_warnings
        # 字段），不能报错，应平滑退回旧的通用文案
        self._write_snapshot(tmp_path, monkeypatch, {})
        result = mf.macro_gate_check("ASTS")
        assert result["block"] is True
        assert "今日FOMC/CPI/非农发布" in result["reason"]
