"""options_watch.py 测试（不联网）。"""
from datetime import date

import pytest

import src.options_watch as ow
from src.options_watch import evaluate, intrinsic

TR = dict(ow.SEED_TRADES[0])
Q = {"long_bid": 3.00, "long_ask": 3.20, "short_bid": 1.80, "short_ask": 1.95}   # 对手价1.05、中间价1.225


class TestEvaluate:
    def test_hold_normally(self):
        r = evaluate(TR, date(2026, 10, 1), 61.0, 1.0, Q)
        assert r["action"] == "hold" and r["mark_natural"] == 1.05 and r["mark_mid"] == pytest.approx(1.225, abs=0.01)

    def test_invalidation_needs_close_and_volume(self):
        assert evaluate(TR, date(2026, 10, 1), 64.0, 1.2, Q)["action"] == "hold"      # 没放量
        r = evaluate(TR, date(2026, 10, 1), 64.0, 1.6, Q)
        assert r["action"] == "close" and "失效条件" in r["reason"] and r["exit_value"] == 1.05

    def test_time_exit_when_profit_below_target(self):
        r = evaluate(TR, date(2026, 10, 14), 61.0, 1.0, Q)
        assert r["action"] == "close" and "时间止损" in r["reason"] and "$1.30" in r["reason"]

    def test_time_exit_skipped_when_profit_reached(self):
        good = {"long_bid": 3.60, "long_ask": 3.80, "short_bid": 2.10, "short_ask": 2.20}   # 对手价1.40≥1.30
        assert evaluate(TR, date(2026, 10, 14), 59.5, 1.0, good)["action"] == "hold"

    def test_expiry_settles_at_intrinsic(self):
        r = evaluate(TR, date(2026, 10, 16), 59.0, 1.0, None)
        assert r["action"] == "close" and r["exit_value"] == 2.0

    def test_intrinsic(self):
        assert intrinsic(TR, 61.0) == 1.0 and intrinsic(TR, 63.0) == 0.0 and intrinsic(TR, 55.0) == 2.0


def test_daily_run_closes_and_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(ow, "_FILE", str(tmp_path / "o.json"))
    msgs = ow.run_options_watch(today=date(2026, 10, 1), market=lambda tr: (61.0, 1.0, Q))
    assert msgs == [] and "持仓中" in ow.status_line() and "对手价$1.05" in ow.status_line()
    msgs = ow.run_options_watch(today=date(2026, 10, 2), market=lambda tr: (64.2, 1.8, Q))
    assert len(msgs) == 1 and "失效条件" in msgs[0] and "$+5" in msgs[0]
    assert ow._load()[0]["status"] == "closed" and "已平仓" in ow.status_line()
    # 已平仓的不再检查
    assert ow.run_options_watch(today=date(2026, 10, 3), market=lambda tr: 1 / 0) == []
