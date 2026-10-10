"""options_shadow.py 测试（合成数据，不联网）。"""
from datetime import date, datetime

import numpy as np
import pytest

import src.options_auto as oa
import src.options_shadow as osh
from tests.test_options_auto import FakeClient, FakeMarket, _row

D0 = date(2026, 10, 12)          # 影子样本起始日
EXP = date(2026, 11, 16)         # 35天
SPY = [_row(598, 2.80, 2.84, -0.20, exp=EXP), _row(596, 2.30, 2.36, -0.17, exp=EXP)]
QQQ = [_row(100, 3.0, 3.1, -0.5, exp=EXP, root="QQQ", iv=0.24), _row(92, 0.80, 0.83, -0.20, exp=EXP, root="QQQ"),
       _row(90, 0.35, 0.38, -0.12, exp=EXP, root="QQQ")]
CHAINS = {"SPY": SPY, "QQQ": QQQ}
S_SHORT, S_LONG = SPY[0]["symbol"], SPY[1]["symbol"]          # 收入 2.80-2.36=0.44


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(oa, "_FILE", str(tmp_path / "auto.json"))
    monkeypatch.setattr(oa, "_SCAN_LOG", str(tmp_path / "scan.json"))
    monkeypatch.setattr(osh, "_FILE", str(tmp_path / "shadow.json"))


REC = {"opened": "2026-10-12", "expiry": "2026-11-16", "credit": 0.40, "width": 2.0}


def test_check_one_take_profit_at_resting_price():
    r = dict(REC)
    up = osh.check_one(r, 0.15, date(2026, 10, 13))
    assert up["exit_debit"] == 0.20 and up["pnl"] == 0.20 and up["reason"].startswith("止盈")


def test_check_one_stop_needs_two_checks_and_resets():
    r = dict(REC)
    assert osh.check_one(r, 1.30, date(2026, 10, 13)) is None and r["stop_breach"] is True
    assert osh.check_one(r, 0.50, date(2026, 10, 13)) is None and r["stop_breach"] is False
    assert osh.check_one(r, 1.30, date(2026, 10, 13)) is None
    assert osh.check_one(r, None, date(2026, 10, 13)) is None and r["stop_breach"] is True   # 无报价不中断
    up = osh.check_one(r, 1.25, date(2026, 10, 14))
    assert up["exit_debit"] == 1.25 and up["pnl"] == pytest.approx(-0.85)


def test_check_one_time_exit_and_same_day_hold():
    assert osh.check_one(dict(REC), 0.01, date(2026, 10, 12)) is None            # 开仓当天不平
    up = osh.check_one(dict(REC), 0.30, date(2026, 10, 26))                       # 剩21天
    assert up["reason"].startswith("时间") and up["exit_debit"] == 0.30


def test_nw_t_matches_plain_t_without_autocorrelation_and_shrinks_with_overlap():
    rng = np.random.default_rng(1)
    x = list(rng.normal(0.05, 0.3, 400))
    plain = np.mean(x) / (np.std(x) / np.sqrt(len(x)))
    assert osh.nw_t(x, 0) == pytest.approx(plain, rel=1e-6)
    y = list(np.convolve(rng.normal(0.05, 0.3, 420), np.ones(15) / 15, mode="valid")[:400])   # 重叠持有
    assert abs(osh.nw_t(y, 15)) < abs(osh.nw_t(y, 0))


def test_daily_open_ignores_buying_power_and_held(store):
    """真实仓位已占满（购买力0）时，影子仍按规则每天各开一笔。"""
    c = FakeClient(bp="0")
    r = oa.run(today=D0, client=c, market=FakeMarket(CHAINS))
    assert c.orders == []                                               # 真实：购买力不够，不开
    rows = osh._load()
    assert [(x["arm"], x["underlying"]) for x in rows] == [("control", "SPY"), ("scan", "QQQ")]
    assert rows[0]["credit"] == 0.44 and rows[0]["mid"] == pytest.approx(0.49)
    assert "影子开control SPY" in r["status"]
    oa.run(today=D0, client=c, market=FakeMarket(CHAINS))
    assert len(osh._load()) == 2                                        # 同一天不重复开


def test_no_shadow_before_start_or_high_vix(store):
    oa.run(today=date(2026, 10, 9), client=FakeClient(), market=FakeMarket(CHAINS))
    assert osh._load() == []
    oa.run(today=D0, client=FakeClient(bp="0"), market=FakeMarket(CHAINS, vix=40))
    assert osh._load() == []


def test_guard_closes_shadow_and_needs_guard(store):
    oa.run(today=D0, client=FakeClient(bp="0"), market=FakeMarket(CHAINS))
    assert oa.needs_guard(D0) is True                                   # 只有影子持仓也要起子进程
    q = {S_SHORT: {"bid": 1.0, "ask": 1.05}, S_LONG: {"bid": 0.90, "ask": 0.95}}   # 平仓成本0.15≤0.22
    r = oa.guard(client=FakeClient(), market=FakeMarket(CHAINS, q), now=oa.ET.localize(datetime(2026, 10, 13, 11, 5)),
                 sleep=lambda s: None)
    spy = [x for x in osh._load() if x["underlying"] == "SPY"][0]
    assert spy["status"] == "closed" and spy["exit_debit"] == 0.22 and spy["pnl"] == 0.22
    assert "影子持仓" in r["status"] and "平仓：SPY" in r["status"]
    s = osh.summarize()
    assert "对照SPY：已平1笔" in s and "样本未满120" in s and "扫描：已平0笔，持仓1笔" in s


def test_shadow_error_does_not_break_real_run(store, monkeypatch):
    monkeypatch.setattr(osh, "open_daily", lambda *a, **k: 1 / 0)
    c = FakeClient()
    r = oa.run(today=D0, client=c, market=FakeMarket(CHAINS))
    assert len(c.orders) == 2 and "影子样本出错" in r["status"]           # 真实两笔照常挂单


def test_calibration_compares_same_day_same_arm():
    rows = [{"opened": "2026-10-12", "arm": "control", "underlying": "SPY", "credit": 0.30}]
    real = [{"submitted": "2026-10-12", "arm": "control", "underlying": "SPY", "credit": 0.26},
            {"submitted": "2026-10-13", "arm": "control", "underlying": "SPY", "credit": 0.20}]
    assert osh.calibration(rows, real) == [-0.04]
