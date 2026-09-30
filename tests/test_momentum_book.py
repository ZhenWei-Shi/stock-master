"""
momentum_book.py 与相关paper_trading改动的测试（不联网）。
"""
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import src.momentum_book as mb
import src.paper_trading as pt


class TestCalendar:
    def test_last_trading_day(self):
        assert mb.is_last_trading_day(date(2026, 9, 30)) is True
        assert mb.is_last_trading_day(date(2026, 9, 29)) is False
        # 2026-10-31是周六 → 10/30(周五)是月末最后交易日
        assert mb.is_last_trading_day(date(2026, 10, 30)) is True


def _closes(n=300):
    idx = pd.bdate_range("2025-06-02", periods=n)
    t = np.arange(n)
    return pd.DataFrame({"UP": 100 * 1.003 ** t, "MID": 100 * 1.001 ** t,
                         "FLAT": np.full(n, 100.0), "DOWN": 100 * 0.998 ** t}, index=idx)


class TestRankAndPlan:
    def test_rank_skips_last_month(self):
        r = mb.rank_momentum(_closes())
        assert list(r.index) == ["UP", "MID", "FLAT", "DOWN"]
        c = _closes()
        expected = c["UP"].iloc[-1 - mb.SKIP] / c["UP"].iloc[-1 - mb.LOOKBACK] - 1
        assert r["UP"] == pytest.approx(expected)

    def test_rank_needs_history(self):
        assert mb.rank_momentum(_closes(100)).empty

    def test_plan_fractional_buys_expensive_and_skips_vetoed(self):
        ranked = pd.Series([0.9, 0.8, 0.7, 0.6], index=["BIG", "VETO", "NOPX", "A"])
        prices = {"BIG": 1069.0, "VETO": 50.0, "A": 100.0}
        plan = mb.plan_rebalance(ranked, prices, held=[], book_value=2000, risk_ok=lambda t: t != "VETO")
        # 单仓预算=2000×39%=780：零股后BIG（股价>预算）也能买，VETO被风控否决，NOPX无价格
        assert plan["targets"] == ["BIG", "A"]
        assert plan["shares"] == {"BIG": 0.7296, "A": 7.8}
        assert ("VETO", "风控否决") in plan["skipped"] and ("NOPX", "无价格") in plan["skipped"]

    def test_risk_ok_uses_own_stop_width(self):
        assert mb.risk_ok({"ok": True, "stop_pct": 13.6})      # AXTI 9/30：风控层12%会否决
        assert mb.risk_ok({"ok": True, "stop_pct": 20.0})
        assert not mb.risk_ok({"ok": True, "stop_pct": 20.5})
        assert not mb.risk_ok({"ok": True, "stop_pct": None})
        assert not mb.risk_ok({"ok": False, "stop_pct": 8.0})  # 其他风控否决照常生效
        assert "stop_distance" in mb.RISK_IGNORE

    def test_fractional_shares_never_exceed_budget(self):
        for px in (1069.0, 333.33, 7.77, 0.9999):
            n = mb.fractional_shares(780, px)
            assert n * px <= 780 and 780 - n * px < px * 1e-4 + 1e-9

    def test_plan_keeps_holdings_still_in_top(self):
        ranked = pd.Series([0.9, 0.8, 0.7], index=["A", "C", "B"])
        plan = mb.plan_rebalance(ranked, {"A": 10.0, "B": 10.0, "C": 10.0}, held=["A", "B"], book_value=2000)
        assert plan["targets"] == ["A", "C"] and plan["sell"] == ["B"] and plan["buy"] == ["C"]
        assert "A" not in plan["shares"]


class TestPaperTradingChanges:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            pt._ledger_path("typo")
        assert pt._ledger_path("momentum").endswith("momentum_trades.json")

    def test_momentum_positions_skip_time_stop(self):
        opened = datetime.now(pt.ET) - timedelta(days=25)
        pos = {"stop_loss": 80.0, "target": 10000.0, "opened_at": str(opened)}
        assert pt._check_position_alert(100.0, {**pos, "strategy": "Agent/LONG/AggressiveSwing"},
                                        datetime.now(pt.ET))[1] == "time_stop"
        assert pt._check_position_alert(100.0, {**pos, "strategy": "Momentum/Monthly"},
                                        datetime.now(pt.ET)) == (None, None)
        # 价格止损对动量仓位照常生效
        assert pt._check_position_alert(79.0, {**pos, "strategy": "Momentum/Monthly"},
                                        datetime.now(pt.ET))[1] == "stop_loss"


def test_rebalance_end_to_end_with_mocks(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "_MOM", str(tmp_path / "mom.json"))
    monkeypatch.setattr(mb, "_LOGFILE", str(tmp_path / "log.json"))
    monkeypatch.setattr(mb, "universe", lambda wl=None: ["UP", "MID", "FLAT", "DOWN"])
    closes = _closes()
    monkeypatch.setattr("yfinance.download", lambda *a, **k: pd.concat({"Close": closes}, axis=1))
    monkeypatch.setattr("src.risk_layer.risk_check", lambda t, **k: {"ok": True, "stop_pct": 8.0})

    r = mb.rebalance(force=True)
    assert r["plan"]["targets"] == ["UP", "MID"]
    book = pt.list_positions("momentum")
    assert sorted(p["ticker"] for p in book["open"]) == ["MID", "UP"]
    assert all(p["strategy"] == "Momentum/Monthly" for p in book["open"])
    assert "月度动量换仓" in mb.format_rebalance(r)

    # 非月末且不强制：不动
    monkeypatch.setattr(mb, "is_last_trading_day", lambda d: False)
    assert "skipped" in mb.rebalance()


def test_two_positions_at_weight_limit_fit_exposure_cap(tmp_path, monkeypatch):
    # 两只都刚好用满39%预算，加0.05%滑点后总仓位仍须低于80%上限
    monkeypatch.setattr(pt, "_MOM", str(tmp_path / "mom.json"))
    pt.init_account(2000, mode="momentum")
    for t in ("A", "B"):
        r = pt.open_position(t, 10, 78.0, stop_loss=62.4, target=7800, strategy="Momentum/Monthly", mode="momentum")
        assert r["ok"], r


def test_fractional_positions_at_weight_limit_fit_exposure_cap(tmp_path, monkeypatch):
    # 零股：高价股按金额买满39%，两只加滑点后仍须低于80%上限
    monkeypatch.setattr(pt, "_MOM", str(tmp_path / "mom.json"))
    pt.init_account(2000, mode="momentum")
    for t, px in (("MU", 1069.0), ("LITE", 333.33)):
        n = mb.fractional_shares(2000 * mb.WEIGHT, px)
        r = pt.open_position(t, n, px, stop_loss=round(px * 0.8, 2), target=px * 100,
                             strategy="Momentum/Monthly", mode="momentum")
        assert r["ok"], r


# ── Alpaca镜像 ────────────────────────────────────────────────

def test_alpaca_diff():
    assert mb.alpaca_diff({"MU": 0.73, "AXTI": 20.5}, {"MU": 0.73, "MRNA": 3.0}) == (["MRNA"], {"AXTI": 20.5})
    assert mb.alpaca_diff({}, {}) == ([], {})


class _FakeAlpaca:
    def __init__(self, positions):
        from types import SimpleNamespace as NS
        self.pos = [NS(symbol=t, qty=str(q), asset_class=c) for t, q, c in positions]
        self.closed, self.orders = [], []

    def get_all_positions(self):
        return self.pos

    def get_orders(self, req):
        from types import SimpleNamespace as NS
        return [NS(symbol=t) for t in getattr(self, "pending", [])]

    def close_position(self, t):
        self.closed.append(t)

    def submit_order(self, req):
        from types import SimpleNamespace as NS
        self.orders.append(req)
        return NS(status="accepted")


def test_sync_alpaca_mirrors_local_book(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "_MOM", str(tmp_path / "mom.json"))
    pt.init_account(2000, mode="momentum")
    pt.open_position("MU", 0.7296, 1069.0, stop_loss=855.2, target=106900, strategy=mb.STRATEGY, mode="momentum")
    c = _FakeAlpaca([("MRNA", 3, "us_equity"), ("SPY261104P00598000", -1, "us_option")])
    out = mb.sync_alpaca(client=c)
    assert c.closed == ["MRNA"]                                  # 期权腿不碰
    assert [(o.symbol, float(o.qty), str(o.side.value)) for o in c.orders] == [("MU", 0.7296, "buy")]
    assert len(out) == 2

    c3 = _FakeAlpaca([("MRNA", 3, "us_equity")])
    c3.pending = ["MU"]                                          # 买单还没成交 → 不重复下
    out = mb.sync_alpaca(client=c3)
    assert c3.closed == ["MRNA"] and not c3.orders and "未成交挂单" in out[0]

    c2 = _FakeAlpaca([("MRNA", 3, "us_equity")])
    assert "空跑" in mb.sync_alpaca(dry_run=True, client=c2)[0] and not c2.closed


def test_sync_alpaca_does_nothing_without_book(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "_MOM", str(tmp_path / "missing.json"))
    c = _FakeAlpaca([("MRNA", 3, "us_equity")])
    assert mb.sync_alpaca(client=c) == [] and not c.closed


def test_sync_alpaca_unconfigured(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    assert mb.sync_alpaca() == []


def test_alpaca_buy_falls_back_to_whole_shares():
    class C:
        def __init__(self):
            self.qty = []

        def submit_order(self, req):
            from types import SimpleNamespace as NS
            self.qty.append(float(req.qty))
            if float(req.qty) != int(float(req.qty)):
                raise RuntimeError("asset AXTI is not fractionable")
            return NS(status="accepted")
    c = C()
    assert "按整数股" in mb._alpaca_buy(c, "AXTI", 26.47) and c.qty == [26.47, 26.0]
    c = C()
    assert "失败" in mb._alpaca_buy(c, "BIG", 0.73) and c.qty == [0.73]
