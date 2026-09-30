"""SPY卖put价差自动开仓：选合约、平仓规则、下单符号、挂单同步。全部不联网。"""
from datetime import date, datetime
from types import SimpleNamespace

import pytest

import src.options_auto as oa
from src.alpaca_options import occ_symbol

TODAY = date(2026, 9, 30)
EXP = date(2026, 11, 4)      # 35天
EXP_FAR = date(2026, 11, 13)  # 44天


def _row(k, bid, ask, delta, exp=EXP):
    return {"symbol": occ_symbol("SPY", exp, "P", k), "expiry": exp, "strike": float(k),
            "bid": bid, "ask": ask, "delta": delta}


CHAIN = [_row(600, 3.10, 3.14, -0.26), _row(598, 2.80, 2.84, -0.205), _row(597, 2.66, 2.70, -0.19),
         _row(596, 2.52, 2.56, -0.18), _row(598, 3.5, 3.6, -0.2, EXP_FAR)]


class TestPickSpread:
    def test_picks_target_expiry_and_delta_rejects_thin_credit(self):
        # 35天优先于44天；delta最接近0.20的是598；宽2收入0.24<0.40、宽1收入0.10<0.20 → 不开
        p = oa.pick_spread(CHAIN, 650, TODAY, 2000)
        assert not p["ok"] and "卖598P" in p["note"] and "宽2" in p["note"] and "宽1" in p["note"]

    def test_credit_and_loss_limits(self):
        rows = [_row(598, 2.80, 2.84, -0.20), _row(596, 2.30, 2.36, -0.17), _row(597, 2.55, 2.58, -0.18)]
        p = oa.pick_spread(rows, 650, TODAY, 2000)
        # 宽2：收入0.44≥0.40，最大亏损$156≤$160
        assert p["ok"] and p["width"] == 2.0 and p["credit"] == 0.44 and p["max_loss"] == 156
        assert p["short_sym"] == "SPY261104P00598000" and p["long_sym"] == "SPY261104P00596000"
        # 账户小了 → 宽2超8%，退到宽1：0.22≥0.20，亏损$78
        p = oa.pick_spread(rows, 650, TODAY, 1500)
        assert p["ok"] and p["width"] == 1.0 and p["credit"] == 0.22

    def test_no_fitting_width_explains(self):
        rows = [_row(598, 2.80, 2.84, -0.20), _row(596, 2.60, 2.66, -0.17)]
        p = oa.pick_spread(rows, 650, TODAY, 2000)
        assert not p["ok"] and "收入" in p["note"] and "没有597" in p["note"]

    def test_no_expiry_in_window(self):
        assert "30-45" in oa.pick_spread([_row(598, 1, 1.1, -0.2, date(2026, 10, 16))], 650, TODAY, 2000)["note"]

    def test_delta_computed_when_greeks_missing(self):
        t = 35 / 365
        price, delta = oa.bs_put(650, 624, t, 0.18)
        assert oa.implied_put_delta(price, 650, 624, t) == pytest.approx(delta, abs=1e-3)
        assert 0.15 <= abs(delta) <= 0.25
        rows = [_row(624, round(price - 0.02, 2), round(price + 0.02, 2), None),
                _row(622, 0.10, round(price - 0.50, 2), None)]
        p = oa.pick_spread(rows, 650, TODAY, 2000)
        assert p["ok"] and p["short_strike"] == 624 and p["short_delta"] == pytest.approx(delta, abs=0.01)


class TestExit:
    TR = {"expiry": "2026-11-04", "credit": 0.40, "opened": "2026-10-01"}

    def test_rules(self):
        d = lambda debit, day: oa.exit_decision(self.TR, debit, day)["action"]
        assert d(0.20, date(2026, 10, 5)) == "close"          # 止盈50%
        assert d(1.20, date(2026, 10, 5)) == "close"          # 止损3倍
        assert d(0.30, date(2026, 10, 14)) == "close"         # 剩21天
        assert d(0.30, date(2026, 10, 5)) == "hold"
        assert d(None, date(2026, 10, 5)) == "hold"

    def test_no_close_on_open_day(self):
        assert oa.exit_decision(self.TR, 0.01, date(2026, 10, 1))["action"] == "hold"

    def test_close_debit_natural(self):
        assert oa.close_debit({"bid": 1.0, "ask": 1.10}, {"bid": 0.70, "ask": 0.75}) == 0.40
        assert oa.close_debit(None, {"bid": 1}) is None


def test_parse_occ():
    assert oa.parse_occ("SPY261104P00598500") == ("SPY", EXP, "P", 598.5)


# ── 主流程（假Alpaca客户端） ──────────────────────────────────

class FakeClient:
    def __init__(self, positions=(), order_status="new", fill=None):
        self.orders, self.positions = [], list(positions)
        self.order_status, self.fill = order_status, fill

    def get_account(self):
        return SimpleNamespace(portfolio_value="2000")

    def get_all_positions(self):
        return self.positions

    def submit_order(self, req):
        self.orders.append(req)
        return SimpleNamespace(id=f"o{len(self.orders)}", status="accepted")

    def get_order_by_id(self, oid):
        return SimpleNamespace(status=self.order_status, filled_avg_price=self.fill,
                               filled_at=datetime(2026, 9, 30, 14, 31))


class FakeMarket:
    def __init__(self, rows, quotes=None, vix=18.0):
        self.rows, self.q, self.vix = rows, quotes or {}, vix

    def context(self):
        return {"spot": 650.0, "above_ma200": True, "vix": self.vix}

    def put_chain(self, spot, today):
        return self.rows

    def quotes(self, syms):
        return self.q


ROWS = [_row(598, 2.80, 2.84, -0.20), _row(596, 2.30, 2.36, -0.17)]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(oa, "_FILE", str(tmp_path / "auto.json"))


def test_open_sends_negative_credit_limit(store):
    c = FakeClient()
    r = oa.run(today=TODAY, client=c, market=FakeMarket(ROWS))
    assert len(c.orders) == 1
    o = c.orders[0]
    assert float(o.limit_price) == -0.44                       # 贷方为负
    sides = {l.symbol: (str(l.side.value), str(l.position_intent.value)) for l in o.legs}
    assert sides["SPY261104P00598000"] == ("sell", "sell_to_open")
    assert sides["SPY261104P00596000"] == ("buy", "buy_to_open")
    assert oa._load()[0]["status"] == "pending_open" and "挂开仓单" in r["msgs"][0]


def test_one_at_a_time_and_vix_guard(store):
    c = FakeClient()
    oa.run(today=TODAY, client=c, market=FakeMarket(ROWS))
    oa.run(today=TODAY, client=c, market=FakeMarket(ROWS))     # 已有挂单，不再开
    assert len(c.orders) == 1

    oa._save([])
    c2 = FakeClient()
    r = oa.run(today=TODAY, client=c2, market=FakeMarket(ROWS, vix=40))
    assert not c2.orders and "VIX" in r["status"]

    c3 = FakeClient(positions=[SimpleNamespace(asset_class="us_option")])
    r = oa.run(today=TODAY, client=c3, market=FakeMarket(ROWS))
    assert not c3.orders and "其他期权持仓" in r["status"]


def test_dry_run_writes_nothing(store):
    c = FakeClient()
    r = oa.run(dry_run=True, today=TODAY, client=c, market=FakeMarket(ROWS))
    assert not c.orders and oa._load() == [] and "[空跑]" in r["msgs"][0]


def test_fill_then_take_profit_close(store):
    c = FakeClient()
    oa.run(today=TODAY, client=c, market=FakeMarket(ROWS))
    # 次日：开仓单已成交（Alpaca成交价以负数表示贷方，取绝对值），价差跌到0.20 → 止盈
    c.order_status, c.fill = "filled", "-0.44"
    q = {"SPY261104P00598000": {"bid": 0.95, "ask": 1.00}, "SPY261104P00596000": {"bid": 0.80, "ask": 0.85}}
    r = oa.run(today=date(2026, 10, 1), client=c, market=FakeMarket(ROWS, q))
    tr = oa._load()[0]
    assert tr["credit"] == 0.44 and tr["opened"] == "2026-09-30"
    assert tr["status"] == "pending_close" and "止盈" in tr["close_reason"]
    close = c.orders[1]
    assert float(close.limit_price) == 0.20                    # 借方为正
    assert {str(l.position_intent.value) for l in close.legs} == {"buy_to_close", "sell_to_close"}
    # 平仓单成交 → 结算盈亏
    c.fill = "0.20"
    r = oa.run(today=date(2026, 10, 2), client=c, market=FakeMarket(ROWS, q), allow_open=False)
    tr = oa._load()[0]
    assert tr["status"] == "closed" and tr["pnl"] == 24.0 and any("盈亏$+24" in m for m in r["msgs"])


def test_unfilled_open_order_frees_slot(store):
    c = FakeClient()
    oa.run(today=TODAY, client=c, market=FakeMarket(ROWS))
    c.order_status = "expired"
    oa.run(today=date(2026, 10, 1), client=c, market=FakeMarket(ROWS))
    trades = oa._load()
    assert [t["status"] for t in trades] == ["not_filled", "pending_open"] and len(c.orders) == 2


def test_unconfigured_is_inert(store, monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    assert "未配置" in oa.run(today=TODAY)["status"]
