"""ETF卖put价差自动开仓：选合约、张数、扫描排序、平仓规则、下单符号、挂单同步。全部不联网。"""
from datetime import date, datetime
from types import SimpleNamespace

import pytest

import src.options_auto as oa
from src.alpaca_options import occ_symbol

TODAY = date(2026, 9, 30)
EXP = date(2026, 11, 4)      # 35天
EXP_FAR = date(2026, 11, 13)  # 44天


def _row(k, bid, ask, delta, exp=EXP, root="SPY", iv=None):
    return {"symbol": occ_symbol(root, exp, "P", k), "expiry": exp, "strike": float(k),
            "bid": bid, "ask": ask, "delta": delta, "iv": iv}


CHAIN = [_row(600, 3.10, 3.14, -0.26), _row(598, 2.80, 2.84, -0.205), _row(597, 2.66, 2.70, -0.19),
         _row(596, 2.52, 2.56, -0.18), _row(598, 3.5, 3.6, -0.2, EXP_FAR)]


class TestPickSpread:
    def test_picks_target_expiry_and_delta(self):
        # 35天优先于44天；delta最接近0.20的是598；宽2收入0.24，每张亏$176≤账户10%($200)
        p = oa.pick_spread(CHAIN, 650, TODAY, 2000)
        assert p["ok"] and p["expiry"] == EXP and p["short_strike"] == 598
        assert p["width"] == 2.0 and p["credit"] == 0.24

    def test_delta20_spread_is_feasible(self):
        # 2026-09-30 10:30实况：SPY卖738P，宽2收入0.21、宽1收入0.17。原"≥宽度20%"规则全挡，修正后可开
        # 宽2每张最大亏损$179：8%($160)放不下，10%($200)可以
        rows = [_row(738, 5.00, 5.04, -0.20), _row(736, 4.75, 4.79, -0.19), _row(737, 4.79, 4.83, -0.195)]
        p = oa.pick_spread(rows, 780, TODAY, 2000)
        assert p["ok"] and p["width"] == 2.0 and p["credit"] == 0.21 and p["max_loss"] == 179

    def test_falls_back_to_next_expiry(self):
        # 35天那期（EXP）卖出腿下方没有行权价（TLT 11/6式的断档）→ 改用44天那期
        rows = [_row(74.5, 0.61, 0.63, -0.20, root="TLT"), _row(70, 0.12, 0.13, -0.05, root="TLT"),
                _row(75, 0.80, 0.83, -0.20, EXP_FAR, "TLT"), _row(74, 0.55, 0.57, -0.15, EXP_FAR, "TLT")]
        p = oa.pick_spread(rows, 77.64, TODAY, 2000)
        assert p["ok"] and p["expiry"] == EXP_FAR and (p["short_strike"], p["long_strike"]) == (75, 74)

    def test_all_expiries_fail_lists_each(self):
        rows = [_row(74.5, 0.61, 0.63, -0.20, root="TLT"), _row(75, 0.80, 0.83, -0.20, EXP_FAR, "TLT")]
        p = oa.pick_spread(rows, 77.64, TODAY, 2000)
        assert not p["ok"] and "11-04：" in p["note"] and "11-13：" in p["note"]

    def test_uses_existing_strike_grid(self):
        # 2026-09-30 TLT实况：卖74.5P，下方只有74/73/72这类整数行权价，没有72.5/73.5
        rows = [_row(74.5, 1.00, 1.03, -0.20, root="TLT"), _row(74, 0.85, 0.88, -0.17, root="TLT"),
                _row(73, 0.60, 0.63, -0.14, root="TLT"), _row(72, 0.42, 0.45, -0.10, root="TLT")]
        p = oa.pick_spread(rows, 80, TODAY, 2000)
        assert p["ok"] and p["long_strike"] == 73 and p["width"] == 1.5 and p["credit"] == 0.37

    def test_credit_and_loss_limits(self):
        rows = [_row(598, 2.80, 2.84, -0.20), _row(596, 2.30, 2.36, -0.17), _row(597, 2.55, 2.58, -0.18)]
        p = oa.pick_spread(rows, 650, TODAY, 2000)
        # 宽2：收入0.44，最大亏损$156≤账户10%($200) → 1张
        assert p["ok"] and p["width"] == 2.0 and p["credit"] == 0.44 and p["max_loss"] == 156 and p["qty"] == 1
        assert p["short_sym"] == "SPY261104P00598000" and p["long_sym"] == "SPY261104P00596000"
        # 账户小了（10%=$150）→ 宽2一张都放不下，退到宽1：收入0.22，亏损$78
        p = oa.pick_spread(rows, 650, TODAY, 1500)
        assert p["ok"] and p["width"] == 1.0 and p["credit"] == 0.22

    def test_wide_market_rejected(self):
        # 对手价0.44，但中间价0.80 → 买卖价差太宽
        rows = [_row(598, 2.80, 3.40, -0.20), _row(596, 2.10, 2.36, -0.17)]
        p = oa.pick_spread(rows, 650, TODAY, 2000)
        assert not p["ok"] and "买卖价差太宽" in p["note"]

    def test_no_fitting_width_explains(self):
        rows = [_row(598, 2.80, 2.84, -0.20), _row(596, 2.74, 2.78, -0.17)]
        p = oa.pick_spread(rows, 650, TODAY, 2000)
        assert not p["ok"] and "收入$0.02（中间价$0.06）<$0.10" in p["note"] and "598下方没有可用行权价" in p["note"]

    def test_no_expiry_in_window(self):
        assert "30-45" in oa.pick_spread([_row(598, 1, 1.1, -0.2, date(2026, 10, 16))], 650, TODAY, 2000)["note"]

    def test_delta_computed_when_greeks_missing(self):
        t = 35 / 365
        price, delta = oa.bs_put(650, 624, t, 0.18)
        assert oa.implied_put_delta(price, 650, 624, t) == pytest.approx(delta, abs=1e-3)
        assert oa.implied_vol(price, 650, 624, t) == pytest.approx(0.18, abs=1e-3)
        assert 0.15 <= abs(delta) <= 0.25
        rows = [_row(624, round(price - 0.02, 2), round(price + 0.02, 2), None),
                _row(622, round(price - 0.54, 2), round(price - 0.50, 2), None)]
        p = oa.pick_spread(rows, 650, TODAY, 2000)
        assert p["ok"] and p["short_strike"] == 624 and p["short_delta"] == pytest.approx(delta, abs=0.01)


def test_qty_scales_with_account_and_buying_power():
    assert oa.size_qty(156, 2000, None) == 1
    assert oa.size_qty(156, 8000, None) == 5          # 账户涨了多开（10%=$800）
    assert oa.size_qty(156, 8000, 400) == 2           # 受期权购买力限制
    assert oa.size_qty(156, 1500, None) == 0
    rows = [_row(598, 2.80, 2.84, -0.20), _row(596, 2.30, 2.36, -0.17)]
    p = oa.pick_spread(rows, 650, TODAY, 8000)
    assert p["qty"] == 5 and p["max_loss"] == 780


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


def test_legacy_records_count_as_control():
    assert oa.arm_of({"underlying": "SPY"}) == "control"


# ── 主流程（假Alpaca客户端） ──────────────────────────────────

class FakeClient:
    def __init__(self, positions=(), order_status="new", fill=None, value="2000", bp=None):
        self.orders, self.positions = [], list(positions)
        self.order_status, self.fill = order_status, fill
        self.value, self.bp = value, bp

    def get_account(self):
        return SimpleNamespace(portfolio_value=self.value, options_buying_power=self.bp)

    def get_all_positions(self):
        return self.positions

    def submit_order(self, req):
        self.orders.append(req)
        return SimpleNamespace(id=f"o{len(self.orders)}", status="accepted")

    def get_order_by_id(self, oid):
        return SimpleNamespace(status=self.order_status, filled_avg_price=self.fill,
                               filled_at=datetime(2026, 9, 30, 14, 31))


class FakeMarket:
    """chains: {标的: rows}；没给的标的链为空。rv20默认0.15。"""
    def __init__(self, chains, quotes=None, vix=18.0):
        self.chains, self.q, self._vix = chains, quotes or {}, vix

    def vix(self):
        return self._vix

    def underlying(self, sym):
        return {"spot": 650.0 if sym == "SPY" else 100.0, "rv20": 0.15, "above_ma200": True}

    def put_chain(self, sym, spot, today):
        return self.chains.get(sym, [])

    def quotes(self, syms):
        return self.q


ROWS = {"SPY": [_row(598, 2.80, 2.84, -0.20), _row(596, 2.30, 2.36, -0.17)]}
# QQQ：平值(100)IV 0.24、RV 0.15 → IV/RV 1.6；IWM：IV 0.12 → 0.8（低于1不做）
QQQ = [_row(100, 3.0, 3.1, -0.5, root="QQQ", iv=0.24), _row(92, 0.80, 0.83, -0.20, root="QQQ"),
       _row(90, 0.35, 0.38, -0.12, root="QQQ")]
IWM = [_row(100, 2.0, 2.1, -0.5, root="IWM", iv=0.12), _row(92, 0.80, 0.83, -0.20, root="IWM"),
       _row(90, 0.35, 0.38, -0.12, root="IWM")]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(oa, "_FILE", str(tmp_path / "auto.json"))
    monkeypatch.setattr(oa, "_SCAN_LOG", str(tmp_path / "scan.json"))


def test_open_sends_negative_credit_limit(store):
    c = FakeClient()
    r = oa.run(today=TODAY, client=c, market=FakeMarket(ROWS))
    assert len(c.orders) == 1
    o = c.orders[0]
    assert float(o.limit_price) == -0.44 and float(o.qty) == 1   # 贷方为负
    sides = {l.symbol: (str(l.side.value), str(l.position_intent.value)) for l in o.legs}
    assert sides["SPY261104P00598000"] == ("sell", "sell_to_open")
    assert sides["SPY261104P00596000"] == ("buy", "buy_to_open")
    tr = oa._load()[0]
    assert tr["status"] == "pending_open" and tr["arm"] == "control" and "挂开仓单" in r["msgs"][0]


def test_scan_picks_highest_iv_rv_and_logs_all(store):
    c = FakeClient()
    m = FakeMarket({**ROWS, "QQQ": QQQ, "IWM": IWM})
    r = oa.run(today=TODAY, client=c, market=m)
    assert len(c.orders) == 2
    trades = oa._load()
    assert [(t["arm"], t["underlying"]) for t in trades] == [("control", "SPY"), ("scan", "QQQ")]
    assert trades[1]["context"]["iv_rv"] == 1.6
    rows = {x["underlying"]: x for x in oa._load(oa._SCAN_LOG)[0]["rows"]}
    assert rows["QQQ"]["iv_rv"] == 1.6 and rows["IWM"]["iv_rv"] == 0.8 and len(rows) == len(oa.SCAN_UNIVERSE)
    assert "QQQ 1.6" in r["status"]
    oa.run(today=TODAY, client=c, market=m)       # 两个仓位都有挂单，不再开
    assert len(c.orders) == 2


def test_scan_skips_when_options_cheap(store):
    c = FakeClient()
    r = oa.run(today=TODAY, client=c, market=FakeMarket({"IWM": IWM}))
    assert not c.orders and "扫描不开" in r["status"] and "对照SPY不开" in r["status"]


def test_guards(store):
    c2 = FakeClient()
    r = oa.run(today=TODAY, client=c2, market=FakeMarket(ROWS, vix=40))
    assert not c2.orders and "VIX" in r["status"]

    c3 = FakeClient(positions=[SimpleNamespace(asset_class="us_option", symbol="ASTS261016P00060000")])
    r = oa.run(today=TODAY, client=c3, market=FakeMarket(ROWS))
    assert not c3.orders and "不是本模块开的" in r["status"]


def test_own_positions_do_not_block(store):
    c = FakeClient()
    oa.run(today=TODAY, client=c, market=FakeMarket(ROWS))
    c.order_status, c.fill = "filled", "-0.44"
    c.positions = [SimpleNamespace(asset_class="us_option", symbol="SPY261104P00598000"),
                   SimpleNamespace(asset_class="us_option", symbol="SPY261104P00596000")]
    oa.run(today=date(2026, 10, 1), client=c, market=FakeMarket({**ROWS, "QQQ": QQQ}))
    assert [t["underlying"] for t in oa._load()] == ["SPY", "QQQ"]


def test_dry_run_writes_nothing(store):
    c = FakeClient()
    r = oa.run(dry_run=True, today=TODAY, client=c, market=FakeMarket(ROWS))
    assert not c.orders and oa._load() == [] and "[空跑]" in r["msgs"][0]
    assert oa._load(oa._SCAN_LOG) == []


def test_fill_then_take_profit_close(store):
    c = FakeClient()
    oa.run(today=TODAY, client=c, market=FakeMarket(ROWS))
    # 次日：开仓单已成交（Alpaca成交价以负数表示贷方，取绝对值），价差跌到0.20 → 止盈
    c.order_status, c.fill = "filled", "-0.44"
    q = {"SPY261104P00598000": {"bid": 0.95, "ask": 1.00}, "SPY261104P00596000": {"bid": 0.80, "ask": 0.85}}
    oa.run(today=date(2026, 10, 1), client=c, market=FakeMarket(ROWS, q), allow_open=False)
    tr = oa._load()[0]
    assert tr["credit"] == 0.44 and tr["opened"] == "2026-09-30"
    assert tr["status"] == "pending_close" and "止盈" in tr["close_reason"]
    close = c.orders[1]
    assert float(close.limit_price) == 0.20                    # 借方为正
    assert {str(l.position_intent.value) for l in close.legs} == {"buy_to_close", "sell_to_close"}
    c.fill = "0.20"
    r = oa.run(today=date(2026, 10, 2), client=c, market=FakeMarket(ROWS, q), allow_open=False)
    tr = oa._load()[0]
    assert tr["status"] == "closed" and tr["pnl"] == 24.0 and any("盈亏$+24" in m for m in r["msgs"])
    assert "已平1笔" in oa.status_line()


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


def test_weekly_summary_per_arm():
    trades = [{"id": 1, "arm": "control", "underlying": "SPY", "expiry": "2026-11-06", "short_strike": 738.0,
               "long_strike": 736.0, "qty": 1, "status": "not_filled"},
              {"id": 2, "arm": "scan", "underlying": "IWM", "expiry": "2026-11-06", "short_strike": 265.0,
               "long_strike": 263.0, "qty": 1, "credit": 0.25, "status": "open",
               "last_check": {"close_debit": 0.15}},
              {"id": 3, "arm": "scan", "underlying": "QQQ", "expiry": "2026-10-30", "short_strike": 500.0,
               "long_strike": 498.0, "qty": 1, "credit": 0.40, "status": "closed", "pnl": 20.0}]
    text = oa.weekly_summary(trades)
    assert "对照SPY：无持仓、未平过仓" in text
    assert "IWM 11-06 265/263P 收入$0.25，浮动$+10" in text and "已平1笔，胜1，累计$+20" in text


# ── 盘中改价（2026-10-01） ─────────────────────────────────────

def test_next_limit_steps_down_to_floor():
    assert oa.next_limit(0.35, 0.35) == 0.33
    assert oa.next_limit(0.26, 0.35) == 0.25        # 底价 = 0.35×70% = 0.245 → 0.25
    assert oa.next_limit(0.25, 0.35) is None
    assert oa.next_limit(0.12, 0.12) == 0.10        # 底价不低于MIN_CREDIT
    assert oa.next_limit(0.10, 0.12) is None


class RepriceClient(FakeClient):
    """按订单id记状态；cancel_after_polls次查询后撤单才生效（模拟异步撤单）。"""
    def __init__(self, cancel_to="canceled", cancel_after_polls=0, **kw):
        super().__init__(**kw)
        self.status, self.cancels = {}, []
        self.cancel_to, self.cancel_after_polls = cancel_to, cancel_after_polls

    def submit_order(self, req):
        o = super().submit_order(req)
        self.status[o.id] = "new"
        return o

    def cancel_order_by_id(self, oid):
        self.cancels.append(oid)
        self.status[oid] = "pending_cancel"
        self._polls = 0

    def get_order_by_id(self, oid):
        st = self.status.get(oid, "new")
        if st == "pending_cancel":
            self._polls += 1
            if self._polls > self.cancel_after_polls:
                st = self.status[oid] = self.cancel_to
        return SimpleNamespace(status=st, filled_avg_price=None, filled_at=None)


def _pending_spy(client):
    oa.run(today=TODAY, client=client, market=FakeMarket(ROWS))
    trades = oa._load()
    trades[0]["last_submit_at"] = "2026-09-30T10:30:00-04:00"
    oa._save(trades)
    return trades[0]


def _at(h, m):
    return oa.ET.localize(datetime(2026, 9, 30, h, m))


def test_reprice_cancels_and_resubmits_lower(store):
    c = RepriceClient(cancel_after_polls=2)
    _pending_spy(c)
    r = oa.reprice(client=c, now=_at(11, 5), sleep=lambda s: None)
    tr = oa._load()[0]
    assert c.cancels == ["o1"] and len(c.orders) == 2
    assert float(c.orders[1].limit_price) == -0.42                  # 0.44 → 0.42，仍是贷方
    assert {str(l.position_intent.value) for l in c.orders[1].legs} == {"sell_to_open", "buy_to_open"}
    assert tr["open_order_id"] == "o2" and tr["limit_credit"] == 0.42 and tr["planned_credit"] == 0.44
    assert tr["reprices"][0]["old_order"] == "o1" and "0.44→$0.42" in r["status"]


def test_reprice_waits_and_stops_at_floor(store):
    c = RepriceClient()
    _pending_spy(c)
    assert "未到改价时间" in oa.reprice(client=c, now=_at(10, 45), sleep=lambda s: None)["status"]
    assert len(c.orders) == 1
    trades = oa._load()
    trades[0]["limit_credit"] = 0.31                                 # 底价 0.44×70% = 0.31
    oa._save(trades)
    assert "已到底价" in oa.reprice(client=c, now=_at(11, 5), sleep=lambda s: None)["status"]
    assert len(c.orders) == 1 and not c.cancels


def test_reprice_does_not_resubmit_if_filled_during_cancel(store):
    c = RepriceClient(cancel_to="filled")
    _pending_spy(c)
    r = oa.reprice(client=c, now=_at(11, 5), sleep=lambda s: None)
    assert len(c.orders) == 1 and "不重挂" in r["status"]
    assert oa._load()[0]["open_order_id"] == "o1"                   # 成交留给_sync_orders记录


def test_reprice_gives_up_if_cancel_unconfirmed(store):
    c = RepriceClient(cancel_after_polls=99)
    _pending_spy(c)
    r = oa.reprice(client=c, now=_at(11, 5), sleep=lambda s: None)
    assert len(c.orders) == 1 and "pending_cancel" in r["status"]


def test_reprice_respects_max_loss_cap(store):
    c = RepriceClient(value="1580")                                  # 10% = $158，降价后每张亏$158
    _pending_spy(c)
    trades = oa._load()
    trades[0]["limit_credit"] = 0.44
    oa._save(trades)
    c.value = "1570"
    r = oa.reprice(client=c, now=_at(11, 5), sleep=lambda s: None)
    assert "不再改价" in r["status"] and len(c.orders) == 1


def test_reprice_ignores_old_days(store):
    c = RepriceClient()
    _pending_spy(c)
    r = oa.reprice(client=c, now=oa.ET.localize(datetime(2026, 10, 1, 11, 5)), sleep=lambda s: None)
    assert len(c.orders) == 1 and r["status"] == "没有需要改价的挂单"
    assert oa.has_pending_open_today(date(2026, 10, 1)) is False
