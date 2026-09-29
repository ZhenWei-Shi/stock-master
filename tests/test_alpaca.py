"""
Alpaca接入测试：未配置时不起作用、数据格式转换、日线兜底顺序、期权价差报价与规则检查。
全部不联网，不需要Alpaca密钥。
"""
from datetime import date

import pandas as pd
import pytest

import src.alpaca_client as ac
import src.fetcher as fetcher
from src.alpaca_options import occ_symbol, vertical_quote, check_rules, submit_vertical

COLS = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)


class TestClientUnconfigured:
    def test_inert_without_keys(self):
        assert ac.is_configured() is False
        assert ac.daily_bars("ASTS", "2026-09-01") is None
        assert ac.paper_trading_client() is None

    def test_configured_needs_both_keys(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        assert ac.is_configured() is False
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        assert ac.is_configured() is True


class TestBarsToOhlcv:
    def test_multiindex_utc_to_et_dates(self):
        idx = pd.MultiIndex.from_tuples(
            [("ASTS", pd.Timestamp("2026-09-22 04:00", tz="UTC")),
             ("ASTS", pd.Timestamp("2026-09-23 04:00", tz="UTC"))], names=["symbol", "timestamp"])
        raw = pd.DataFrame({"open": [61.91, 63.5], "high": [65.41, 63.79], "low": [60.99, 59.92],
                            "close": [63.69, 59.98], "volume": [10.48e6, 8.1e6],
                            "trade_count": [1, 2], "vwap": [1, 2]}, index=idx)
        out = ac.bars_to_ohlcv(raw)
        assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert [d.date() for d in out.index] == [date(2026, 9, 22), date(2026, 9, 23)]
        assert str(out.index.tz) == "America/New_York"
        assert out.loc[out.index[0], "Close"] == 63.69

    def test_empty(self):
        assert ac.bars_to_ohlcv(pd.DataFrame()).empty


# 2026-09-22缺失的日线场景（与test_fetcher一致）
DAYS_WITH_GAP = [d for d in pd.bdate_range("2026-09-14", "2026-09-23") if d.date() != date(2026, 9, 22)]


def _daily(days):
    idx = pd.DatetimeIndex(days).tz_localize("America/New_York")
    return pd.DataFrame({c: 1.0 for c in COLS}, index=idx)


class _FakeTicker:
    ticker = "ASTS"

    def __init__(self, daily, hourly=None):
        self.daily, self.hourly, self.calls = daily, hourly, []

    def history(self, period, interval):
        self.calls.append(interval)
        return self.daily if interval == "1d" else self.hourly


class TestDailyHistoryAlpacaFallback:
    def test_alpaca_fills_gap_before_hourly(self, monkeypatch):
        bar = pd.DataFrame({"Open": [61.91], "High": [65.41], "Low": [60.99], "Close": [63.69],
                            "Volume": [10.48e6]},
                           index=pd.DatetimeIndex([pd.Timestamp("2026-09-22", tz="America/New_York")]))
        monkeypatch.setattr(ac, "daily_bars", lambda sym, start, end=None: bar)
        tk = _FakeTicker(_daily(DAYS_WITH_GAP))
        out = fetcher.daily_history(tk, "1y")
        assert tk.calls == ["1d"]
        assert out.attrs["filled_dates"] == ["2026-09-22"] and out.attrs["filled_source"] == "alpaca_sip"
        assert out.loc[out.index.date == date(2026, 9, 22), "Volume"].iloc[0] == 10.48e6

    def test_unconfigured_falls_back_to_hourly(self):
        hourly = pd.DataFrame({"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [5.0]},
                              index=pd.DatetimeIndex([pd.Timestamp("2026-09-22 10:30", tz="America/New_York")]))
        tk = _FakeTicker(_daily(DAYS_WITH_GAP), hourly)
        out = fetcher.daily_history(tk, "1y")
        assert tk.calls == ["1d", "1h"] and out.attrs["filled_source"] == "yf_hourly"


class TestOptions:
    def test_occ_symbol(self):
        assert occ_symbol("asts", date(2026, 10, 16), "p", 62) == "ASTS261016P00062000"
        assert occ_symbol("SPY", date(2026, 10, 16), "C", 512.5) == "SPY261016C00512500"

    def test_vertical_quote_matches_options_log_trade_1(self):
        # options-log第1笔：买62P ask 4.15、卖60P bid 3.15 → 对手价1.00；中间价3.95-3.25=0.70
        chain = {"L": {"bid": 3.75, "ask": 4.15}, "S": {"bid": 3.15, "ask": 3.35}}
        q = vertical_quote(chain, "L", "S")
        assert q["natural"] == 1.00 and q["mid"] == 0.70 and q["friction_pct"] == 30.0

    def test_vertical_quote_missing_leg(self):
        q = vertical_quote({"L": {"bid": 1, "ask": 2}}, "L", "S")
        assert q["ok"] is False and "S" in q["note"]

    TODAY = date(2026, 9, 29)

    def test_rules_pass(self):
        assert check_rules(1.00, 2.0, date(2026, 10, 16), 1615.27, 0, today=self.TODAY) == []

    def test_rules_max_loss(self):
        p = check_rules(1.50, 2.0, date(2026, 10, 16), 1615.27, 0, today=self.TODAY)
        assert len(p) == 1 and "8%" in p[0]

    def test_rules_qty_scales_loss(self):
        assert check_rules(1.00, 2.0, date(2026, 10, 16), 1615.27, 0, qty=2, today=self.TODAY)

    def test_rules_dte_and_open_limit(self):
        p = check_rules(1.00, 2.0, date(2026, 10, 2), 1615.27, 1, today=self.TODAY)
        assert any("14天" in x for x in p) and any("上限1笔" in x for x in p)

    def test_rules_debit_must_be_below_width(self):
        assert any("不合理" in x for x in check_rules(2.0, 2.0, date(2026, 10, 16), 1e6, 0, today=self.TODAY))

    def test_submit_without_keys(self):
        r = submit_vertical("ASTS", date(2026, 10, 16), 62, 60, "P", 1.0, confirm=True)
        assert r["ok"] is False and "ALPACA" in r["note"]
