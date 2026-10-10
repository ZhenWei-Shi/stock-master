"""etf_momentum_book.py 测试（合成数据，不联网）。"""
import numpy as np
import pandas as pd
import pytest

import src.etf_momentum_book as eb


def _closes(growth: dict, days: int = 300) -> pd.DataFrame:
    idx = pd.bdate_range("2025-01-01", periods=days)
    return pd.DataFrame({t: 100 * (1 + g) ** np.arange(days) for t, g in growth.items()}, index=idx)


def test_plan_first_build_and_partial_swap():
    ranked = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1, 0.05], index=list("ABCDEF"))
    prices = {t: 10.0 for t in "ABCDEF"}
    p = eb.plan(ranked, prices, [], 2000)
    assert p["targets"] == list("ABCDE") and p["turnover"] == 1.0
    assert p["cost"] == pytest.approx(4.0)                     # 2000×2×1×10bp
    assert p["holdings"]["A"] == pytest.approx((2000 - 4) / 5 / 10)
    p2 = eb.plan(ranked, prices, list("ABCDF"), 2000)            # F掉出、E进入
    assert p2["buy"] == ["E"] and p2["sell"] == ["F"] and p2["turnover"] == pytest.approx(0.2)
    assert p2["cost"] == pytest.approx(0.8)


def test_plan_skips_missing_prices():
    ranked = pd.Series([0.5, 0.4], index=["A", "B"])
    p = eb.plan(ranked, {"B": 5.0}, [], 1000)
    assert p["targets"] == ["B"]


def test_rebalance_writes_book_and_values(tmp_path, monkeypatch):
    path = str(tmp_path / "book.json")
    monkeypatch.setattr(eb, "_momentum_book_value", lambda: 1874.55)
    g = {t: 0.0001 * i for i, t in enumerate(eb.ETF_UNIVERSE)} | {"SPY": 0.0003}
    cl = _closes(g)
    r = eb.rebalance(force=True, path=path, closes=cl)
    top5 = list(reversed(eb.ETF_UNIVERSE))[:5]
    assert r["targets"] == top5 and r["value_before"] == 2000 and r["momentum_book_value"] == 1874.55
    book = eb._load(path)
    assert set(book["holdings"]) == set(top5) and len(book["history"]) == 1
    prices = {t: float(cl[t].iloc[-1]) for t in cl}
    assert eb.book_value(book, prices) == pytest.approx(2000 - 4, abs=0.05)   # 只少了建仓成本（零股取整误差）
    # 第二次：排名不变 → 不换仓、零成本
    r2 = eb.rebalance(force=True, path=path, closes=cl)
    assert r2["buy"] == [] and r2["cost"] == 0 and "只调回等权" in eb.format_rebalance(r2)
    assert "上次调仓" in eb.status_line(path)


def test_dry_run_does_not_write(tmp_path, monkeypatch):
    path = str(tmp_path / "book.json")
    monkeypatch.setattr(eb, "_momentum_book_value", lambda: None)
    cl = _closes({t: 0.0001 * i for i, t in enumerate(eb.ETF_UNIVERSE)})
    r = eb.rebalance(dry_run=True, force=True, path=path, closes=cl)
    assert r["targets"] and eb._load(path) == {} and "空跑" in eb.format_rebalance(r)
    assert "尚未建仓" in eb.status_line(path)


def test_skips_when_not_month_end(monkeypatch):
    monkeypatch.setattr(eb, "is_last_trading_day", lambda d: False)
    assert eb.rebalance()["skipped"]
