"""
paper_trading.py 核心风控/资金逻辑回归测试

这是影响面第3高的模块（见对话中的核心功能排序）：错误不只是给错建议，
而是直接影响真实/模拟账户资金。覆盖：

  - open_position()  单仓上限 / 总仓位上限 / 同标的防摊平（对应用户历史上
    "越补越深"的亏损反模式）/ 最大持仓数 / 资金不足
  - close_position()  P&L计算 / 连续亏损熔断 / 回撤熔断
  - update_trailing_stop()  止损只升不降
  - performance_report()  Kelly公式（直接测生产代码，而非test_calculations.py
    里那份"待提取"的内联副本）/ Sharpe样本量门槛

所有测试用 monkeypatch 把 paper_trading._LOG 和 feedback._FB_LOG 重定向到
pytest tmp_path，绝不触碰 data/paper_trades.json 等真实生产数据文件。
close_position() 只在"待平仓时刻仍有其他持仓"才会发起 yfinance 请求获取
实时价，本文件所有测试场景平仓时都没有其他持仓在场，因此不需要额外mock网络。
"""
import pytest

import src.paper_trading as pt
import src.feedback as fb


@pytest.fixture
def isolated_paper_log(tmp_path, monkeypatch):
    log_path = tmp_path / "paper_trades.json"
    monkeypatch.setattr(pt, "_LOG", str(log_path))
    fb_path = tmp_path / "feedback_log.json"
    monkeypatch.setattr(fb, "_FB_LOG", str(fb_path))
    return log_path


def _round_trip(ticker, entry, exit_price, shares=1):
    o = pt.open_position(ticker, shares=shares, entry_price=entry,
                          stop_loss=entry * 0.9, target=entry * 1.1, slippage_pct=0)
    assert o["ok"], o
    return pt.close_position(o["trade_id"], exit_price=exit_price, slippage_pct=0)


def _write_closed_trades(log_path, pnl_pcts):
    """绕开open/close直接写入已平仓记录，用于独立测试performance_report的统计公式。"""
    data = pt._load(str(log_path))
    for i, pct in enumerate(pnl_pcts):
        data.setdefault("trades", []).append({
            "event": "close", "id": f"t{i}", "ticker": "X",
            "pnl": pct * 10, "pnl_pct": pct, "reason": "test",
            "at": "2026-07-28T15:00:00-04:00",
        })
    pt._save(data, str(log_path))


# ─────────────────────────────────────────────────────────────────────────────
# open_position
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenPosition:
    def test_invalid_entry_price_rejected(self, isolated_paper_log):
        pt.init_account(2000, mode="paper")
        r = pt.open_position("AAPL", shares=1, entry_price=0, stop_loss=1, target=2)
        assert r["ok"] is False

    def test_invalid_shares_rejected(self, isolated_paper_log):
        pt.init_account(2000, mode="paper")
        r = pt.open_position("AAPL", shares=0, entry_price=100, stop_loss=90, target=110)
        assert r["ok"] is False

    def test_successful_open_deducts_cash_with_slippage(self, isolated_paper_log):
        pt.init_account(2000, mode="paper")
        r = pt.open_position("AAPL", shares=5, entry_price=100, stop_loss=90, target=110, slippage_pct=0)
        assert r["ok"] is True
        assert r["total_cost"] == pytest.approx(500.0)
        assert r["cash_left"] == pytest.approx(1500.0)

    def test_slippage_raises_execution_price(self, isolated_paper_log):
        pt.init_account(2000, mode="paper")
        r = pt.open_position("AAPL", shares=1, entry_price=100, stop_loss=90, target=110, slippage_pct=1.0)
        assert r["exec_price"] == pytest.approx(101.0)

    def test_exceeds_single_position_cap_rejected(self, isolated_paper_log):
        pt.init_account(2000, mode="paper")
        r = pt.open_position("AAPL", shares=15, entry_price=100, stop_loss=90, target=110, slippage_pct=0)
        assert r["ok"] is False
        assert "单仓上限" in r["error"]

    def test_duplicate_ticker_blocks_averaging_down(self, isolated_paper_log):
        """回归保护：同标的已有持仓时禁止加仓，对应用户历史"越补越深"亏损反模式"""
        pt.init_account(2000, mode="paper")
        r1 = pt.open_position("AAPL", shares=1, entry_price=100, stop_loss=90, target=110, slippage_pct=0)
        assert r1["ok"] is True
        r2 = pt.open_position("AAPL", shares=1, entry_price=90, stop_loss=80, target=100, slippage_pct=0)
        assert r2["ok"] is False
        assert "禁止同向加仓" in r2["error"]

    def test_max_concurrent_positions_enforced(self, isolated_paper_log):
        pt.init_account(2000, mode="paper")
        assert pt.open_position("AAA", shares=1, entry_price=100, stop_loss=90, target=110, slippage_pct=0)["ok"]
        assert pt.open_position("BBB", shares=1, entry_price=100, stop_loss=90, target=110, slippage_pct=0)["ok"]
        r3 = pt.open_position("CCC", shares=1, entry_price=100, stop_loss=90, target=110, slippage_pct=0)
        assert r3["ok"] is False
        assert "达到上限" in r3["error"]

    def test_total_exposure_cap_enforced(self, isolated_paper_log, monkeypatch):
        """单独放宽并发持仓数上限，隔离出"总仓位80%"这条独立风控线"""
        monkeypatch.setattr(pt, "MAX_CONCURRENT_POSITIONS", 5)
        pt.init_account(1000, mode="paper")
        assert pt.open_position("AAA", shares=3, entry_price=100, stop_loss=90, target=110, slippage_pct=0)["ok"]
        assert pt.open_position("BBB", shares=3, entry_price=100, stop_loss=90, target=110, slippage_pct=0)["ok"]
        # 已用60%，第三笔再加30% → 90%超过80%上限
        r3 = pt.open_position("CCC", shares=3, entry_price=100, stop_loss=90, target=110, slippage_pct=0)
        assert r3["ok"] is False
        assert "总仓位" in r3["error"]

    def test_insufficient_cash_rejected(self, isolated_paper_log):
        pt.init_account(1000, mode="paper")
        # 手动把现金压低到50，但current_value仍是1000（单仓/总仓位上限按current_value算，
        # 都能通过），只让"资金不足"这一条独立触发
        data = pt._load(str(isolated_paper_log))
        data["account"]["cash"] = 50.0
        pt._save(data, str(isolated_paper_log))
        r = pt.open_position("AAA", shares=3, entry_price=100, stop_loss=90, target=110, slippage_pct=0)
        assert r["ok"] is False
        assert "资金不足" in r["error"]


# ─────────────────────────────────────────────────────────────────────────────
# close_position
# ─────────────────────────────────────────────────────────────────────────────

class TestClosePosition:
    def test_pnl_calculated_correctly(self, isolated_paper_log):
        pt.init_account(2000, mode="paper")
        c = _round_trip("AAPL", entry=100, exit_price=110, shares=10)
        assert c["ok"] is True
        assert c["pnl"] == pytest.approx(100.0)   # (110-100)*10
        assert c["pnl_pct"] == pytest.approx(10.0)
        assert c["result"] == "盈利"

    def test_close_nonexistent_trade_id_fails(self, isolated_paper_log):
        pt.init_account(2000, mode="paper")
        r = pt.close_position("does-not-exist", exit_price=100)
        assert r["ok"] is False

    def test_close_already_closed_position_fails(self, isolated_paper_log):
        pt.init_account(2000, mode="paper")
        o = pt.open_position("AAPL", shares=1, entry_price=100, stop_loss=90, target=110, slippage_pct=0)
        pt.close_position(o["trade_id"], exit_price=105, slippage_pct=0)
        r2 = pt.close_position(o["trade_id"], exit_price=105, slippage_pct=0)
        assert r2["ok"] is False


class TestCircuitBreaker:
    def test_five_consecutive_losses_triggers_breaker(self, isolated_paper_log):
        pt.init_account(2000, mode="paper")
        for i in range(4):
            c = _round_trip(f"T{i}", 100, 90)
            assert c["circuit_breaker"]["active"] is False
        c5 = _round_trip("T4", 100, 90)
        assert c5["circuit_breaker"]["active"] is True
        assert c5["circuit_breaker"]["consecutive_losses"] == 5

    def test_win_resets_consecutive_loss_streak(self, isolated_paper_log):
        pt.init_account(2000, mode="paper")
        _round_trip("A", 100, 90)       # 亏损1
        _round_trip("B", 100, 90)       # 亏损2
        c = _round_trip("C", 100, 110)  # 盈利，应清零连亏计数
        assert c["circuit_breaker"]["consecutive_losses"] == 0

    def test_large_drawdown_triggers_breaker_independent_of_loss_streak(self, isolated_paper_log):
        """单笔就能把回撤打穿-6%阈值时，即使连亏计数远未到5也要触发熔断"""
        pt.init_account(2000, mode="paper")
        c = _round_trip("BIG", entry=10, exit_price=8, shares=100)  # -10%单笔回撤
        assert c["circuit_breaker"]["active"] is True
        assert "回撤" in c["circuit_breaker"]["reason"]
        assert c["circuit_breaker"]["consecutive_losses"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# update_trailing_stop
# ─────────────────────────────────────────────────────────────────────────────

class TestTrailingStop:
    def test_stop_moves_up_on_new_high(self, isolated_paper_log):
        pt.init_account(2000, mode="paper")
        o = pt.open_position("AAPL", shares=1, entry_price=100, stop_loss=92, target=120, slippage_pct=0)
        r = pt.update_trailing_stop(o["trade_id"], current_price=110, trail_pct=8.0)
        assert r["updated"] is True
        assert r["new_stop"] == pytest.approx(110 * 0.92, abs=0.01)

    def test_stop_never_moves_down_on_pullback(self, isolated_paper_log):
        pt.init_account(2000, mode="paper")
        o = pt.open_position("AAPL", shares=1, entry_price=100, stop_loss=92, target=120, slippage_pct=0)
        r1 = pt.update_trailing_stop(o["trade_id"], current_price=110, trail_pct=8.0)
        stop_after_high = r1["new_stop"]
        r2 = pt.update_trailing_stop(o["trade_id"], current_price=105, trail_pct=8.0)  # 从高点回调
        assert r2["updated"] is False
        assert r2["current_stop"] == pytest.approx(stop_after_high)


# ─────────────────────────────────────────────────────────────────────────────
# performance_report —— Kelly / Sharpe
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformanceReportKelly:
    def test_kelly_matches_formula_on_real_trade_data(self, isolated_paper_log):
        """3胜(各+10%)/2负(各-5%)：win_rate=0.6, rr=2.0 → Kelly=0.6-0.4/2=0.4"""
        pt.init_account(2000, mode="paper")
        _write_closed_trades(isolated_paper_log, [10, 10, 10, -5, -5])
        r = pt.performance_report()
        assert r["kelly"]["full_kelly_pct"] == pytest.approx(40.0, abs=0.5)
        assert r["kelly"]["half_kelly_pct"] == pytest.approx(20.0, abs=0.5)
        assert r["kelly"]["negative_edge"] is False

    def test_negative_edge_flagged_when_kelly_below_zero(self, isolated_paper_log):
        """win_rate=0.3, avg_win=5, avg_loss=-10 → rr=0.5 → Kelly=0.3-0.7/0.5=-1.1<0"""
        pt.init_account(2000, mode="paper")
        _write_closed_trades(isolated_paper_log, [5, 5, 5, -10, -10, -10, -10, -10, -10, -10])
        r = pt.performance_report()
        assert r["kelly"]["negative_edge"] is True
        assert "不应交易" in r["kelly"]["note"]

    def test_sharpe_is_zero_with_fewer_than_5_trades(self, isolated_paper_log):
        pt.init_account(2000, mode="paper")
        _write_closed_trades(isolated_paper_log, [10, -5, 8])
        r = pt.performance_report()
        assert r["risk_metrics"]["sharpe_ratio"] == 0.0
        assert r["risk_metrics"]["sortino_ratio"] == 0.0

    def test_no_trades_returns_no_data_note(self, isolated_paper_log):
        pt.init_account(2000, mode="paper")
        r = pt.performance_report()
        assert r["ok"] is True
        assert "note" in r


# ─────────────────────────────────────────────────────────────────────────────
# reset_circuit_breaker
# ─────────────────────────────────────────────────────────────────────────────

class TestResetCircuitBreaker:
    def test_requires_explicit_confirm(self, isolated_paper_log):
        pt.init_account(2000, mode="paper")
        r = pt.reset_circuit_breaker(confirm=False)
        assert r["ok"] is False

    def test_confirm_true_clears_breaker(self, isolated_paper_log):
        pt.init_account(2000, mode="paper")
        for i in range(5):
            _round_trip(f"T{i}", 100, 90)
        assert pt.performance_report()["risk_metrics"]["circuit_breaker"] is True
        r = pt.reset_circuit_breaker(confirm=True)
        assert r["ok"] is True
        assert pt.performance_report()["risk_metrics"]["circuit_breaker"] is False
