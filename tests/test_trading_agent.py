"""
trading_agent.py 回归测试

覆盖范围：_apply_conditional_sizing —— CONDITIONAL 信号减半仓位逻辑
（2026-08-13新增，修复debate.py"建议缩小仓位50%"从未真正影响下单量的问题）。

不覆盖 run_scan() 本身——联网请求的编排层，超出"直接导入生产代码测纯逻辑"
这套测试哲学的适用范围。run_monitor()只测警报过滤（2026-10-02），盯市和平仓打桩。
"""
from src.trading_agent import _apply_conditional_sizing


class TestConditionalSizing:
    def test_conditional_halves_shares(self):
        assert _apply_conditional_sizing(4, "CONDITIONAL") == 2

    def test_go_keeps_full_shares(self):
        assert _apply_conditional_sizing(4, "GO") == 4

    def test_odd_shares_floor_division(self):
        # 3股减半后floor到1股，不是1.5股
        assert _apply_conditional_sizing(3, "CONDITIONAL") == 1

    def test_single_share_conditional_rounds_to_zero(self):
        # 1股减半后是0——调用方需要据此跳过整笔交易，而不是强制开1股
        assert _apply_conditional_sizing(1, "CONDITIONAL") == 0

    def test_none_conclusion_keeps_full_shares(self):
        # debate报错/未运行时 debate_conclusion 可能是 None，不应误判成CONDITIONAL
        assert _apply_conditional_sizing(4, None) == 4

    def test_wait_conclusion_keeps_full_shares(self):
        # WAIT不会进入go_signals（run_scan里已过滤），但函数本身对未知值要保守放行
        assert _apply_conditional_sizing(4, "WAIT") == 4


def test_run_monitor_drops_alerts_of_auto_closed_positions(monkeypatch):
    """2026-10-02：AMD时间止损平仓后又推了"建议平仓"警报。已平掉的不再出现在alerts里，
    平仓失败的和止盈提醒照常保留。mark_to_market/close_position打桩，不联网不写盘。"""
    import src.paper_trading as pt
    from src.trading_agent import run_monitor

    def _p(tid, tk, alert, atype):
        return {"id": tid, "ticker": tk, "current_price": 10.0, "alert": alert, "alert_type": atype}

    opens = [_p("a", "AMD", "time-stop alert", "time_stop"),
             _p("b", "NVDA", "stop alert", "stop_loss"),
             _p("c", "MU", "target alert", "target"),
             _p("d", "XOM", "stop alert, close fails", "stop_loss"),
             _p("e", "QQQ", None, None)]
    monkeypatch.setattr(pt, "mark_to_market", lambda mode: {
        "open_positions": opens, "alerts": [p["alert"] for p in opens if p["alert"]]})

    def fake_close(tid, price, exit_reason, mode):
        if tid == "d":
            raise RuntimeError("boom")
        return {"ok": True, "pnl": 1.0}
    monkeypatch.setattr(pt, "close_position", fake_close)

    r = run_monitor("paper", auto_stop=True)
    assert [c["ticker"] for c in r["auto_closed"]] == ["AMD", "NVDA"]
    assert r["alerts"] == ["target alert", "stop alert, close fails"]
    # auto_stop=False时不平仓，警报全部保留
    r2 = run_monitor("paper", auto_stop=False)
    assert r2["auto_closed"] == [] and len(r2["alerts"]) == 4
