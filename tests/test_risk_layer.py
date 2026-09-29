"""
risk_layer.py 测试：只读风控类gate、忽略信号类gate、ignore参数、宏观否决（不联网）。
另测scheduler在九关降级后不再根据GO开仓/推送。
"""
import src.scheduler as sch
from src.risk_layer import evaluate, format_risk, RISK_GATES, SIGNAL_GATES


def _cold(**gates):
    base = {g: {"pass": True, "note": ""} for g in RISK_GATES + SIGNAL_GATES}
    for k, v in gates.items():
        base[k] = v
    return {"ticker": "ASTS", "price": 61.3, "atr": 3.99, "stop_pct": 9.8,
            "verdict": "ABORT", "reason": "以下检查未通过（一票否决）：trend", "gates": base}


class TestEvaluate:
    def test_signal_gate_failures_are_ignored(self):
        # 9/29 ASTS的真实状态：只有trend否决 → 在风控层里应当通过
        r = evaluate(_cold(trend={"pass": False, "note": "价格低于MA200"},
                           earnings_quality={"pass": False, "note": "CANSLIM D"}))
        assert r["ok"] is True and r["vetoes"] == [] and r["stop_pct"] == 9.8

    def test_risk_gate_failure_vetoes(self):
        r = evaluate(_cold(stop_distance={"pass": False, "note": "ATR止损14%过大"}))
        assert r["ok"] is False and r["vetoes"] == ["stop_distance"]
        assert "14%" in r["notes"]["stop_distance"]

    def test_warn_is_warning_not_veto(self):
        r = evaluate(_cold(news_event={"pass": "warn", "note": "class action"}))
        assert r["ok"] is True and r["warnings"] == ["news_event"]

    def test_ignore_for_event_strategies(self):
        cold = _cold(earnings_blackout={"pass": False, "note": "财报在3天后"})
        assert evaluate(cold)["ok"] is False
        assert evaluate(cold, ignore=("earnings_blackout",))["ok"] is True

    def test_macro_early_return(self):
        r = evaluate({"verdict": "ABORT", "reason": "宏观否决：今日FOMC决议", "score": 0, "gates": {}})
        assert r["ok"] is False and r["vetoes"] == ["macro"]

    def test_format(self):
        text = format_risk(evaluate(_cold(debt_event={"pass": False, "note": "7/15发债"})))
        assert "否决" in text and "debt_event" in text


def test_scheduler_scan_does_not_trade_or_push(monkeypatch):
    assert sch.NINE_GATE_AS_SIGNAL is False
    calls, sent = {}, []

    def fake_run_scan(**kw):
        calls.update(kw)
        return {"go_signals": [{"ticker": "NVDA", "price": 1.0}], "results": []}

    monkeypatch.setattr("src.trading_agent.build_dynamic_watchlist",
                        lambda core, max_total: {"tickers": core, "note": ""})
    monkeypatch.setattr(sch, "news_prefilter", lambda wl: {"passed": wl, "neutral": [], "blocked": []})
    monkeypatch.setattr(sch, "run_scan", fake_run_scan, raising=False)
    monkeypatch.setattr("src.trading_agent.run_scan", fake_run_scan)
    monkeypatch.setattr(sch, "send_telegram", lambda msg: sent.append(msg))
    monkeypatch.setattr("src.paper_trading.list_positions", lambda mode: {"account": {}})
    sch.full_scan_cycle(["NVDA"], 1600, use_telegram=True)
    assert calls["auto_paper"] is False and sent == []
