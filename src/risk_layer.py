"""
风控层（2026-09-29：九关模型降级而来）

回测证明九关的"信号类"gate（trend/rsi/volume/near_high/动能确认/打分）挑出的买点
与随机入场无差别（wiki stock-master/overview"技术面近似回测结论"），用户决定把九关
从"信号源"降级为"风控层"：不再决定买什么，只负责回答"这笔交易现在能不能做、
风险是否可控"。任何策略（事件实验室、月度动量等）开仓前都先过这一层。

保留的风控类检查（都来自cold_decision，不重复实现）：
  stop_distance     ATR止损宽度是否可控（激进模式≤12%）
  earnings_blackout 财报前7天（事件策略本身就是赌财报时，用ignore跳过）
  debt_event        近期发债/可转债公告（历史上单日-9%~-15%）
  news_event        硬性负面新闻关键词（只警示）
  pdt_rule          PDT日内交易次数
  time_window       开盘/收盘缓冲、已收盘
  vix               VIX恐慌
  macro_breadth     跨资产risk-off（只警示）
  宏观否决          FOMC/CPI/非农当天（cold_decision提前返回）

不再使用的信号类检查：trend、rsi、volume、vwap、near_high、momentum_conviction、
sector_rotation、earnings_quality、score/bonus。
"""
RISK_GATES = ("stop_distance", "earnings_blackout", "debt_event", "news_event",
              "pdt_rule", "time_window", "vix", "macro_breadth")
SIGNAL_GATES = ("trend", "rsi", "volume", "vwap", "near_high", "momentum_conviction",
                "sector_rotation", "earnings_quality")
MACRO_VETO_PREFIX = "宏观否决"


def evaluate(cold: dict, ignore: tuple = ()) -> dict:
    """
    从cold_decision的结果里只读风控类检查（纯函数）。
    ignore：本策略有意承担的风险，例如财报事件策略传("earnings_blackout",)。
    """
    vetoes, warnings, notes = [], [], {}
    if str(cold.get("reason", "")).startswith(MACRO_VETO_PREFIX):
        vetoes.append("macro")
        notes["macro"] = cold["reason"]
    gates = cold.get("gates") or {}
    for g in RISK_GATES:
        if g in ignore or g not in gates:
            continue
        p = gates[g].get("pass", True)
        if p is False:
            vetoes.append(g)
            notes[g] = gates[g].get("note", "")
        elif p == "warn":
            warnings.append(g)
            notes[g] = gates[g].get("note", "")
    return {
        "ok": not vetoes,
        "ticker": cold.get("ticker"),
        "price": cold.get("price"),
        "atr": cold.get("atr"),
        "stop_pct": cold.get("stop_pct"),
        "vetoes": vetoes,
        "warnings": warnings,
        "notes": notes,
    }


def risk_check(ticker: str, direction: str = "LONG", portfolio: float | None = None,
               ignore: tuple = ()) -> dict:
    """对单只股票跑cold_decision（激进模式，与账户配置一致），只返回风控结论。"""
    from .cold_model import cold_decision
    if portfolio is None:
        try:
            from .paper_trading import list_positions
            portfolio = list_positions("paper").get("account", {}).get("current_value") or 2000
        except Exception:
            portfolio = 2000
    return evaluate(cold_decision(ticker, portfolio=portfolio, direction=direction,
                                  aggressive_mode=True), ignore)


def format_risk(r: dict) -> str:
    head = f"{r.get('ticker')} 风控{'通过' if r['ok'] else '否决'}"
    if r.get("stop_pct") is not None:
        head += f"（1.5ATR止损约{r['stop_pct']:.1f}%）"
    lines = [head]
    for g in r["vetoes"]:
        lines.append(f"  ❌ {g}：{r['notes'].get(g, '')}")
    for g in r["warnings"]:
        lines.append(f"  ⚠️ {g}：{r['notes'].get(g, '')}")
    return "\n".join(lines)
