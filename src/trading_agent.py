"""
自动化交易 Agent

架构：
  1. 信号生成层   — 运行 cold_decision + debate，生成候选信号
  2. 风控过滤层   — PDT检查、熔断器检查、相关性检查
  3. 执行层       — 自动开模拟仓 / 发送真实仓通知
  4. 监控层       — 盯市检查止损/目标，触发自动平仓
  5. 报告层       — 每日汇总 P&L，更新 Kelly 参数

运行方式：
  python -m src.trading_agent scan --watchlist NVDA,AAPL,TSLA --account 2000
  python -m src.trading_agent monitor
  python -m src.trading_agent report

适合 $2,000 激进账户的运行参数：
  - 每个交易日10:00 AM ET执行一次扫描
  - 自动开模拟仓（paper=True），真实仓需手动确认
  - 连续亏5笔或回撤10%触发熔断，停止所有新仓
"""

import sys
import json
import time
import argparse
from datetime import datetime
import pytz

ET = pytz.timezone("America/New_York")

_DEFAULT_WATCHLIST = [
    "NVDA", "AMD", "AAPL", "MSFT", "META",
    "TSLA", "GOOGL", "AMZN", "AVGO", "ORCL",
    "AXTI", "AAOI", "COHR", "LITE", "VRT",
]


def build_dynamic_watchlist(core: list | None = None,
                             max_total: int = 20) -> dict:
    """
    构建动态 watchlist：用户固定自选股 + 当前最热板块代表股。

    返回 dict：
      tickers      — 最终扫描列表（max 20只）
      core         — 用户固定股
      sector_add   — 板块轮动动态追加的股
      sectors_used — 当前使用的热门板块列表
      note         — 说明字符串
    """
    try:
        from .sector_rotation import build_dynamic_watchlist as _sr_build
        result = _sr_build(core=core or [], max_total=max_total)
        sector_names = "、".join(s["name"] for s in result.get("sectors_used", []))
        result["note"] = (
            f"核心 {len(result['core'])} 只 + "
            f"板块轮动 {len(result['sector_add'])} 只（{sector_names}）"
            f"= 合计 {result['total']} 只"
        )
        return result
    except Exception as e:
        fallback = (core or []) + _DEFAULT_WATCHLIST
        fallback = list(dict.fromkeys(fallback))[:max_total]
        return {
            "tickers":      fallback,
            "core":         core or [],
            "sector_add":   [],
            "sectors_used": [],
            "note":         f"板块轮动加载失败（{e}），使用默认列表",
        }


# ─────────────────────────────────────────────────────────────
# 主扫描函数（核心入口）
# ─────────────────────────────────────────────────────────────

def run_scan(watchlist: list, account_value: float = 2000,
             direction: str = "LONG", auto_paper: bool = True,
             mode: str = "paper") -> dict:
    """
    运行完整扫描流程：
    1. 对 watchlist 中每只股票运行 cold_decision（激进模式）
    2. GO 信号进入辩论引擎（debate）
    3. 辩论结论 >= CONDITIONAL → 自动开模拟仓
    4. 返回完整扫描报告

    参数：
      watchlist    — 要扫描的股票列表
      account_value — 账户净值
      auto_paper   — True = GO信号自动开模拟仓
      mode         — "paper"（模拟）或 "real"（真实）
    """
    from .cold_model import cold_decision
    from .debate import generate_trade_debate
    from .paper_trading import (open_position, list_positions, mark_to_market,
                                 MAX_CONCURRENT_POSITIONS, MAX_TOTAL_EXPOSURE_PCT)
    from .pdt_guard import check_pdt_risk, get_rolling_day_trades

    now        = datetime.now(ET)
    scan_time  = now.strftime("%Y-%m-%d %H:%M ET")
    results    = []
    go_signals = []
    errors     = []

    # PDT 状态检查（从持久化日志读取真实5日滚动次数）
    pdt_used  = get_rolling_day_trades()
    pdt_check = check_pdt_risk(account_value, "margin", pdt_used)

    print(f"\n[Agent] 开始扫描 {len(watchlist)} 只股票 @ {scan_time}")
    print(f"[Agent] 账户：${account_value:,.0f} | PDT状态：{pdt_check['status']}")
    print(f"[Agent] 扫描列表：{', '.join(watchlist[:8])}{'...' if len(watchlist)>8 else ''}\n")

    for i, ticker in enumerate(watchlist[:20]):
        print(f"  [{i+1}/{min(len(watchlist),20)}] 分析 {ticker}...")
        try:
            cold = cold_decision(
                ticker,
                portfolio=account_value,
                direction=direction,
                is_intraday=False,     # 小账户用摆动模式
                aggressive_mode=True,  # $2k自动激进
            )

            row = {
                "ticker":  ticker,
                "price":   cold.get("price"),
                "verdict": cold.get("verdict"),
                "score":   cold.get("score"),
                "reason":  cold.get("reason", "")[:80],
            }

            # GO 信号进入辩论层
            if cold.get("verdict") == "GO":
                print(f"    → GO信号！进入辩论分析...")
                try:
                    debate = generate_trade_debate(
                        ticker, direction, cold, account_value
                    )
                    row["debate_conclusion"] = debate.get("verdict", {}).get("conclusion")
                    row["debate_score"]      = debate.get("verdict", {}).get("net_score")
                    row["debate_one_line"]   = debate.get("verdict", {}).get("one_line", "")[:100]
                    row["bull_conviction"]   = debate.get("bull_analyst", {}).get("conviction")
                    row["bear_severity"]     = debate.get("bear_analyst", {}).get("severity")
                    row["kelly_usd"]         = debate.get("risk_officer", {}).get(
                        "kelly_criterion", {}).get("half_kelly_usd")

                    # 保存入场区间到 row，供 Telegram 通知使用
                    ep = cold.get("entry_plan", {})
                    row["entry_price"] = ep.get("entry_price")
                    row["stop_loss"]   = ep.get("stop_loss")
                    row["target_1"]    = ep.get("target_1")
                    row["shares"]      = ep.get("shares")

                    debate_result = debate.get("verdict", {}).get("conclusion", "WAIT")
                    if debate_result in ("GO", "CONDITIONAL"):
                        go_signals.append({**row, "cold": cold, "debate": debate})
                        print(f"    → 辩论：{debate_result}，净分{row['debate_score']}")
                    else:
                        print(f"    → 辩论否决：{debate_result}")
                except Exception as de:
                    row["debate_error"] = str(de)
                    print(f"    → 辩论报错：{de}")
            else:
                print(f"    → {cold.get('verdict')} (分数{cold.get('score')})")

            results.append(row)

        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})
            print(f"    → 错误：{e}")

    # 自动开模拟仓
    auto_opened = []
    if auto_paper and go_signals:
        print(f"\n[Agent] 发现 {len(go_signals)} 个高质量信号，开模拟仓...")
        try:
            existing = list_positions(mode)
            open_list = existing.get("open", [])
            current_open_count = len(open_list)
            current_exposure   = sum(p.get("entry_price", 0) * p.get("shares", 0)
                                     for p in open_list)
        except Exception:
            current_open_count = 0
            current_exposure   = 0
        slots = max(0, MAX_CONCURRENT_POSITIONS - current_open_count)
        if slots == 0:
            print(f"[Agent] 已有{current_open_count}个持仓（上限{MAX_CONCURRENT_POSITIONS}），跳过本次开仓")
        for sig in go_signals[:slots]:  # 剩余可用仓位槽
            ep = sig["cold"].get("entry_plan", {})
            if not ep or ep.get("shares", 0) < 1:
                continue
            try:
                result = open_position(
                    ticker      = sig["ticker"],
                    shares      = ep.get("shares", 1),
                    entry_price = ep.get("entry_price", sig["price"]),
                    stop_loss   = ep.get("stop_loss", sig["price"] * 0.95),
                    target      = ep.get("target_1", sig["price"] * 1.09),
                    strategy    = f"Agent/{direction}/AggressiveSwing",
                    mode        = mode,
                    cold_result = sig.get("cold"),
                    debate_result = sig.get("debate"),
                )
                if result.get("ok"):
                    auto_opened.append({
                        "ticker":    sig["ticker"],
                        "trade_id":  result["trade_id"],
                        "exec_price": result["exec_price"],
                        "shares":    ep.get("shares"),
                        "cost":      result["total_cost"],
                    })
                    print(f"  ✅ 开仓 {sig['ticker']} × {ep.get('shares')}股 @ ${result['exec_price']:.2f}")
                else:
                    print(f"  ❌ 开仓失败 {sig['ticker']}：{result.get('error')}")
            except Exception as oe:
                print(f"  ❌ 开仓异常 {sig['ticker']}：{oe}")

    # 市值更新
    mtm = {}
    try:
        mtm = mark_to_market(mode)
    except Exception:
        pass

    scan_result = {
        "scan_time":     scan_time,
        "account_value": account_value,
        "watchlist_n":   len(watchlist),
        "scanned_n":     len(results),
        "go_signals_n":  len(go_signals),
        "auto_opened":   auto_opened,
        "results":       results,
        "go_signals":    [{k: v for k, v in s.items() if k not in ("cold", "debate")}
                          for s in go_signals],
        "pdt_status":    pdt_check.get("status"),
        "portfolio_mtm": mtm,
        "errors":        errors,
        "note": (
            f"扫描完成。{len(go_signals)}个GO信号，{len(auto_opened)}个模拟仓已开。"
            f"{'检查API连接可接入真实盘。' if mode == 'paper' else ''}"
        ),
    }

    _save_scan_log(scan_result)
    return scan_result


# ─────────────────────────────────────────────────────────────
# 监控层（止损/目标盯市）
# ─────────────────────────────────────────────────────────────

def run_monitor(mode: str = "paper", auto_stop: bool = True) -> dict:
    """
    检查所有持仓是否触及止损或目标价。

    auto_stop = True：触及止损时自动模拟平仓
    """
    from .paper_trading import mark_to_market, close_position

    mtm    = mark_to_market(mode)
    alerts = mtm.get("alerts", [])
    closed = []

    for pos in mtm.get("open_positions", []):
        alert = pos.get("alert", "")
        if auto_stop and alert and "止损" in alert:
            try:
                r = close_position(
                    pos["id"], pos["current_price"],
                    exit_reason="Agent自动止损",
                    mode=mode,
                )
                if r.get("ok"):
                    closed.append({
                        "ticker": pos["ticker"],
                        "trade_id": pos["id"],
                        "pnl": r["pnl"],
                        "reason": "自动止损",
                    })
                    print(f"[Agent] 自动止损 {pos['ticker']} | P&L: ${r['pnl']:.2f}")
            except Exception as e:
                print(f"[Agent] 止损执行失败 {pos['ticker']}: {e}")

    return {
        "monitor_time": datetime.now(ET).strftime("%Y-%m-%d %H:%M ET"),
        "total_value":  mtm.get("total_value"),
        "drawdown_pct": mtm.get("drawdown_pct"),
        "alerts":       alerts,
        "auto_closed":  closed,
        "positions_checked": len(mtm.get("open_positions", [])),
    }


# ─────────────────────────────────────────────────────────────
# 每日报告
# ─────────────────────────────────────────────────────────────

def daily_report(mode: str = "paper") -> dict:
    """生成当日交易总结报告。"""
    from .paper_trading import performance_report, list_positions, compare_paper_vs_real, mark_to_market

    perf = performance_report(mode)
    pos  = list_positions(mode)
    comp = compare_paper_vs_real()
    mtm  = mark_to_market(mode)

    today = datetime.now(ET).strftime("%Y-%m-%d")
    return {
        "report_date":          today,
        "mode":                 mode,
        "performance":          perf,
        "open_positions":       len(pos.get("open", [])),
        "open_positions_detail": mtm.get("open_positions", []),
        "closed_today":  len([
            t for t in pos.get("closed", [])
            if t.get("closed_at", "")[:10] == today
        ]),
        "comparison":    comp,
        "kelly_today":   perf.get("kelly", {}),
        "action_items": _generate_action_items(perf, pos),
    }


def _generate_action_items(perf: dict, pos: dict) -> list:
    """根据当日绩效生成具体行动建议。"""
    items = []
    risk  = perf.get("risk_metrics", {})
    summ  = perf.get("summary", {})

    if risk.get("circuit_breaker"):
        items.append("🚨 熔断器激活！今日禁止开任何新仓。先复盘最近5笔亏损原因。")

    if risk.get("consecutive_losses", 0) >= 3:
        items.append(f"⚠️ 连续亏损{risk['consecutive_losses']}笔，考虑将仓位缩减50%")

    if risk.get("sharpe_ratio", 1) < 0.5 and summ.get("total_trades", 0) >= 10:
        items.append("策略Sharpe<0.5，风险调整后收益不佳，需检查止损是否执行纪律")

    kelly = perf.get("kelly", {})
    if kelly.get("half_kelly_usd"):
        items.append(f"今日Kelly建议单仓：${kelly['half_kelly_usd']:.0f}")

    if not items:
        items.append("今日纪律执行良好，继续保持。")

    return items


# ─────────────────────────────────────────────────────────────
# 持久化扫描日志
# ─────────────────────────────────────────────────────────────

import os
_SCAN_LOG = os.path.join(os.path.dirname(__file__), "..", "data", "scan_log.json")

def _save_scan_log(result: dict):
    os.makedirs(os.path.dirname(_SCAN_LOG), exist_ok=True)
    existing = []
    if os.path.exists(_SCAN_LOG):
        try:
            with open(_SCAN_LOG, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(result)
    existing = existing[-50:]  # 保留最近50次扫描
    with open(_SCAN_LOG, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2, default=str)


# ─────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradingAgent CLI")
    parser.add_argument("command", choices=["scan", "monitor", "report", "init"])
    parser.add_argument("--watchlist", default=",".join(_DEFAULT_WATCHLIST))
    parser.add_argument("--account",   type=float, default=2000)
    parser.add_argument("--mode",      default="paper", choices=["paper", "real"])
    parser.add_argument("--direction", default="LONG",  choices=["LONG", "SHORT"])
    parser.add_argument("--no-auto",   action="store_true")
    args = parser.parse_args()

    if args.command == "init":
        from src.paper_trading import init_account
        r = init_account(args.account, args.mode)
        print(json.dumps(r, ensure_ascii=False, indent=2))

    elif args.command == "scan":
        tickers = [t.strip().upper() for t in args.watchlist.split(",") if t.strip()]
        result  = run_scan(tickers, args.account, args.direction,
                           auto_paper=not args.no_auto, mode=args.mode)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.command == "monitor":
        result = run_monitor(args.mode)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.command == "report":
        result = daily_report(args.mode)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
