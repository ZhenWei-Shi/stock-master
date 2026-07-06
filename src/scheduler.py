"""
自动化调度器（Agent Scheduler）

运行方式：
  python scheduler.py                  # 前台运行，Ctrl+C 停止
  python scheduler.py --telegram       # 开启 Telegram 通知

交易日程（美东时间 ET）：
  09:45  开盘扫描    — 等开盘15分钟稳定后扫描，避免开盘噪音
  12:00  午间监控    — 检查止损/目标是否触发
  15:30  收盘前扫描  — 下一交易日候选名单
  16:00  每日报告    — 生成 P&L 报告，更新 Kelly 参数
  09:00  周日复盘提醒 — 提示回顾本周交易记录

数据成本：
  yfinance日线数据 — 免费，适合3-10天摆动交易
  LLM API        — 当前系统0消耗（全部确定性规则）
  服务器          — DigitalOcean/AWS t2.micro ≈ $6-8/月
"""

import time
import json
import argparse
import threading
from datetime import datetime, timedelta
import pytz
import requests
import os
import sys

# 把项目根目录加入路径（服务器上用）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading_agent import run_scan, run_monitor, daily_report
from src.paper_trading  import mark_to_market, performance_report

ET = pytz.timezone("America/New_York")

# ── 配置（从环境变量读取，安全不暴露 key）────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")

DEFAULT_WATCHLIST = [
    "NVDA", "AMD", "AAPL", "MSFT", "META",
    "TSLA", "GOOGL", "AMZN", "AVGO", "ORCL",
    "AXTI", "AAOI", "COHR", "LITE", "VRT", "CEG",
]

_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "scheduler_config.json")


# ─────────────────────────────────────────────────────────────
# Telegram 通知（免费，无需付费 API）
# ─────────────────────────────────────────────────────────────

def send_telegram(msg: str) -> bool:
    """
    发送 Telegram 消息。

    设置方法（免费）：
      1. Telegram 搜索 @BotFather，创建 bot，获取 TOKEN
      2. 给 bot 发任意消息，访问：
         https://api.telegram.org/bot<TOKEN>/getUpdates
         找到 chat.id
      3. 设置环境变量：
         TELEGRAM_BOT_TOKEN=你的token
         TELEGRAM_CHAT_ID=你的chat_id
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[Telegram] 未配置（消息：{msg[:50]}...）")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       msg,
            "parse_mode": "HTML",
        }, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"[Telegram] 发送失败：{e}")
        return False


def notify_go_signal(ticker: str, price: float, score: int,
                      verdict: str, entry_plan: dict = None):
    """发送 GO 信号通知。"""
    msg = (
        f"🚀 <b>交易信号</b>\n"
        f"股票：<b>{ticker}</b> @ ${price:.2f}\n"
        f"评分：{score}/100 | 结论：{verdict}\n"
    )
    if entry_plan:
        msg += (
            f"入场：${entry_plan.get('entry_price', price):.2f}\n"
            f"止损：${entry_plan.get('stop_loss', 0):.2f}\n"
            f"目标：${entry_plan.get('target_1', 0):.2f}\n"
            f"仓位：{entry_plan.get('shares', 0)}股 "
            f"(${entry_plan.get('position_usd', 0):.0f})\n"
            f"模式：{entry_plan.get('trade_type', '摆动')}\n"
        )
    msg += f"\n📊 /api/debate-with-cold?ticker={ticker}"
    send_telegram(msg)


def notify_stop_loss(ticker: str, pnl: float):
    """止损触发通知。"""
    emoji = "🔴" if pnl < 0 else "🟢"
    send_telegram(
        f"{emoji} <b>止损平仓</b>\n"
        f"股票：{ticker}\n"
        f"P&L：${pnl:.2f} ({'亏损' if pnl < 0 else '盈利'})\n"
        f"自动止损已执行"
    )


def notify_daily_report(report: dict):
    """每日报告通知。"""
    perf = report.get("performance", {})
    summ = perf.get("summary", {})
    risk = perf.get("risk_metrics", {})

    msg = (
        f"📈 <b>每日报告</b>  {report.get('report_date', '')}\n\n"
        f"账户总值：${summ.get('current_value', 0):,.2f}\n"
        f"总P&L：${summ.get('total_pnl', 0):+.2f} "
        f"({summ.get('total_pnl_pct', 0):+.1f}%)\n"
        f"胜率：{summ.get('win_rate', 0):.0f}% | "
        f"交易次数：{summ.get('total_trades', 0)}\n"
        f"Sharpe：{risk.get('sharpe_ratio', 0):.2f} | "
        f"最大回撤：{risk.get('max_drawdown_pct', 0):.1f}%\n"
    )

    # 当前持仓明细
    positions = report.get("open_positions_detail", [])
    if positions:
        msg += "\n📂 <b>当前持仓</b>\n"
        for p in positions[:5]:
            pnl = p.get("unrealized_pnl", 0)
            pct = p.get("unrealized_pct", 0)
            emoji = "🟢" if pnl >= 0 else "🔴"
            msg += (
                f"{emoji} <b>{p['ticker']}</b> {p['shares']}股\n"
                f"  现价 ${p['current_price']:.2f}  入场 ${p['entry_price']:.2f}\n"
                f"  止损 ${p['stop_loss']:.2f}  目标 ${p['target']:.2f}\n"
                f"  未实现 ${pnl:+.2f} ({pct:+.1f}%)\n"
            )
            if p.get("alert"):
                msg += f"  {p['alert']}\n"
    elif report.get("open_positions", 0) == 0:
        msg += "\n📂 当前无持仓\n"

    if risk.get("circuit_breaker"):
        msg += "\n⚠️ 熔断警示中（回撤/连续亏损超阈值），不再自动拦截开仓，请自行评估仓位"

    kelly = perf.get("kelly", {})
    if kelly.get("half_kelly_usd"):
        msg += f"\n💡 Kelly建议单仓：${kelly['half_kelly_usd']:.0f}"

    send_telegram(msg)


# ─────────────────────────────────────────────────────────────
# 实时新闻预筛（扫描前运行）
# ─────────────────────────────────────────────────────────────

def news_prefilter(tickers: list) -> dict:
    """
    扫描前对股票做新闻情绪预筛，过滤掉有重大负面新闻的标的。

    数据源：yfinance.news（有几小时延迟，适合摆动策略）
    如需实时：升级到 Benzinga API ($49/月) 或 Newsfilter ($19/月)

    返回：
      passed  — 无明显负面新闻，进入技术分析
      blocked — 重大负面新闻，本次跳过
      neutral — 无足够数据，按正常流程
    """
    import yfinance as yf

    passed  = []
    blocked = []
    neutral = []

    # 极度负面词（可能引发股价大跌）
    HARD_BLOCK_KEYWORDS = [
        "fraud", "sec investigation", "delisted", "bankruptcy",
        "doj", "criminal", "restatement", "accounting irregularities",
        "going concern", "subpoena", "class action",
    ]
    # 轻度负面词（仅警告，不封锁）
    SOFT_WARN_KEYWORDS = [
        "downgrade", "miss", "disappointing", "guidance cut",
        "layoff", "recall", "competition", "margin pressure",
    ]

    for ticker in tickers:
        try:
            tk    = yf.Ticker(ticker)
            news  = tk.news or []
            if not news:
                neutral.append(ticker)
                continue

            # 只看最近 48 小时内的新闻
            recent = []
            cutoff = datetime.now(ET).timestamp() - 48 * 3600
            for n in news[:10]:
                if n.get("providerPublishTime", 0) >= cutoff:
                    recent.append(n)

            if not recent:
                neutral.append(ticker)
                continue

            # 合并标题文本
            all_text = " ".join(
                (n.get("title", "") + " " + n.get("summary", "")).lower()
                for n in recent
            )

            hard_hit = [k for k in HARD_BLOCK_KEYWORDS if k in all_text]
            if hard_hit:
                blocked.append({
                    "ticker":  ticker,
                    "reason":  f"重大负面新闻：{', '.join(hard_hit)}",
                    "news_count": len(recent),
                })
            else:
                soft_hit = [k for k in SOFT_WARN_KEYWORDS if k in all_text]
                passed.append({
                    "ticker":  ticker,
                    "warning": f"轻度负面词：{', '.join(soft_hit)}" if soft_hit else None,
                    "news_count": len(recent),
                })

        except Exception:
            neutral.append(ticker)

    return {
        "passed":  [p["ticker"] for p in passed],
        "blocked": blocked,
        "neutral": neutral,
        "warnings": {p["ticker"]: p["warning"] for p in passed if p.get("warning")},
    }


# ─────────────────────────────────────────────────────────────
# 完整扫描流程（含新闻预筛 + 通知）
# ─────────────────────────────────────────────────────────────

def full_scan_cycle(watchlist: list, account: float, mode: str = "paper",
                    use_telegram: bool = True):
    """
    完整扫描周期：
      1. 新闻预筛 → 过滤重大负面新闻股票
      2. 技术+基本面扫描（cold_decision + debate）
      3. GO信号自动开模拟仓
      4. Telegram 推送信号通知
    """
    now = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    print(f"\n{'='*50}")
    print(f"[Scheduler] 开始完整扫描周期 @ {now}")
    print(f"[Scheduler] 账户：${account:,.0f} | 模式：{mode}")

    # ── 动态 watchlist 扩充（合并用户自选 + 热门板块代表股）────
    try:
        from src.trading_agent import build_dynamic_watchlist
        dyn = build_dynamic_watchlist(core=watchlist, max_total=20)
        watchlist = dyn["tickers"]
        print(f"[Scheduler] 动态列表：{dyn['note']}")
    except Exception as e:
        print(f"[Scheduler] 动态列表构建失败，使用原列表：{e}")

    # ── Step 1：新闻预筛 ──────────────────────────────────
    print(f"\n[1/3] 新闻预筛 {len(watchlist)} 只股票...")
    news_result = news_prefilter(watchlist)
    scan_list   = news_result["passed"] + news_result["neutral"]
    blocked     = news_result["blocked"]

    if blocked:
        blocked_names = [b["ticker"] for b in blocked]
        print(f"  ❌ 封锁：{blocked_names}（重大负面新闻）")
        if use_telegram:
            send_telegram(
                f"⚠️ 新闻过滤\n"
                f"以下股票因负面新闻跳过：{', '.join(blocked_names)}"
            )
    print(f"  ✅ 进入分析：{scan_list}")

    # ── Step 2：技术扫描 ─────────────────────────────────
    print(f"\n[2/3] 技术+基本面扫描 {len(scan_list)} 只...")
    if not scan_list:
        print("  所有股票被新闻过滤，本次跳过")
        return {}

    scan_result = run_scan(
        watchlist  = scan_list,
        account_value = account,
        direction  = "LONG",
        auto_paper = True,
        mode       = mode,
    )

    # ── Step 3：推送 GO 信号 ────────────────────────────
    go_signals = scan_result.get("go_signals", [])
    print(f"\n[3/3] 推送通知（{len(go_signals)} 个GO信号）...")
    if use_telegram:
        if go_signals:
            for sig in go_signals:
                msg_sig = (
                    f"🚀 <b>Agent GO信号</b>\n"
                    f"<b>{sig['ticker']}</b> @ ${sig.get('price', 0):.2f}\n"
                    f"冷静评分：{sig.get('score', 0)}\n"
                    f"辩论结论：{sig.get('debate_conclusion', 'N/A')}\n"
                    f"看多力度：{sig.get('bull_conviction', 'N/A')} | "
                    f"风险：{sig.get('bear_severity', 'N/A')}\n"
                    f"Kelly仓位：${sig.get('kelly_usd', 0):.0f}\n"
                )
                entry  = sig.get("entry_price")
                stop   = sig.get("stop_loss")
                target = sig.get("target_1")
                shares = sig.get("shares")
                if entry and stop and target:
                    rr = (target - entry) / (entry - stop) if entry > stop else 0
                    limit_px = round(entry * 1.005, 2)   # 最大追价 0.5%，超过此价跳过本笔
                    msg_sig += (
                        f"—\n"
                        f"入场区间：${entry:.2f}"
                        f"{f'  ×{shares}股' if shares else ''}\n"
                        f"止损：${stop:.2f} | 目标：${target:.2f}\n"
                        f"R:R = 1:{rr:.1f}\n"
                        f"⚡ <b>限价上限：${limit_px:.2f}</b>  超过此价跳过本笔\n"
                    )
                # 附加长持评分（带超时，避免 yfinance 网络挂起阻塞 GO 信号推送）
                try:
                    from src.long_hold import long_hold_eval, format_longhold_inline
                    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
                    with ThreadPoolExecutor(max_workers=1) as ex:
                        future = ex.submit(long_hold_eval, sig["ticker"])
                        lh = future.result(timeout=20)   # 最多等 20 秒
                    lh_line = format_longhold_inline(lh)
                    if lh_line:
                        msg_sig += lh_line + "\n"
                except (FuturesTimeout, Exception):
                    pass
                msg_sig += (
                    f"详情：/api/debate-with-cold?ticker={sig['ticker']}&account={account:.0f}\n"
                    f"—\n"
                    f"✅ 执行后请记录：<code>/logexec {sig['ticker']} &lt;实际成交价&gt;</code>\n"
                    f"❌ 跳过（超限价）请记录：<code>/logskip {sig['ticker']}</code>"
                )
                send_telegram(msg_sig)
        else:
            send_telegram(f"📊 扫描完成，本次无GO信号（已扫{len(scan_list)}只）")

    print(f"\n[Scheduler] 扫描周期完成 ✓\n{'='*50}")
    return scan_result


def monitor_cycle(mode: str = "paper", use_telegram: bool = True):
    """监控持仓，自动止损，推送警报。"""
    result  = run_monitor(mode, auto_stop=True)
    alerts  = result.get("alerts", [])
    closed  = result.get("auto_closed", [])

    if closed and use_telegram:
        for c in closed:
            notify_stop_loss(c["ticker"], c["pnl"])

    if alerts and use_telegram:
        send_telegram(
            f"⚡ 持仓警报\n" + "\n".join(f"• {a}" for a in alerts)
        )
    return result


def report_cycle(mode: str = "paper", use_telegram: bool = True):
    """每日报告 + GEX 快照 + 推送。"""
    report = daily_report(mode)
    if use_telegram:
        notify_daily_report(report)

    # GEX 快照（独立发送，避免日报消息过长）
    if use_telegram:
        try:
            from src.gex_scanner import gex_daily_snapshot, format_gex_telegram
            results = gex_daily_snapshot()
            send_telegram(format_gex_telegram(results))
            print("[GEX] 收盘快照已推送")
        except Exception as e:
            print(f"[GEX] 快照推送失败：{e}")

    return report


# ─────────────────────────────────────────────────────────────
# 调度器主循环（轻量级，无需 APScheduler）
# ─────────────────────────────────────────────────────────────

def is_market_day() -> bool:
    """简单判断：周一至周五（不含节假日，节假日影响较小可接受）。"""
    return datetime.now(ET).weekday() < 5


def run_scheduler(watchlist: list, account: float, mode: str = "paper",
                  use_telegram: bool = True):
    """
    主调度循环。每分钟检查是否到了预定时间。

    美东时间计划：
      09:45 → 开盘扫描
      12:00 → 午间监控
      15:30 → 收盘前扫描
      16:05 → 每日报告
    """
    def _latest_watchlist():
        """每次扫描前重新读取 watchlist.txt，支持 Telegram 实时更新。"""
        return load_watchlist(",".join(watchlist))

    def _macro_refresh():
        """每日 09:00 晨报：宏观快照 + 板块轮动 + 动态 watchlist 预告。
        任务1（宏观）/ 任务2（板块）/ 任务4（13D/G）并行下载，
        任务3（动态watchlist）等待任务2板块缓存就绪后串行执行。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        macro_msgs  = []   # 收集各任务的 Telegram 消息，按顺序统一发送
        sector_ok   = False

        def _task_macro():
            try:
                from src.macro_filter import full_macro_report, format_macro_telegram
                report = full_macro_report(_latest_watchlist())
                print(f"[Macro] 宏观快照已刷新：{report.get('master_action', '')}")
                return ("macro", format_macro_telegram(report))
            except Exception as e:
                print(f"[Macro] 宏观刷新失败：{e}")
                return ("macro", None)

        def _task_sector():
            try:
                from src.sector_rotation import fetch_sector_rankings, format_telegram_report
                fetch_sector_rankings(force=True)
                print("[Sector] 板块轮动已刷新")
                return ("sector", format_telegram_report())
            except Exception as e:
                print(f"[Sector] 板块轮动刷新失败：{e}")
                return ("sector", None)

        def _task_13dg():
            try:
                from src.sec_13dg_monitor import run_13dg_monitor
                wl  = _latest_watchlist()
                r13 = run_13dg_monitor(watchlist=wl,
                                       send_fn=send_telegram if use_telegram else None)
                if r13["new_count"]:
                    print(f"[13D/G] 推送 {r13['new_count']} 条新申报：{r13['pushed']}")
                else:
                    print("[13D/G] 无新机构大仓申报")
                return ("13dg", None)   # 13D/G 内部已自行推送
            except Exception as e:
                print(f"[13D/G] 监控失败：{e}")
                return ("13dg", None)

        # ── 任务1 / 2 / 4 并行执行 ─────────────────────────────
        results = {}
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {
                ex.submit(_task_macro):  "macro",
                ex.submit(_task_sector): "sector",
                ex.submit(_task_13dg):   "13dg",
            }
            for fut in as_completed(futures):
                kind, msg = fut.result()
                results[kind] = msg

        # ── 按顺序发送 Telegram 消息（宏观→板块）──────────────
        if use_telegram:
            for key in ("macro", "sector"):
                if results.get(key):
                    send_telegram(results[key])

        # ── 任务3：动态 watchlist（依赖任务2的板块缓存）────────
        try:
            from src.trading_agent import build_dynamic_watchlist
            core_wl = _latest_watchlist()
            dyn = build_dynamic_watchlist(core=core_wl, max_total=20)
            if use_telegram and dyn.get("sector_add"):
                sector_info = "\n".join(
                    f"  [{s['rank']}] {s['name']}（{s['etf']}）热度{s['heat']:+.1f}%"
                    + ("↑" if s["accel"] else "")
                    for s in dyn.get("sectors_used", [])
                )
                sector_add_display = dyn['sector_add']
                add_suffix = f"（共{len(sector_add_display)}只）" if len(sector_add_display) > 10 else ""
                send_telegram(
                    f"📋 <b>今日扫描列表（{dyn['total']}只）</b>\n\n"
                    f"核心持续：{', '.join(dyn['core'][:8]) or '无'}\n"
                    f"板块追加{add_suffix}：{', '.join(sector_add_display[:10])}\n\n"
                    f"🎯 当前热门板块：\n{sector_info}"
                )
            print(f"[Sector] 动态 watchlist：{dyn['total']} 只（{dyn.get('note','')}）")
        except Exception as e:
            print(f"[Sector] 动态 watchlist 构建失败：{e}")

    def _afternoon_13dg():
        """14:00 午后再查一次13D/G（大陆时间07:00前提交的申报可能当天才出现）。"""
        try:
            from src.sec_13dg_monitor import run_13dg_monitor
            wl = _latest_watchlist()
            r13 = run_13dg_monitor(watchlist=wl,
                                   send_fn=send_telegram if use_telegram else None)
            if r13["new_count"]:
                print(f"[13D/G] 午后推送 {r13['new_count']} 条新申报")
        except Exception as e:
            print(f"[13D/G] 午后监控失败：{e}")

    SCHEDULE = {
        (9,   0): ("macro_refresh",   _macro_refresh),
        (14,  0): ("afternoon_13dg",  _afternoon_13dg),
        (9,  45): ("morning_scan",    lambda: full_scan_cycle(_latest_watchlist(), account, mode, use_telegram)),
        (12,  0): ("noon_monitor",   lambda: monitor_cycle(mode, use_telegram)),
        (15, 30): ("closing_scan",   lambda: full_scan_cycle(_latest_watchlist(), account, mode, use_telegram)),
        (16,  5): ("daily_report",   lambda: report_cycle(mode, use_telegram)),
    }

    executed_today = set()

    if use_telegram:
        send_telegram(
            f"🤖 <b>TradingAgent 启动</b>\n"
            f"账户：${account:,.0f} | 模式：{mode}\n"
            f"监控股票：{', '.join(watchlist[:8])}{'...' if len(watchlist)>8 else ''}\n"
            f"计划：09:45 / 12:00 / 15:30 / 16:05 ET"
        )

    print(f"[Scheduler] Agent 启动，按 Ctrl+C 停止")
    print(f"[Scheduler] Telegram：{'已配置' if TELEGRAM_BOT_TOKEN else '未配置（建议配置）'}")
    print(f"[Scheduler] 监控 {len(watchlist)} 只股票 @ ${account:,.0f}")

    while True:
        try:
            now  = datetime.now(ET)
            hhmm = (now.hour, now.minute)
            date_str = now.strftime("%Y-%m-%d")

            if is_market_day():
                for (h, m), (label, func) in SCHEDULE.items():
                    key = f"{date_str}_{label}"
                    if hhmm == (h, m) and key not in executed_today:
                        executed_today.add(key)
                        print(f"\n[Scheduler] 执行：{label} @ {now.strftime('%H:%M ET')}")
                        try:
                            func()
                        except Exception as e:
                            print(f"[Scheduler] {label} 执行出错：{e}")
                            if use_telegram:
                                send_telegram(f"❌ Agent错误：{label}\n{str(e)[:200]}")

            # 每天午夜清空已执行记录
            if hhmm == (0, 1):
                executed_today.clear()

            time.sleep(30)  # 每30秒检查一次（减少任务触发延迟）

        except KeyboardInterrupt:
            print("\n[Scheduler] 已停止")
            if use_telegram:
                send_telegram("⛔ TradingAgent 已停止")
            break
        except Exception as e:
            print(f"[Scheduler] 主循环异常：{e}")
            time.sleep(60)


# ─────────────────────────────────────────────────────────────
# 部署信息（运行时打印）
# ─────────────────────────────────────────────────────────────

def print_deployment_guide():
    guide = """
╔══════════════════════════════════════════════════════════════╗
║              部署到云服务器（全天候自动运行）                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  推荐方案：DigitalOcean Droplet（$6/月，1GB内存）               ║
║  或：AWS EC2 t2.micro（免费1年）                               ║
║  或：Render.com（免费tier，每月750小时）                         ║
║                                                              ║
║  1. 服务器上安装依赖：                                          ║
║     pip install -r requirements.txt                           ║
║                                                              ║
║  2. 设置环境变量（Telegram通知）：                               ║
║     export TELEGRAM_BOT_TOKEN="你的token"                     ║
║     export TELEGRAM_CHAT_ID="你的chat_id"                     ║
║                                                              ║
║  3. 后台运行（nohup）：                                         ║
║     nohup python scheduler.py --telegram &                    ║
║     tail -f nohup.out  # 查看日志                             ║
║                                                              ║
║  4. 或用 systemd 服务（开机自启）：                              ║
║     见 deploy/trading-agent.service                           ║
║                                                              ║
║  每月成本估算：                                                 ║
║    服务器：$0-8                                                ║
║    数据：$0（yfinance免费）                                    ║
║    通知：$0（Telegram免费）                                    ║
║    LLM：$0（当前系统无LLM调用）                                 ║
║    ─────────────────────────                                  ║
║    总计：$0-8/月                                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(guide)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def load_watchlist(watchlist_arg: str) -> list:
    """
    读取自选股列表。优先级：
      1. watchlist.txt 文件（每行一个代码）
      2. --watchlist 命令行参数
      3. 内置默认列表
    """
    wl_file = os.path.join(os.path.dirname(__file__), "..", "watchlist.txt")
    if os.path.exists(wl_file):
        with open(wl_file, "r", encoding="utf-8") as f:
            tickers = [
                line.strip().upper()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
        if tickers:
            print(f"[Scheduler] 从 watchlist.txt 加载 {len(tickers)} 只股票")
            return tickers
    return [t.strip().upper() for t in watchlist_arg.split(",") if t.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradingAgent Scheduler")
    parser.add_argument("--watchlist", default=",".join(DEFAULT_WATCHLIST))
    parser.add_argument("--account",   type=float, default=2000)
    parser.add_argument("--mode",      default="paper", choices=["paper", "real"])
    parser.add_argument("--telegram",  action="store_true", help="开启 Telegram 通知")
    parser.add_argument("--test",      action="store_true", help="立即运行一次扫描（测试用）")
    parser.add_argument("--deploy",    action="store_true", help="打印部署指南")
    args = parser.parse_args()

    tickers = load_watchlist(args.watchlist)

    # 启动 Telegram 指令监听
    if args.telegram:
        from src.telegram_bot import start_bot_thread
        start_bot_thread()

    if args.deploy:
        print_deployment_guide()
    elif args.test:
        print("[Test] 立即执行一次完整扫描...")
        result = full_scan_cycle(tickers, args.account, args.mode, args.telegram)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        run_scheduler(tickers, args.account, args.mode, args.telegram)
