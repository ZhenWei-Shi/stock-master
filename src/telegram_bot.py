"""
Telegram Bot 指令处理器

支持的指令（手机直接发给 bot）：
  /add NVDA AAPL TSLA   — 添加股票到自选
  /remove NVDA          — 从自选删除
  /list                 — 查看当前自选股列表
  /scan                 — 立即触发一次扫描
  /status               — 查看 Agent 运行状态
  /help                 — 显示帮助
"""

import os
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeoutError
from datetime import datetime
import pytz

ET = pytz.timezone("America/New_York")


def _timed_thread(fn, timeout: int, send_fn, label: str):
    """daemon 线程包装器：超时后推送超时提示，防止网络故障时永久阻塞。"""
    def _wrapper():
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fn)
            try:
                fut.result(timeout=timeout)
            except FutTimeoutError:
                try:
                    send_fn(f"⏰ {label} 超时（>{timeout}s），请检查网络或稍后重试")
                except Exception:
                    pass
            except Exception as e:
                try:
                    send_fn(f"⚠️ {label} 执行出错：{e}")
                except Exception:
                    pass
    threading.Thread(target=_wrapper, daemon=True).start()

WATCHLIST_FILE = os.path.join(os.path.dirname(__file__), "..", "watchlist.txt")
_last_update_id = 0


def _token():
    return os.getenv("TELEGRAM_BOT_TOKEN", "")


def _chat_id():
    return os.getenv("TELEGRAM_CHAT_ID", "")


def send(msg: str):
    token = _token()
    if not token:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": _chat_id(), "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# 股票自动分类
# ─────────────────────────────────────────────────────────────

# 关键词 → 中文分类标签
_SECTOR_MAP = {
    "semiconductor":        "半导体",
    "semiconductors":       "半导体",
    "electronic":           "电子元器件",
    "software":             "软件/SaaS",
    "cloud":                "云计算",
    "internet":             "互联网",
    "artificial intelligence": "人工智能",
    "data center":          "数据中心",
    "networking":           "网络设备",
    "cybersecurity":        "网络安全",
    "biotechnology":        "生物科技",
    "pharmaceutical":       "制药",
    "healthcare":           "医疗",
    "financial":            "金融科技",
    "bank":                 "银行",
    "energy":               "能源",
    "oil":                  "石油天然气",
    "electric":             "新能源/电动车",
    "consumer":             "消费",
    "retail":               "零售",
    "aerospace":            "航空航天",
    "defense":              "国防",
    "real estate":          "房地产/REIT",
    "utilities":            "公用事业",
    "communication":        "通信",
    "media":                "传媒",
    "entertainment":        "娱乐",
    "e-commerce":           "电商",
}

_SUPPLY_CHAIN_MAP = {
    # AI 供应链层级
    "NVDA": ("AI芯片", "供应链第1层：算力核心"),
    "AMD":  ("AI芯片/CPU", "供应链第1层：算力核心"),
    "INTC": ("CPU/晶圆代工", "供应链第1层：算力核心"),
    "AVGO": ("AI网络芯片", "供应链第1层：互联芯片"),
    "MRVL": ("数据中心芯片", "供应链第1层：算力核心"),
    "QCOM": ("移动芯片", "供应链第1层：端侧AI"),
    "ARM":  ("芯片IP授权", "供应链第1层：底层架构"),
    # 半导体设备
    "AMAT": ("半导体设备", "供应链第0层：制造设备"),
    "LRCX": ("半导体设备", "供应链第0层：刻蚀设备"),
    "KLAC": ("半导体检测", "供应链第0层：良率控制"),
    "ASML": ("光刻机", "供应链第0层：最上游"),
    # 先进封装/HBM
    "COHR": ("光子/先进封装", "供应链第0.5层：封装材料"),
    "LITE": ("光模块", "供应链第1.5层：数据中心互联"),
    "AXTI": ("砷化镓衬底", "供应链第0层：化合物半导体"),
    "AAOI": ("光模块", "供应链第1.5层：AI网络互联"),
    # 云/数据中心
    "MSFT": ("云计算/AI应用", "供应链第3层：AI使能者"),
    "GOOGL":("云计算/AI搜索","供应链第3层：AI使能者"),
    "AMZN": ("云计算/电商", "供应链第3层：AI基础设施"),
    "META": ("AI社交/广告", "供应链第3层：AI应用"),
    "ORCL": ("云数据库", "供应链第2层：企业AI"),
    # 电力/散热
    "VRT":  ("数据中心散热", "供应链第1层：基础设施"),
    "CEG":  ("核电/清洁能源", "供应链第0层：AI电力"),
    "VST":  ("电力供应", "供应链第0层：AI电力"),
    # 消费科技
    "AAPL": ("消费电子/生态", "供应链第4层：终端设备"),
    "TSLA": ("电动车/AI机器人", "跨界：能源+AI"),
}


def classify_ticker(ticker: str) -> dict:
    """
    自动分析股票所属行业、供应链位置、风险等级。
    数据源：yfinance info（免费）
    """
    import yfinance as yf

    result = {
        "ticker":         ticker,
        "name":           ticker,
        "sector":         "未知",
        "industry":       "未知",
        "category":       "未知",
        "supply_chain":   None,
        "market_cap":     None,
        "market_cap_label": "未知",
        "risk_level":     "中",
        "note":           "",
    }

    # 先查内置供应链映射
    if ticker in _SUPPLY_CHAIN_MAP:
        result["category"], result["supply_chain"] = _SUPPLY_CHAIN_MAP[ticker]

    try:
        info = yf.Ticker(ticker).info
        result["name"] = info.get("shortName") or info.get("longName") or ticker

        sector   = (info.get("sector") or "").lower()
        industry = (info.get("industry") or "").lower()
        combined = sector + " " + industry

        # 行业分类
        for kw, label in _SECTOR_MAP.items():
            if kw in combined:
                result["sector"]   = label
                result["industry"] = info.get("industry", "")
                if not result["category"] or result["category"] == "未知":
                    result["category"] = label
                break

        # 市值分级
        cap = info.get("marketCap") or 0
        result["market_cap"] = cap
        if cap >= 1_000_000_000_000:
            result["market_cap_label"] = f"超大盘（${cap/1e12:.1f}T）"
            result["risk_level"] = "低"
        elif cap >= 100_000_000_000:
            result["market_cap_label"] = f"大盘（${cap/1e9:.0f}B）"
            result["risk_level"] = "低中"
        elif cap >= 10_000_000_000:
            result["market_cap_label"] = f"中盘（${cap/1e9:.0f}B）"
            result["risk_level"] = "中"
        elif cap >= 2_000_000_000:
            result["market_cap_label"] = f"小盘（${cap/1e9:.1f}B）"
            result["risk_level"] = "中高"
        elif cap > 0:
            result["market_cap_label"] = f"微盘（${cap/1e6:.0f}M）"
            result["risk_level"] = "高"

        # 额外标注
        beta = info.get("beta") or 1.0
        if beta > 2.0:
            result["note"] += f"⚡ 高Beta({beta:.1f})，波动剧烈 "
        if info.get("shortPercentOfFloat", 0) > 0.15:
            result["note"] += f"🐻 空头比例{info['shortPercentOfFloat']*100:.0f}%，注意轧空 "

    except Exception as e:
        result["note"] = f"数据获取失败：{e}"

    return result


def format_classification(r: dict) -> str:
    lines = [f"📌 <b>{r['ticker']}</b> — {r['name']}"]
    if r["category"] != "未知":
        lines.append(f"分类：{r['category']}")
    if r["supply_chain"]:
        lines.append(f"供应链：{r['supply_chain']}")
    if r["sector"] != "未知":
        lines.append(f"行业：{r['sector']}")
    lines.append(f"市值：{r['market_cap_label']}")
    lines.append(f"风险：{r['risk_level']}")
    if r["note"]:
        lines.append(r["note"].strip())
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 自选股文件读写
# ─────────────────────────────────────────────────────────────

def read_watchlist() -> list:
    if not os.path.exists(WATCHLIST_FILE):
        return []
    with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
        return [
            line.strip().upper()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]


def write_watchlist(tickers: list):
    with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
        f.write("# 自选股列表（通过 Telegram Bot 管理）\n")
        f.write(f"# 最后更新：{datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}\n\n")
        for t in tickers:
            f.write(t + "\n")


# ─────────────────────────────────────────────────────────────
# 指令处理
# ─────────────────────────────────────────────────────────────

def handle_command(text: str):
    text = text.strip()
    parts = text.split()
    cmd = parts[0].lower() if parts else ""

    if cmd == "/help" or cmd == "/start":
        send(
            "📋 <b>TradingAgent 指令</b>\n\n"
            "/add NVDA AAPL    添加股票\n"
            "/remove NVDA      删除股票\n"
            "/list             查看自选股\n"
            "/scan             立即扫描（含板块轮动扩充）\n"
            "/sector           板块轮动排名（强制刷新）\n"
            "/hotlist          查看当日动态扫描列表\n"
            "/gex [NVDA TSLA]           GEX伽马敞口快照（默认SPY/QQQ/NVDA）\n"
            "/longhold NVDA AAPL        长期持仓质量评估（1年以上视角）\n"
            "/insider NVDA AMD          SEC Form 4 内部人买卖记录（近90天）\n"
            "/13dg                      SEC 13D/G机构大仓新申报（近3天）\n"
            "/logexec NVDA 142.00 143.50  记录信号价→实际成交价（执行追踪）\n"
            "/logskip NVDA              记录跳过（超出限价）\n"
            "/execreport                执行偏差统计报告\n"
            "/status                    运行状态\n"
            "/help                      显示帮助"
        )

    elif cmd == "/list":
        wl = read_watchlist()
        if wl:
            send(f"📊 <b>当前自选股（{len(wl)}只）</b>\n\n" + "\n".join(wl))
        else:
            send("自选股列表为空，用 /add NVDA 添加")

    elif cmd == "/add":
        new_tickers = [p.upper() for p in parts[1:] if p.isalpha()]
        if not new_tickers:
            send("用法：/add NVDA AAPL TSLA")
            return
        wl = read_watchlist()
        added = []
        for t in new_tickers:
            if t not in wl:
                wl.append(t)
                added.append(t)
        write_watchlist(wl)
        if not added:
            send(f"这些股票已在列表中：{', '.join(new_tickers)}")
            return
        send(f"✅ 已添加 {len(added)} 只，正在分析分类...")
        # 后台分类分析，逐个发送结果
        def classify_and_report():
            for t in added:
                try:
                    r = classify_ticker(t)
                    send(format_classification(r))
                except Exception as e:
                    send(f"{t} 分类失败：{e}")
            send(f"\n📋 当前自选股共 {len(wl)} 只\n发 /scan 立即扫描")
        _timed_thread(classify_and_report, timeout=90, send_fn=send, label="/add 分类")

    elif cmd == "/remove":
        del_tickers = [p.upper() for p in parts[1:] if p.isalpha()]
        if not del_tickers:
            send("用法：/remove NVDA")
            return
        wl = read_watchlist()
        removed = [t for t in del_tickers if t in wl]
        wl = [t for t in wl if t not in del_tickers]
        write_watchlist(wl)
        if removed:
            send(f"🗑 已删除：{', '.join(removed)}\n剩余 {len(wl)} 只")
        else:
            send(f"未找到：{', '.join(del_tickers)}")

    elif cmd == "/scan":
        try:
            from src.scheduler import full_scan_cycle
            wl = read_watchlist()
            account = float(os.getenv("AGENT_ACCOUNT", "2000"))
            if not wl:
                send("⚠️ 自选股为空，将使用板块轮动热股扫描（约2-3分钟）...")
            else:
                send(f"⏳ 开始扫描 {len(wl)} 只自选股 + 板块热股，请稍候（约1-3分钟）...")
            _timed_thread(
                lambda: full_scan_cycle(wl, account, "paper", True),
                timeout=480, send_fn=send, label="/scan"
            )
        except Exception as e:
            send(f"扫描启动失败：{e}")

    elif cmd == "/sector":
        send("⏳ 正在拉取板块轮动数据（强制刷新）...")
        def _do_sector():
            try:
                from src.sector_rotation import fetch_sector_rankings, format_telegram_report
                fetch_sector_rankings(force=True)
                send(format_telegram_report())
            except Exception as e:
                send(f"板块轮动获取失败：{e}")
        _timed_thread(_do_sector, timeout=90, send_fn=send, label="/sector")

    elif cmd == "/hotlist":
        send("⏳ 正在构建动态扫描列表（如缓存过期需约30秒）...")
        def _do_hotlist():
            try:
                from src.trading_agent import build_dynamic_watchlist
                wl  = read_watchlist()
                dyn = build_dynamic_watchlist(core=wl, max_total=20)
                lines = [f"📋 <b>今日动态扫描列表（{dyn['total']}只）</b>\n"]
                if dyn["core"]:
                    lines.append(f"📌 <b>固定自选</b>：{', '.join(dyn['core'])}")
                if dyn["sector_add"]:
                    lines.append(f"\n⚡ <b>板块轮动追加</b>：{', '.join(dyn['sector_add'])}")
                for s in dyn.get("sectors_used", []):
                    accel = "↑加速" if s["accel"] else ""
                    lines.append(f"  [{s['rank']}] {s['name']}（{s['etf']}）{s['heat']:+.1f}% {accel}")
                lines.append(f"\n💡 {dyn.get('note','')}")
                send("\n".join(lines))
            except Exception as e:
                send(f"动态列表构建失败：{e}")
        _timed_thread(_do_hotlist, timeout=120, send_fn=send, label="/hotlist")

    elif cmd == "/gex":
        # /gex 或 /gex NVDA TSLA AMD
        custom = [p.upper() for p in parts[1:] if p.isalpha()]
        from src.gex_scanner import GEX_DEFAULT_TICKERS
        tickers_to_scan = custom if custom else GEX_DEFAULT_TICKERS
        send(f"⏳ 正在计算 {', '.join(tickers_to_scan)} 的 GEX 快照（约30-60秒）...")
        def _do_gex():
            try:
                from src.gex_scanner import gex_daily_snapshot, format_gex_telegram
                results = gex_daily_snapshot(tickers_to_scan)
                send(format_gex_telegram(results))
            except Exception as e:
                send(f"GEX 计算失败：{e}")
        _timed_thread(_do_gex, timeout=150, send_fn=send, label="/gex")

    elif cmd == "/longhold":
        # /longhold 或 /longhold NVDA AAPL MSFT
        custom = [p.upper() for p in parts[1:] if p.isalpha()]
        if not custom:
            wl = read_watchlist()
            custom = wl[:8] if wl else ["NVDA", "AAPL", "MSFT", "GOOGL"]
        send(f"⏳ 正在评估 {', '.join(custom)} 的长期持仓质量（每只约10-15秒）...")
        def _do_longhold():
            try:
                from src.long_hold import long_hold_scan, format_longhold_telegram
                results = long_hold_scan(custom)
                send(format_longhold_telegram(results))
            except Exception as e:
                send(f"长持评估失败：{e}")
        _timed_thread(_do_longhold, timeout=200, send_fn=send, label="/longhold")

    elif cmd == "/status":
        wl = read_watchlist()
        now = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
        send(
            f"🤖 <b>Agent 状态</b>\n\n"
            f"时间：{now}\n"
            f"固定自选：{len(wl)} 只\n"
            f"{'股票：' + ', '.join(wl[:5]) + ('...' if len(wl)>5 else '') if wl else '暂无自选股'}\n\n"
            f"定时任务：09:00 晨报 / 09:45 扫描 / 12:00 监控 / 15:30 扫描 / 16:05 日报\n"
            f"发 /hotlist 查看今日动态扫描列表"
        )

    elif cmd == "/insider":
        tickers_arg = [p.upper() for p in parts[1:] if p.isalpha()] or ["NVDA"]
        send(f"⏳ 查询 {', '.join(tickers_arg)} 的 SEC Form 4 内部人交易记录...")
        def _run_insider():
            from src.insider_tracker import insider_summary, format_insider_telegram
            for t in tickers_arg[:4]:
                try:
                    r = insider_summary(t)
                    send(format_insider_telegram(r))
                except Exception as e:
                    send(f"⚠️ {t} 内部人查询失败：{e}")
        _timed_thread(_run_insider, timeout=120, send_fn=send, label="/insider")

    elif cmd == "/13dg":
        send("⏳ 查询 SEC 最新13D/G机构大仓申报（近3天）...")
        def _run_13dg():
            from src.sec_13dg_monitor import check_new_13dg, format_13dg_telegram
            wl = read_watchlist()
            filings = check_new_13dg(watchlist=wl if wl else None)
            if not filings:
                send("✅ 最近3天无新13D/G申报（针对当前自选股）")
                return
            for f in filings[:5]:
                send(format_13dg_telegram(f))
        _timed_thread(_run_13dg, timeout=90, send_fn=send, label="/13dg")

    elif cmd in ("/logexec", "/logskip"):
        # 执行偏差日志（P0-C：关闭信号→实际执行的黑洞监控）
        # 用法：/logexec NVDA 142.00 143.50   或   /logskip NVDA [信号价]
        from src.paper_trading import log_execution
        ticker_arg = parts[1].upper() if len(parts) > 1 else ""
        if not ticker_arg:
            send("用法：/logexec NVDA 142.00 143.50  或  /logskip NVDA 142.00")
            return
        if cmd == "/logexec":
            if len(parts) < 4:
                send("用法：/logexec NVDA <信号价> <实际成交价>\n例：/logexec NVDA 142.00 143.50")
                return
            try:
                signal_px = float(parts[2])
                actual_px = float(parts[3])
                r = log_execution(ticker_arg, signal_price=signal_px, actual_price=actual_px,
                                  signal_time="", action="entered",
                                  note="Telegram手动记录")
                dev = r.get("deviation_pct", 0)
                send(f"✅ 已记录 {ticker_arg}：信号${signal_px:.2f}→实际${actual_px:.2f}，"
                     f"偏差{dev:+.2f}%")
            except ValueError:
                send(f"价格格式错误，用法：/logexec NVDA 142.00 143.50")
        else:  # /logskip
            try:
                signal_px = float(parts[2]) if len(parts) >= 3 else 0.0
            except ValueError:
                signal_px = 0.0
            log_execution(ticker_arg, signal_price=signal_px, actual_price=0,
                          signal_time="", action="skipped",
                          note="超出限价，Telegram手动记录")
            send(f"⏭ 已记录 {ticker_arg} 跳过（信号超出限价{f'，信号价${signal_px:.2f}' if signal_px else ''}）")

    elif cmd == "/execreport":
        from src.paper_trading import execution_deviation_report
        r = execution_deviation_report()
        if r.get("note"):
            send(f"📊 <b>执行偏差报告</b>\n{r['note']}\n\n"
                 f"总信号：{r.get('total_signals',0)}  "
                 f"已执行：{r.get('entered',0)}  "
                 f"跳过：{r.get('skipped',0)}  "
                 f"手动改单：{r.get('manual_override',0)}\n"
                 f"平均偏差：{r.get('avg_deviation_pct',0):+.2f}%  "
                 f"最大偏差：{r.get('max_deviation_pct',0):+.2f}%")
        else:
            send(str(r))

    else:
        send(f"未知指令：{cmd}\n发 /help 查看支持的指令")


# ─────────────────────────────────────────────────────────────
# 轮询 Telegram 消息（后台线程）
# ─────────────────────────────────────────────────────────────

def poll_loop():
    """持续轮询 Telegram 消息，处理用户指令。"""
    global _last_update_id
    token = _token()
    if not token:
        print("[Bot] 未配置 TELEGRAM_BOT_TOKEN，指令功能关闭")
        return

    print("[Bot] Telegram 指令监听已启动")
    while True:
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{token}/getUpdates",
                params={"offset": _last_update_id + 1, "timeout": 30},
                timeout=35,
            )
            if resp.status_code != 200:
                time.sleep(5)
                continue

            updates = resp.json().get("result", [])
            for update in updates:
                _last_update_id = update["update_id"]
                msg = update.get("message", {})
                text = msg.get("text", "")
                # 只响应自己发的消息（安全过滤）
                from_id = str(msg.get("chat", {}).get("id", ""))
                if from_id != _chat_id():
                    continue
                if text.startswith("/"):
                    handle_command(text)

        except Exception as e:
            print(f"[Bot] 轮询异常：{e}")
            time.sleep(10)


def start_bot_thread():
    """在后台线程启动 Bot 指令监听。"""
    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()
    return t
