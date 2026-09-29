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
_wl_lock = threading.Lock()  # 防止读写竞态清空watchlist
_bot_started = False
_bot_start_lock = threading.Lock()


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
    except Exception as e:
        # 过滤掉含 token 的异常信息，防止 token 泄漏到日志
        err_str = str(e)
        safe_err = "[敏感信息已过滤]" if token and token in err_str else err_str
        print(f"[Telegram] 发送失败：{safe_err}")


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


def read_watchlist() -> list:
    with _wl_lock:
        if not os.path.exists(WATCHLIST_FILE):
            return []
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            return [
                line.strip().upper()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]


def write_watchlist(tickers: list):
    with _wl_lock:
        tmp = WATCHLIST_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("# 自选股列表（通过 Telegram Bot 管理）\n")
            f.write(f"# 最后更新：{datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}\n\n")
            for t in tickers:
                f.write(t + "\n")
        os.replace(tmp, WATCHLIST_FILE)  # 原子替换，防止并发截断


# ─────────────────────────────────────────────────────────────
# 指令处理
# ─────────────────────────────────────────────────────────────

# 2026-09-29精简：九关降级后不再使用的（/scan /hotlist /check /longhold /logexec /logskip
# /execreport），以及很少用的查询（/sector /oi /uoalist /shortvol /fedwatch /insider /13dg）
RETIRED_COMMANDS = {"/scan", "/sector", "/hotlist", "/oi", "/uoalist", "/shortvol", "/fedwatch",
                    "/longhold", "/check", "/insider", "/13dg", "/logexec", "/logskip", "/execreport"}


def handle_command(text: str):
    text = text.strip()
    parts = text.split()
    cmd = parts[0].lower() if parts else ""

    if cmd == "/help" or cmd == "/start":
        send(
            "📋 <b>TradingAgent 指令</b>（2026-09-29精简）\n\n"
            "<b>自选股</b>\n"
            "/list            查看自选股\n"
            "/add NVDA AAPL   添加\n"
            "/remove NVDA     删除\n\n"
            "<b>策略</b>\n"
            "/events          事件实验室+隔夜放量的进度\n"
            "/perf            模拟盘绩效（回撤/Sharpe/对比SPY）\n"
            "/risk NVDA       风控检查（不是买卖信号）\n\n"
            "<b>期权</b>\n"
            "/gex [NVDA]      伽马敞口（默认SPX/SPY/QQQ）\n"
            "/uoa NVDA        异常大单检测+加入每小时监控\n"
            "/uoa NVDA off    停止监控　/uoa 查看监控列表\n\n"
            "/status          运行状态"
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

    elif cmd == "/gex":
        # /gex（默认大盘SPX/SPY/QQQ）或 /gex NVDA TSLA AMD（查任意个股，"^SPX"等指数代码也支持）
        custom = [p.upper() for p in parts[1:] if p.lstrip("^").isalpha()]
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

    elif cmd == "/uoa":
        # /uoa NVDA —— 立即检测一次 + 自动加入监控列表（此后scheduler每小时
        #             扫描，有新增/升级的异常大单会自动推送，不用再手动查）
        # /uoa NVDA off —— 停止监控该股票
        args = parts[1:]
        if not args:
            from src.smart_money import get_uoa_watchlist
            wl = get_uoa_watchlist()
            send(f"👁 <b>UOA自动监控列表</b>（{len(wl)}只）\n{', '.join(wl) if wl else '空，用 /uoa TICKER 添加'}\n"
                 f"盘中10:00-15:00每小时扫描，有新增/升级的异常大单自动推送")
            return
        if not args[0].isalpha():
            send("用法：/uoa NVDA（检测+加入监控）　/uoa NVDA off（停止监控）　/uoa（查看列表）")
            return
        uoa_ticker = args[0].upper()

        if len(args) > 1 and args[1].lower() == "off":
            from src.smart_money import remove_uoa_watch
            wl = remove_uoa_watch(uoa_ticker)
            send(f"🔕 已停止监控 {uoa_ticker}\n当前监控列表（{len(wl)}只）：{', '.join(wl) if wl else '空'}")
            return

        from src.smart_money import add_uoa_watch
        wl = add_uoa_watch(uoa_ticker)
        send(f"⏳ 正在检测 {uoa_ticker} 期权异常大单（约20-40秒），已加入自动监控列表（{len(wl)}只）...")
        def _do_uoa():
            try:
                from src.smart_money import detect_large_orders, format_large_orders_telegram
                result = detect_large_orders(uoa_ticker)
                send(format_large_orders_telegram(result))
            except Exception as e:
                send(f"异常大单检测失败：{e}")
        _timed_thread(_do_uoa, timeout=120, send_fn=send, label="/uoa")

    elif cmd == "/events":
        # /events —— 事件实验室进度：各假设样本数/结果、即将到来的财报
        from src.event_lab import summarize
        from src.overnight_lab import summarize as overnight_summary
        send(summarize() + "\n" + overnight_summary())

    elif cmd == "/risk":
        # /risk NVDA —— 风控层检查（九关2026-09-29降级后只保留风控类gate：止损宽度/
        # 财报/发债/新闻/PDT/时段/VIX/宏观），不给买卖信号
        if len(parts) < 2:
            send("用法：/risk NVDA（只看风控能不能做，不是买卖信号）")
        else:
            tk = parts[1].upper()

            def _do_risk():
                from src.risk_layer import risk_check, format_risk
                send(format_risk(risk_check(tk)))
            _timed_thread(_do_risk, 120, send, f"/risk {tk}")

    elif cmd == "/perf":
        # /perf —— 模拟盘绩效（资金曲线：回撤/Sharpe/对比SPY + 逐笔胜率区间）。
        # quantstats较重，子进程生成，不常驻bot内存
        send("📊 正在生成模拟盘绩效报告（约30秒）...")

        def _do_perf():
            from src.performance_report import run_in_subprocess
            send(run_in_subprocess())
        _timed_thread(_do_perf, 300, send, "/perf 绩效报告")

    elif cmd == "/status":
        wl = read_watchlist()
        now = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
        send(
            f"🤖 <b>Agent 状态</b>\n\n"
            f"时间：{now}\n"
            f"固定自选：{len(wl)} 只\n"
            f"{'股票：' + ', '.join(wl[:5]) + ('...' if len(wl)>5 else '') if wl else '暂无自选股'}\n\n"
            f"定时任务（ET）：09:00晨报 / 10-15点每小时监控 / 15:40月度动量（月末）/ "
            f"15:45隔夜放量登记 / 15:50事件实验室 / 16:05日报 / 16:20假突破记录 / 周五16:30周报"
        )

    elif cmd in RETIRED_COMMANDS:
        send(f"{cmd} 已下线（2026-09-29精简指令）。发 /help 查看现有指令")

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
    """在后台线程启动 Bot 指令监听。重入保护：Flask reloader 等场景下只启动一次。"""
    global _bot_started
    with _bot_start_lock:
        if _bot_started:
            return None
        _bot_started = True
    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()
    return t
