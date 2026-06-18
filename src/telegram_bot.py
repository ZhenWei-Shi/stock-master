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
from datetime import datetime
import pytz

ET = pytz.timezone("America/New_York")

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
            "/scan             立即扫描\n"
            "/status           运行状态\n"
            "/help             显示帮助"
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
        if added:
            send(f"✅ 已添加：{', '.join(added)}\n当前共 {len(wl)} 只\n\n重启Agent后生效，或发 /scan 立即扫描")
        else:
            send(f"这些股票已在列表中：{', '.join(new_tickers)}")

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
        send("⏳ 开始扫描，请稍候（约1-3分钟）...")
        try:
            from src.scheduler import full_scan_cycle
            wl = read_watchlist()
            if not wl:
                send("自选股列表为空，请先 /add 股票")
                return
            account = float(os.getenv("AGENT_ACCOUNT", "2000"))
            threading.Thread(
                target=full_scan_cycle,
                args=(wl, account, "paper", True),
                daemon=True,
            ).start()
        except Exception as e:
            send(f"扫描启动失败：{e}")

    elif cmd == "/status":
        wl = read_watchlist()
        now = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
        send(
            f"🤖 <b>Agent 状态</b>\n\n"
            f"时间：{now}\n"
            f"自选股：{len(wl)} 只\n"
            f"{'股票：' + ', '.join(wl[:5]) + ('...' if len(wl)>5 else '') if wl else '暂无自选股'}\n\n"
            f"定时任务：09:45 / 12:00 / 15:30 / 16:05 ET"
        )

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
