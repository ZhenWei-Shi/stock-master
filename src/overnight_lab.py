"""
隔夜放量前向登记 H4（2026-09-29，用户想法；只记录、不下单）

回测（wiki stock-master/overview"隔夜放量策略回测"，PR#26）：放量上涨的股票收盘买、
次日开盘卖，扣成本后每笔约+0.12%~+0.17%（t≈1.5-1.7），方向为正但不显著；
开盘后止损/动态止盈没有带来额外收益。用户决定登记为前向假设。

事先登记的规则（数据出来前定好，不事后修改）：
  - 每个交易日15:45，在股票池（与月度动量相同：watchlist+板块代表股，去掉ETF）里找
    "当天截至此时的成交量 ≥ 前20日均量×RVOL_MIN 且 现价 > 昨收"的股票，按量比取前TOP_K只
    （15:45离收盘还差最后一截成交，所以阈值用1.8而不是回测的2.0）
  - 买入价 = 15:45时的价格；卖出价 = 下一个交易日开盘价
  - 每条信号同时从当天非信号股票里随机抽1只，用同样的价格口径记作对照
  - 成本：每笔往返扣COST_ROUNDTRIP（10bp）
  - 判定：≥MIN_TRADES笔后，扣成本平均收益>0且t>2才算成立；对照组用来确认不是市场整体隔夜上涨

扩池（2026-10-10用户决定，PR#52，POOL_V2_START起生效）：原股票池约91只，9个交易日只出5笔信号，
  攒满200笔约需1.5年。股票池改为原池 ∪ 标普500 ∪ 纳指100（universe_large，约530只）；量比门槛、
  每天前TOP_K只、对照、成本、判定标准都不变。近6个月日线估算：封顶5只后每天约3.7笔（旧池约1.1笔），
  按15:45成交量偏少折算，200笔约3-5个月。
  - 新记录带pool="v2"；**判定只用v2样本**（POOL_V2_START起），v1（9/29-10/9旧池）单独列出、不参与判定
  - 约530只的下载放在子进程里跑（run_in_subprocess），避免撑大常驻scheduler的内存

用法：python -m src.overnight_lab [--summary]
"""
import json
import math
import os
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytz

ET = pytz.timezone("America/New_York")
_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
_FILE = os.path.join(_DATA, "overnight_lab.json")

RVOL_MIN = 1.8
TOP_K = 5
COST_ROUNDTRIP = 0.0010
MIN_TRADES = 200
POOL_V2_START = "2026-10-12"


def detect(closes: pd.DataFrame, volumes: pd.DataFrame, top_k: int = TOP_K) -> tuple:
    """
    最后一行是今天（盘中）。返回 (信号列表[(ticker, rvol, price)], 非信号但有效的ticker列表)（纯函数）。
    量比 = 今天成交量 / 前20日均量（不含今天）。
    """
    if len(closes) < 22:
        return [], []
    v_today, avg20 = volumes.iloc[-1], volumes.iloc[-21:-1].mean()
    c_today, c_prev = closes.iloc[-1], closes.iloc[-2]
    rvol = v_today / avg20
    valid = rvol.notna() & c_today.notna() & c_prev.notna() & (avg20 > 0)
    hit = valid & (rvol >= RVOL_MIN) & (c_today > c_prev)
    ranked = rvol[hit].sort_values(ascending=False).head(top_k)
    signals = [(t, round(float(r), 2), round(float(c_today[t]), 4)) for t, r in ranked.items()]
    others = [t for t in valid[valid].index if t not in set(ranked.index)]
    return signals, others


def fill_exits(trades: list, opens: pd.DataFrame) -> int:
    """给还没有卖出价的记录填下一个交易日开盘价（纯函数，原地修改）。"""
    n = 0
    idx = [d.date() if hasattr(d, "date") else d for d in opens.index]
    for tr in trades:
        if tr.get("exit") is not None:
            continue
        entry_day = date.fromisoformat(tr["date"])
        later = [i for i, d in enumerate(idx) if d > entry_day]
        if not later or tr["ticker"] not in opens.columns:
            continue
        px = opens[tr["ticker"]].iloc[later[0]]
        if pd.notna(px):
            tr["exit"] = round(float(px), 4)
            tr["exit_date"] = idx[later[0]].isoformat()
            tr["ret"] = tr["exit"] / tr["entry"] - 1 - COST_ROUNDTRIP
            n += 1
    return n


def _load() -> dict:
    try:
        with open(_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"trades": []}


def _save(state: dict):
    os.makedirs(_DATA, exist_ok=True)
    tmp = _FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    os.replace(tmp, _FILE)


def _download(tickers: list) -> pd.DataFrame:
    import yfinance as yf
    return yf.download(tickers, period="2mo", auto_adjust=False, progress=False, threads=True)


def pool(watchlist: list | None = None) -> list:
    """原股票池（同月度动量，去掉ETF）∪ 标普500 ∪ 纳指100。"""
    from .momentum_book import universe
    from .universe_large import large_cap_universe
    return sorted(set(universe(watchlist)) | set(large_cap_universe()))


def run_overnight_lab(watchlist: list | None = None, download=None, today: date | None = None) -> dict:
    """每天15:45：先给昨天的记录填开盘价，再记录今天的信号和对照。"""
    today = today or datetime.now(ET).date()
    download = download or _download
    raw = download(pool(watchlist))
    closes, volumes, opens = raw["Close"], raw["Volume"], raw["Open"]
    last_day = closes.index[-1].date() if hasattr(closes.index[-1], "date") else closes.index[-1]

    state = _load()
    filled = fill_exits(state["trades"], opens)
    if last_day != today:
        _save(state)
        return {"ok": True, "filled": filled, "new": [], "note": f"数据最后一天{last_day}不是今天，今天不记录"}
    if any(tr["date"] == today.isoformat() for tr in state["trades"]):
        _save(state)
        return {"ok": True, "filled": filled, "new": [], "note": "今天已记录"}

    signals, others = detect(closes, volumes)
    rng = np.random.default_rng(int(today.strftime("%Y%m%d")))
    controls = list(rng.choice(others, size=min(len(signals), len(others)), replace=False)) if others else []
    now = datetime.now(ET).isoformat(timespec="minutes")
    new = []
    for t, rvol, px in signals:
        state["trades"].append({"date": today.isoformat(), "at": now, "ticker": t, "group": "signal",
                                "rvol": rvol, "entry": px, "exit": None, "pool": "v2"})
        new.append(t)
    for t in controls:
        state["trades"].append({"date": today.isoformat(), "at": now, "ticker": str(t), "group": "control",
                                "entry": round(float(closes[t].iloc[-1]), 4), "exit": None, "pool": "v2"})
    _save(state)
    return {"ok": True, "filled": filled, "new": new, "controls": [str(t) for t in controls]}


def _mt(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return len(xs), (xs[0] if xs else None), None
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return len(xs), m, (m / (sd / math.sqrt(len(xs))) if sd > 0 else None)


def summarize(state: dict | None = None) -> str:
    """判定只用扩池后的v2样本；扩池前的v1单独列出。"""
    state = state or _load()
    f = lambda x: "—" if x is None else f"{x * 100:+.3f}%"
    v2 = [t for t in state["trades"] if t.get("pool") == "v2"]
    v1 = [t for t in state["trades"] if t.get("pool") != "v2"]
    n, m, tt = _mt([t.get("ret") for t in v2 if t["group"] == "signal"])
    nc, mc, _ = _mt([t.get("ret") for t in v2 if t["group"] == "control"])
    n1, m1, _ = _mt([t.get("ret") for t in v1 if t["group"] == "signal"])
    verdict = (f"样本未满{MIN_TRADES}，不下结论" if n < MIN_TRADES else
               ("✅ 按事先标准成立" if (m or 0) > 0 and (tt or 0) > 2 else "❌ 按事先标准不成立"))
    return (f"H4 隔夜放量（收盘前买、次日开盘卖，{POOL_V2_START}起约530只池）：{n}笔 平均{f(m)}（已扣成本） "
            f"t={'—' if tt is None else f'{tt:+.1f}'}  对照{nc}笔 平均{f(mc)}  {verdict}"
            + (f"｜扩池前旧池{n1}笔 平均{f(m1)}（不参与判定）" if n1 else ""))


def format_result(r: dict) -> str:
    return (f"填开盘价{r['filled']}笔，今日信号{r['new']}，对照{r.get('controls', [])}"
            + (f"（{r['note']}）" if r.get("note") else ""))


def run_in_subprocess(watchlist: list | None = None, timeout: int = 600) -> str:
    """供scheduler调用：约530只的日线下载放在子进程里，跑完退出释放内存（做法同event_lab）。"""
    import subprocess
    root = os.path.join(os.path.dirname(__file__), "..")
    args = [sys.executable, "-m", "src.overnight_lab"]
    if watchlist:
        args += ["--watchlist", ",".join(watchlist)]
    r = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()
        raise RuntimeError(tail[-1] if tail else f"exit {r.returncode}")
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""


if __name__ == "__main__":
    if "--summary" in sys.argv:
        print(summarize())
    else:
        wl = sys.argv[sys.argv.index("--watchlist") + 1].split(",") if "--watchlist" in sys.argv else None
        print(format_result(run_overnight_lab(wl)))
