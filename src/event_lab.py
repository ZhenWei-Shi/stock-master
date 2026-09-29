"""
事件实验室（2026-09-29新增）：财报事件的期权假设，前向记录、事先定好判断标准

背景：九关降级为风控层后，核心策略转向"事件驱动 + 前向验证"。GitHub调研
（wiki stock-master/options-log"财报期权策略的外部证据"）显示：财报期权里略占优势
的是卖方且扣成本后几乎为零（quanttqueensu/earnings_iv_crush）；唯一显示买方赚钱
的是"财报前买跨式、公布前卖出"（jakehanson/Pre-earnings-Straddle，2012-2018）。
免费数据拿不到历史期权报价，只能从现在起用CBOE延迟报价逐个事件记录。

事先登记的假设（数据出来之前定好，不事后改标准）：
  H1 财报前买跨式：退出日前ENTRY_SESSIONS个交易日15:50按ask买入ATM跨式（到期日在
     财报之后的最近一期），在财报前最后一个交易日15:50按bid卖出同一对合约。
     成立标准：≥60个事件后，扣买卖价差的平均收益>0且t>2
  H2 隐含 vs 实际波动：财报前最后一个交易日的ATM跨式中间价/现价 = 隐含波动幅度；
     财报后第一个交易日15:50的现价相对前一日的涨跌绝对值 = 实际幅度。只做描述统计
  H3 卖跨式扛财报：H2那对合约财报前按bid卖出、财报后第一个交易日按ask买回。
     成立标准：≥100个事件后，扣买卖价差的平均收益（占收到的权利金）>0且t>2

交易日规则（按财报时间是否确认）：
  退出日(exit) = 财报日期之前的最后一个交易日——不论盘前盘后都安全，AMC会少赚
                 财报当天白天的IV上升，换来不会误扛财报
  财报后(post) = BMO：财报当天；AMC或时间未确认：财报日期之后的第一个交易日
  yfinance里时间为15:00的是未确认的占位日期，只记录不进入实验

只记录、不下单；Alpaca配好后可在此基础上加模拟下单（先过risk_layer）。

用法：python -m src.event_lab [--summary]
"""
import json
import math
import os
import sys
from datetime import date, datetime, timedelta

import pandas as pd
import pytz

ET = pytz.timezone("America/New_York")
_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
_FILE = os.path.join(_DATA, "event_lab.json")

ENTRY_SESSIONS = 5            # H1：退出日前5个交易日入场（约一周）
CALENDAR_MAX_AGE_DAYS = 3     # 财报日历缓存多久全量刷新一次
LOOKAHEAD_SESSIONS = 12       # 只跟踪未来12个交易日内的财报
H1_MIN_EVENTS, H3_MIN_EVENTS = 60, 100


# ─────────────────────────────────────────────────────────────
# 交易日与时间规则（纯函数）
# ─────────────────────────────────────────────────────────────

def sessions_between(start: date, end: date) -> list:
    """[start, end]之间的NYSE交易日。"""
    from .fetcher import NYSEHolidayCalendar
    hol = NYSEHolidayCalendar().holidays(pd.Timestamp(start) - timedelta(days=10), pd.Timestamp(end) + timedelta(days=10))
    return [d.date() for d in pd.bdate_range(start, end, freq="C", holidays=hol)]


def classify_timing(ts: pd.Timestamp) -> str:
    """yfinance财报时间 → BMO/AMC/unconfirmed。15:00是未确认的占位值。"""
    t = ts.tz_convert(ET) if ts.tzinfo else ts
    if t.hour >= 16:
        return "AMC"
    if t.hour < 10:
        return "BMO"
    return "unconfirmed"


def schedule_for(event_date: date, timing: str) -> dict:
    """算出entry/exit/post三个交易日。"""
    before = sessions_between(event_date - timedelta(days=30), event_date - timedelta(days=1))
    after = sessions_between(event_date, event_date + timedelta(days=10))
    exit_s = before[-1]
    entry_s = before[-1 - ENTRY_SESSIONS] if len(before) > ENTRY_SESSIONS else None
    if timing == "BMO" and after and after[0] == event_date:
        post_s = event_date
    else:
        post_s = next(d for d in after if d > event_date)
    return {"entry": entry_s, "exit": exit_s, "post": post_s}


# ─────────────────────────────────────────────────────────────
# 期权快照（纯函数：输入CBOE的data）
# ─────────────────────────────────────────────────────────────

def pick_straddle(data: dict, min_expiry: date) -> dict | None:
    """到期日≥min_expiry的最近一期里，同时有Call和Put的最接近现价的行权价。"""
    from .gex_cboe import parse_occ
    spot = float(data.get("current_price") or data.get("close") or 0)
    if spot <= 0:
        return None
    legs = {}
    for o in data.get("options", []):
        p = parse_occ(o.get("option", ""))
        if not p or p[0] < min_expiry:
            continue
        legs.setdefault((p[0], p[2]), {})[p[1]] = o
    pairs = [(k, v) for k, v in legs.items() if "C" in v and "P" in v]
    if not pairs:
        return None
    exp = min(k[0] for k, _ in pairs)
    (_, strike), leg = min(((k, v) for k, v in pairs if k[0] == exp), key=lambda kv: abs(kv[0][1] - spot))
    return {"expiry": exp.isoformat(), "strike": strike, "spot": spot,
            "call": leg["C"]["option"], "put": leg["P"]["option"]}


def quote_pair(data: dict, call_sym: str, put_sym: str) -> dict | None:
    by_sym = {o.get("option"): o for o in data.get("options", [])}
    c, p = by_sym.get(call_sym), by_sym.get(put_sym)
    if not c or not p:
        return None
    f = lambda o, k: float(o.get(k) or 0)
    return {"call_bid": f(c, "bid"), "call_ask": f(c, "ask"), "put_bid": f(p, "bid"), "put_ask": f(p, "ask"),
            "call_iv": f(c, "iv"), "put_iv": f(p, "iv"),
            "bid": f(c, "bid") + f(p, "bid"), "ask": f(c, "ask") + f(p, "ask"),
            "mid": (f(c, "bid") + f(c, "ask") + f(p, "bid") + f(p, "ask")) / 2,
            "spot": float(data.get("current_price") or 0)}


def snapshot(data: dict, stage: str, ev: dict) -> dict:
    """按阶段取需要的报价，写进事件记录（返回更新后的ev）。"""
    now = datetime.now(ET).isoformat(timespec="minutes")
    snap = {"at": now, "spot": float(data.get("current_price") or 0), "iv30": data.get("iv30")}
    min_exp = date.fromisoformat(ev["post"])
    if stage == "entry":
        pick = pick_straddle(data, min_exp)
        if pick:
            ev["h1_contract"] = pick
            snap["h1"] = quote_pair(data, pick["call"], pick["put"])
    elif stage == "exit":
        if ev.get("h1_contract"):
            snap["h1"] = quote_pair(data, ev["h1_contract"]["call"], ev["h1_contract"]["put"])
        pick = pick_straddle(data, min_exp)
        if pick:
            ev["h23_contract"] = pick
            snap["h23"] = quote_pair(data, pick["call"], pick["put"])
    elif stage == "post" and ev.get("h23_contract"):
        snap["h23"] = quote_pair(data, ev["h23_contract"]["call"], ev["h23_contract"]["put"])
    ev.setdefault("snapshots", {})[stage] = snap
    return ev


def event_results(ev: dict) -> dict:
    """由快照算三个假设的结果；缺数据的项为None（纯函数）。"""
    s = ev.get("snapshots", {})
    out = {"h1_ret": None, "h1_ret_mid": None, "implied_move": None, "realized_move": None,
           "h3_ret": None, "iv_crush": None}
    h1_in, h1_out = (s.get("entry") or {}).get("h1"), (s.get("exit") or {}).get("h1")
    if h1_in and h1_out and h1_in["ask"] > 0 and h1_in["mid"] > 0:
        out["h1_ret"] = h1_out["bid"] / h1_in["ask"] - 1
        out["h1_ret_mid"] = h1_out["mid"] / h1_in["mid"] - 1
    pre, post = (s.get("exit") or {}).get("h23"), (s.get("post") or {}).get("h23")
    if pre and pre["spot"] > 0 and pre["mid"] > 0:
        out["implied_move"] = pre["mid"] / pre["spot"]
    if pre and post and pre["spot"] > 0 and post["spot"] > 0:
        out["realized_move"] = abs(post["spot"] / pre["spot"] - 1)
    if pre and post and pre["bid"] > 0:
        out["h3_ret"] = (pre["bid"] - post["ask"]) / pre["bid"]
        out["iv_crush"] = ((pre["call_iv"] + pre["put_iv"]) - (post["call_iv"] + post["put_iv"])) / 2
    return out


# ─────────────────────────────────────────────────────────────
# 存取与日历
# ─────────────────────────────────────────────────────────────

def _load() -> dict:
    try:
        with open(_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"calendar": {}, "calendar_updated": None, "events": {}}


def _save(state: dict):
    os.makedirs(_DATA, exist_ok=True)
    tmp = _FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, _FILE)


def default_universe(watchlist: list | None = None) -> list:
    from .failed_breakout_log import RESEARCH_UNIVERSE
    from .sector_rotation import SECTOR_TICKERS
    u = set(RESEARCH_UNIVERSE) | {t.upper() for t in (watchlist or [])}
    for lst in SECTOR_TICKERS.values():
        u |= set(lst)
    return sorted(u - {"QQQ", "SPY"})


def next_earnings(ticker: str) -> dict | None:
    import yfinance as yf
    ed = yf.Ticker(ticker).get_earnings_dates(limit=4)
    if ed is None or ed.empty:
        return None
    now = pd.Timestamp.now(tz=ET)
    fut = ed[ed.index > now - pd.Timedelta(days=1)].sort_index()
    if fut.empty:
        return None
    ts = fut.index[0]
    return {"ts": ts.isoformat(), "date": ts.tz_convert(ET).date().isoformat(), "timing": classify_timing(ts)}


def refresh_calendar(state: dict, universe: list, force: bool = False) -> int:
    upd = state.get("calendar_updated")
    fresh = upd and (datetime.now(ET) - datetime.fromisoformat(upd)).days < CALENDAR_MAX_AGE_DAYS
    today = datetime.now(ET).date()
    soon = [t for t, c in state.get("calendar", {}).items()
            if c and 0 <= (date.fromisoformat(c["date"]) - today).days <= 16]
    todo = universe if (force or not fresh) else soon
    for t in todo:
        try:
            state["calendar"][t] = next_earnings(t)
        except Exception:
            continue
    if force or not fresh:
        state["calendar_updated"] = datetime.now(ET).isoformat()
    return len(todo)


# ─────────────────────────────────────────────────────────────
# 每日任务
# ─────────────────────────────────────────────────────────────

def run_event_lab(watchlist: list | None = None, fetch=None, today: date | None = None) -> dict:
    """每天15:50运行：更新日历 → 登记新事件 → 按阶段拍快照 → 算结果。"""
    from .gex_cboe import fetch_cboe_chain
    fetch = fetch or fetch_cboe_chain
    today = today or datetime.now(ET).date()
    state = _load()
    refreshed = refresh_calendar(state, default_universe(watchlist))

    horizon = sessions_between(today, today + timedelta(days=30))[:LOOKAHEAD_SESSIONS + 1]
    registered = []
    for t, c in state["calendar"].items():
        if not c or c["timing"] == "unconfirmed":
            continue
        ev_date = date.fromisoformat(c["date"])
        key = f"{t}_{c['date']}"
        if key in state["events"] or not horizon or ev_date > horizon[-1] or ev_date < today:
            continue
        sch = schedule_for(ev_date, c["timing"])
        state["events"][key] = {"ticker": t, "event_date": c["date"], "timing": c["timing"], "ts": c["ts"],
                                **{k: (v.isoformat() if v else None) for k, v in sch.items()},
                                "registered": today.isoformat()}
        registered.append(key)

    cancelled = cancel_moved_events(state, today)

    done = []
    for key, ev in state["events"].items():
        if ev.get("cancelled"):
            continue
        stages = [st for st in ("entry", "exit", "post")
                  if ev.get(st) == today.isoformat() and st not in ev.get("snapshots", {})]
        if not stages:
            continue
        try:
            data = fetch(ev["ticker"])
        except Exception as e:
            ev.setdefault("errors", []).append(f"{today}: {e}")
            continue
        for st in stages:
            snapshot(data, st, ev)
            done.append(f"{ev['ticker']}:{st}")
        ev["results"] = event_results(ev)

    _save(state)
    return {"ok": True, "calendar_refreshed": refreshed, "registered": registered, "snapshots": done,
            "cancelled": cancelled, "tracked": len(state["events"])}


def cancel_moved_events(state: dict, today: date) -> list:
    """
    公司改了财报日期：日期还没到、日历上的确认日期已变的事件作废（原地修改state）。
    否则旧事件会在错误的日子继续拍快照，污染H1/H3统计（2026-09-29重启前清查发现）。
    已经过去、只是漏拍了财报后快照的事件不动——它们的H2隐含幅度仍然有效。
    """
    out = []
    for key, ev in state["events"].items():
        if ev.get("cancelled") or ev["event_date"] < today.isoformat():
            continue
        cal = state.get("calendar", {}).get(ev["ticker"])
        if not cal or cal.get("timing") == "unconfirmed" or cal["date"] == ev["event_date"]:
            continue
        ev["cancelled"] = f"{today}: 财报日期由{ev['event_date']}改为{cal['date']}"
        ev["results"] = {}
        out.append(key)
    return out


# ─────────────────────────────────────────────────────────────
# 汇总
# ─────────────────────────────────────────────────────────────

def _mt(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if len(xs) < 2:
        return len(xs), (xs[0] if xs else None), None
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return len(xs), m, (m / (sd / math.sqrt(len(xs))) if sd > 0 else None)


def summarize(state: dict | None = None) -> str:
    state = state or _load()
    res = [e.get("results") or {} for e in state["events"].values() if not e.get("cancelled")]
    n1, m1, t1 = _mt([r.get("h1_ret") for r in res])
    _, m1m, _ = _mt([r.get("h1_ret_mid") for r in res])
    n3, m3, t3 = _mt([r.get("h3_ret") for r in res])
    pairs = [(r["implied_move"], r["realized_move"]) for r in res
             if r.get("implied_move") and r.get("realized_move") is not None]
    fmt = lambda x: "—" if x is None else f"{x * 100:+.2f}%"
    ft = lambda t: "—" if t is None else f"{t:+.1f}"
    lines = [f"🧪 <b>事件实验室</b>  已登记{len(state['events'])}个财报事件",
             f"H1 财报前买跨式：{n1}笔  平均{fmt(m1)}（中间价{fmt(m1m)}） t={ft(t1)}  "
             f"{'达到判断样本量' if n1 >= H1_MIN_EVENTS else f'样本未满{H1_MIN_EVENTS}，不下结论'}",
             f"H3 卖跨式扛财报：{n3}笔  平均{fmt(m3)}（占权利金） t={ft(t3)}  "
             f"{'达到判断样本量' if n3 >= H3_MIN_EVENTS else f'样本未满{H3_MIN_EVENTS}，不下结论'}"]
    if pairs:
        imp = sum(p[0] for p in pairs) / len(pairs)
        real = sum(p[1] for p in pairs) / len(pairs)
        over = sum(1 for p in pairs if p[1] < p[0]) / len(pairs)
        lines.append(f"H2 {len(pairs)}个事件：平均隐含幅度{imp * 100:.1f}% vs 实际{real * 100:.1f}%，"
                     f"实际小于隐含的占{over * 100:.0f}%")
    today = datetime.now(ET).date().isoformat()
    upcoming = sorted((e["event_date"], e["ticker"], e["timing"]) for e in state["events"].values()
                      if e["event_date"] >= today and not e.get("cancelled"))[:8]
    if upcoming:
        lines.append("即将到来：" + "，".join(f"{t} {d[5:]} {tm}" for d, t, tm in upcoming))
    return "\n".join(lines)


def run_in_subprocess(watchlist: list | None = None, timeout: int = 900) -> str:
    """
    供scheduler调用：子进程运行，跑完退出释放内存。2026-09-29服务器实测：刷新105只
    股票的财报日历让进程内存从173MB涨到337MB，而服务器可用内存约345MB，不能在
    常驻的scheduler进程里跑（做法同performance_report.run_in_subprocess）。
    """
    import subprocess
    root = os.path.join(os.path.dirname(__file__), "..")
    args = [sys.executable, "-m", "src.event_lab"]
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
        r = run_event_lab(wl)
        print(f"日历刷新{r['calendar_refreshed']}只，新登记{r['registered']}，快照{r['snapshots']}，"
              f"作废{r['cancelled']}，累计{r['tracked']}个事件")
