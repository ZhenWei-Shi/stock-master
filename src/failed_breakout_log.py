"""
假突破前向样本记录器（2026-09-29新增，观察期：只记录，不推送、不参与打分）

背景：failed_breakout.py 的日K形态单独回测无预测力（见该文件顶部）。用户提出
伽马结构和个股消息可能是区分"真失败"和"噪音"的条件——例如2026-09-29 ASTS
盘中冲到$65.31即回落，$65恰好是正伽马最集中的行权价（GEX King），当天的冲高
又是8-K（高管控制权变更遣散政策）引发的收购猜测推动。

免费数据拿不到历史期权链，只能从现在起往前攒样本：每个交易日收盘后
（scheduler 16:20 ET）扫一遍 watchlist + 研究池，形态触发时记录：
  - 形态本身（failed_breakout.detect_failed_breakout 的结果）
  - 伽马结构（gex_cboe.calc_gex_cboe，失败时退回gex_scanner.calc_gex）：环境、GEX King、翻转点、冲高是否碰到
    上方正伽马墙、收盘在翻转点上方还是下方
  - 消息标签：当天/前一天是否有8-K（EDGAR submissions API）、Yahoo个股新闻
每次运行同时回填之前记录的1/3/5日后收盘涨跌。攒到30笔以上再用summarize()
按条件分组看有无优势，结论写回wiki（stock-master/features-and-gates）。
"""
import json
import os
from datetime import datetime, timedelta

import pytz
import requests
import yfinance as yf

from .failed_breakout import detect_failed_breakout

ET = pytz.timezone("America/New_York")
_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
_LOG_FILE = os.path.join(_DATA, "failed_breakout_log.json")

# 研究池：2026-09-29回测用的29只高beta/成长股。watchlist之外额外扫，
# 目的只是加快攒样本（单靠watchlist约每天1-2笔）。
RESEARCH_UNIVERSE = [
    "ASTS", "RKLB", "LUNR", "PL", "RDW", "NVDA", "AMD", "TSLA", "META", "MU",
    "PLTR", "SOFI", "HOOD", "COIN", "MSTR", "SMCI", "IONQ", "OKLO", "RIVN", "LCID",
    "NIO", "SNAP", "U", "AFRM", "UPST", "RBLX", "DKNG", "ROKU", "PINS",
]

WALL_TOUCH_PCT   = 1.5    # 最高价距伽马墙在此百分比以内（或越过）算"碰到"
WALL_MIN_SHARE   = 0.5    # 正GEX达到上方最大值的50%以上才算"墙"
FORWARD_DAYS     = (1, 3, 5)
FILING_LOOKBACK  = 1      # 8-K申报日期在交易日当天或前N个日历日内算"当天有公告"


# ─────────────────────────────────────────────────────────────
# 伽马结构
# ─────────────────────────────────────────────────────────────

def gamma_context(gex: dict, high: float, close: float, prev_close: float) -> dict:
    """
    从GEX结果提炼跟"冲高被压回"相关的字段（纯函数）。
    伽马墙 = 昨收上方、正GEX达到上方最大值WALL_MIN_SHARE以上的行权价（可能不止一个）。
    只认最大的一个会漏掉近处的墙：2026-09-29 ASTS 45天内最大墙在$70，
    当天冲高被压回的$65是第二大（约为$70的80%）。
    """
    if not gex or gex.get("error"):
        return {"ok": False, "note": (gex or {}).get("error", "无GEX数据")}

    by_strike = {float(k): v for k, v in (gex.get("gex_by_strike") or {}).items()}
    above = {k: v for k, v in by_strike.items() if k > prev_close and v > 0}
    walls = []
    if above:
        top = max(above.values())
        walls = sorted(k for k, v in above.items() if v >= top * WALL_MIN_SHARE)

    # 碰到的墙：最高价达到(墙价-容差)的墙里取最高的一堵
    hit = [w for w in walls if high >= w * (1 - WALL_TOUCH_PCT / 100)]
    wall = hit[-1] if hit else (walls[0] if walls else None)
    touched = bool(hit)
    rejected = touched and close < wall

    flip = gex.get("flip_strike")
    close_vs_flip = None if flip is None else ("above" if close >= flip else "below")

    return {
        "ok": True,
        "source": gex.get("source", "yfinance"),
        "env": gex.get("gex_env"),
        "total_gex_m": gex.get("total_gex_m"),
        "gex_king": gex.get("gex_king"),
        "call_walls": walls,
        "call_wall": wall,
        "call_wall_m": round(above[wall], 2) if wall is not None else None,
        "touched_wall": touched,
        "rejected_at_wall": rejected,
        "flip_strike": flip,
        "close_vs_flip": close_vs_flip,
        "put_wall": gex.get("put_wall"),
        "pc_ratio": gex.get("pc_ratio"),
        "iv30": gex.get("iv30"),
    }


# ─────────────────────────────────────────────────────────────
# 消息标签
# ─────────────────────────────────────────────────────────────

# 8-K item编号 → 含义（SEC Form 8-K General Instructions B）。只收常见项；
# 用于在样本里区分"实质性事件"和例行公告（2026-09-29 ASTS：5.02高管变动）。
ITEM_NAMES = {
    "1.01": "签订重大协议", "1.02": "终止重大协议", "1.03": "破产/接管",
    "2.01": "完成收购/出售资产", "2.02": "业绩发布", "2.03": "新增重大债务",
    "2.04": "债务加速到期", "2.05": "退出/重组成本", "2.06": "重大资产减值",
    "3.01": "退市/不符合上市标准", "3.02": "未注册股票发行", "3.03": "股东权利变更",
    "4.01": "更换审计师", "4.02": "财报不可依赖（需重述）",
    "5.01": "控制权变更", "5.02": "高管/董事变动或薪酬安排", "5.03": "章程修改",
    "5.07": "股东大会投票结果", "7.01": "Reg FD披露", "8.01": "其他事件", "9.01": "财务报表与附件",
}
# 例行/附属项：只有这些item时不算"实质性"事件
ROUTINE_ITEMS = {"7.01", "8.01", "9.01", "5.07"}


def parse_recent_8k(recent: dict, cik: str, trade_date: str) -> list:
    """从EDGAR submissions的filings.recent里挑出交易日当天或前FILING_LOOKBACK天的8-K（纯函数）。"""
    start = (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=FILING_LOOKBACK)).strftime("%Y-%m-%d")
    cols = {k: recent.get(k, []) for k in ("form", "filingDate", "items", "accessionNumber", "primaryDocument")}
    out = []
    for i, form in enumerate(cols["form"]):
        get = lambda k: cols[k][i] if i < len(cols[k]) else ""
        d = get("filingDate")
        if form not in ("8-K", "8-K/A") or not (start <= d <= trade_date):
            continue
        items = [x.strip() for x in (get("items") or "").split(",") if x.strip()]
        acc = get("accessionNumber").replace("-", "")
        out.append({
            "date": d,
            "items": ",".join(items),
            "labels": [ITEM_NAMES.get(x, x) for x in items],
            "material": any(x not in ROUTINE_ITEMS for x in items),
            "url": (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{get('primaryDocument')}"
                    if acc and get("primaryDocument") else None),
        })
    return out


def _recent_8k(ticker: str, trade_date: str) -> list:
    """交易日当天或前FILING_LOOKBACK天内的8-K，附item含义和正文链接。"""
    try:
        from .insider_tracker import get_cik, _EDGAR_HEADERS
        cik = get_cik(ticker)
        if not cik:
            return []
        url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
        r = requests.get(url, headers=_EDGAR_HEADERS, timeout=20)
        r.raise_for_status()
        recent = r.json().get("filings", {}).get("recent", {})
    except Exception:
        return []
    return parse_recent_8k(recent, cik, trade_date)


def news_context(ticker: str, trade_date: str) -> dict:
    filings = _recent_8k(ticker, trade_date)
    news = {}
    try:
        from .news_aggregator import get_news_for_ticker
        n = get_news_for_ticker(ticker, hours=24)
        news = {
            "article_count": n.get("article_count", 0),
            "sentiment": n.get("sentiment"),
            "top_headline": (n.get("top_headline") or "")[:120],
        }
    except Exception as e:
        news = {"news_error": str(e)[:120]}
    return {"has_8k": bool(filings), "has_material_8k": any(f["material"] for f in filings),
            "filings_8k": filings, **news}


# ─────────────────────────────────────────────────────────────
# 存取
# ─────────────────────────────────────────────────────────────

def _load() -> list:
    try:
        with open(_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def _save(records: list):
    os.makedirs(_DATA, exist_ok=True)
    tmp = _LOG_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    os.replace(tmp, _LOG_FILE)


def fill_forward_returns(record: dict, hist) -> bool:
    """用交易日之后的收盘价回填f1/f3/f5（纯函数，hist为日线DataFrame）。返回是否有更新。"""
    dates = [d.strftime("%Y-%m-%d") for d in hist.index]
    if record["date"] not in dates:
        return False
    i = dates.index(record["date"])
    base = record["close"]
    changed = False
    for k in FORWARD_DAYS:
        key = f"f{k}"
        if record.get(key) is None and i + k < len(dates):
            record[key] = round((float(hist["Close"].iloc[i + k]) / base - 1) * 100, 2)
            changed = True
    return changed


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────

def run_failed_breakout_log(watchlist: list | None = None) -> dict:
    today = datetime.now(ET).strftime("%Y-%m-%d")
    universe = sorted(set(t.upper() for t in (watchlist or [])) | set(RESEARCH_UNIVERSE))
    records = _load()
    seen = {(r["date"], r["ticker"]) for r in records}
    pending = {r["ticker"] for r in records if any(r.get(f"f{k}") is None for k in FORWARD_DAYS)}

    new, filled, errors = [], 0, []
    for t in sorted(set(universe) | pending):
        try:
            hist = yf.Ticker(t).history(period="1y")
            if hist.empty:
                continue
        except Exception as e:
            errors.append(f"{t}: {e}")
            continue

        for r in records:
            if r["ticker"] == t and fill_forward_returns(r, hist):
                filled += 1

        if t not in universe:
            continue
        bar_date = hist.index[-1].strftime("%Y-%m-%d")
        if bar_date != today or (today, t) in seen or len(hist) < 2:
            continue
        fb = detect_failed_breakout(hist)
        if not fb["triggered"]:
            continue

        high = float(hist["High"].iloc[-1])
        close = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2])
        # 优先用CBOE期权链（自带IV/gamma，翻转点算法更完整），失败时退回yfinance版本。
        # 两者GEX数值口径不同（yfinance版约为CBOE版的100倍），墙按相对比例判断不受影响，
        # 但分析时应按gamma["source"]分开看绝对值。
        try:
            from .gex_cboe import calc_gex_cboe
            gex = calc_gex_cboe(t)
            if gex.get("error"):
                from .gex_scanner import calc_gex
                gex = calc_gex(t)
            gamma = gamma_context(gex, high, close, prev_close)
        except Exception as e:
            gamma = {"ok": False, "note": str(e)}

        ma200 = hist["Close"].rolling(200).mean().iloc[-1]
        new.append({
            "date": today,
            "ticker": t,
            "close": round(close, 2),
            "high": round(high, 2),
            "prev_close": round(prev_close, 2),
            "below_ma200": bool(close < ma200) if ma200 == ma200 else None,
            "in_watchlist": t in {w.upper() for w in (watchlist or [])},
            "signals": fb["signals"],
            "close_pos": fb["close_pos"],
            "vol_ratio": fb["vol_ratio"],
            "gamma": gamma,
            "news": news_context(t, today),
            **{f"f{k}": None for k in FORWARD_DAYS},
        })

    records.extend(new)
    _save(records)
    return {"ok": True, "new": [r["ticker"] for r in new], "filled": filled,
            "total": len(records), "errors": errors}


# ─────────────────────────────────────────────────────────────
# 分组统计（样本够了再看）
# ─────────────────────────────────────────────────────────────

def summarize(records: list | None = None, horizon: int = 3) -> list:
    """按条件分组统计f{horizon}：返回[(分组名, 样本数, 平均涨跌%, 下跌比例%)]。"""
    records = _load() if records is None else records
    key = f"f{horizon}"
    done = [r for r in records if r.get(key) is not None]

    groups = {
        "全部触发": lambda r: True,
        "碰到伽马墙后收回": lambda r: r["gamma"].get("rejected_at_wall") is True,
        "未碰伽马墙": lambda r: r["gamma"].get("ok") and not r["gamma"].get("touched_wall"),
        "正伽马环境": lambda r: r["gamma"].get("env") == "正伽马",
        "负伽马环境": lambda r: r["gamma"].get("env") == "负伽马",
        "收盘在翻转点下方": lambda r: r["gamma"].get("close_vs_flip") == "below",
        "当天有8-K": lambda r: r["news"].get("has_8k") is True,
        "无8-K": lambda r: r["news"].get("has_8k") is False,
        "实质性8-K（非例行公告）": lambda r: r["news"].get("has_material_8k") is True,
        "伽马墙+8-K": lambda r: r["gamma"].get("rejected_at_wall") is True and r["news"].get("has_8k") is True,
        "低于MA200": lambda r: r.get("below_ma200") is True,
    }
    out = []
    for name, cond in groups.items():
        xs = [r[key] for r in done if cond(r)]
        if xs:
            out.append((name, len(xs), round(sum(xs) / len(xs), 2),
                        round(sum(1 for x in xs if x < 0) / len(xs) * 100)))
        else:
            out.append((name, 0, None, None))
    return out


if __name__ == "__main__":
    import sys
    if "--summary" in sys.argv:
        for h in FORWARD_DAYS:
            print(f"\n== {h}日后 ==")
            for name, n, avg, dn in summarize(horizon=h):
                print(f"{name:12s} n={n:3d}  平均{avg if avg is not None else '-':>6}%  下跌{dn if dn is not None else '-'}%")
    else:
        print(run_failed_breakout_log())
