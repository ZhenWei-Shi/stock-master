"""
发债/可转债公告事件门（Debt / Convertible Notes Offering Gate）

背景：2026-07-15 ASTS公告10亿美元可转债发行，收盘后1分钟内盘后暴跌约14%。
回溯ASTS历史4次可比发债事件，单日跌幅高度一致（-9%~-15%），机制是可转债
套利盘做空正股对冲delta敞口+稀释预期，属普遍规律而非个股巧合，非仅ASTS偶发。

数据源：复用 sec_13dg_monitor.py 的 SEC EDGAR EFTS 全文检索模板，
过滤条件从 13D/13G 表单换成 8-K + 新发行专用关键词短语。

集成方式：
  - scheduler.py 09:00/14:00 与 13D/G 监控一起跑一次 run_debt_event_monitor()，
    结果写入 data/debt_events_watchlist.json 快照
  - cold_model.py 的 debt_event gate 只读快照文件（不在决策路径内发起HTTP请求）：
      公告0-1天（当天/次日）→ 一票否决
      2-7天 → warn（观望/缩仓）
      7天以上 → 通过

已知局限（先记录，非本次范围）：
  1. 关键词为"新发行"专用短语（如"proposed offering of convertible"），
     刻意避开 redemption/repurchase 等回购措辞，但不是完整NLP分类，仍可能有漏报/误报
  2. 仅用 EDGAR file_date 计算天数；若公告发生在收盘后而8-K次日才提交，
     "天数"以SEC实际收文日为准，可能比新闻公告晚1天
  3. 这是"防止买在崩盘第一天"的风控闸门，不是"N天后抄底"的择时信号——
     历史案例（见ASTS 2026-02-11那次）显示完整消化可能耗时数月，
     不能把 warn/pass 状态当作抄底确认信号
"""

from __future__ import annotations   # Python 3.9 兼容 X | Y 类型注解

import os
import json
import re
import time
from datetime import datetime, timedelta, date
from urllib.parse import quote

import requests
import pytz

from .sec_13dg_monitor import _EDGAR_HEADERS

# EDGAR EFTS 返回的公司名格式固定为 "Company Name  (TICKER)  (CIK 0001234567)"，
# 直接从中提取ticker比 sec_13dg_monitor.py 里那套公司名模糊匹配（_company_key）更可靠，
# 且不依赖 company_tickers.json 缓存。
_TICKER_IN_DISPLAY_NAME = re.compile(r"\(([A-Z]{1,6})\)\s*\(CIK")

ET_TZ = pytz.timezone("America/New_York")

_DATA          = os.path.join(os.path.dirname(__file__), "..", "data")
_SNAPSHOT_FILE = os.path.join(_DATA, "debt_events_watchlist.json")

# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════

LOOKBACK_DAYS       = 10     # 覆盖否决(0-1天)+警示(2-7天)窗口，留余量防漏
MAX_RESULTS_PER_RUN = 20
REQUEST_DELAY       = 0.2    # EDGAR 请求间隔

VETO_DAYS = 1     # 公告当天/次日一票否决
WARN_DAYS = 7     # 2-7天内warn，之后pass

# 新发行专用查询（AND 组合两个短语，而非单一连续短语）。
# 原因：真实press release标题几乎总在"offering of"和"convertible/notes"之间
# 插入发行金额（如"Proposed Private Offering of $1.0 Billion of Convertible
# Senior Notes"），导致单一连续短语精确匹配（EDGAR q= 的引号短语要求逐词相邻）
# 完全落空——实测用ASTS 2026-07-15真实案例验证过这个坑。AND组合同时刻意
# 避开 redemption/repurchase/tender offer 等回购措辞，降低误报。
_OFFERING_QUERIES = [
    '"proposed offering" AND "convertible senior notes"',
    '"proposed private offering" AND "notes"',
    '"pricing of" AND "convertible senior notes"',
    '"proposed offering" AND "senior notes"',
]

# ══════════════════════════════════════════════════════════════


def _search_edgar_8k(query: str, startdt: str, enddt: str) -> list[dict]:
    """
    用 EDGAR EFTS API 检索最近8-K中匹配 query（含引号短语/AND）的申报。

    ⚠️ 两个实测确认的EDGAR EFTS坑（均用ASTS 2026-07-15真实可转债公告验证过）：
      1. 缺少 `enddt` 参数时，`dateRange=custom&startdt=...` 会被完全忽略，
         返回不受日期限制、按相关度排序的全历史结果（实测偏差达20+年）。
      2. `_source=` 字段过滤参数被忽略，且公司名字段是 `display_names`（数组，
         格式固定为"Company Name  (TICKER)  (CIK 0001234567)"），不是
         `entity_name`——旧写法 `_source.get("entity_name")` 实测恒为空字符串。
    """
    url = (
        "https://efts.sec.gov/LATEST/search-index"
        f"?q={quote(query)}"
        "&forms=8-K"
        f"&dateRange=custom&startdt={startdt}&enddt={enddt}"
    )
    try:
        r = requests.get(url, headers=_EDGAR_HEADERS, timeout=20)
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
    except Exception:
        return []

    results = []
    for h in hits[:MAX_RESULTS_PER_RUN]:
        src = h.get("_source", {})
        display_names = src.get("display_names") or []
        ticker = None
        for name in display_names:
            m = _TICKER_IN_DISPLAY_NAME.search(name)
            if m:
                ticker = m.group(1)
                break
        results.append({
            "accession":    h.get("_id", ""),
            "display_name": display_names[0] if display_names else "",
            "ticker":       ticker,
            "file_date":    src.get("file_date", ""),
            "query":        query,
        })
    return results


def _load_snapshot() -> dict:
    if not os.path.exists(_SNAPSHOT_FILE):
        return {}
    try:
        with open(_SNAPSHOT_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_snapshot(snapshot: dict):
    os.makedirs(_DATA, exist_ok=True)
    tmp = _SNAPSHOT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _SNAPSHOT_FILE)


def check_debt_events(watchlist: list[str]) -> dict:
    """
    扫描 watchlist 中每只股票近 LOOKBACK_DAYS 天内是否有发债/可转债发行8-K，
    更新 data/debt_events_watchlist.json 快照。

    返回：{ok, new_events}，new_events 是本次新发现（快照中此前没有）的事件，
    可直接用于 Telegram 推送。
    """
    if not watchlist:
        return {"ok": True, "new_events": []}

    today   = datetime.now(ET_TZ).date()
    startdt = (today - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    enddt   = today.strftime("%Y-%m-%d")
    watchlist_set = {t.upper() for t in watchlist}

    all_hits = []
    for query in _OFFERING_QUERIES:
        time.sleep(REQUEST_DELAY)
        all_hits.extend(_search_edgar_8k(query, startdt, enddt))

    # 按ticker匹配watchlist，每只股票只保留最新一条
    matched: dict[str, dict] = {}
    for h in all_hits:
        ticker = h.get("ticker")
        if not ticker or ticker not in watchlist_set:
            continue
        existing = matched.get(ticker)
        if not existing or h["file_date"] > existing["file_date"]:
            matched[ticker] = h

    snapshot = _load_snapshot()
    new_events = []
    for ticker, hit in matched.items():
        prev = snapshot.get(ticker)
        if not prev or prev.get("accession") != hit["accession"]:
            new_events.append({"ticker": ticker, **hit})
        snapshot[ticker] = {
            "accession":    hit["accession"],
            "display_name": hit["display_name"],
            "file_date":    hit["file_date"],
            "checked_at":   str(datetime.now(ET_TZ)),
        }

    _save_snapshot(snapshot)
    return {"ok": True, "new_events": new_events}


def grade_debt_event(days_since: int | None) -> dict:
    """
    纯函数：根据距离发债公告天数，给出gate分级（pass/warn/veto）。
    独立抽出，便于pytest直接测试评分边界，不依赖网络/文件IO。
    """
    if days_since is None or days_since < 0:
        return {"pass": True, "note": "近期无发债/可转债发行公告"}
    if days_since <= VETO_DAYS:
        return {
            "pass": False,
            "note": (f"{days_since}天前公告发债/可转债发行，冲击期禁止新开仓"
                     "（套利盘做空正股对冲delta敞口，历史单日跌幅约9%-15%）"),
        }
    if days_since <= WARN_DAYS:
        return {
            "pass": "warn",
            "note": f"{days_since}天前公告发债/可转债发行，短期波动期未完全消化，建议观望或缩仓",
        }
    return {"pass": True, "note": f"{days_since}天前发债公告，冲击期已过"}


def check_ticker_debt_event(ticker: str) -> dict:
    """
    从快照读取该ticker最近的发债事件（不发起HTTP请求，供 cold_model.py 高频调用）。
    快照缺失/过期/损坏时一律按"无事件"处理，不用陈旧数据一票否决。
    """
    snapshot = _load_snapshot()
    entry = snapshot.get(ticker.upper())
    if not entry:
        return grade_debt_event(None)
    try:
        file_date = date.fromisoformat(entry["file_date"])
        days_since = (datetime.now(ET_TZ).date() - file_date).days
    except Exception:
        return grade_debt_event(None)
    if days_since > LOOKBACK_DAYS + WARN_DAYS:
        return grade_debt_event(None)
    return grade_debt_event(days_since)


def format_debt_event_telegram(event: dict) -> str:
    """格式化单条发债事件为 Telegram HTML 消息。"""
    entity = event.get("display_name", "未知公司")
    file_d = event.get("file_date", "")
    ticker = event.get("ticker", "")
    return (
        f"💵 <b>发债/可转债发行公告</b>\n"
        f"<b>标的：{ticker}（{entity}）</b>\n"
        f"公告日期：{file_d}\n"
        f"⚠️ 历史规律：公告后单日跌幅常见-9%~-15%（可转债套利盘做空正股对冲delta敞口），"
        f"短期不建议新开仓，完整消化可能需数周至数月（大白话：这不是抄底信号，是避雷提醒）\n"
        f"🔗 EDGAR搜索：<code>{entity}</code> 8-K"
    )


def run_debt_event_monitor(watchlist: list[str] | None = None,
                           send_fn=None) -> dict:
    """执行一次扫描，推送新发现的发债/可转债事件。"""
    wl = watchlist or []
    if not wl:
        return {"ok": True, "new_count": 0, "pushed": [], "note": "watchlist为空，跳过"}

    result = check_debt_events(wl)
    pushed = []
    for ev in result.get("new_events", []):
        msg = format_debt_event_telegram(ev)
        if send_fn:
            try:
                send_fn(msg)
                time.sleep(0.5)
            except Exception:
                pass
        pushed.append(ev.get("ticker", ""))

    return {
        "ok":        True,
        "new_count": len(pushed),
        "pushed":    pushed,
        "note":      (f"发现 {len(pushed)} 条新发债/可转债公告并推送：{pushed}"
                      if pushed else "无新发债/可转债公告"),
    }
