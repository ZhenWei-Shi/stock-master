"""
SEC 13D/G 机构大仓监控

触发条件：任何机构持股超过5%时，必须在72小时内向SEC提交 13D 或 13G。
  - 13D：主动型持股（可能推动管理层变革或并购，强烈买入信号）
  - 13G：被动型持股（指数基金/ETF流入，温和信号）
  - 13D/A, 13G/A：持仓变动修正（增减仓均有意义）

数据来源：SEC EDGAR EFTS 全文检索 API（免费，实时）

集成方式：
  - scheduler.py 每天09:00和14:00各查一次全局列表
  - 有新申报时推送 Telegram 通知（带出价暗示分析）
  - long_hold._score_insider() 也可查询此数据
"""

from __future__ import annotations   # Python 3.9 兼容 X | Y 类型注解

import os
import json
import time
import threading
import requests
from datetime import datetime, timedelta
from urllib.parse import quote
import pytz

_13DG_LOCK = threading.Lock()   # 防止 scheduler 定时任务与 Telegram /13dg 命令并发写入 seen 文件

ET_TZ = pytz.timezone("America/New_York")

_DATA        = os.path.join(os.path.dirname(__file__), "..", "data")
_SEEN_FILE   = os.path.join(_DATA, "sec_13dg_seen.json")   # 已推送的申报，防重复
_NAMES_FILE  = os.path.join(_DATA, "sec_ticker_names.json") # ticker→公司全名（7天缓存）

_EDGAR_HEADERS = {
    "User-Agent": "stock-master-personal 1758162368szw@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════

LOOKBACK_DAYS       = 3      # 检查最近N天的新申报（防止漏掉周末）
FORMS_TO_WATCH      = ["SC 13D", "SC 13G", "SC 13D/A", "SC 13G/A"]
MAX_RESULTS_PER_RUN = 20     # 单次最多处理N条结果
REQUEST_DELAY       = 0.2    # EDGAR 请求间隔

# ══════════════════════════════════════════════════════════════


def _load_seen() -> set:
    """加载已推送过的申报 accession number 集合。"""
    if not os.path.exists(_SEEN_FILE):
        return set()
    try:
        with open(_SEEN_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("seen", []))
    except Exception:
        return set()


def _save_seen(seen: set):
    os.makedirs(_DATA, exist_ok=True)
    seen_list = sorted(seen)[-500:]
    tmp = _SEEN_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"seen": seen_list, "updated": str(datetime.now(ET_TZ))},
                  f, ensure_ascii=False, indent=2)
    os.replace(tmp, _SEEN_FILE)


def _ticker_name_map() -> dict:
    """从 EDGAR company_tickers.json 构建 ticker→公司全名 映射（7天缓存）。"""
    if os.path.exists(_NAMES_FILE):
        if time.time() - os.path.getmtime(_NAMES_FILE) < 7 * 86400:
            try:
                with open(_NAMES_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json",
                         headers=_EDGAR_HEADERS, timeout=20)
        r.raise_for_status()
        mapping = {v["ticker"].upper(): v.get("title", "").upper()
                   for v in r.json().values()}
        os.makedirs(_DATA, exist_ok=True)
        tmp = _NAMES_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(mapping, f)
        os.replace(tmp, _NAMES_FILE)
        return mapping
    except Exception:
        return {}


_COMPANY_STOP_WORDS = {"INC", "CORP", "LLC", "LTD", "CO", "THE", "GROUP",
                       "HOLDINGS", "HOLDING", "PLC", "NV", "SA", "AG", "SE"}


def _company_key(name: str) -> str:
    """提取公司名核心词（去后缀/停用词），用于模糊匹配。"""
    words = name.upper().rstrip(".").split()
    key_words = [w for w in words if w not in _COMPANY_STOP_WORDS and len(w) > 2]
    return " ".join(key_words[:2])   # 取前2个关键词


def _matches_watchlist(entity_name: str, watchlist: list[str]) -> bool:
    """检查 entity_name（EDGAR 申报中的受益公司名）是否对应 watchlist 中的某只股票。"""
    name_map = _ticker_name_map()
    entity_key = _company_key(entity_name)
    if not entity_key:
        return False
    for ticker in watchlist:
        company = name_map.get(ticker.upper(), "")
        if not company:
            continue
        ck = _company_key(company)
        if ck and (ck in entity_key or entity_key in ck):
            return True
    return False


def _search_edgar(form_type: str, days: int = LOOKBACK_DAYS) -> list[dict]:
    """
    用 EDGAR EFTS API 检索最近申报。

    返回：[{accession, entity_name, file_date, form_type}]
    """
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    # 只用 forms= 过滤表单类型，不用 q= 全文搜索（会匹配到无关文档）
    url = (
        "https://efts.sec.gov/LATEST/search-index"
        f"?forms={quote(form_type, safe='')}"
        f"&dateRange=custom&startdt={cutoff}"
        "&_source=entity_name,file_date,accession_no"
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
        results.append({
            "accession":   h.get("_id", ""),
            "entity_name": src.get("entity_name", ""),
            "file_date":   src.get("file_date", ""),
            "form_type":   form_type,
        })
    return results


def _get_filing_detail(accession: str) -> dict:
    """
    从申报 index 提取：被持股公司（issuer）、持股方（filer）、持股比例。
    """
    # accession 格式：0001234567-26-012345
    # index URL 格式：https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{acc}-index.json
    # 但 efts 结果里 _id 格式不固定，先尝试 EDGAR 全文搜索 secondary
    try:
        acc_clean = accession.replace("-", "")
        # 从 accession 中提取 CIK（前10位）
        parts = accession.split("-")
        if len(parts) < 1:
            return {}
        cik_part = parts[0].lstrip("0") or "0"
        idx_url  = (f"https://www.sec.gov/Archives/edgar/data/{cik_part}"
                    f"/{acc_clean}/{accession}-index.json")
        ir = requests.get(idx_url, headers=_EDGAR_HEADERS, timeout=15)
        if ir.status_code != 200:
            return {}
        idx = ir.json()
        company = idx.get("company-name", "")
        # 试图找 .txt 主文档
        items = idx.get("directory", {}).get("item", [])
        txt_files = [i["name"] for i in items if i["name"].endswith(".txt")]
        return {"company": company, "doc_count": len(items), "txt": txt_files[:1]}
    except Exception:
        return {}


def check_new_13dg(watchlist: list[str] | None = None) -> list[dict]:
    """
    查询 EDGAR 最近的 13D/G 申报。

    参数：
      watchlist — 如果提供，只返回与 watchlist 中股票相关的申报
                  如果为 None，返回全市场所有新申报（用于广泛监控）

    返回：新申报列表，每项含格式化信息，可直接推 Telegram。
    """
    with _13DG_LOCK:   # 防止 scheduler(09:00/14:00) 与 /13dg Telegram 命令并发覆盖 seen 文件
        seen     = _load_seen()
        new_ones = []

        for form_type in FORMS_TO_WATCH:
            time.sleep(REQUEST_DELAY)
            results = _search_edgar(form_type, LOOKBACK_DAYS)
            for r in results:
                acc = r["accession"]
                if acc in seen:
                    continue

                # watchlist 过滤：用公司全名匹配（ticker与公司名无子串关系，不能直接比）
                if watchlist and not _matches_watchlist(r["entity_name"], watchlist):
                    continue

                new_ones.append(r)
                seen.add(acc)

        _save_seen(seen)
        return new_ones


def format_13dg_telegram(filing: dict) -> str:
    """格式化单条13D/G为 Telegram HTML 消息。"""
    form  = filing.get("form_type", "13D/G")
    entity = filing.get("entity_name", "未知公司")
    date  = filing.get("file_date", "")

    is_activist = "13D" in form and "/A" not in form  # 新建的13D最有意义
    is_amend    = "/A" in form
    icon = "🐳" if is_activist else ("📊" if "13G" in form else "📋")

    type_desc = {
        "SC 13D":   "新进主动型大仓（≥5%，可能推动变革/并购）",
        "SC 13G":   "新进被动型大仓（≥5%，指数/ETF建仓）",
        "SC 13D/A": "主动型仓位变动修正",
        "SC 13G/A": "被动型仓位变动修正",
    }.get(form, form)

    lines = [
        f"{icon} <b>SEC {form} 申报</b>",
        f"<b>标的：{entity}</b>",
        f"日期：{date}",
        f"类型：{type_desc}",
    ]

    if is_activist:
        lines.append("⚠️ <b>主动型持股</b>：机构可能推动管理层变革、私有化或并购，注意溢价风险")
    if is_amend:
        lines.append("📝 修正申报：请关注增减仓方向（持股比例变化）")

    lines.append(f"🔗 EDGAR搜索：<code>{entity}</code>")
    return "\n".join(lines)


def run_13dg_monitor(watchlist: list[str] | None = None,
                     send_fn=None) -> dict:
    """
    执行一次监控，推送所有新申报。

    参数：
      watchlist — 目标标的列表（None = 全市场）
      send_fn   — Telegram 发送函数（如 send_telegram）

    返回：{ok, new_count, pushed}
    """
    new_filings = check_new_13dg(watchlist)
    pushed = []

    for f in new_filings:
        msg = format_13dg_telegram(f)
        if send_fn:
            try:
                send_fn(msg)
                time.sleep(0.5)
            except Exception:
                pass
        pushed.append(f.get("entity_name", ""))

    return {
        "ok":        True,
        "new_count": len(new_filings),
        "pushed":    pushed,
        "note":      (f"发现 {len(new_filings)} 条新13D/G申报并推送"
                      if new_filings else "无新13D/G申报"),
    }
