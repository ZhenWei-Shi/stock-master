"""
SEC Form 4 内部人交易追踪

数据来源：SEC EDGAR 官方 API（免费，无需注册，24小时内公开）

核心逻辑：
  - 内部人开市买入（代码 P）= 管理层相信当前价格低估，是质量最高的看多信号之一
  - 内部人卖出（代码 S）= 弱看空（RSU解锁、多元化、税收规划均会触发，噪音高）
  - 授权股/激励股（代码 A/F/M）= 薪酬安排，排除
  - 净买入 > $10万 且无大量卖出 → 计入正分

两处集成：
  1. long_hold._score_insider()  → 长持基本面评分
  2. cold_model bonus 区域        → 替代失效的 UOA，作为机构级买入确认

EDGAR API 文档：https://www.sec.gov/developer
速率限制：10请求/秒，建议 0.15s 间隔
"""

from __future__ import annotations   # Python 3.9 兼容 X | Y 类型注解

import os
import json
import time
import requests
import xml.etree.ElementTree as ET_XML
from datetime import datetime, timedelta
import pytz

ET_TZ = pytz.timezone("America/New_York")

_DATA   = os.path.join(os.path.dirname(__file__), "..", "data")
_CACHE  = os.path.join(_DATA, "insider_cache")

# SEC 要求 User-Agent 含联系邮箱，否则会被拒绝（429）
_EDGAR_HEADERS = {
    "User-Agent": "stock-master-personal 1758162368szw@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

# ══════════════════════════════════════════════════════════════
# 配置常量
# ══════════════════════════════════════════════════════════════

INSIDER_LOOKBACK_DAYS       = 90       # 分析最近N天的申报
INSIDER_CACHE_HOURS         = 24       # 结果缓存时间（小时）
CIK_CACHE_DAYS              = 7        # ticker→CIK 映射缓存天数
MAX_FILINGS_PARSE           = 12       # 每只股票最多解析N份Form 4
REQUEST_DELAY               = 0.15     # 两次EDGAR请求间最小间隔（秒）

# 买入信号阈值
BUY_STRONG_USD              = 500_000  # 净买入>$50万 = 强信号 +12分
BUY_MOD_USD                 = 100_000  # 净买入>$10万 = 中信号 +6分
BUY_WEAK_USD                = 20_000   # 净买入>$2万  = 弱信号 +3分

# 只计开市购入（P），排除薪酬相关代码
PURCHASE_CODES              = {"P"}               # 开市买入
SALE_CODES                  = {"S"}               # 开市卖出（仅这个算真卖出）
EXCLUDE_CODES               = {"A", "F", "M", "X", "G", "L", "C", "E", "W"}

# ══════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────
# Ticker → CIK 映射（带7天缓存）
# ─────────────────────────────────────────────────────────────

_CIK_FILE = os.path.join(_DATA, "sec_cik_map.json")


def _load_cik_map() -> dict:
    """加载 ticker→CIK 映射，过期则从 EDGAR 重新下载。"""
    try:
        mtime = os.path.getmtime(_CIK_FILE)
        if time.time() - mtime < CIK_CACHE_DAYS * 86400:
            with open(_CIK_FILE, "r") as f:
                return json.load(f)
    except OSError:
        pass  # 文件不存在，继续下载

    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        r = requests.get(url, headers=_EDGAR_HEADERS, timeout=20)
        r.raise_for_status()
        raw = r.json()
        mapping = {v["ticker"].upper(): str(v["cik_str"]) for v in raw.values()}
        os.makedirs(_DATA, exist_ok=True)
        tmp = _CIK_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(mapping, f)
        os.replace(tmp, _CIK_FILE)
        return mapping
    except Exception:
        return {}


def get_cik(ticker: str) -> str | None:
    """返回 SEC CIK（10位数字字符串），找不到返回 None。"""
    m = _load_cik_map()
    return m.get(ticker.upper())


# ─────────────────────────────────────────────────────────────
# 获取最近 Form 4 申报列表
# ─────────────────────────────────────────────────────────────

def _get_recent_form4s(cik: str, days: int = INSIDER_LOOKBACK_DAYS) -> list[dict]:
    """从 EDGAR submissions API 取近期 Form 4 申报元数据。"""
    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    try:
        r = requests.get(url, headers=_EDGAR_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms      = recent.get("form", [])
    dates      = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    result = []
    for i, ft in enumerate(forms):
        if ft not in ("4", "4/A"):
            continue
        fd = dates[i] if i < len(dates) else ""
        if fd < cutoff:
            continue
        result.append({
            "accession":    accessions[i] if i < len(accessions) else "",
            "date":         fd,
            "primary_doc":  primary_docs[i] if i < len(primary_docs) else "",
            "cik":          cik,
        })
    return result


# ─────────────────────────────────────────────────────────────
# 解析单份 Form 4 XML
# ─────────────────────────────────────────────────────────────

def _fetch_form4_xml(cik: str, accession: str, primary_doc: str) -> str:
    """下载 Form 4 XML 文本。primary_doc 优先，失败则扫描 index。"""
    cik_int     = int(cik)
    acc_nodash  = accession.replace("-", "")

    # 直接尝试 primary_doc（最快路径）
    if primary_doc and primary_doc.endswith(".xml"):
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary_doc}"
        try:
            r = requests.get(url, headers=_EDGAR_HEADERS, timeout=15)
            if r.status_code == 200 and "<ownershipDocument" in r.text:
                return r.text
        except Exception:
            pass

    # 退路：从 index JSON 找 XML
    idx_url = (f"https://www.sec.gov/Archives/edgar/data/{cik_int}"
               f"/{acc_nodash}/{accession}-index.json")
    try:
        ir = requests.get(idx_url, headers=_EDGAR_HEADERS, timeout=15)
        if ir.status_code == 200:
            items = ir.json().get("directory", {}).get("item", [])
            for item in items:
                name = item.get("name", "")
                if (name.endswith(".xml")
                        and not name.startswith("_")
                        and "index" not in name.lower()):
                    xml_url = (f"https://www.sec.gov/Archives/edgar/data"
                               f"/{cik_int}/{acc_nodash}/{name}")
                    xr = requests.get(xml_url, headers=_EDGAR_HEADERS, timeout=15)
                    if xr.status_code == 200:
                        return xr.text
    except Exception:
        pass
    return ""


def _parse_form4_xml(xml_text: str) -> dict:
    """
    解析 Form 4 XML，提取所有非衍生品交易明细。

    返回：
      reporter_name, reporter_title, is_director, is_officer
      transactions: [{type, shares, price, value, date, acquired_disposed}]
    """
    try:
        root = ET_XML.fromstring(xml_text)
    except ET_XML.ParseError:
        return {}

    # 申报人信息
    reporter_name  = ""
    is_director    = False
    is_officer     = False
    officer_title  = ""

    for owner in root.findall(".//reportingOwner"):
        reporter_name = (owner.findtext(".//rptOwnerName") or "").strip()
        rel = owner.find(".//reportingOwnerRelationship")
        if rel is not None:
            is_director   = rel.findtext("isDirector") == "1"
            is_officer    = rel.findtext("isOfficer")  == "1"
            officer_title = (rel.findtext("officerTitle") or "").strip()

    transactions = []

    for tbl_tag in ("nonDerivativeTable", "derivativeTable"):
        tbl = root.find(f".//{tbl_tag}")
        if tbl is None:
            continue
        for tx_el in tbl:
            if "Transaction" not in tx_el.tag:
                continue

            tx_code = (tx_el.findtext(".//transactionCode") or "").strip().upper()
            if not tx_code:
                continue

            try:
                shares_el   = tx_el.find(".//transactionShares/value")
                price_el    = tx_el.find(".//transactionPricePerShare/value")
                date_el     = tx_el.find(".//transactionDate/value")
                ad_el       = tx_el.find(".//transactionAcquiredDisposedCode/value")

                shares  = float(shares_el.text)  if shares_el  is not None else 0
                price   = float(price_el.text)   if price_el   is not None else 0
                date    = date_el.text            if date_el    is not None else ""
                ad_code = (ad_el.text or "").upper() if ad_el is not None else ""
                value   = shares * price
            except (ValueError, TypeError):
                continue

            transactions.append({
                "type":    tx_code,
                "shares":  shares,
                "price":   price,
                "value":   value,
                "date":    date,
                "ad_code": ad_code,   # A=Acquired, D=Disposed
            })

    return {
        "reporter": reporter_name,
        "title":       officer_title,
        "is_director": is_director,
        "is_officer":  is_officer,
        "transactions": transactions,
    }


# ─────────────────────────────────────────────────────────────
# 主入口：内部人活动汇总（带缓存）
# ─────────────────────────────────────────────────────────────

def insider_summary(ticker: str, days: int = INSIDER_LOOKBACK_DAYS,
                    force_refresh: bool = False) -> dict:
    """
    汇总最近 N 天内部人开市买卖记录。

    返回：
      bias:         "bullish" | "bearish" | "neutral"
      net_buy_usd:  净买入金额（正=买入，负=卖出）
      buy_usd:      合计开市买入金额
      sell_usd:     合计开市卖出金额
      buy_count:    开市购买笔数
      sell_count:   开市卖出笔数
      significant:  重大交易列表（$2万以上）
      score_delta:  建议加减分（已算入信心权重）
      note:         一行说明
    """
    os.makedirs(_CACHE, exist_ok=True)
    cache_file = os.path.join(_CACHE, f"{ticker.upper()}.json")

    # 读缓存
    if not force_refresh and os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < INSIDER_CACHE_HOURS * 3600:
            with open(cache_file) as f:
                return json.load(f)

    cik = get_cik(ticker)
    if not cik:
        result = {"ok": False, "reason": f"未找到 {ticker} 的 CIK（可能是ETF或非美股）"}
        _cache_write(cache_file, result)
        return result

    filings = _get_recent_form4s(cik, days)
    if not filings:
        result = {"ok": True, "ticker": ticker, "bias": "neutral", "net_buy_usd": 0,
                  "buy_usd": 0, "sell_usd": 0, "buy_count": 0, "sell_count": 0,
                  "significant": [], "score_delta": 0,
                  "note": f"过去{days}天无Form 4申报记录"}
        _cache_write(cache_file, result)
        return result

    buy_usd  = sell_usd  = 0.0
    buy_cnt  = sell_cnt  = 0
    significant = []

    for f in filings[:MAX_FILINGS_PARSE]:
        time.sleep(REQUEST_DELAY)
        xml_text = _fetch_form4_xml(f["cik"], f["accession"], f["primary_doc"])
        if not xml_text:
            continue
        parsed = _parse_form4_xml(xml_text)
        if not parsed:
            continue

        reporter = parsed["reporter"]
        title    = parsed["title"]

        for tx in parsed["transactions"]:
            code  = tx["type"]
            value = tx["value"]

            if code in PURCHASE_CODES:
                buy_usd += value
                buy_cnt += 1
                if value >= BUY_WEAK_USD:
                    significant.append({
                        "type":     "买入",
                        "reporter": reporter,
                        "title":    title,
                        "value_usd": int(value),
                        "shares":   int(tx["shares"]),
                        "price":    round(tx["price"], 2),
                        "date":     tx["date"],
                    })
            elif code in SALE_CODES:
                sell_usd += value
                sell_cnt += 1
                if value >= BUY_MOD_USD:
                    significant.append({
                        "type":     "卖出",
                        "reporter": reporter,
                        "title":    title,
                        "value_usd": int(value),
                        "shares":   int(tx["shares"]),
                        "price":    round(tx["price"], 2),
                        "date":     tx["date"],
                    })
            # 排除薪酬/授权（EXCLUDE_CODES），不计入

    net = buy_usd - sell_usd

    # 评分：买入信号有强度，卖出信号轻惩（原因多样，噪音高）
    if net >= BUY_STRONG_USD and buy_cnt >= 2:
        bias, score_delta = "bullish", 12
        note_core = f"强烈内部人买入信号（{buy_cnt}笔，净${net/1000:.0f}k）"
    elif net >= BUY_MOD_USD:
        bias, score_delta = "bullish", 6
        note_core = f"内部人净买入 ${net/1000:.0f}k"
    elif net >= BUY_WEAK_USD:
        bias, score_delta = "bullish", 3
        note_core = f"内部人小额净买入 ${net/1000:.0f}k"
    elif sell_usd > 2_000_000 and buy_usd < sell_usd * 0.1:
        bias, score_delta = "bearish", -4
        note_core = f"内部人大量净卖出 ${sell_usd/1000:.0f}k（买入仅${buy_usd/1000:.0f}k）"
    else:
        bias, score_delta = "neutral", 0
        note_core = (f"买卖均衡或活动极少（买${buy_usd/1000:.0f}k/卖${sell_usd/1000:.0f}k）"
                     if (buy_usd + sell_usd) > 0 else "无内部人开市交易记录")

    result = {
        "ok":          True,
        "ticker":      ticker,
        "days":        days,
        "bias":        bias,
        "net_buy_usd": int(net),
        "buy_usd":     int(buy_usd),
        "sell_usd":    int(sell_usd),
        "buy_count":   buy_cnt,
        "sell_count":  sell_cnt,
        "significant": significant,
        "score_delta": score_delta,
        "note":        f"过去{days}天 Form 4：{note_core}",
        "filings_found": len(filings),
    }
    _cache_write(cache_file, result)
    return result


def _cache_write(path: str, data: dict):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# 格式化（Telegram 展示用）
# ─────────────────────────────────────────────────────────────

def format_insider_telegram(result: dict) -> str:
    if not result.get("ok"):
        return f"⚠️ 内部人数据：{result.get('reason', '获取失败')}"

    bias_icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(
        result["bias"], "⚪"
    )
    lines = [f"{bias_icon} <b>内部人交易（近{result['days']}天）</b>"]
    lines.append(result["note"])

    for tx in result.get("significant", [])[:3]:
        icon = "📈" if tx["type"] == "买入" else "📉"
        lines.append(
            f"  {icon} {tx['reporter']} ({tx['title'] or '内部人'})"
            f" {tx['type']} {tx['shares']:,}股 @ ${tx['price']:.2f}"
            f" = ${tx['value_usd']/1000:.0f}k  ({tx['date']})"
        )
    return "\n".join(lines)
