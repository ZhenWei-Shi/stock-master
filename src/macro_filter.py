"""
宏观事件过滤器 + 行业传导链分析

解决的核心问题：
  你朋友说的"海峡封锁→油价涨→科技股跌"——这叫宏观传导链。
  个人投资者通常在事件发生后才反应，机构提前已经布局。
  这个模块让你做到：
    1. 知道今天有没有重大经济事件（FOMC/CPI/非农）→ 主动避开
    2. 检测当前主导的宏观主题（通胀/衰退恐慌/AI繁荣/地缘风险）
    3. 根据主题自动调整各板块得分（某些股票今天不该碰）
    4. 过滤日常噪音，只保留影响市场结构的信息

数据来源（全免费）：
  - FOMC 日历：美联储提前公布全年，硬编码
  - CPI/非农：BLS 固定规律（每月第2周/第1个周五）
  - 市场新闻：yfinance（SPY/VIX/USO/TLT/GLD 板块 ETF）
  - 实时价格变化：从 ETF 当日涨跌反向推断宏观主题

传导链逻辑：
  油价↑ 5%+ → 航空/运输↓，能源↑
  VIX↑ 30%+ → 全面风险规避，只做防御
  TLT（长债）↑  → 降息预期，成长股↑
  TLT↓       → 加息预期，价值/金融↑
  GLD↑ 2%+   → 避险，地缘风险或通胀
  DXY↑ 1%+   → 美元强，新兴市场/大宗商品↓
"""

import json
import os
import re
from datetime import datetime, timedelta, date
from typing import Optional

import yfinance as yf
import pytz

ET = pytz.timezone("America/New_York")
_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(_DATA, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 宏观过滤配置常量（修改参数在此处）
# ══════════════════════════════════════════════════════════════

# ── 经济事件禁入窗口 ─────────────────────────────────────────
FOMC_BLOCK_DAYS             = 1        # FOMC决议日当天±1天禁入
CPI_BLOCK_DAYS              = 1        # CPI发布日当天禁入
NFP_BLOCK_DAYS              = 1        # 非农发布日当天禁入

# ── ETF宏观信号阈值 ──────────────────────────────────────────
VIX_SURGE_PCT               = 15.0     # VIX单日涨>15% = 地缘风险
VIX_CAUTION_PCT             = 8.0      # VIX单日涨>8%  = 警戒
OIL_SPIKE_PCT               = 3.0      # 原油涨>3%  = 能源冲击
OIL_CRASH_PCT               = -3.0     # 原油跌>3%  = 需求崩塌
GOLD_SAFE_HAVEN_PCT         = 1.5      # 黄金涨>1.5% = 避险
BOND_DOVISH_PCT             = 1.0      # TLT涨>1%  = 降息预期
DEFENSE_SPIKE_PCT           = 2.0      # 国防股涨>2% = 地缘
TECH_BOOM_PCT               = 2.0      # 科技涨>2%  = AI繁荣信号

# ── 宏观快照有效期 ────────────────────────────────────────────
SNAPSHOT_MAX_AGE_HOURS      = 2        # >2小时的快照提示刷新

# ── 宏观门关惩罚/加分 ────────────────────────────────────────
MACRO_PENALTY_AVOID         = 25       # 受害板块扣分
MACRO_BONUS_FAVOR           = 15       # 受益板块加分
MACRO_PENALTY_VIX_HIGH      = 20       # VIX剧烈波动扣分
MACRO_PENALTY_VIX_MILD      = 10       # VIX温和波动扣分
MACRO_PENALTY_CAP           = 35       # 总扣分上限
MACRO_BONUS_CAP             = 20       # 总加分上限

# ── 负面新闻硬性关键词 ──────────────────────────────────────
HARD_NEGATIVE_KEYWORDS = [
    "fraud", "sec investigation", "bankruptcy", "scandal",
    "accounting restatement", "criminal charges", "ceo arrested",
    "delisting", "going concern", "chapter 11",
]

# ══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# 1. 经济日历（FOMC / CPI / 非农 / 财季）
# ─────────────────────────────────────────────────────────────

# FOMC 决议日（美联储提前公布）——第二天公布 → 当天禁入
_FOMC_DATES_2025 = [
    "2025-01-29", "2025-03-19", "2025-05-07",
    "2025-06-18", "2025-07-30", "2025-09-17",
    "2025-10-29", "2025-12-10",
]
_FOMC_DATES_2026 = [
    "2026-01-28", "2026-03-18", "2026-05-06",
    "2026-06-17", "2026-07-29", "2026-09-16",
    "2026-10-28", "2026-12-09",
]
_ALL_FOMC = set(_FOMC_DATES_2025 + _FOMC_DATES_2026)


def _this_month_cpi_day() -> Optional[str]:
    """CPI 发布日：每月第2个周三（估算，BLS 实际日期可能略有不同）"""
    today = date.today()
    first = date(today.year, today.month, 1)
    wed_count = 0
    for d in range(1, 20):
        day = first + timedelta(days=d - 1)
        if day.weekday() == 2:  # 周三
            wed_count += 1
            if wed_count == 2:
                return str(day)
    return None


def _this_month_nfp_day() -> Optional[str]:
    """非农就业：每月第1个周五"""
    today = date.today()
    first = date(today.year, today.month, 1)
    for d in range(1, 8):
        day = first + timedelta(days=d - 1)
        if day.weekday() == 4:  # 周五
            return str(day)
    return None


def get_economic_calendar() -> dict:
    """
    返回最近30天内的重大经济事件。

    每个事件包含：
      name, date, days_away, risk_level, action
    """
    today    = date.today()
    events   = []
    warnings = []

    # FOMC
    for fd in _ALL_FOMC:
        d = date.fromisoformat(fd)
        days = (d - today).days
        if -1 <= days <= 30:  # 昨天到未来30天
            risk  = "🔴 极高" if abs(days) <= 1 else "🟡 高" if days <= 7 else "🟢 正常"
            if days == -1:
                action = f"昨日（{fd}）FOMC已公布决议，今日市场仍在消化，谨慎开新仓"
            elif days == 0:
                action = "今日 FOMC 决议！禁止开新仓，等待决议后方向确认"
            elif days == 1:
                action = "明日 FOMC 决议！今日降低新仓比例，注意提前风险"
            elif days <= 5:
                action = f"FOMC 在 {days} 天后，注意仓位风险，不宜过重"
            else:
                action = f"FOMC 在 {days} 天后，正常交易"
            events.append({
                "name":      "FOMC 利率决议",
                "date":      fd,
                "days_away": days,
                "risk_level": risk,
                "action":    action,
                "impact":    "利率方向影响全市场，不确定性最高",
            })
            if days == -1:
                warnings.append(f"📌 昨日FOMC已决议（{fd}），今日市场消化期，谨慎建仓")
            elif days == 0:
                warnings.append(f"🚨 今日FOMC决议（{fd}），禁止开新仓")
            elif days == 1:
                warnings.append(f"⚠️ 明日FOMC决议（{fd}），今日降低新仓")

    # CPI
    cpi_day = _this_month_cpi_day()
    if cpi_day:
        d = date.fromisoformat(cpi_day)
        days = (d - today).days
        if -1 <= days <= 14:
            events.append({
                "name":      "CPI 通胀数据",
                "date":      cpi_day,
                "days_away": days,
                "risk_level": "🔴 高" if abs(days) <= 1 else "🟡 中",
                "action": (
                    "CPI 发布日！高通胀→科技受压，低通胀→成长股受益"
                    if abs(days) <= 1 else
                    f"CPI 在 {days} 天后，布局前先确认通胀预期"
                ),
                "impact": "通胀数据决定美联储下一步，影响利率预期",
            })
            if abs(days) <= 1:
                warnings.append(f"⚠️ CPI 发布日（{cpi_day}），等待数据再行动")

    # 非农
    nfp_day = _this_month_nfp_day()
    if nfp_day:
        d = date.fromisoformat(nfp_day)
        days = (d - today).days
        if -1 <= days <= 7:
            events.append({
                "name":      "非农就业报告（NFP）",
                "date":      nfp_day,
                "days_away": days,
                "risk_level": "🔴 高" if abs(days) <= 1 else "🟡 中",
                "action": (
                    "NFP 发布日！就业强→美联储鹰派，就业弱→降息预期"
                    if abs(days) <= 1 else
                    f"NFP 在 {days} 天后，就业数据影响利率方向"
                ),
                "impact": "就业是美联储最重要的参考指标",
            })
            if abs(days) <= 1:
                warnings.append(f"⚠️ 非农就业数据发布日（{nfp_day}），等数据再建仓")

    # 财报季（美股：1月/4月/7月/10月）
    month = today.month
    in_earnings_season = month in (1, 4, 7, 10)
    if in_earnings_season:
        events.append({
            "name":      "财报密集期",
            "date":      str(today),
            "days_away": 0,
            "risk_level": "🟡 中",
            "action":    "财报季：个股黑天鹅风险高，优先等待财报公布后再建仓",
            "impact":    "重磅财报（NVDA/AAPL等）会影响整个板块情绪",
        })

    events.sort(key=lambda x: abs(x["days_away"]))

    return {
        "ok":       True,
        "today":    str(today),
        "events":   events,
        "warnings": warnings,
        "in_earnings_season": in_earnings_season,
        "high_risk_today": len(warnings) > 0,
    }


# ─────────────────────────────────────────────────────────────
# 2. 宏观主题传导链
# ─────────────────────────────────────────────────────────────

# 每个主题：检测关键词 + 受益/受损板块 + 关键股票 + 冷静模型分调整
_TRANSMISSION_CHAINS = {
    "FED_HAWKISH": {
        "label":    "美联储鹰派（加息/维持高利率）",
        "keywords": ["rate hike", "hawkish", "inflation too high", "higher for longer",
                     "tighten", "fed hikes", "rate increase", "policy rate"],
        "sectors_favor": ["Financials", "Energy", "Healthcare", "Consumer Staples"],
        "sectors_avoid": ["Technology", "Real Estate", "Utilities", "Consumer Discretionary"],
        "tickers_favor": ["JPM", "BAC", "GS", "WFC", "XOM", "CVX"],
        "tickers_avoid": ["MSFT", "NVDA", "AMZN", "TSLA", "META", "GOOGL", "VNQ"],
        "score_delta":   {"Technology": -20, "Financials": +15, "Real Estate": -25, "Utilities": -15},
        "logic": "高利率压低成长股DCF估值，银行净利差扩大受益",
    },
    "FED_DOVISH": {
        "label":    "美联储鸽派（降息/暂停加息）",
        "keywords": ["rate cut", "dovish", "pause", "pivot", "easing", "lower rates",
                     "accommodation", "fed cuts", "rate reduction"],
        "sectors_favor": ["Technology", "Real Estate", "Consumer Discretionary", "Biotech"],
        "sectors_avoid": ["Financials"],
        "tickers_favor": ["MSFT", "NVDA", "AMZN", "TSLA", "GOOGL", "ARKK"],
        "tickers_avoid": ["JPM", "GS"],
        "score_delta":   {"Technology": +20, "Financials": -10, "Real Estate": +20},
        "logic": "低利率提升成长股估值，融资成本下降利好高杠杆行业",
    },
    "OIL_SPIKE": {
        "label":    "油价飙升（地缘/减产）",
        "keywords": ["oil spike", "crude surge", "opec cut", "supply shock", "oil price jump",
                     "energy crisis", "strait closure", "pipeline", "oil ban"],
        "sectors_favor": ["Energy", "Defense"],
        "sectors_avoid": ["Airlines", "Transportation", "Consumer Discretionary"],
        "tickers_favor": ["XOM", "CVX", "COP", "SLB", "OXY", "VLO", "MPC"],
        "tickers_avoid": ["AAL", "DAL", "UAL", "UPS", "FDX", "AMZN"],
        "score_delta":   {"Airlines": -30, "Energy": +25, "Transportation": -20},
        "logic": "油价上涨：能源公司盈利扩大，航空/物流成本急升",
    },
    "OIL_CRASH": {
        "label":    "油价暴跌（需求下滑/增产）",
        "keywords": ["oil crash", "crude plunge", "oil surplus", "opec increase",
                     "demand slowdown", "oil glut"],
        "sectors_favor": ["Airlines", "Transportation", "Consumer Discretionary"],
        "sectors_avoid": ["Energy"],
        "tickers_favor": ["AAL", "DAL", "JBLU", "FDX", "UPS"],
        "tickers_avoid": ["XOM", "CVX", "COP", "SLB"],
        "score_delta":   {"Energy": -30, "Airlines": +20},
        "logic": "油价下跌：航空运营成本降低，消费者汽油负担减轻",
    },
    "TARIFFS_CHINA": {
        "label":    "对华关税/贸易战",
        "keywords": ["tariff china", "china tariff", "trade war", "import duty",
                     "trade restrictions", "export control", "china trade",
                     "tariffs on", "trade tension"],
        "sectors_favor": ["Defense", "Domestic Manufacturing", "Cybersecurity"],
        "sectors_avoid": ["Technology", "Consumer Electronics", "Semiconductors"],
        "tickers_favor": ["LMT", "RTX", "NOC", "AMAT", "LRCX", "KLAC"],
        "tickers_avoid": ["AAPL", "NVDA", "QCOM", "AVGO", "AMD", "MU", "INTC"],
        "score_delta":   {"Technology": -15, "Defense": +10, "Semiconductors": -20},
        "logic": "关税增加进口成本，中国收入依赖高的科技公司受损",
    },
    "GEOPOLITICAL_TENSION": {
        "label":    "地缘政治紧张（战争/制裁）",
        "keywords": ["war", "conflict", "missile", "sanctions", "military",
                     "invasion", "taiwan", "strait", "nuclear", "attack",
                     "geopolitical", "tension escalate"],
        "sectors_favor": ["Defense", "Cybersecurity", "Gold", "Energy"],
        "sectors_avoid": ["Travel", "Luxury", "Airlines", "Global Supply Chain"],
        "tickers_favor": ["LMT", "RTX", "NOC", "BA", "CRWD", "PANW", "GLD"],
        "tickers_avoid": ["MAR", "HLT", "BKNG", "LVS", "AAL"],
        "score_delta":   {"Defense": +25, "Travel": -25, "Luxury": -15},
        "logic": "地缘风险推升避险资产，国防支出增加，全球化企业受损",
    },
    "RECESSION_FEAR": {
        "label":    "衰退恐慌",
        "keywords": ["recession", "economic slowdown", "gdp contraction", "downturn",
                     "soft landing failed", "hard landing", "stagflation",
                     "yield curve invert", "credit crunch"],
        "sectors_favor": ["Consumer Staples", "Healthcare", "Utilities", "Gold"],
        "sectors_avoid": ["Consumer Discretionary", "Technology", "Industrials", "Materials"],
        "tickers_favor": ["JNJ", "PG", "KO", "WMT", "NEE", "DUK", "GLD"],
        "tickers_avoid": ["TSLA", "AMZN", "HD", "CAT", "DE", "NVDA"],
        "score_delta":   {"Consumer Staples": +15, "Technology": -20,
                          "Consumer Discretionary": -25, "Industrials": -15},
        "logic": "衰退期防御性需求（食品/医疗/公用事业）稳定，周期性行业盈利下滑",
    },
    "AI_BOOM": {
        "label":    "AI/科技资本支出热潮",
        "keywords": ["ai capex", "data center", "gpu demand", "ai investment",
                     "artificial intelligence spending", "cloud capex",
                     "nvidia earnings", "ai infrastructure", "hyperscaler"],
        "sectors_favor": ["Semiconductors", "Data Centers", "Power/Cooling", "Cybersecurity"],
        "sectors_avoid": [],
        "tickers_favor": ["NVDA", "AMD", "AMAT", "VRT", "CEG", "VST", "MSFT",
                          "AVGO", "MRVL", "COHR", "LITE"],
        "tickers_avoid": [],
        "score_delta":   {"Technology": +20, "Semiconductors": +25, "Energy": +10},
        "logic": "AI资本支出拉动算力/存储/电力/散热全产业链需求",
    },
    "CHIP_EXPORT_CONTROLS": {
        "label":    "半导体出口管制",
        "keywords": ["export control", "chip ban", "semiconductor restriction",
                     "entity list", "chip equipment ban", "huawei ban",
                     "advanced chips", "ai chip ban"],
        "sectors_favor": ["Domestic Semiconductor Equipment"],
        "sectors_avoid": ["Semiconductors (China revenue)"],
        "tickers_favor": ["AMAT", "LRCX", "KLAC"],
        "tickers_avoid": ["NVDA", "AMD", "QCOM", "MU", "INTC"],
        "score_delta":   {"Semiconductors": -15},
        "logic": "芯片出口管制：有中国收入的芯片公司直接受损，国内设备商受益于国产替代需求",
    },
    "BANKING_STRESS": {
        "label":    "银行业压力/金融风险",
        "keywords": ["bank failure", "banking crisis", "credit risk", "bank run",
                     "svb", "financial stress", "credit default", "bank collapse",
                     "contagion", "banking sector"],
        "sectors_favor": ["Gold", "Consumer Staples", "Utilities"],
        "sectors_avoid": ["Financials", "Real Estate"],
        "tickers_favor": ["GLD", "JNJ", "PG"],
        "tickers_avoid": ["JPM", "BAC", "WFC", "GS", "MS", "C"],
        "score_delta":   {"Financials": -30, "Real Estate": -20},
        "logic": "银行危机传导系统性风险，金融股直接受损，避险资产受益",
    },
    "INFLATION_HOT": {
        "label":    "通胀超预期（CPI/PCE 高于预期）",
        "keywords": ["inflation surges", "cpi hot", "inflation higher than expected",
                     "core inflation", "price surge", "inflation accelerates",
                     "pce hot", "inflation beat"],
        "sectors_favor": ["Energy", "Materials", "Commodities", "Financials"],
        "sectors_avoid": ["Technology", "Real Estate", "Utilities", "Bonds"],
        "tickers_favor": ["XOM", "CVX", "FCX", "NEM", "ALB", "JPM"],
        "tickers_avoid": ["MSFT", "NVDA", "AMZN", "VNQ"],
        "score_delta":   {"Technology": -15, "Energy": +15, "Materials": +10},
        "logic": "高通胀：实物资产/能源/材料保值，成长股折现率上升",
    },
    "INFLATION_COOL": {
        "label":    "通胀降温（CPI/PCE 低于预期）",
        "keywords": ["inflation cools", "cpi miss", "inflation lower", "disinflation",
                     "deflation", "price decline", "cpi below", "inflation slows"],
        "sectors_favor": ["Technology", "Real Estate", "Consumer Discretionary", "Biotech"],
        "sectors_avoid": ["Energy", "Materials"],
        "tickers_favor": ["MSFT", "NVDA", "AMZN", "TSLA", "GOOGL"],
        "tickers_avoid": ["XOM", "CVX", "FCX"],
        "score_delta":   {"Technology": +15, "Real Estate": +20, "Energy": -10},
        "logic": "通胀下行降息预期上升，成长股和房地产重新受益",
    },
}


# ─────────────────────────────────────────────────────────────
# 3. ETF 价格变化 → 实时宏观主题推断
# ─────────────────────────────────────────────────────────────

_MACRO_ETF_MAP = {
    "USO":  "oil",      # 原油 ETF
    "GLD":  "gold",     # 黄金
    "TLT":  "bonds",    # 长期国债（降息预期↑ = TLT↑）
    "UUP":  "dollar",   # 美元指数
    "XLF":  "financials",
    "XLK":  "technology",
    "XLE":  "energy",
    "XLU":  "utilities",
    "XLV":  "healthcare",
    "XLP":  "staples",
    "IYT":  "transportation",
    "ITA":  "defense",
    "^VIX": "vix",
}


def get_etf_signals() -> dict:
    """
    通过关键 ETF 的当日涨跌推断当前主导宏观主题。

    方法：市场已经把宏观信息定价进去了，ETF 的相对表现就是最实时的宏观信号。
    """
    try:
        etf_list = [k for k in _MACRO_ETF_MAP if not k.startswith("^")]
        vix_tick = "^VIX"

        raw = yf.download(etf_list + [vix_tick],
                          period="2d", interval="1d",
                          group_by="ticker", auto_adjust=True, progress=False)

        signals = {}
        inferred_themes = []

        for etf, category in _MACRO_ETF_MAP.items():
            try:
                if len(etf_list) + 1 == 1:
                    d = raw
                else:
                    if etf not in raw.columns.get_level_values(0):
                        continue
                    d = raw[etf].dropna()

                if len(d) < 2:
                    continue
                chg = (float(d["Close"].iloc[-1]) - float(d["Close"].iloc[-2])) / float(d["Close"].iloc[-2]) * 100
                signals[category] = round(chg, 2)
            except Exception:
                continue

        # 推断宏观主题（基于 ETF 变化幅度阈值）
        vix     = signals.get("vix", 0)
        oil     = signals.get("oil", 0)
        gold    = signals.get("gold", 0)
        bonds   = signals.get("bonds", 0)
        dollar  = signals.get("dollar", 0)
        tech    = signals.get("technology", 0)
        finance = signals.get("financials", 0)
        defense = signals.get("defense", 0)
        energy  = signals.get("energy", 0)

        if vix > 15:
            inferred_themes.append("GEOPOLITICAL_TENSION")
        elif vix > 8:
            inferred_themes.append("RECESSION_FEAR")

        if oil > 3:
            inferred_themes.append("OIL_SPIKE")
        elif oil < -3:
            inferred_themes.append("OIL_CRASH")

        if gold > 1.5:
            inferred_themes.append("GEOPOLITICAL_TENSION")

        if bonds > 1 and tech > 1:
            inferred_themes.append("FED_DOVISH")
        elif bonds < -1 and finance > 0.5:
            inferred_themes.append("FED_HAWKISH")

        if defense > 2 and gold > 1:
            inferred_themes.append("GEOPOLITICAL_TENSION")

        if tech > 2 and signals.get("vix", 0) < 3:
            inferred_themes.append("AI_BOOM")

        # 去重，保留最显著主题
        inferred_themes = list(dict.fromkeys(inferred_themes))[:3]

        return {
            "ok":              True,
            "etf_changes_pct": signals,
            "inferred_themes": inferred_themes,
            "vix_level":       vix,
            "risk_environment": (
                "🔴 高风险（VIX飙升）" if vix > 10 else
                "🟡 中等风险"          if vix > 3  else
                "🟢 低风险"
            ),
        }

    except Exception as e:
        return {"ok": False, "reason": str(e), "inferred_themes": []}


# ─────────────────────────────────────────────────────────────
# 4. 新闻关键词分类（从 yfinance 获取）
# ─────────────────────────────────────────────────────────────

_HARD_NEGATIVE = HARD_NEGATIVE_KEYWORDS  # 统一使用顶部配置常量


def get_news_themes(tickers: list = None) -> dict:
    """
    抓取市场新闻，识别主导宏观主题。

    扫描对象：SPY（大盘）+ 用户指定标的
    返回：active_themes（当前主导主题）+ hard_blocked（硬性禁入股票）
    """
    if tickers is None:
        tickers = ["SPY"]

    scan_tickers = list(set(["SPY", "^VIX"] + (tickers or [])))
    all_headlines = []
    hard_blocked  = {}

    for tick in scan_tickers:
        try:
            news = yf.Ticker(tick).news or []
            for item in news[:8]:  # 每个标的最新8条
                title   = (item.get("title") or "").lower()
                summary = (item.get("summary") or "").lower()
                text    = title + " " + summary

                # 硬性负面（个股）
                if tick not in ("SPY", "^VIX"):
                    for kw in _HARD_NEGATIVE:
                        if kw in text:
                            hard_blocked[tick.upper()] = kw
                            break

                all_headlines.append(text)
        except Exception:
            continue

    # 匹配宏观主题
    theme_hits = {}
    for theme_id, chain in _TRANSMISSION_CHAINS.items():
        hits = 0
        for kw in chain["keywords"]:
            if any(kw in h for h in all_headlines):
                hits += 1
        if hits > 0:
            theme_hits[theme_id] = hits

    # 按命中次数排序
    active_themes = sorted(theme_hits.items(), key=lambda x: x[1], reverse=True)

    return {
        "ok":            True,
        "headlines_scanned": len(all_headlines),
        "active_themes": [(tid, cnt) for tid, cnt in active_themes[:4]],
        "hard_blocked":  hard_blocked,
        "note": (
            "新闻关键词匹配，可能有误判。主题命中次数越多，信号越可信。"
            "只有 hard_blocked 的股票才会被系统硬性屏蔽。"
        ),
    }


# ─────────────────────────────────────────────────────────────
# 5. 完整宏观报告（主入口）
# ─────────────────────────────────────────────────────────────

def full_macro_report(watchlist: list = None) -> dict:
    """
    一键生成完整宏观过滤报告。

    输出：
      1. 今日经济事件（是否应该暂停交易）
      2. 当前宏观主题（ETF 推断 + 新闻确认）
      3. 各板块分调整建议
      4. 受益/受损股票列表
      5. 综合操作建议
    """
    calendar   = get_economic_calendar()
    etf_sigs   = get_etf_signals()
    news_themes= get_news_themes(watchlist or [])

    # 合并主题（ETF 推断 + 新闻确认双重权重）
    all_themes = {}
    for tid in etf_sigs.get("inferred_themes", []):
        all_themes[tid] = all_themes.get(tid, 0) + 2  # ETF权重高
    for tid, cnt in news_themes.get("active_themes", []):
        all_themes[tid] = all_themes.get(tid, 0) + cnt

    top_themes = sorted(all_themes.items(), key=lambda x: x[1], reverse=True)[:3]

    # 合并板块分调整
    sector_scores = {}
    sectors_favor = []
    sectors_avoid = []
    tickers_favor = []
    tickers_avoid = list(news_themes.get("hard_blocked", {}).keys())

    for tid, _ in top_themes:
        chain = _TRANSMISSION_CHAINS.get(tid, {})
        for sec, delta in chain.get("score_delta", {}).items():
            sector_scores[sec] = sector_scores.get(sec, 0) + delta
        sectors_favor.extend(chain.get("sectors_favor", []))
        sectors_avoid.extend(chain.get("sectors_avoid", []))
        tickers_favor.extend(chain.get("tickers_favor", []))
        tickers_avoid.extend(chain.get("tickers_avoid", []))

    # 去重
    sectors_favor = list(dict.fromkeys(sectors_favor))
    sectors_avoid = list(dict.fromkeys(sectors_avoid))
    tickers_favor = list(dict.fromkeys(tickers_favor))[:10]
    tickers_avoid = list(dict.fromkeys(tickers_avoid))[:10]

    # 综合操作建议
    high_risk_today = calendar.get("high_risk_today", False)
    vix_high = etf_sigs.get("vix_level", 0) > 8

    if high_risk_today:
        master_action = "🛑 今日重大经济事件发布——暂停所有新开仓，等待数据消化后再操作"
    elif vix_high:
        master_action = "⚠️ 市场波动上升——减少仓位，只做高确定性信号，止损收紧"
    elif top_themes:
        main_theme = _TRANSMISSION_CHAINS.get(top_themes[0][0], {})
        master_action = f"当前主导主题：{main_theme.get('label', '')}——{main_theme.get('logic', '')}"
    else:
        master_action = "✅ 宏观环境平静，正常执行技术信号"

    # 保存快照（供 cold_model 读取）
    snapshot = {
        "generated_at":    str(datetime.now(ET)),
        "top_themes":      [(t, c) for t, c in top_themes],
        "sector_scores":   sector_scores,
        "tickers_avoid":   tickers_avoid,
        "tickers_favor":   tickers_favor,
        "high_risk_today": high_risk_today,
        "vix_change_pct":  etf_sigs.get("vix_level", 0),
    }
    snap_path = os.path.join(_DATA, "macro_snapshot.json")
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)

    return {
        "ok":              True,
        "generated_at":    str(datetime.now(ET)),
        "economic_calendar": calendar,
        "etf_signals":     etf_sigs,
        "news_themes": [
            {**_TRANSMISSION_CHAINS[tid], "id": tid, "confidence": cnt}
            for tid, cnt in top_themes
            if tid in _TRANSMISSION_CHAINS
        ],
        "sector_adjustments": sector_scores,
        "sectors_favor":   sectors_favor,
        "sectors_avoid":   sectors_avoid,
        "tickers_favor":   tickers_favor,
        "tickers_avoid":   tickers_avoid,
        "hard_blocked":    news_themes.get("hard_blocked", {}),
        "master_action":   master_action,
        "risk_environment": etf_sigs.get("risk_environment", "未知"),
    }


# ─────────────────────────────────────────────────────────────
# 6. cold_model 集成接口
# ─────────────────────────────────────────────────────────────

def macro_gate_check(ticker: str) -> dict:
    """
    为单个股票的决策提供宏观层面的加分/扣分/否决。

    cold_model 调用此函数，得到：
      block:    True → 宏观环境直接否决（硬性负面新闻/重大事件日）
      penalty:  0-30  → 扣分（宏观不利该股）
      bonus:    0-20  → 加分（宏观有利该股）
      reason:   说明
    """
    try:
        snap_path = os.path.join(_DATA, "macro_snapshot.json")
        if not os.path.exists(snap_path):
            return {"block": False, "penalty": 0, "bonus": 0, "reason": "宏观快照不存在，跳过"}

        with open(snap_path, "r", encoding="utf-8") as f:
            snap = json.load(f)

        # 检查快照是否太旧（超过2小时重新生成）
        gen_at = snap.get("generated_at", "")
        if gen_at:
            try:
                age_h = (datetime.now() - datetime.fromisoformat(gen_at[:19])).total_seconds() / 3600
                if age_h > 2:
                    return {"block": False, "penalty": 0, "bonus": 0,
                            "reason": f"宏观快照已{age_h:.1f}小时，建议刷新"}
            except Exception:
                pass

        ticker = ticker.upper()
        block  = False
        penalty= 0
        bonus  = 0
        reasons= []

        # 重大经济事件日 → 直接否决
        if snap.get("high_risk_today"):
            block = True
            reasons.append("今日FOMC/CPI/非农发布，禁止开新仓")

        # 硬性负面新闻
        if ticker in snap.get("tickers_avoid", []):
            # 查看原因
            top_themes = snap.get("top_themes", [])
            for tid, _ in top_themes:
                chain = _TRANSMISSION_CHAINS.get(tid, {})
                if ticker in chain.get("tickers_avoid", []):
                    penalty += 25
                    reasons.append(f"宏观主题[{chain.get('label',tid)}]不利于{ticker}")
                    break

        # 受益板块加分
        if ticker in snap.get("tickers_favor", []):
            bonus += 15
            reasons.append(f"宏观主题对{ticker}有利")

        # VIX 异常 → 全面减仓
        vix_chg = snap.get("vix_change_pct", 0)
        if vix_chg > 15:
            penalty += 20
            reasons.append(f"VIX当日涨{vix_chg:.0f}%，市场极度恐慌")
        elif vix_chg > 8:
            penalty += 10
            reasons.append(f"VIX当日涨{vix_chg:.0f}%，波动加剧")

        return {
            "block":   block,
            "penalty": min(penalty, 35),
            "bonus":   min(bonus, 20),
            "reason":  " | ".join(reasons) if reasons else "宏观环境正常",
        }

    except Exception as e:
        return {"block": False, "penalty": 0, "bonus": 0, "reason": f"宏观检查异常：{e}"}


# ─────────────────────────────────────────────────────────────
# Telegram 格式化
# ─────────────────────────────────────────────────────────────

def format_macro_telegram(report: dict) -> str:
    if not report.get("ok"):
        return f"❌ 宏观报告失败"

    today_str = datetime.now(ET).strftime("%Y-%m-%d")
    lines = [f"🌍 <b>宏观环境报告</b>  {today_str}", report.get("risk_environment", ""), ""]

    cal = report.get("economic_calendar", {})
    if cal.get("warnings"):
        lines.append("⚠️ <b>今日重大事件</b>")
        for w in cal["warnings"]:
            lines.append(f"  {w}")
        lines.append("")

    themes = report.get("news_themes", [])
    if themes:
        lines.append("📰 <b>当前宏观主题</b>")
        for chain in themes[:2]:
            lines.append(f"  · {chain.get('label', '')}")
        lines.append("")

    favor = report.get("sectors_favor", [])[:3]
    avoid = report.get("sectors_avoid", [])[:3]
    if favor:
        lines.append(f"✅ 受益板块：{', '.join(favor)}")
    if avoid:
        lines.append(f"🚫 规避板块：{', '.join(avoid)}")

    lines.append("")
    lines.append(f"💡 {report.get('master_action', '')}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# CLI 运行
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    watchlist = sys.argv[1].split(",") if len(sys.argv) > 1 else ["NVDA", "AAPL", "TSLA"]
    report = full_macro_report(watchlist)

    print("\n" + "="*60)
    print("  宏观环境报告")
    print("="*60)
    print(f"  风险环境：{report['risk_environment']}")
    print(f"  操作建议：{report['master_action']}")

    if report.get("economic_calendar", {}).get("warnings"):
        print("\n⚠️  重大事件警告：")
        for w in report["economic_calendar"]["warnings"]:
            print(f"  {w}")

    if report.get("news_themes"):
        print("\n📰 当前宏观主题：")
        for chain in report["news_themes"][:3]:
            print(f"  [{chain['id']}] {chain.get('label', '')}")
            print(f"    逻辑：{chain.get('logic', '')}")
            print(f"    受益：{', '.join(chain.get('tickers_favor', [])[:5])}")
            print(f"    规避：{', '.join(chain.get('tickers_avoid', [])[:5])}")

    if report.get("sector_adjustments"):
        print("\n📊 板块分调整：")
        for sec, delta in sorted(report["sector_adjustments"].items(),
                                  key=lambda x: x[1], reverse=True):
            sign = "+" if delta > 0 else ""
            print(f"  {sec}: {sign}{delta}")

    print("="*60)
