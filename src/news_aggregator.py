"""
多源新闻聚合器（完全免费）

数据源（按权威度排序）：
  1. Yahoo Finance 每股 RSS    — 最相关，有ticker过滤
  2. Reuters Business RSS      — 权威，延迟低
  3. MarketWatch RSS           — 市场专注
  4. AP Business RSS           — 权威综合
  5. Seeking Alpha RSS         — 分析师意见

比 yfinance.news 快 1-4 小时，覆盖面更广。
"""

import feedparser
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

# ── 情绪关键词 ─────────────────────────────────────────────────
_HARD_BLOCK = [
    "fraud", "sec investigation", "delisted", "bankruptcy", "doj",
    "criminal", "restatement", "accounting irregularities",
    "going concern", "subpoena", "class action", "securities fraud",
]
_BEAR_WORDS = [
    "downgrade", "miss", "disappointing", "guidance cut", "guidance lowered",
    "layoff", "recall", "competition", "margin pressure", "revenue decline",
    "profit warning", "earnings miss", "short seller", "short report",
]
_BULL_WORDS = [
    "upgrade", "beat", "record", "raised guidance", "buyback", "dividend",
    "partnership", "contract", "approval", "fda approved", "strong demand",
    "outperform", "accelerating growth", "record revenue", "blowout",
]

_RSS_SOURCES = [
    ("reuters",     "https://feeds.reuters.com/reuters/businessNews",    3),
    ("ap",          "https://rsshub.app/apnews/topics/business",          2),
    ("marketwatch", "https://feeds.marketwatch.com/marketwatch/topstories/", 2),
]


def _parse_feed(url: str, timeout: int = 8) -> list:
    try:
        feed = feedparser.parse(url)
        return feed.entries or []
    except Exception:
        return []


def _ticker_rss(ticker: str) -> list:
    """Yahoo Finance 每股 RSS（最相关）"""
    url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    return _parse_feed(url)


def _entry_age_hours(entry) -> float:
    """返回新闻发布距今小时数（无法解析返回999）"""
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                dt = datetime(*t[:6], tzinfo=timezone.utc)
                return (datetime.now(timezone.utc) - dt).total_seconds() / 3600
            except Exception:
                pass
    return 999.0


def _score_text(text: str) -> int:
    """简单情绪打分：正 = 看多，负 = 看空，0 = 中性"""
    text = text.lower()
    score = 0
    score += sum(1 for w in _BULL_WORDS if w in text)
    score -= sum(1 for w in _BEAR_WORDS if w in text)
    return score


def get_news_for_ticker(ticker: str, hours: int = 48) -> dict:
    """
    聚合单只股票的最新新闻。

    返回：
      hard_block  — 存在极度负面新闻，建议跳过该标的
      sentiment   — bull / bear / neutral
      score       — 情绪分（正=看多，负=看空）
      articles    — 文章列表
      top_headline — 最新最相关标题
    """
    articles = []

    # 1. Yahoo Finance 每股 RSS（精准）
    for entry in _ticker_rss(ticker):
        age = _entry_age_hours(entry)
        if age > hours:
            continue
        title   = getattr(entry, "title", "") or ""
        summary = getattr(entry, "summary", "") or ""
        articles.append({
            "source":  "yahoo",
            "title":   title,
            "summary": summary[:200],
            "age_h":   round(age, 1),
            "weight":  3,  # 最高权重：股票专属
        })

    # 2. 综合 RSS，按 ticker 关键词过滤
    # 【2026-07-21修复】必须加单词边界，否则"ON"(半导体)/"F"(福特)/"ALL"(好事达)
    # 这类短代码/常见单词ticker会误命中任何包含该子串的不相关文本
    # （如"ON"命中"...earnings on Friday..."），已用真实测试验证过修复前会误判。
    company_pat = re.compile(r"\b" + re.escape(ticker) + r"\b", re.IGNORECASE)
    for source_name, url, weight in _RSS_SOURCES:
        for entry in _parse_feed(url):
            age   = _entry_age_hours(entry)
            if age > hours:
                continue
            title   = getattr(entry, "title", "") or ""
            summary = getattr(entry, "summary", "") or ""
            text    = title + " " + summary
            if not company_pat.search(text):
                continue
            articles.append({
                "source":  source_name,
                "title":   title,
                "summary": summary[:200],
                "age_h":   round(age, 1),
                "weight":  weight,
            })

    # 去重（相同标题只保留一条）
    seen = set()
    unique = []
    for a in articles:
        key = a["title"][:60]
        if key not in seen:
            seen.add(key)
            unique.append(a)
    articles = sorted(unique, key=lambda x: x["age_h"])

    # 情绪分析
    combined = " ".join(a["title"] + " " + a["summary"] for a in articles)
    combined_lower = combined.lower()

    hard_hits = [w for w in _HARD_BLOCK if w in combined_lower]
    bull_hits  = [w for w in _BULL_WORDS if w in combined_lower]
    bear_hits  = [w for w in _BEAR_WORDS if w in combined_lower]

    raw_score = len(bull_hits) - len(bear_hits)
    if hard_hits:
        sentiment = "HARD_BLOCK"
    elif raw_score >= 2:
        sentiment = "bull"
    elif raw_score <= -2:
        sentiment = "bear"
    else:
        sentiment = "neutral"

    top = articles[0]["title"] if articles else "（无近期新闻）"

    return {
        "ticker":       ticker,
        "article_count": len(articles),
        "hours_window": hours,
        "hard_block":   bool(hard_hits),
        "hard_reasons": hard_hits,
        "sentiment":    sentiment,
        "score":        raw_score,
        "bull_keywords": bull_hits[:5],
        "bear_keywords": bear_hits[:5],
        "top_headline": top,
        "articles":     articles[:8],
    }


def batch_news_filter(tickers: list, hours: int = 48) -> dict:
    """
    批量新闻过滤，替代 scheduler 的 news_prefilter。
    返回 passed / blocked / neutral / warnings。
    """
    passed, blocked, neutral, warnings = [], [], [], {}

    for ticker in tickers:
        result = get_news_for_ticker(ticker, hours)
        if result["hard_block"]:
            blocked.append({
                "ticker": ticker,
                "reason": f"负面新闻：{', '.join(result['hard_reasons'])}",
                "headline": result["top_headline"],
            })
        elif result["article_count"] == 0:
            neutral.append(ticker)
        else:
            passed.append(ticker)
            if result["sentiment"] == "bear":
                warnings[ticker] = f"情绪偏空（{', '.join(result['bear_keywords'][:3])}）"

    return {
        "passed":   passed,
        "blocked":  blocked,
        "neutral":  neutral,
        "warnings": warnings,
    }


def format_news_telegram(ticker: str, result: dict) -> str:
    """格式化单股新闻摘要"""
    emoji = {"bull": "📈", "bear": "📉", "neutral": "📋",
             "HARD_BLOCK": "🚨"}.get(result["sentiment"], "📋")
    lines = [f"{emoji} <b>{ticker}</b> 近{result['hours_window']}h新闻（{result['article_count']}条）"]
    if result["hard_block"]:
        lines.append(f"⛔ 封锁：{', '.join(result['hard_reasons'])}")
    lines.append(f"情绪：{result['sentiment']}  分：{result['score']:+d}")
    if result["top_headline"]:
        lines.append(f"头条：{result['top_headline'][:100]}")
    return "\n".join(lines)
