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
import os
import json
import pytz
from datetime import datetime, timezone, timedelta
from typing import Optional

ET = pytz.timezone("America/New_York")
_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
_NEWS_EVENT_SNAPSHOT = os.path.join(_DATA, "news_event_watchlist.json")
_NEWS_EVENT_MAX_AGE_HOURS = 6   # 09:00/14:00两次刷新，间隔约5小时，留余量判定陈旧

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


# ─────────────────────────────────────────────────────────────
# cold_model 集成接口：news_event gate（2026-07-21新增）
# ─────────────────────────────────────────────────────────────
# 设计结论（多轮审查后确定，详见对话记录，不在代码里重复展开）：
#   1. 硬否决判断只信Yahoo个股专属RSS（weight=3那一档），不用综合RSS
#      按ticker正则过滤出的结果——后者存在"文章确实提到了这只股票，但
#      敏感词说的是别的公司"这种归属歧义，无法用正则完全排除。
#   2. 不做否定语境过滤（如"avoids delisted"）——"公司denies fraud"这类
#      新闻本身往往就是坏消息（指控存在，不是被排除），用词语邻近关系
#      做否定判断反而会把真正的坏消息过滤掉，风险比不过滤更大。
#   3. 因此只做warn强力扣分，不做pass=False硬否决（跟debt_event不同，
#      debt_event基于SEC结构化文件事实，这里是关键词模糊匹配，精确度
#      不到"一票否决"的信任门槛）。
#   4. 按ticker分key存快照，模式抄debt_event_monitor.py，cold_model只读
#      快照零网络请求。

def _yahoo_hard_block_check(ticker: str, hours: int = 48) -> dict:
    """
    只用Yahoo个股专属RSS做硬性负面新闻关键词检测，记录具体命中的文章
    （而不是笼统的"命中了这些词"），供人工核实用。
    """
    articles = []
    for entry in _ticker_rss(ticker):
        age = _entry_age_hours(entry)
        if age > hours:
            continue
        title   = getattr(entry, "title", "") or ""
        summary = getattr(entry, "summary", "") or ""
        articles.append({"title": title, "summary": summary[:200], "age_h": round(age, 1)})

    hit_articles = []
    hit_words = set()
    for a in articles:
        text = (a["title"] + " " + a["summary"]).lower()
        matched = [w for w in _HARD_BLOCK if w in text]
        if matched:
            hit_words.update(matched)
            hit_articles.append({"title": a["title"], "matched": matched})

    return {
        "article_count": len(articles),
        "hard_block":    bool(hit_articles),
        "hard_reasons":  sorted(hit_words),
        "hit_articles":  hit_articles[:3],   # 最多留3条，够人工核实即可
    }


def _load_news_event_snapshot() -> dict:
    if not os.path.exists(_NEWS_EVENT_SNAPSHOT):
        return {}
    try:
        with open(_NEWS_EVENT_SNAPSHOT, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_news_event_snapshot(snapshot: dict):
    os.makedirs(_DATA, exist_ok=True)
    tmp = _NEWS_EVENT_SNAPSHOT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, _NEWS_EVENT_SNAPSHOT)


def run_news_event_monitor(watchlist: list) -> dict:
    """
    扫描watchlist每只股票的Yahoo个股新闻，写入按ticker分key的快照文件。
    供scheduler在09:00/14:00调用（跟debt_event_monitor同一节奏）。

    健康监控（2026-07-21新增）：Reuters/AP两个RSS源此前已静默失效很久
    都没人发现，为避免Yahoo源哪天也这样"悄悄失明"却让gate一直误判成
    "无负面新闻"，这里检测——如果watchlist里≥半数股票本次刷新拉到0篇
    文章（对活跃股票这不正常），打印告警日志，不影响本次快照正常写入。
    """
    wl = watchlist or []
    if not wl:
        return {"ok": True, "checked": 0, "hard_blocked": [], "note": "watchlist为空，跳过"}

    snapshot = _load_news_event_snapshot()
    zero_article_count = 0
    hard_blocked = []

    for ticker in wl:
        ticker = ticker.upper()
        try:
            r = _yahoo_hard_block_check(ticker)
        except Exception as e:
            r = {"article_count": 0, "hard_block": False, "hard_reasons": [], "hit_articles": [], "error": str(e)}

        if r["article_count"] == 0:
            zero_article_count += 1
        if r["hard_block"]:
            hard_blocked.append(ticker)

        snapshot[ticker] = {
            "article_count": r["article_count"],
            "hard_block":    r["hard_block"],
            "hard_reasons":  r["hard_reasons"],
            "hit_articles":  r["hit_articles"],
            "checked_at":    str(datetime.now(ET)),
        }

    _save_news_event_snapshot(snapshot)

    if len(wl) >= 3 and zero_article_count >= len(wl) / 2:
        print(f"[NewsEvent] ⚠️ 健康告警：{zero_article_count}/{len(wl)}只股票本次拉到0篇Yahoo新闻，"
              f"疑似RSS源失效（历史上Reuters/AP就发生过这种静默失效），建议人工核实feed是否还能访问")

    return {
        "ok": True,
        "checked": len(wl),
        "hard_blocked": hard_blocked,
        "zero_article_count": zero_article_count,
        "note": (f"发现{len(hard_blocked)}只命中硬性负面关键词：{hard_blocked}"
                 if hard_blocked else "无命中，全部正常"),
    }


def check_ticker_news_event(ticker: str) -> dict:
    """
    从快照读取该ticker的新闻事件检查结果（不发起HTTP请求，供cold_model.py调用）。
    快照缺失/过期时一律按"无负面新闻"处理，不用陈旧数据做判断。
    只返回 pass=True 或 "warn"，不返回 False——精确度不足以支撑硬否决，
    详见文件顶部本节的设计结论。
    """
    snapshot = _load_news_event_snapshot()
    entry = snapshot.get(ticker.upper())
    if not entry:
        return {"pass": True, "note": "新闻快照无记录（不在watchlist或尚未刷新），按无负面新闻处理"}

    try:
        checked_at = datetime.fromisoformat(str(entry["checked_at"]))
        age_h = (datetime.now(ET) - checked_at).total_seconds() / 3600
        if age_h > _NEWS_EVENT_MAX_AGE_HOURS:
            return {"pass": True, "note": f"新闻快照已{age_h:.1f}小时，建议刷新，暂按无负面新闻处理"}
    except Exception:
        pass

    if not entry.get("hard_block"):
        return {"pass": True, "note": f"近48h新闻正常（{entry.get('article_count', 0)}篇）"}

    reasons = "、".join(entry.get("hard_reasons", []))
    hit = entry.get("hit_articles", [])
    example = f"，如《{hit[0]['title'][:60]}》" if hit else ""
    return {
        "pass": "warn",
        "note": f"近48h新闻命中敏感词[{reasons}]{example}，关键词匹配精确度有限仅作强力警示，建议人工核实后再决定",
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
