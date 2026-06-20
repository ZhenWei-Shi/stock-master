"""
九关数据爬虫：
  关卡4 - 散户密度：Reddit + StockTwits + Google Trends
  关卡5 - 机构持仓变化：yfinance institutional_holders
  关卡8 - 财报验证：yfinance earnings history
"""
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime

_HEADERS = {
    "User-Agent": "StockRadar/1.0 (personal research tool)",
    "Accept": "application/json",
}

# ─────────────────────────────────────────────────────────
# 关卡4：散户密度
# ─────────────────────────────────────────────────────────

def get_reddit_sentiment(ticker: str) -> dict:
    """
    Reddit JSON API（免费，无需key）
    扫描 wallstreetbets / stocks / investing 三个社区
    """
    subs = ["wallstreetbets", "stocks", "investing"]
    total_mentions = 0
    top_posts = []

    for sub in subs:
        try:
            url = f"https://www.reddit.com/r/{sub}/search.json"
            params = {"q": ticker, "sort": "new", "limit": 25, "t": "week"}
            r = requests.get(url, headers=_HEADERS, params=params, timeout=8)
            if r.status_code != 200:
                continue
            data = r.json()
            posts = data.get("data", {}).get("children", [])
            total_mentions += len(posts)
            for p in posts[:2]:
                d = p.get("data", {})
                top_posts.append({
                    "sub": sub,
                    "title": d.get("title", "")[:80],
                    "score": d.get("score", 0),
                    "comments": d.get("num_comments", 0),
                })
        except Exception:
            continue

    # 热度分：7天提及 < 10 → 低 / 10-50 → 中 / > 50 → 高（散户扎堆）
    if total_mentions < 10:
        heat = "low"
        score = max(0, total_mentions * 3)
    elif total_mentions < 50:
        heat = "medium"
        score = 30 + (total_mentions - 10) * 1
    else:
        heat = "high"
        score = min(100, 70 + (total_mentions - 50))

    return {
        "source": "Reddit",
        "mentions_7d": total_mentions,
        "heat": heat,
        "score": score,
        "top_posts": top_posts[:4],
    }


def get_stocktwits_sentiment(ticker: str) -> dict:
    """
    StockTwits 公开 API（免费，无需key）
    返回最近30条消息的 Bullish/Bearish 比例
    """
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        r = requests.get(url, headers=_HEADERS, timeout=8)
        if r.status_code != 200:
            return {"source": "StockTwits", "error": f"HTTP {r.status_code}"}

        data = r.json()
        messages = data.get("messages", [])
        if not messages:
            return {"source": "StockTwits", "error": "无消息数据"}

        bull = sum(1 for m in messages
                   if m.get("entities", {}).get("sentiment", {}) and
                      m["entities"]["sentiment"].get("basic") == "Bullish")
        bear = sum(1 for m in messages
                   if m.get("entities", {}).get("sentiment", {}) and
                      m["entities"]["sentiment"].get("basic") == "Bearish")
        total = len(messages)
        bull_pct = round(bull / total * 100, 1) if total else 0
        bear_pct = round(bear / total * 100, 1) if total else 0

        # 极端情绪 → 反向信号（神父反身性原则）
        # 散户热度 = 消息总数（越多越热门）
        msg_score = min(100, total * 3)

        return {
            "source": "StockTwits",
            "total_messages": total,
            "bull_pct": bull_pct,
            "bear_pct": bear_pct,
            "neutral_pct": round(100 - bull_pct - bear_pct, 1),
            "score": msg_score,
            "contrarian_signal": bull_pct > 80 or bear_pct > 80,
        }
    except Exception as e:
        return {"source": "StockTwits", "error": str(e)}


def get_google_trends(ticker: str) -> dict:
    """
    Google Trends（pytrends，免费）
    搜索热度 0-100，峰值 = 散户疯狂 = 警告信号
    """
    try:
        from pytrends.request import TrendReq
        pt = TrendReq(hl="en-US", tz=360, timeout=(5, 15))
        pt.build_payload([ticker], cat=0, timeframe="now 7-d", geo="US")
        df = pt.interest_over_time()

        if df.empty or ticker not in df.columns:
            return {"source": "GoogleTrends", "error": "无数据"}

        current = int(df[ticker].iloc[-1])
        peak = int(df[ticker].max())
        avg = float(df[ticker].mean())

        # 热度分直接用 Google 给的 0-100
        return {
            "source": "GoogleTrends",
            "current_score": current,
            "peak_7d": peak,
            "avg_7d": round(avg, 1),
            "score": current,
            "heat": "high" if current > 70 else "medium" if current > 30 else "low",
        }
    except Exception as e:
        return {"source": "GoogleTrends", "error": str(e)}


def get_retail_density(ticker: str) -> dict:
    """
    综合散户密度评分（关卡4）
    合并 Reddit + StockTwits + Google Trends
    """
    reddit = get_reddit_sentiment(ticker)
    twits = get_stocktwits_sentiment(ticker)
    trends = get_google_trends(ticker)

    scores = []
    if "score" in reddit:
        scores.append(reddit["score"])
    if "score" in twits:
        scores.append(twits["score"])
    if "score" in trends:
        scores.append(trends["score"])

    avg_score = round(sum(scores) / len(scores), 1) if scores else None

    if avg_score is None:
        gate_status = "manual"
        gate_detail = "无法获取散户密度数据，请人工判断"
    elif avg_score < 25:
        gate_status = "pass"
        gate_detail = f"散户热度低（{avg_score}/100），大众未发现，符合神父原则"
    elif avg_score < 55:
        gate_status = "warn"
        gate_detail = f"散户热度中等（{avg_score}/100），有人开始关注，需留意"
    else:
        gate_status = "fail"
        gate_detail = f"散户热度高（{avg_score}/100），大众扎堆，需等血洗后再进"

    return {
        "avg_score": avg_score,
        "gate_status": gate_status,
        "gate_detail": gate_detail,
        "reddit": reddit,
        "stocktwits": twits,
        "trends": trends,
    }


# ─────────────────────────────────────────────────────────
# 关卡5：机构持仓变化
# ─────────────────────────────────────────────────────────

def get_institutional_analysis(ticker: str) -> dict:
    """
    yfinance institutional_holders + major_holders
    分析机构是在增仓还是减仓
    """
    try:
        t = yf.Ticker(ticker)
        holders = t.institutional_holders
        major = t.major_holders

        inst_pct = None
        if major is not None and not major.empty:
            for _, row in major.iterrows():
                label = str(row.iloc[1]).lower() if len(row) > 1 else ""
                if "institution" in label:
                    try:
                        inst_pct = float(str(row.iloc[0]).replace("%", "")) / 100
                    except Exception:
                        pass

        top_holders = []
        recent_buyers = []
        recent_sellers = []

        if holders is not None and not holders.empty:
            for _, row in holders.iterrows():
                pct_out = row.get("% Out") if "% Out" in holders.columns else None
                # yfinance 不同版本字段名不一致，逐一尝试
                _chg_col = next((c for c in ("pctChange", "% Change", "Change") if c in holders.columns), None)
                pct_change = row.get(_chg_col) if _chg_col else None

                try:
                    shares_val = int(row.get("Shares", 0)) if row.get("Shares") is not None else None
                except Exception:
                    shares_val = None
                holder_data = {
                    "name": str(row.get("Holder", "Unknown")),
                    "shares": shares_val,
                    "pct_out": float(pct_out) if pct_out is not None else None,
                    "pct_change": float(pct_change) if pct_change is not None else None,
                    "date": str(row.get("Date Reported", ""))[:10],
                }
                top_holders.append(holder_data)

                if pct_change is not None:
                    try:
                        chg = float(pct_change)
                        if chg > 0.01:
                            recent_buyers.append(holder_data["name"])
                        elif chg < -0.01:
                            recent_sellers.append(holder_data["name"])
                    except Exception:
                        pass

        # 判断趋势
        net_direction = len(recent_buyers) - len(recent_sellers)

        if inst_pct and inst_pct > 0.35 and net_direction >= 0:
            gate_status = "pass"
            gate_detail = f"机构持仓{inst_pct:.1%}，净买入机构{len(recent_buyers)}家，趋势积极"
        elif inst_pct and inst_pct > 0.15:
            gate_status = "warn"
            gate_detail = f"机构持仓{inst_pct:.1%}，需持续观察机构动向"
        elif net_direction < -1:
            gate_status = "fail"
            gate_detail = f"多家机构减仓（{len(recent_sellers)}家卖出），机构在出货"
        else:
            gate_status = "manual"
            gate_detail = "机构数据不足，请查 SEC EDGAR 13F 手动确认"

        return {
            "inst_pct": inst_pct,
            "top_holders": top_holders[:8],
            "recent_buyers": recent_buyers[:5],
            "recent_sellers": recent_sellers[:5],
            "net_direction": net_direction,
            "gate_status": gate_status,
            "gate_detail": gate_detail,
        }
    except Exception as e:
        return {
            "gate_status": "manual",
            "gate_detail": f"获取机构数据失败：{e}",
            "top_holders": [],
        }


# ─────────────────────────────────────────────────────────
# 关卡8：财报验证
# ─────────────────────────────────────────────────────────

def get_earnings_analysis(ticker: str) -> dict:
    """
    yfinance 历史财报：EPS实际 vs 预期，过去4个季度
    连续超预期 = 论文验证信号
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.earnings_history

        if hist is None or hist.empty:
            return {
                "gate_status": "manual",
                "gate_detail": "无历史财报数据，请手动核查",
                "quarters": [],
            }

        quarters = []
        for idx, row in hist.head(8).iterrows():
            eps_est = row.get("epsEstimate")
            eps_act = row.get("epsActual")
            date = str(idx)[:10]  # date是index，不是名为"quarter"的列

            if eps_est is None or eps_act is None:
                continue

            eps_est_f = float(eps_est)
            eps_act_f = float(eps_act)
            beat = eps_act_f > eps_est_f
            # surprise_pct：百分比超预期 = (实际-预期)/|预期|×100，epsDifference是美元绝对差值
            if eps_est_f != 0:
                surprise_pct_val = round((eps_act_f - eps_est_f) / abs(eps_est_f) * 100, 1)
            else:
                surprise_pct_val = None
            quarters.append({
                "date": date,
                "eps_est": round(eps_est_f, 3),
                "eps_act": round(eps_act_f, 3),
                "surprise_pct": surprise_pct_val,
                "beat": beat,
            })

        recent = quarters[:4]
        beats = sum(1 for q in recent if q.get("beat") is True)
        total = len(recent)
        beat_rate = beats / total if total > 0 else 0

        if beat_rate >= 0.75:
            gate_status = "pass"
            gate_detail = f"过去{total}季度超预期{beats}次（{beat_rate:.0%}），财报持续验证论文"
        elif beat_rate >= 0.5:
            gate_status = "warn"
            gate_detail = f"过去{total}季度超预期{beats}次（{beat_rate:.0%}），表现一般"
        else:
            gate_status = "fail"
            gate_detail = f"过去{total}季度仅超预期{beats}次（{beat_rate:.0%}），财报未验证论文"

        return {
            "beat_rate": round(beat_rate, 2),
            "beats": beats,
            "total_quarters": total,
            "quarters": quarters,
            "gate_status": gate_status,
            "gate_detail": gate_detail,
        }
    except Exception as e:
        return {
            "gate_status": "manual",
            "gate_detail": f"财报数据获取失败：{e}",
            "quarters": [],
        }
