"""
基本面 + 市场情绪数据：
  - 空头兴趣 / 空头挤压评分
  - 分析师评级（升/降级历史）
  - 新闻情绪（关键词评分）
  - 内部人交易（yfinance + SEC EDGAR）
  - 历史财报反应（涨跌幅）
  - CNN Fear & Greed（免费 JSON）
  - 板块轮动强度（SPDR ETF RS）
"""
import yfinance as yf
import numpy as np
import requests
from datetime import datetime, timedelta


# ── 空头兴趣 ────────────────────────────────────────────────
def get_short_interest(info: dict, ticker: str = "") -> dict:
    """
    Short Squeeze 完整信号 v2

    五因素联动模型：
      1. Short % of Float（> 20% = 高危空压）
      2. Days to Cover（< 3 天 = 空头极难逃脱）
      3. Float Size（低流通股 = 价格更容易被逼空）
      4. 机构增持趋势（空头同时机构买入 = 经典逼空前置）
      5. 期权 Call OI 异动（散户 Call 大增 = 散户发现逼空机会）

    满分 10 分，≥ 7 = 高逼空风险
    """
    sr   = info.get("shortRatio")
    spf  = info.get("shortPercentOfFloat")
    ss   = info.get("sharesShort")
    flt  = info.get("floatShares")
    inst = info.get("institutionPercentHeld")
    shares_out = info.get("sharesOutstanding")

    score, signals, details = 0, [], {}

    # ── 因素1：空头占比 ──────────────────────────────────
    if spf is not None and _valid(spf):
        pct = float(spf) * 100
        details["short_pct_float"] = round(pct, 1)
        if pct > 30:
            score += 3; signals.append(f"空头占浮动股 {pct:.1f}%（>30%，极高空压，历史 GME/AMC 级别）")
        elif pct > 20:
            score += 2; signals.append(f"空头占浮动股 {pct:.1f}%（>20%，高危空压）")
        elif pct > 10:
            score += 1; signals.append(f"空头占浮动股 {pct:.1f}%（中等空压）")
        else:
            details["note_si"] = f"空头占比{pct:.1f}%，空压不高"

    # ── 因素2：Days to Cover ─────────────────────────────
    if sr is not None and _valid(sr):
        days = float(sr)
        details["days_to_cover"] = round(days, 1)
        if days < 2:
            score += 3; signals.append(f"回补天数仅 {days:.1f} 天（空头极难逃脱）")
        elif days < 4:
            score += 2; signals.append(f"回补天数 {days:.1f} 天（逃脱困难）")
        elif days < 7:
            score += 1; signals.append(f"回补天数 {days:.1f} 天（可控但有压力）")

    # ── 因素3：低流通股（Float < 50M = 容易被逼空）───────
    if flt is not None and _valid(flt):
        flt_m = float(flt) / 1e6
        details["float_shares_m"] = round(flt_m, 1)
        if flt_m < 20:
            score += 2; signals.append(f"超低流通股 {flt_m:.0f}M（<20M，极易被逼空）")
        elif flt_m < 50:
            score += 1; signals.append(f"低流通股 {flt_m:.0f}M（<50M，有逼空空间）")
        elif flt_m > 500:
            details["note_float"] = f"高流通股 {flt_m:.0f}M，逼空难度大"

    # ── 因素4：机构持仓趋势（代理：当前持仓比例）────────
    # 真实 QoQ 数据在 institutional_13f.py
    if inst is not None and _valid(inst):
        inst_pct = float(inst) * 100
        details["inst_pct"] = round(inst_pct, 1)
        if inst_pct > 40:
            score += 1; signals.append(f"机构持仓 {inst_pct:.0f}%，若持续增持可触发逼空")

    # ── 因素5：期权异动信号（需传入 gamma_score）────────
    # 在 app.py 层面整合，这里给出占位
    details["call_oi_note"] = "需结合 /api/options-full 的 gamma_score 判断"

    # ── 综合评级 ─────────────────────────────────────────
    score = min(10, score)
    if score >= 7:
        potential = "极高"
        note = "多因素共振，Short Squeeze 风险极高，做空者谨慎，做多者关注催化剂"
    elif score >= 5:
        potential = "高"
        note = "具备 Short Squeeze 潜力，需要催化剂（财报/新闻/产品发布）触发"
    elif score >= 3:
        potential = "中"
        note = "有一定空压，但不足以单独触发逼空"
    else:
        potential = "低"
        note = "空压不足，逼空概率低"

    return {
        "short_ratio":        round(float(sr),  2) if sr and _valid(sr) else None,
        "short_pct_float":    details.get("short_pct_float"),
        "shares_short":       int(float(ss)) if ss and _valid(ss) else None,
        "float_shares_m":     details.get("float_shares_m"),
        "inst_pct":           details.get("inst_pct"),
        "squeeze_score":      score,
        "squeeze_score_max":  10,
        "squeeze_potential":  potential,
        "signals":            signals,
        "details":            details,
        "verdict":            note,
        "classic_setup": (
            score >= 5
            and details.get("short_pct_float", 0) > 15
            and details.get("float_shares_m", 999) < 100
        ),
    }


# ── 分析师评级 ───────────────────────────────────────────────
def get_analyst_ratings(ticker: str) -> dict:
    try:
        t    = yf.Ticker(ticker)
        recs = t.recommendations
        if recs is None or recs.empty:
            return {"error": "无评级数据", "upgrades": [], "downgrades": [], "summary": {}}

        upgrades, downgrades = [], []
        if "firm" in recs.columns:
            for idx, row in recs.tail(30).iterrows():
                action = str(row.get("action", "")).lower()
                item = {
                    "date":       str(idx)[:10],
                    "firm":       str(row.get("firm", "")),
                    "from_grade": str(row.get("fromGrade", "")),
                    "to_grade":   str(row.get("toGrade", "")),
                }
                if "up" in action or "init" in action or "reit" in action:
                    upgrades.append(item)
                elif "down" in action:
                    downgrades.append(item)

        summary = {}
        if "strongBuy" in recs.columns:
            last = recs.tail(1).iloc[0]
            sb, b, h, s, ss = (int(last.get(k, 0)) for k in
                               ["strongBuy", "buy", "hold", "sell", "strongSell"])
            total = sb + b + h + s + ss
            summary = {
                "strongBuy": sb, "buy": b, "hold": h,
                "sell": s, "strongSell": ss, "total": total,
                "bull_pct": round((sb + b) / total * 100, 1) if total else 0,
            }

        return {
            "upgrades":         upgrades[-5:],
            "downgrades":       downgrades[-5:],
            "summary":          summary,
            "total_upgrades":   len(upgrades),
            "total_downgrades": len(downgrades),
        }
    except Exception as e:
        return {"error": str(e), "upgrades": [], "downgrades": [], "summary": {}}


# ── 新闻情绪 v2（否定词修复）────────────────────────────────
_BULL = {"beat","surge","rally","upgrade","buy","record","growth","exceed","above",
         "bullish","outperform","raise","profit","positive","strong","soar","jump",
         "spike","boom","breakout","win","gain","top","high",
         "buyback","repurchase","dividend","partnership","contract","awarded",
         "approval","approved","topped","exceeded"}
_BEAR = {"miss","drop","fall","downgrade","sell","loss","below","concern","cut",
         "reduce","bearish","underperform","layoff","negative","weak","decline",
         "crash","warn","warning","probe","investigation","fraud","collapse","tank",
         "fails","failed","missing","disappoints","disappointing",
         "recall","default","bankruptcy","resign","fired","subpoena","lawsuit",
         "dilution","offering"}

# 否定词：严格限于语法否定词，语义失败词已移入 _BEAR
_NEGATORS = {"not", "no", "never", "didn't", "won't", "can't", "cannot"}

def _score_title(title: str) -> int:
    """
    带否定词处理的情绪评分。
    规则：否定词 + 正面词 = 负面（"didn't beat" = bear）
          否定词 + 负面词 = 正面（"no layoffs" = bull）
    """
    tokens = title.lower().split()
    score  = 0
    for i, word in enumerate(tokens):
        # 清理标点
        clean = word.strip(".,!?;:'\"()")
        # 前2个词里有否定词
        prev_neg = any(tokens[max(0, i-2):i][j].strip(".,!?") in _NEGATORS
                       for j in range(len(tokens[max(0, i-2):i])))
        if clean in _BULL:
            score += -1 if prev_neg else +1
        elif clean in _BEAR:
            score += +1 if prev_neg else -1
    return score


def get_news_sentiment(ticker: str) -> dict:
    try:
        news = yf.Ticker(ticker).news or []
        items, bull, bear = [], 0, 0
        for n in news[:15]:
            title = n.get("title", "")
            sc    = _score_title(title)
            sent  = "bull" if sc > 0 else "bear" if sc < 0 else "neutral"
            if sc > 0: bull += 1
            elif sc < 0: bear += 1
            ts = n.get("providerPublishTime", 0)
            items.append({
                "title":     title[:96],
                "source":    n.get("publisher", ""),
                "date":      datetime.fromtimestamp(ts).strftime("%m/%d") if ts else "",
                "sentiment": sent,
                "score":     sc,
                "link":      n.get("link", ""),
            })
        total = len(items)
        overall = ("bullish"  if bull  > bear * 1.5 else
                   "bearish"  if bear  > bull * 1.5 else "neutral")
        return {
            "items": items, "bull_count": bull, "bear_count": bear,
            "neutral_count": total - bull - bear, "sentiment": overall,
            "note": "⚠️ 基于关键词匹配估算，非NLP情感模型；否定词翻转已启用（'didn't beat'→看空，'no layoffs'→看多）",
        }
    except Exception as e:
        return {"error": str(e), "items": [], "sentiment": "neutral"}


# ── 内部人交易 ───────────────────────────────────────────────
def get_insider_trades(ticker: str) -> dict:
    try:
        t       = yf.Ticker(ticker)
        insider = t.insider_transactions
        if insider is None or insider.empty:
            return _sec_insider(ticker)

        trades = []
        for _, row in insider.head(15).iterrows():
            sh = row.get("Shares"); val = row.get("Value")
            trades.append({
                "date":        str(row.get("Start Date", ""))[:10],
                "insider":     str(row.get("Insider Trading", "")),
                "relation":    str(row.get("Relationship", "")),
                "transaction": str(row.get("Transaction", "")),
                "shares":      int(float(sh))  if sh  is not None and _valid(sh)  else None,
                "value":       int(float(val)) if val is not None and _valid(val) else None,
            })

        # 过滤期权行权（非真实买卖意愿），按金额而非笔数判断方向
        def is_exercise(t): return "exercise" in t["transaction"].lower()
        buy_val  = sum(t["value"] or 0 for t in trades
                       if ("purchase" in t["transaction"].lower()
                           or "acquisition" in t["transaction"].lower())
                       and not is_exercise(t))
        sell_val = sum(t["value"] or 0 for t in trades
                       if "sale" in t["transaction"].lower() and not is_exercise(t))
        buys  = [t for t in trades if "purchase" in t["transaction"].lower() and not is_exercise(t)]
        sells = [t for t in trades if "sale"     in t["transaction"].lower() and not is_exercise(t)]
        if buy_val > 0 and sell_val == 0:
            sig = "bullish_strong"
        elif buy_val > 0 and buy_val > sell_val * 1.5:
            sig = "bullish"
        elif sell_val > buy_val * 1.5:
            sig = "bearish"
        else:
            sig = "neutral"
        return {"trades": trades[:10], "buys": len(buys), "sells": len(sells),
                "buy_value": buy_val, "sell_value": sell_val,
                "signal": sig, "source": "yfinance"}
    except Exception:
        return _sec_insider(ticker)

def _sec_insider(ticker: str) -> dict:
    try:
        since = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        url   = (f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
                 f"&forms=4&dateRange=custom&startdt={since}")
        r = requests.get(url, headers={"User-Agent": "StockRadar/1.0 research@example.com"},
                         timeout=8)
        hits  = r.json().get("hits", {}).get("hits", []) if r.status_code == 200 else []
        trades = [{"date":    h["_source"].get("file_date", "")[:10],
                   "insider": (h["_source"].get("display_names") or ["Unknown"])[0],
                   "relation": "", "transaction": "Form 4",
                   "shares": None, "value": None} for h in hits[:10]]
        return {"trades": trades, "buys": 0, "sells": 0,
                "signal": "unknown", "source": "SEC EDGAR"}
    except Exception as e:
        return {"error": str(e), "trades": [], "buys": 0, "sells": 0, "signal": "unknown"}

def _valid(v):
    try: return not np.isnan(float(v))
    except Exception: return False


# ── 历史财报反应 ──────────────────────────────────────────────
def get_earnings_reaction(ticker: str) -> dict:
    try:
        t      = yf.Ticker(ticker)
        eh     = t.earnings_history
        prices = t.history(period="3y")
        if eh is None or eh.empty or prices.empty:
            return {"error": "数据不足", "reactions": []}

        prices.index = prices.index.tz_localize(None)
        reactions = []
        for idx, row in eh.head(8).iterrows():
            try:
                q_date = str(idx)[:10]  # date是index，不是名为"quarter"的列
                q_dt   = datetime.strptime(q_date, "%Y-%m-%d")
                window = prices.loc[q_dt - timedelta(days=4): q_dt + timedelta(days=4)]
                if len(window) < 2:
                    continue
                dates = window.index.tolist()
                ei    = min(range(len(dates)), key=lambda i: abs((dates[i] - q_dt).days))
                prev  = float(window["Close"].iloc[ei - 1]) if ei > 0 else None
                earn  = float(window["Close"].iloc[ei])
                nxt   = float(window["Close"].iloc[ei + 1]) if ei + 1 < len(window) else None
                eps_e = row.get("epsEstimate"); eps_a = row.get("epsActual")
                reactions.append({
                    "date":     q_date,
                    "day_chg":  round((earn - prev) / prev * 100, 2) if prev else None,
                    "next_chg": round((nxt  - earn) / earn  * 100, 2) if nxt  else None,
                    "eps_est":  round(float(eps_e), 3) if eps_e is not None else None,
                    "eps_act":  round(float(eps_a), 3) if eps_a is not None else None,
                    "beat":     bool(float(eps_a) > float(eps_e)) if (eps_e is not None and eps_a is not None) else None,
                })
            except Exception:
                continue

        chgs   = [r["day_chg"] for r in reactions if r["day_chg"] is not None]
        avg_mv = round(float(np.mean(np.abs(chgs))), 2) if chgs else None
        return {"reactions": reactions, "avg_move": avg_mv}
    except Exception as e:
        return {"error": str(e), "reactions": []}


# ── CNN Fear & Greed ─────────────────────────────────────────
def get_fear_greed() -> dict:
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        r   = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}"}
        data  = r.json()
        score = data.get("fear_and_greed", {}).get("score")
        label = data.get("fear_and_greed", {}).get("rating", "")
        hist  = data.get("fear_and_greed_historical", {}).get("data", [])
        trend = [{"date": h.get("x", "")[:10], "score": round(float(h.get("y", 0)), 1)}
                 for h in hist[-30:]] if hist else []
        return {
            "score": round(float(score), 1) if score is not None else None,
            "label": label,
            "color": ("var(--red)" if score and score < 25 else
                      "var(--orange)" if score and score < 45 else
                      "var(--muted)" if score and score < 55 else
                      "var(--green)" if score and score < 75 else "#39d353"),
            "trend": trend,
        }
    except Exception as e:
        return {"error": str(e)}


# ── 板块轮动强度 ──────────────────────────────────────────────
_SECTORS = {
    "XLK": "科技", "XLF": "金融", "XLE": "能源", "XLV": "医疗",
    "XLY": "消费(非必需)", "XLP": "消费(必需)", "XLI": "工业",
    "XLB": "原材料", "XLRE": "房地产", "XLU": "公用事业", "XLC": "通信",
}

def get_sector_rotation() -> dict:
    try:
        tickers = list(_SECTORS.keys())
        spy3m   = yf.Ticker("SPY").history(period="3mo")["Close"]
        spy_ret = float((spy3m.iloc[-1] - spy3m.iloc[0]) / spy3m.iloc[0] * 100) if len(spy3m) >= 2 else 0

        sectors = []
        for sym, name in _SECTORS.items():
            try:
                h = yf.Ticker(sym).history(period="3mo")["Close"]
                if len(h) < 2:
                    continue
                ret = float((h.iloc[-1] - h.iloc[0]) / h.iloc[0] * 100)
                sectors.append({
                    "sym":         sym,
                    "name":        name,
                    "ret_3m":      round(ret, 1),
                    "vs_spy":      round(ret - spy_ret, 1),
                    "momentum":    "hot"  if ret - spy_ret > 5  else
                                   "warm" if ret - spy_ret > 0  else "cold",
                })
            except Exception:
                continue

        sectors.sort(key=lambda x: x["ret_3m"], reverse=True)
        return {"sectors": sectors, "spy_ret": round(spy_ret, 1)}
    except Exception as e:
        return {"error": str(e), "sectors": []}


# ── 财报质量评分（P1 新增）────────────────────────────────────

def get_earnings_quality(ticker: str) -> dict:
    """
    财报质量深度评分（满分 100）

    六个维度：
      1. EPS 超预期幅度（Beat Magnitude）— 超 5% 才算有效超预期
      2. 营收超预期（Revenue Beat）
      3. 营收增长加速（Acceleration）— 本季 YoY > 上季 YoY = 最强信号
      4. 利润率趋势（Margin Expansion）
      5. 指引方向（Guidance）— 上调/维持/下调
      6. 连续超预期次数（Consistency）— 机构最看重的信任度指标

    O'Neil CANSLIM 要求：EPS 增长 ≥ 25%，营收增长 ≥ 25%，且加速
    """
    try:
        t    = yf.Ticker(ticker)
        info = t.info
        eh   = t.earnings_history
        qf   = t.quarterly_financials
        qi   = t.quarterly_income_stmt

        score      = 0
        signals    = []
        dimensions = {}

        # ── 1. EPS 超预期幅度 ────────────────────────────────
        eps_beats = []
        if eh is not None and not eh.empty:
            for _, row in eh.head(4).iterrows():
                est = row.get("epsEstimate")
                act = row.get("epsActual")
                if est is not None and act is not None and _valid(est) and _valid(act):
                    est_f, act_f = float(est), float(act)
                    if abs(est_f) > 0.001:
                        beat_pct = (act_f - est_f) / abs(est_f) * 100
                        eps_beats.append(round(beat_pct, 1))

        if eps_beats:
            latest_beat = eps_beats[0]
            avg_beat    = float(np.mean(eps_beats))
            if latest_beat > 10:
                score += 25; signals.append(f"EPS超预期{latest_beat:.1f}%（强烈超预期）")
            elif latest_beat > 5:
                score += 18; signals.append(f"EPS超预期{latest_beat:.1f}%（有效超预期）")
            elif latest_beat > 0:
                score += 8;  signals.append(f"EPS超预期{latest_beat:.1f}%（轻微超预期）")
            else:
                signals.append(f"EPS不及预期{abs(latest_beat):.1f}%")
            dimensions["eps_beat"] = {"latest_pct": latest_beat, "avg_4q": round(avg_beat, 1),
                                       "history": eps_beats}

        # ── 2. 营收超预期 ────────────────────────────────────
        rev_surprise = info.get("revenuePerShare")  # 用增长率替代
        rev_growth   = info.get("revenueGrowth")
        rev_qoq      = info.get("revenueGrowth")    # yfinance 同一字段

        if rev_growth is not None and _valid(rev_growth):
            rg = float(rev_growth) * 100
            if rg > 30:
                score += 20; signals.append(f"营收 YoY +{rg:.0f}%（CANSLIM 25%+达标）")
            elif rg > 20:
                score += 14; signals.append(f"营收 YoY +{rg:.0f}%（增速强劲）")
            elif rg > 10:
                score += 8;  signals.append(f"营收 YoY +{rg:.0f}%（增速一般）")
            elif rg > 0:
                score += 3;  signals.append(f"营收 YoY +{rg:.0f}%（微增）")
            else:
                signals.append(f"营收 YoY {rg:.0f}%（负增长）")
            dimensions["revenue_growth_yoy"] = round(rg, 1)

        # ── 3. 营收增长加速（最重要信号）────────────────────
        accel_signal = None
        if qi is not None and not qi.empty:
            try:
                rev_rows = [c for c in qi.index if "Revenue" in str(c) or "revenue" in str(c).lower()]
                if rev_rows:
                    rev_series = qi.loc[rev_rows[0]].dropna().sort_index()
                    if len(rev_series) >= 4:
                        q_rets = []
                        for i in range(1, min(4, len(rev_series))):
                            prev_v = float(rev_series.iloc[-(i+1)])
                            curr_v = float(rev_series.iloc[-i])
                            if abs(prev_v) > 0:
                                q_rets.append((curr_v - prev_v) / abs(prev_v) * 100)
                        if len(q_rets) >= 2:
                            latest_qoq = q_rets[0]
                            prev_qoq   = q_rets[1]
                            accel = latest_qoq - prev_qoq
                            if accel > 5:
                                score += 20
                                signals.append(f"营收加速！QoQ {latest_qoq:.1f}% vs 上季 {prev_qoq:.1f}%（+{accel:.1f}pp）")
                                accel_signal = "accelerating"
                            elif accel > 0:
                                score += 10
                                signals.append(f"营收微加速 QoQ {latest_qoq:.1f}% vs 上季 {prev_qoq:.1f}%")
                                accel_signal = "mild_accelerating"
                            elif accel > -5:
                                score += 3
                                accel_signal = "stable"
                            else:
                                signals.append(f"营收减速！QoQ {latest_qoq:.1f}% vs 上季 {prev_qoq:.1f}%（{accel:.1f}pp）")
                                accel_signal = "decelerating"
                            dimensions["acceleration"] = {
                                "latest_qoq": round(latest_qoq, 1),
                                "prev_qoq":   round(prev_qoq, 1),
                                "delta_pp":   round(accel, 1),
                                "signal":     accel_signal,
                            }
            except Exception:
                pass

        # ── 4. 利润率趋势 ────────────────────────────────────
        gross_m = info.get("grossMargins")
        op_m    = info.get("operatingMargins")
        profit_m = info.get("profitMargins")

        margin_score = 0
        if gross_m and _valid(gross_m):
            gm = float(gross_m) * 100
            if gm > 60:   margin_score += 5
            elif gm > 40: margin_score += 3
            elif gm > 20: margin_score += 1

        if op_m and _valid(op_m):
            om = float(op_m) * 100
            if om > 25:   margin_score += 5
            elif om > 15: margin_score += 3
            elif om > 5:  margin_score += 1

        score += margin_score
        if margin_score >= 8:
            gm_str = f"{float(gross_m)*100:.0f}%" if gross_m is not None else "—"
            om_str = f"{float(op_m)*100:.0f}%"   if op_m   is not None else "—"
            signals.append(f"利润率优秀（毛利{gm_str} / 经营{om_str}）")
        elif margin_score >= 4:
            signals.append(f"利润率尚可")
        dimensions["margins"] = {
            "gross": round(float(gross_m)*100, 1) if gross_m else None,
            "operating": round(float(op_m)*100, 1) if op_m else None,
            "net": round(float(profit_m)*100, 1) if profit_m else None,
        }

        # ── 5. 指引方向（用分析师目标价变化代理）────────────
        target = info.get("targetMeanPrice")
        price  = info.get("currentPrice") or info.get("regularMarketPrice")
        if target and price and _valid(target) and _valid(price):
            upside = (float(target) - float(price)) / float(price) * 100
            if upside > 20:
                score += 10; signals.append(f"分析师目标价隐含上涨空间{upside:.0f}%")
            elif upside > 10:
                score += 6
            elif upside > 0:
                score += 3
            else:
                signals.append(f"分析师目标价低于现价，下行风险{abs(upside):.0f}%")
            dimensions["analyst_upside"] = round(upside, 1)

        # ── 6. 连续超预期次数（机构信任度）─────────────────
        if eps_beats:
            consec_beats = sum(1 for b in eps_beats if b > 0)
            if consec_beats == 4:
                score += 10; signals.append("连续4季度超预期（机构高信任度）")
            elif consec_beats == 3:
                score += 6;  signals.append("连续3季度超预期")
            elif consec_beats == 2:
                score += 3;  signals.append("近2季超预期")
            dimensions["consecutive_beats"] = consec_beats

        # ── 综合评级 ─────────────────────────────────────────
        score = min(100, score)
        if score >= 80:
            grade = "A"
            verdict = "高质量财报，强烈支撑买入"
        elif score >= 65:
            grade = "B"
            verdict = "良好财报，基本面支撑"
        elif score >= 45:
            grade = "C"
            verdict = "一般财报，需结合趋势判断"
        elif score >= 25:
            grade = "D"
            verdict = "财报质量差，谨慎"
        else:
            grade = "F"
            verdict = "财报拖累股价，优先回避"

        canslim_pass = (
            rev_growth is not None and float(rev_growth) * 100 >= 25
            and bool(eps_beats) and eps_beats[0] > 0
            and accel_signal in ("accelerating", "mild_accelerating")
        )

        return {
            "ticker":        ticker,
            "score":         score,
            "grade":         grade,
            "verdict":       verdict,
            "signals":       signals,
            "dimensions":    dimensions,
            "canslim_pass":  canslim_pass,
            "canslim_note":  "营收≥25% + EPS超预期 + 加速 = CANSLIM 高分" if canslim_pass else "未完全满足 CANSLIM 基本面标准",
        }
    except Exception as e:
        return {"error": str(e), "score": 0, "grade": "N/A"}
