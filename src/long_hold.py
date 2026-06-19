"""
长期持仓质量评估（Long-Hold Screener）

评估时间尺度：1 年以上
核心逻辑：业务护城河 + 资产负债表健康 + 长期趋势 + 估值合理性

与 cold_model（摆动交易）的区别：
  - 不看日内/周内技术形态（RSI区间、VWAP 无意义）
  - 看多年增长曲线，不是近期动量
  - 允许在整理期分批建仓（不需要"近52周新高"）
  - 容忍 20-30% 短期回撤，只要基本面不变就继续持有

评分 0-100：
  ≥ 70 = HOLD（高置信长持候选）
  50-69 = WATCH（需更多数据或等待更好入场点）
  < 50  = SKIP（周期性/衰退/估值过高，不适合长持）

硬性否决（无论分数）：
  - 营收连续两年萎缩（负增长）
  - 自由现金流持续为负且无盈利路径
  - 债务/资产 > 80%（债务炸弹）
  - 股份年化稀释 > 15%（股权破坏）
"""

import yfinance as yf
import numpy as np
from datetime import datetime
import pytz

ET = pytz.timezone("America/New_York")

# ══════════════════════════════════════════════════════════════
# 评分权重常量
# ══════════════════════════════════════════════════════════════
W_GROWTH_MAX     = 35   # 业务增长质量上限
W_BALANCE_MAX    = 25   # 资产负债健康上限
W_TREND_MAX      = 25   # 长期趋势上限
W_VALUATION_MAX  = 15   # 估值合理性上限

HOLD_THRESHOLD   = 70   # 长持候选门槛
WATCH_THRESHOLD  = 50   # 观察区门槛

# 行业护城河先验评分（基于 yfinance sector 字段）
MOAT_SECTOR_BONUS = {
    "Technology":            8,
    "Healthcare":            6,
    "Communication Services":5,
    "Consumer Discretionary":4,
    "Industrials":           3,
    "Financial Services":    3,
    "Basic Materials":       1,
    "Energy":                1,
    "Utilities":             1,
    "Real Estate":           1,
}
# ══════════════════════════════════════════════════════════════


def _safe(val, default=None):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────
# 各维度评分函数
# ─────────────────────────────────────────────────────────────

def _score_growth(info: dict) -> tuple[int, list, list]:
    """
    业务增长质量（0-35分）
    评估维度：营收增速、EPS增速、毛利率、自由现金流
    """
    score, pos, neg = 0, [], []

    # 营收增长（yfinance: revenueGrowth，YoY）
    rev_g = _safe(info.get("revenueGrowth"))
    if rev_g is not None:
        pct = rev_g * 100
        if pct >= 25:
            score += 14; pos.append(f"营收增速 {pct:.1f}%（高速成长）")
        elif pct >= 15:
            score += 9;  pos.append(f"营收增速 {pct:.1f}%（稳健成长）")
        elif pct >= 8:
            score += 5;  pos.append(f"营收增速 {pct:.1f}%（温和成长）")
        elif pct >= 0:
            score += 2;  neg.append(f"营收增速仅 {pct:.1f}%（成长乏力）")
        else:
            neg.append(f"营收萎缩 {pct:.1f}%（警告：业务收缩）")

    # EPS 增长（earningsGrowth，TTM YoY）
    eps_g = _safe(info.get("earningsGrowth"))
    if eps_g is not None:
        pct = eps_g * 100
        if pct >= 25:
            score += 10; pos.append(f"EPS增速 {pct:.1f}%（盈利加速）")
        elif pct >= 15:
            score += 6;  pos.append(f"EPS增速 {pct:.1f}%（盈利改善）")
        elif pct >= 0:
            score += 3;  pos.append(f"EPS增速 {pct:.1f}%（微增）")
        else:
            neg.append(f"EPS下滑 {pct:.1f}%（利润萎缩）")

    # 毛利率（grossMargins）— 衡量定价权/护城河
    gm = _safe(info.get("grossMargins"))
    if gm is not None:
        pct = gm * 100
        if pct >= 60:
            score += 8; pos.append(f"毛利率 {pct:.1f}%（极强定价权）")
        elif pct >= 40:
            score += 5; pos.append(f"毛利率 {pct:.1f}%（良好利润空间）")
        elif pct >= 25:
            score += 2; pos.append(f"毛利率 {pct:.1f}%（一般）")
        else:
            neg.append(f"毛利率 {pct:.1f}%（低，竞争激烈或资产重）")

    # 自由现金流（freeCashflow，正值 = 造血能力）
    fcf = _safe(info.get("freeCashflow"))
    if fcf is not None:
        if fcf > 0:
            score += 3; pos.append(f"FCF正向（{fcf/1e9:.2f}B），不依赖融资")
        else:
            neg.append(f"FCF为负（{fcf/1e9:.2f}B），依赖持续融资")

    return min(score, W_GROWTH_MAX), pos, neg


def _score_balance(info: dict) -> tuple[int, list, list]:
    """
    资产负债表健康（0-25分）
    评估维度：债务比率、流动比率、股份稀释
    """
    score, pos, neg = 0, [], []

    # 债务/权益比（debtToEquity，越低越安全）
    de = _safe(info.get("debtToEquity"))
    if de is not None:
        de_norm = de / 100  # yfinance 返回的是百分比形式
        if de_norm < 0.2:
            score += 15; pos.append(f"债务/权益 {de_norm:.2f}（极低负债，财务堡垒）")
        elif de_norm < 0.5:
            score += 10; pos.append(f"债务/权益 {de_norm:.2f}（健康）")
        elif de_norm < 1.0:
            score += 5;  pos.append(f"债务/权益 {de_norm:.2f}（可接受）")
        elif de_norm < 2.0:
            score += 2;  neg.append(f"债务/权益 {de_norm:.2f}（偏高，需关注利息覆盖）")
        else:
            neg.append(f"债务/权益 {de_norm:.2f}（过高负债，长持风险大）")

    # 流动比率（currentRatio，>2 安全，<1 危险）
    cr = _safe(info.get("currentRatio"))
    if cr is not None:
        if cr >= 2.0:
            score += 7; pos.append(f"流动比率 {cr:.2f}（强流动性）")
        elif cr >= 1.5:
            score += 5; pos.append(f"流动比率 {cr:.2f}（良好）")
        elif cr >= 1.0:
            score += 2; pos.append(f"流动比率 {cr:.2f}（基本安全）")
        else:
            neg.append(f"流动比率 {cr:.2f}（<1，短期偿债压力）")

    # 股份稀释检查（sharesOutstanding vs implied dilution via floatShares）
    shares = _safe(info.get("sharesOutstanding"))
    float_s = _safe(info.get("floatShares"))
    if shares and float_s and shares > 0:
        dilution_est = (shares - float_s) / shares
        if dilution_est < 0.05:
            score += 3; pos.append("股权结构稳定（无明显稀释）")
        elif dilution_est > 0.20:
            neg.append(f"疑似高稀释比例 {dilution_est*100:.1f}%（需验证增发历史）")

    return min(score, W_BALANCE_MAX), pos, neg


def _score_trend(ticker: str, info: dict) -> tuple[int, list, list]:
    """
    长期价格趋势（0-25分）
    评估维度：1年RS vs SPY、周均线结构、52周位置
    """
    score, pos, neg = 0, [], []

    try:
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="2y", interval="1wk")  # 周线，2年
        spy  = yf.Ticker("SPY").history(period="2y", interval="1wk")

        if hist.empty or spy.empty:
            neg.append("无足够历史数据评估长期趋势")
            return 0, pos, neg

        # 周线 MA50（≈1年）
        close = hist["Close"]
        ma50w = close.rolling(50).mean()  # 50周 ≈ 1年
        ma26w = close.rolling(26).mean()  # 26周 ≈ 半年

        price_now = float(close.iloc[-1])
        ma50_now  = float(ma50w.iloc[-1]) if not np.isnan(ma50w.iloc[-1]) else None
        ma26_now  = float(ma26w.iloc[-1]) if not np.isnan(ma26w.iloc[-1]) else None

        # 价格在50周均线上方（长期上升趋势确认）
        if ma50_now:
            if price_now > ma50_now * 1.05:
                score += 10; pos.append(f"价格在50周均线上方 {(price_now/ma50_now-1)*100:.1f}%（强势上升趋势）")
            elif price_now > ma50_now:
                score += 7;  pos.append("价格略高于50周均线（趋势偏多）")
            elif price_now > ma50_now * 0.92:
                score += 3;  neg.append("价格接近或略低于50周均线（趋势弱化）")
            else:
                neg.append(f"价格低于50周均线 {(1-price_now/ma50_now)*100:.1f}%（长期趋势受损）")

        # 均线排列（26周 > 50周 = 多头排列）
        if ma26_now and ma50_now:
            if ma26_now > ma50_now:
                score += 5; pos.append("周线均线多头排列（26周 > 50周）")
            else:
                neg.append("周线均线空头排列（26周 < 50周）")

        # 1年 RS vs SPY（相对强弱，长期的核心指标）
        if len(close) >= 52 and len(spy["Close"]) >= 52:
            ret_1y  = float(close.iloc[-1] / close.iloc[-52] - 1) if len(close) >= 52 else None
            spy_1y  = float(spy["Close"].iloc[-1] / spy["Close"].iloc[-52] - 1) if len(spy["Close"]) >= 52 else None

            if ret_1y is not None and spy_1y is not None:
                excess = ret_1y - spy_1y
                if excess > 0.20:
                    score += 10; pos.append(f"1年超额回报 +{excess*100:.1f}% vs SPY（强势龙头）")
                elif excess > 0.05:
                    score += 7;  pos.append(f"1年超额回报 +{excess*100:.1f}% vs SPY（跑赢大盘）")
                elif excess > -0.05:
                    score += 4;  pos.append(f"1年与SPY持平（{excess*100:+.1f}%）")
                else:
                    neg.append(f"1年跑输SPY {excess*100:.1f}%（弱于大盘）")

    except Exception as e:
        neg.append(f"趋势数据获取失败：{e}")

    return min(score, W_TREND_MAX), pos, neg


def _score_valuation(info: dict) -> tuple[int, list, list]:
    """
    估值合理性（0-15分）
    核心逻辑：长持不怕贵，但怕"贵且不增长"
    用 PEG 比率 = PE / EPS增速 衡量
    """
    score, pos, neg = 0, [], []

    # PEG（yfinance: pegRatio）
    peg = _safe(info.get("pegRatio"))
    if peg is not None and peg > 0:
        if peg < 1.0:
            score += 12; pos.append(f"PEG {peg:.2f}（<1，相对增速低估）")
        elif peg < 1.5:
            score += 9;  pos.append(f"PEG {peg:.2f}（合理，增速匹配估值）")
        elif peg < 2.5:
            score += 5;  pos.append(f"PEG {peg:.2f}（略贵但成长股可接受）")
        elif peg < 4.0:
            score += 2;  neg.append(f"PEG {peg:.2f}（偏贵，需要高增速维持）")
        else:
            neg.append(f"PEG {peg:.2f}（过高，增速难以消化估值）")

    # P/S 兜底（适用于无盈利成长股，PEG无法使用时）
    elif peg is None or peg <= 0:
        ps = _safe(info.get("priceToSalesTrailing12Months"))
        rev_g = _safe(info.get("revenueGrowth"))
        if ps is not None and rev_g is not None:
            growth_pct = rev_g * 100
            if ps < 5 and growth_pct > 20:
                score += 10; pos.append(f"P/S {ps:.1f}x，增速{growth_pct:.0f}%（成长股估值合理）")
            elif ps < 10 and growth_pct > 30:
                score += 7;  pos.append(f"P/S {ps:.1f}x，高速增速 {growth_pct:.0f}% 支撑估值")
            elif ps > 20:
                neg.append(f"P/S {ps:.1f}x（过高，需极强增速才能维持）")
            else:
                score += 3;  pos.append(f"P/S {ps:.1f}x（参考）")
        elif ps is not None:
            if ps < 3:
                score += 8; pos.append(f"P/S {ps:.1f}x（价值合理）")
            elif ps < 8:
                score += 5; pos.append(f"P/S {ps:.1f}x（中等估值）")
            else:
                neg.append(f"P/S {ps:.1f}x（估值偏高）")

    # 行业护城河先验加分
    sector = info.get("sector", "")
    bonus = MOAT_SECTOR_BONUS.get(sector, 0)
    if bonus:
        score += bonus
        pos.append(f"行业护城河加分：{sector}（+{bonus}）")

    return min(score, W_VALUATION_MAX), pos, neg


# ─────────────────────────────────────────────────────────────
# 硬性否决检查
# ─────────────────────────────────────────────────────────────

def _hard_veto(info: dict) -> str | None:
    """返回否决原因字符串，无问题则返回 None。"""
    # 营收萎缩
    rev_g = _safe(info.get("revenueGrowth"))
    if rev_g is not None and rev_g < -0.10:
        return f"营收萎缩 {rev_g*100:.1f}%（业务实质性收缩）"

    # 债务炸弹：总负债/总资产 > 80%
    # yfinance 用 debtToEquity 近似，极高值警告
    de = _safe(info.get("debtToEquity"))
    if de is not None and de > 300:  # D/E > 3.0（百分比形式>300）
        return f"债务/权益 {de/100:.1f}（极高杠杆，破产风险）"

    # 严重亏损且无增长迹象
    fcf = _safe(info.get("freeCashflow"))
    rev_g2 = _safe(info.get("revenueGrowth"), 0)
    if fcf is not None and fcf < -1e9 and rev_g2 < 0.10:
        return f"FCF严重为负（{fcf/1e9:.1f}B）且营收增速仅{rev_g2*100:.0f}%，烧钱无出路"

    return None


# ─────────────────────────────────────────────────────────────
# 主评估函数
# ─────────────────────────────────────────────────────────────

def long_hold_eval(ticker: str) -> dict:
    """
    对单只股票做长期持仓质量评估。

    返回：
      verdict   — HOLD / WATCH / SKIP
      score     — 0-100
      breakdown — 各维度得分
      positives — 优点列表
      negatives — 风险/缺陷列表
      moat      — 护城河摘要
      sell_triggers — 建议的卖出触发条件
    """
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info
        name = info.get("longName") or info.get("shortName") or ticker
    except Exception as e:
        return {"ticker": ticker, "error": f"数据获取失败：{e}"}

    # 硬性否决
    veto = _hard_veto(info)
    if veto:
        return {
            "ticker":  ticker,
            "name":    name,
            "verdict": "SKIP",
            "score":   0,
            "veto":    veto,
            "breakdown": {},
            "positives": [],
            "negatives": [veto],
            "moat":    "否决，未评估",
            "sell_triggers": [],
        }

    # 四维评分
    g_score, g_pos, g_neg = _score_growth(info)
    b_score, b_pos, b_neg = _score_balance(info)
    t_score, t_pos, t_neg = _score_trend(ticker, info)
    v_score, v_pos, v_neg = _score_valuation(info)

    total = g_score + b_score + t_score + v_score
    positives = g_pos + b_pos + t_pos + v_pos
    negatives = g_neg + b_neg + t_neg + v_neg

    # 评级
    if total >= HOLD_THRESHOLD:
        verdict = "HOLD"
    elif total >= WATCH_THRESHOLD:
        verdict = "WATCH"
    else:
        verdict = "SKIP"

    # 护城河摘要
    sector     = info.get("sector", "未知")
    industry   = info.get("industry", "未知")
    market_cap = _safe(info.get("marketCap"))
    cap_str    = f"{market_cap/1e9:.0f}B" if market_cap else "未知"
    moat_parts = [f"{sector} / {industry}", f"市值 ${cap_str}"]

    gm = _safe(info.get("grossMargins"))
    if gm and gm > 0.50:
        moat_parts.append(f"高毛利率 {gm*100:.0f}%（定价权护城河）")
    inst = _safe(info.get("institutionPercentHeld"))
    if inst and inst > 0.70:
        moat_parts.append(f"机构持仓 {inst*100:.0f}%（机构认可）")

    moat = "；".join(moat_parts)

    # 卖出触发条件（持有期间需监控的基本面红线）
    sell_triggers = [
        "连续两季度营收增速下滑超过5%",
        "毛利率季度环比下降超过3个百分点",
        "管理层大规模减持（内部人 >5% 净卖出）",
        "债务大幅扩张（D/E 升至当前的2倍以上）",
        "出现盈利能力强劲的直接竞争对手抢夺市场份额",
    ]

    return {
        "ticker":   ticker,
        "name":     name,
        "verdict":  verdict,
        "score":    total,
        "breakdown": {
            "growth":     f"{g_score}/{W_GROWTH_MAX}",
            "balance":    f"{b_score}/{W_BALANCE_MAX}",
            "trend":      f"{t_score}/{W_TREND_MAX}",
            "valuation":  f"{v_score}/{W_VALUATION_MAX}",
        },
        "positives":     positives,
        "negatives":     negatives,
        "moat":          moat,
        "sell_triggers": sell_triggers,
    }


def long_hold_scan(tickers: list) -> list:
    """批量评估，按分数降序返回。"""
    results = [long_hold_eval(t) for t in tickers]
    results.sort(key=lambda r: r.get("score", 0), reverse=True)
    return results


# ─────────────────────────────────────────────────────────────
# Telegram 格式化
# ─────────────────────────────────────────────────────────────

_VERDICT_ICON = {"HOLD": "🟢", "WATCH": "🟡", "SKIP": "🔴"}
_VERDICT_LABEL = {
    "HOLD":  "长持候选",
    "WATCH": "观察等待",
    "SKIP":  "不适合长持",
}


def format_longhold_telegram(results: list) -> str:
    today = datetime.now(ET).strftime("%Y-%m-%d")
    lines = [f"📦 <b>长期持仓质量评估</b>  {today}", ""]

    for r in results:
        if r.get("error"):
            lines.append(f"⚠️ {r.get('ticker','?')}：{r['error']}")
            continue

        verdict = r["verdict"]
        icon    = _VERDICT_ICON.get(verdict, "⚪")
        label   = _VERDICT_LABEL.get(verdict, verdict)

        lines.append(
            f"{icon} <b>{r['ticker']}</b>  {r.get('score',0)}/100"
            f"  [{label}]"
        )

        if r.get("veto"):
            lines.append(f"  ❌ 否决：{r['veto']}")
            lines.append("")
            continue

        bd = r.get("breakdown", {})
        lines.append(
            f"  增长{bd.get('growth','?')}  "
            f"负债{bd.get('balance','?')}  "
            f"趋势{bd.get('trend','?')}  "
            f"估值{bd.get('valuation','?')}"
        )

        pos = r.get("positives", [])
        neg = r.get("negatives", [])

        if pos:
            lines.append("  <b>优势：</b>")
            for p in pos[:3]:  # 只显示前3条
                lines.append(f"    ✅ {p}")

        if neg:
            lines.append("  <b>风险：</b>")
            for n in neg[:2]:
                lines.append(f"    ⚠️ {n}")

        if r.get("moat"):
            lines.append(f"  护城河：{r['moat']}")

        if verdict == "HOLD" and r.get("sell_triggers"):
            lines.append("  <b>卖出触发（任一出现立即复评）：</b>")
            lines.append(f"    • {r['sell_triggers'][0]}")
            lines.append(f"    • {r['sell_triggers'][1]}")

        lines.append("")

    lines.append("─────────────────────────")
    lines.append("💡 长持策略：基本面不变 = 持有，无视短期回撤")
    lines.append("   评估基于 yfinance 公开数据（TTM财报 + 2年周线）")
    return "\n".join(lines)


def format_longhold_inline(result: dict) -> str:
    """用于摆动交易 GO 信号旁显示的单行长持摘要。"""
    if result.get("error") or result.get("veto"):
        return ""
    icon    = _VERDICT_ICON.get(result["verdict"], "⚪")
    label   = _VERDICT_LABEL.get(result["verdict"], result["verdict"])
    bd      = result.get("breakdown", {})
    return (
        f"\n{icon} <b>长持评分</b> {result['score']}/100（{label}）"
        f"  增长{bd.get('growth','?')} 负债{bd.get('balance','?')}"
        f" 趋势{bd.get('trend','?')} 估值{bd.get('valuation','?')}"
    )
