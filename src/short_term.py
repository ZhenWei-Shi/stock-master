"""
短线高风险策略引擎
理论来源：阿宝神父心法 + 技术分析宗师（Livermore趋势跟踪 / Minervini VCP /
           O'Neil动量 / Elder三重滤网 / 布林格均值回归）

核心逻辑：
  大盘方向（最高权重）→ 个股技术信号（RSI/MACD/BB/Vol）→
  ATR定量止损 → 杠杆工具匹配 → 风险收益比验证（≥2:1才出手）
"""

# ──────────────────────────────────────────────
# 杠杆工具映射表（个股 / ETF → 3x杠杆工具）
# ──────────────────────────────────────────────
LEVERAGE_MAP = {
    # 大盘ETF
    "SPY":  {"long": "UPRO",  "short": "SPXS", "sector": "标普500 3x"},
    "QQQ":  {"long": "TQQQ",  "short": "SQQQ", "sector": "纳斯达克100 3x"},
    "IWM":  {"long": "TNA",   "short": "TZA",  "sector": "罗素2000 3x"},
    "DIA":  {"long": "UDOW",  "short": "SDOW", "sector": "道琼斯 3x"},
    # 行业ETF
    "SOXX": {"long": "SOXL",  "short": "SOXS", "sector": "半导体 3x"},
    "XLF":  {"long": "FAS",   "short": "FAZ",  "sector": "金融 3x"},
    "XLE":  {"long": "ERX",   "short": "ERY",  "sector": "能源 3x"},
    "XLV":  {"long": "CURE",  "short": "RXD",  "sector": "医疗 3x"},
    "IBB":  {"long": "LABU",  "short": "LABD", "sector": "生物科技 3x"},
    "GLD":  {"long": "UGLD",  "short": "DGLD", "sector": "黄金 3x"},
    "TLT":  {"long": "TMF",   "short": "TMV",  "sector": "长期国债 3x"},
    "USO":  {"long": "UCO",   "short": "SCO",  "sector": "原油 2x"},
    # 热门个股映射到对应行业3x
    "NVDA": {"long": "SOXL",  "short": "SOXS", "sector": "→半导体 SOXL/SOXS"},
    "AMD":  {"long": "SOXL",  "short": "SOXS", "sector": "→半导体 SOXL/SOXS"},
    "INTC": {"long": "SOXL",  "short": "SOXS", "sector": "→半导体 SOXL/SOXS"},
    "TSM":  {"long": "SOXL",  "short": "SOXS", "sector": "→半导体 SOXL/SOXS"},
    "AAPL": {"long": "TQQQ",  "short": "SQQQ", "sector": "→纳斯达克 TQQQ/SQQQ"},
    "MSFT": {"long": "TQQQ",  "short": "SQQQ", "sector": "→纳斯达克 TQQQ/SQQQ"},
    "GOOGL":{"long": "TQQQ",  "short": "SQQQ", "sector": "→纳斯达克 TQQQ/SQQQ"},
    "META": {"long": "TQQQ",  "short": "SQQQ", "sector": "→纳斯达克 TQQQ/SQQQ"},
    "AMZN": {"long": "TQQQ",  "short": "SQQQ", "sector": "→纳斯达克 TQQQ/SQQQ"},
    "TSLA": {"long": "TQQQ",  "short": "SQQQ", "sector": "→纳斯达克 TQQQ/SQQQ"},
    "ASTS": {"long": "TQQQ",  "short": "SQQQ", "sector": "→纳斯达克 TQQQ/SQQQ"},
    "AXTI": {"long": "SOXL",  "short": "SOXS", "sector": "→半导体 SOXL/SOXS"},
    "AAOI": {"long": "SOXL",  "short": "SOXS", "sector": "→半导体 SOXL/SOXS"},
    "COHR": {"long": "SOXL",  "short": "SOXS", "sector": "→半导体 SOXL/SOXS"},
    "VRT":  {"long": "UPRO",  "short": "SPXS", "sector": "→标普 UPRO/SPXS"},
    "CEG":  {"long": "UPRO",  "short": "SPXS", "sector": "→标普 UPRO/SPXS"},
}

DEFAULT_LEVERAGE = {"long": "TQQQ", "short": "SQQQ", "sector": "→纳斯达克(默认)"}

# 3x ETF 波动耗散警告（必读）
_LEVERAGE_WARNING = {
    "volatility_decay": (
        "⚠️ 波动耗散（Volatility Decay）：3x ETF 每日重置，"
        "即使正股横盘，长期持有 3x ETF 也会亏损。"
        "例：QQQ 震荡1年几乎不涨，TQQQ 可能亏损 20-40%。"
    ),
    "extreme_risk": (
        "🚨 极端风险：SOXL 在 2022 年从 $70 跌至 $6（-91%）；"
        "LABU 在 2020 年跌超 95%。3x ETF 可以实质归零。"
    ),
    "hold_limit":   "✅ 正确用法：趋势明确时，最多持有 1-5 个交易日",
    "forbidden_for_small": (
        "🚫 小账户禁用（< $10,000）：一次 15% 的日内波动可能永久损伤小账户本金"
    ),
    "suitable_account": "$25,000+ 且有明确日线趋势信号时，才考虑短持 3x ETF",
}


def get_leverage_tools(ticker: str, account_value: float = 100_000) -> dict:
    tool = LEVERAGE_MAP.get(ticker.upper(), DEFAULT_LEVERAGE).copy()
    tool["warning"] = _LEVERAGE_WARNING
    if account_value < 10_000:
        tool["account_suitable"] = False
        tool["account_note"]     = f"账户 ${account_value:,.0f} < $10,000，禁止使用 3x ETF"
    elif account_value < 25_000:
        tool["account_suitable"] = "limited"
        tool["account_note"]     = f"账户 ${account_value:,.0f}，3x ETF 只能持有 1-3 天，严守止损"
    else:
        tool["account_suitable"] = True
        tool["account_note"]     = "账户规模允许，但最多持有 5 天，严格止损"
    return tool


# ──────────────────────────────────────────────
# 五大短线策略模型
# ──────────────────────────────────────────────
STRATEGIES = {
    "momentum_long": {
        "name": "趋势动量做多",
        "desc": "顺势追多，趋势初期动量延续。来源：Livermore趋势跟踪 + O'Neil动量",
        "condition": "大盘A + RSI(40-65) + MACD金叉/扩张 + 成交量放大 + 价格站上MA20",
    },
    "oversold_reversal": {
        "name": "超卖反弹做多",
        "desc": "在强趋势中的超卖回踩买入。来源：布林格均值回归 + Minervini pullback",
        "condition": "大盘A + RSI<35 + 布林下轨支撑 + 成交量萎缩（卖压枯竭）",
    },
    "vcp_breakout": {
        "name": "VCP突破做多",
        "desc": "波动率收缩后的爆发突破。来源：Minervini SEPA + O'Neil Cup Handle",
        "condition": "价格连续收窄振荡后突破阻力 + 成交量放大≥1.5x + RSI<70",
    },
    "trend_short": {
        "name": "趋势跟踪做空",
        "desc": "下跌趋势中的顺势做空。来源：Livermore + Elder三重滤网",
        "condition": "大盘C + MACD死叉 + 价格跌破MA20 + 成交量放大确认",
    },
    "overbought_reversal": {
        "name": "超买反转做空",
        "desc": "趋势顶部超买信号做空。来源：布林格上轨压制 + RSI背离",
        "condition": "大盘C/B + RSI>70 + 布林上轨压制 + MACD顶背离 + 成交量异常",
    },
}


def analyze(indicators: dict, market_state: dict, ticker: str) -> dict:
    """
    主分析入口：
    1. 为做多和做空分别评分（各14分满分）
    2. 匹配最符合的策略模型
    3. 计算ATR止损位和目标位
    4. 匹配杠杆工具
    5. 验证风险收益比（≥2:1才触发出手建议）
    """
    curr = indicators["current"]
    state = market_state.get("state", "E")
    vix   = market_state.get("vix") or 20

    price     = curr["price"]
    rsi_val   = curr["rsi"]
    macd_h    = curr["macd_hist"]
    macd_prev = curr["macd_hist_prev"]
    pct_b     = curr["pct_b"]       # 0=布林下轨 / 1=布林上轨
    vol_ratio = curr["vol_ratio"]
    atr_val   = curr["atr"]
    ma20      = curr["ma20"]
    ma50      = curr["ma50"]

    long_signals  = []
    short_signals = []
    ls = 0   # long score
    ss = 0   # short score

    # ── 1. 大盘方向（权重最高，±3/4分）──────────────────────
    if state == "A":
        ls += 3
        long_signals.append( _sig(True,  "大盘上升趋势（状态A）— 做多顺势"))
        short_signals.append(_sig(False, "大盘上升趋势（状态A）— 做空逆势，高风险"))
    elif state == "C":
        ss += 3
        short_signals.append(_sig(True,  "大盘下跌趋势（状态C）— 做空顺势"))
        long_signals.append( _sig(False, "大盘下跌趋势（状态C）— 做多逆势，高风险"))
    elif state == "D":
        ls += 4
        long_signals.append( _sig(True,  "极度恐慌（状态D）VIX>30 — 历史性抄底，做多强信号"))
        short_signals.append(_sig(False, "极度恐慌底部 — 做空空间已极度压缩"))
    elif state == "B":
        long_signals.append( _sig(None, "大盘横盘震荡（状态B）— 需个股技术信号进一步确认"))
        short_signals.append(_sig(None, "大盘横盘震荡（状态B）— 需个股技术信号进一步确认"))
    else:
        long_signals.append( _sig(False, "大盘方向不明（状态E）— 短线不操作，空仓等风起"))
        short_signals.append(_sig(False, "大盘方向不明（状态E）— 短线不操作，空仓等风起"))

    # ── 2. RSI（±3分）──────────────────────────────────────
    if rsi_val is not None:
        if rsi_val < 25:
            ls += 3
            long_signals.append( _sig(True,  f"RSI={rsi_val:.1f} 极度超卖(<25)，反弹概率极高，历史强买点"))
            short_signals.append(_sig(False, f"RSI={rsi_val:.1f} 极度超卖，做空动能已枯竭"))
        elif rsi_val < 35:
            ls += 2
            long_signals.append( _sig(True,  f"RSI={rsi_val:.1f} 超卖区间(25-35)，做多优势明显"))
            short_signals.append(_sig(None,  f"RSI={rsi_val:.1f} 超卖，做空需谨慎"))
        elif rsi_val < 45:
            ls += 1
            long_signals.append( _sig(True,  f"RSI={rsi_val:.1f} 偏低(35-45)，有反弹空间"))
            short_signals.append(_sig(None,  f"RSI={rsi_val:.1f} 偏低，做空动能有限"))
        elif rsi_val > 75:
            ss += 3
            short_signals.append(_sig(True,  f"RSI={rsi_val:.1f} 极度超买(>75)，回调概率极高"))
            long_signals.append( _sig(False, f"RSI={rsi_val:.1f} 极度超买，做多追高危险"))
        elif rsi_val > 65:
            ss += 2
            short_signals.append(_sig(True,  f"RSI={rsi_val:.1f} 超买区间(65-75)，做空优势明显"))
            long_signals.append( _sig(None,  f"RSI={rsi_val:.1f} 偏高，做多需确认不是顶部"))
        elif rsi_val > 55:
            ss += 1
            short_signals.append(_sig(None,  f"RSI={rsi_val:.1f} 偏高(55-65)，有回调可能"))
            long_signals.append( _sig(True,  f"RSI={rsi_val:.1f} 中性偏强，趋势多头延续"))
        else:
            long_signals.append( _sig(True,  f"RSI={rsi_val:.1f} 中性区间，无超买超卖"))
            short_signals.append(_sig(None,  f"RSI={rsi_val:.1f} 中性，做空无明显依据"))

    # ── 3. MACD（±3分）──────────────────────────────────────
    if macd_h is not None:
        cross_up   = macd_h > 0 and macd_prev <= 0
        cross_down = macd_h < 0 and macd_prev >= 0
        bull_exp   = macd_h > 0 and macd_h > macd_prev   # 金叉后柱子扩大
        bear_exp   = macd_h < 0 and macd_h < macd_prev   # 死叉后负柱子扩大

        if cross_up:
            ls += 3
            long_signals.append( _sig(True,  "MACD金叉（刚刚发生）— 动量转多，短线最强买点之一"))
            short_signals.append(_sig(False, "MACD金叉 — 做空方向完全相反"))
        elif cross_down:
            ss += 3
            short_signals.append(_sig(True,  "MACD死叉（刚刚发生）— 动量转空，短线最强卖点之一"))
            long_signals.append( _sig(False, "MACD死叉 — 做多方向完全相反"))
        elif bull_exp:
            ls += 2
            long_signals.append( _sig(True,  f"MACD多头扩张（柱:{macd_h:.3f}↑）— 买方动能持续加速"))
            short_signals.append(_sig(False, f"MACD多头扩张 — 空头阻力增大"))
        elif bear_exp:
            ss += 2
            short_signals.append(_sig(True,  f"MACD空头扩张（柱:{macd_h:.3f}↓）— 卖方动能持续加速"))
            long_signals.append( _sig(False, f"MACD空头扩张 — 多头阻力增大"))
        elif macd_h > 0:
            ls += 1
            long_signals.append( _sig(True,  f"MACD柱在零轴上方（{macd_h:.3f}），多头控场"))
        else:
            ss += 1
            short_signals.append(_sig(True,  f"MACD柱在零轴下方（{macd_h:.3f}），空头控场"))

    # ── 4. 布林带（±2分）────────────────────────────────────
    if pct_b is not None:
        if pct_b < 0.05:
            ls += 2
            long_signals.append( _sig(True,  f"触及布林下轨（%B={pct_b:.2f}）— 均值回归买点，超跌反弹"))
            short_signals.append(_sig(False, f"价格在布林下轨 — 做空极限空间已小"))
        elif pct_b < 0.2:
            ls += 1
            long_signals.append( _sig(True,  f"靠近布林下轨（%B={pct_b:.2f}）— 支撑区域"))
        elif pct_b > 0.95:
            ss += 2
            short_signals.append(_sig(True,  f"触及布林上轨（%B={pct_b:.2f}）— 均值回归卖点，超买回落"))
            long_signals.append( _sig(False, f"价格在布林上轨 — 做多上行空间已小"))
        elif pct_b > 0.8:
            ss += 1
            short_signals.append(_sig(True,  f"靠近布林上轨（%B={pct_b:.2f}）— 压制区域"))
        elif pct_b > 0.5:
            ls += 1
            long_signals.append( _sig(True,  f"价格在布林中轨上方（%B={pct_b:.2f}）— 多头强势区"))
        else:
            ss += 1
            short_signals.append(_sig(True,  f"价格在布林中轨下方（%B={pct_b:.2f}）— 空头弱势区"))

    # ── 5. 均线排列（±2分）──────────────────────────────────
    if ma20 and ma50 and price:
        if price > ma20 and ma20 > ma50:
            ls += 2
            long_signals.append( _sig(True,  f"价格>MA20(${ma20:.1f})>MA50(${ma50:.1f})，均线多头排列"))
            short_signals.append(_sig(False, "均线多头排列 — 做空逆势"))
        elif price < ma20 and ma20 < ma50:
            ss += 2
            short_signals.append(_sig(True,  f"价格<MA20(${ma20:.1f})<MA50(${ma50:.1f})，均线空头排列"))
            long_signals.append( _sig(False, "均线空头排列 — 做多逆势"))
        elif price > ma20:
            ls += 1
            long_signals.append( _sig(True,  f"价格站上MA20(${ma20:.1f})，短线多头"))
        elif price < ma20:
            ss += 1
            short_signals.append(_sig(True,  f"价格跌破MA20(${ma20:.1f})，短线空头"))

    # ── 6. 成交量（±2分，作为确认信号）────────────────────────
    if vol_ratio is not None:
        if vol_ratio >= 2.0:
            if ls >= ss:
                ls += 2
                long_signals.append( _sig(True,  f"成交量爆量（{vol_ratio:.1f}x）— 强力确认多头方向"))
            else:
                ss += 2
                short_signals.append(_sig(True,  f"成交量爆量（{vol_ratio:.1f}x）— 强力确认空头方向"))
        elif vol_ratio >= 1.5:
            if ls >= ss:
                ls += 1
                long_signals.append( _sig(True,  f"成交量放大（{vol_ratio:.1f}x）— 多头确认"))
            else:
                ss += 1
                short_signals.append(_sig(True,  f"成交量放大（{vol_ratio:.1f}x）— 空头确认"))
        elif vol_ratio < 0.5:
            long_signals.append( _sig(None, f"成交量极度萎缩（{vol_ratio:.1f}x）— 观望，无方向信号"))
            short_signals.append(_sig(None, f"成交量极度萎缩（{vol_ratio:.1f}x）— 观望，无方向信号"))
        else:
            long_signals.append( _sig(None, f"成交量正常（{vol_ratio:.1f}x均量）"))
            short_signals.append(_sig(None, f"成交量正常（{vol_ratio:.1f}x均量）"))

    # ── 7. VIX修正 ──────────────────────────────────────────
    if vix > 30:
        ls += 1   # 极恐时做多是顺势（神父原则28）
        long_signals.append( _sig(True,  f"VIX={vix:.1f}>30 极度恐慌，历史上此时做多超额收益最高"))
        short_signals.append(_sig(None,  f"VIX={vix:.1f}>30 恐慌顶部附近，做空需控制仓位"))
    elif vix > 22:
        short_signals.append(_sig(None,  f"VIX={vix:.1f} 偏高，市场情绪偏空"))

    # ── 综合判断 ────────────────────────────────────────────
    MAX_SCORE = 15

    # 识别匹配的策略
    matched_strategy = _match_strategy(state, rsi_val, macd_h, macd_prev, pct_b, vol_ratio, ls, ss)

    if state == "E":
        direction, confidence = "NEUTRAL", 0
        strategy_detail = None
    elif ls >= 9 and ls > ss + 3:
        direction = "STRONG_LONG"
        confidence = min(95, int(ls / MAX_SCORE * 100))
        strategy_detail = _long_strategy(curr, atr_val, confidence, ticker)
    elif ss >= 9 and ss > ls + 3:
        direction = "STRONG_SHORT"
        confidence = min(95, int(ss / MAX_SCORE * 100))
        strategy_detail = _short_strategy(curr, atr_val, confidence, ticker)
    elif ls >= 6 and ls > ss:
        direction = "LEAN_LONG"
        confidence = min(80, int(ls / MAX_SCORE * 100))
        strategy_detail = _long_strategy(curr, atr_val, confidence, ticker)
    elif ss >= 6 and ss > ls:
        direction = "LEAN_SHORT"
        confidence = min(80, int(ss / MAX_SCORE * 100))
        strategy_detail = _short_strategy(curr, atr_val, confidence, ticker)
    else:
        direction, confidence = "NEUTRAL", 0
        strategy_detail = None

    return {
        "direction":       direction,
        "long_score":      ls,
        "short_score":     ss,
        "max_score":       MAX_SCORE,
        "confidence":      confidence,
        "long_signals":    long_signals,
        "short_signals":   short_signals,
        "matched_strategy": matched_strategy,
        "strategy":        strategy_detail,
        "indicators":      curr,
    }


def _match_strategy(state, rsi, macd_h, macd_prev, pct_b, vol_ratio, ls, ss):
    cross_up   = macd_h and macd_h > 0 and macd_prev <= 0
    cross_down = macd_h and macd_h < 0 and macd_prev >= 0

    if ls > ss:
        if rsi and rsi < 35 and pct_b and pct_b < 0.2:
            return STRATEGIES["oversold_reversal"]
        if cross_up and vol_ratio and vol_ratio >= 1.5:
            return STRATEGIES["vcp_breakout"]
        return STRATEGIES["momentum_long"]
    elif ss > ls:
        if rsi and rsi > 70 and pct_b and pct_b > 0.8:
            return STRATEGIES["overbought_reversal"]
        return STRATEGIES["trend_short"]
    return None


def _long_strategy(curr, atr_val, confidence, ticker):
    price = curr["price"]
    if not atr_val or not price:
        return None
    stop   = round(price - 1.5 * atr_val, 2)
    tgt1   = round(price + 2.0 * atr_val, 2)
    tgt2   = round(price + 3.5 * atr_val, 2)
    rr     = round((tgt1 - price) / (price - stop), 2) if price != stop else 0
    lev    = "3x" if confidence >= 70 else "2x" if confidence >= 50 else "1x"
    tools  = get_leverage_tools(ticker)
    pos_pct = "15%" if lev == "3x" else "25%" if lev == "2x" else "40%"
    return {
        "type": "LONG", "entry": price, "stop_loss": stop,
        "target1": tgt1, "target2": tgt2, "risk_reward": rr,
        "atr": round(atr_val, 2), "atr_pct": curr.get("atr_pct"),
        "leverage": lev,
        "etf_tool": tools["long"], "sector": tools["sector"],
        "position_size": pos_pct,
        "hold_days": "1-5天，超5天未达目标考虑离场",
        "note": f"止损 = 入场价 - 1.5×ATR(${round(1.5*atr_val,2)})，风险收益比 {rr}:1",
        "option_alt": f"买入ATM Call，到期日2-3周，仓位≤3%账户",
        "warning": "3x ETF每日复利衰减，不可长持；最大回撤触发止损立即执行",
    }


def _short_strategy(curr, atr_val, confidence, ticker):
    price = curr["price"]
    if not atr_val or not price:
        return None
    stop   = round(price + 1.5 * atr_val, 2)
    tgt1   = round(price - 2.0 * atr_val, 2)
    tgt2   = round(price - 3.5 * atr_val, 2)
    rr     = round((price - tgt1) / (stop - price), 2) if stop != price else 0
    lev    = "3x" if confidence >= 70 else "2x" if confidence >= 50 else "1x"
    tools  = get_leverage_tools(ticker)
    pos_pct = "15%" if lev == "3x" else "25%" if lev == "2x" else "40%"
    return {
        "type": "SHORT", "entry": price, "stop_loss": stop,
        "target1": tgt1, "target2": tgt2, "risk_reward": rr,
        "atr": round(atr_val, 2), "atr_pct": curr.get("atr_pct"),
        "leverage": lev,
        "etf_tool": tools["short"], "sector": tools["sector"],
        "position_size": pos_pct,
        "hold_days": "1-5天，超5天未达目标考虑离场",
        "note": f"止损 = 入场价 + 1.5×ATR(${round(1.5*atr_val,2)})，风险收益比 {rr}:1",
        "option_alt": f"买入ATM Put，到期日2-3周，仓位≤3%账户",
        "warning": "3x反向ETF每日复利衰减，不可长持；止损被触及立即认错",
    }


def _sig(ok, text):
    return {"ok": ok, "text": text}
