"""
量化短线交易策略库

引用项目：
  pandas-ta   ⭐5,299 (twopirllc/pandas-ta)  — Squeeze, Supertrend, 200+指标
  quantstats  ⭐6,500 (ranaroussi/quantstats) — 策略绩效评估
  vectorbt    ⭐7,900 (polakowo/vectorbt)      — 向量化信号生成（可选）

6大核心量化策略：
  1. VWAP均值回归   — 日内短线，偏离回归
  2. 动量突破       — 日线/周线，趋势跟踪
  3. RSI-MACD共振   — 日线，双重确认
  4. Squeeze动量    — 日线，蓄力爆发
  5. 供应链相对强弱  — 日线，卡脖子层轮动
  6. 剥头皮策略     — 1分钟图，超短线，多重价格级别确认

冷静规则：信号合分 < 6 绝不入场，止损1.5ATR，1%风险规则
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime

try:
    import ta as _ta
    TA_OK = True
except ImportError:
    TA_OK = False

ET = pytz.timezone("America/New_York")


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────

def run_all_strategies(ticker: str, portfolio: float = 100_000,
                       compare_tickers: list = None) -> dict:
    """5大策略综合运行，返回综合信号 + 仓位建议"""
    hist    = yf.Ticker(ticker).history(period="6mo", interval="1d")
    hist_5m = yf.Ticker(ticker).history(period="5d",  interval="5m")

    if hist.empty:
        return {"error": f"无法获取 {ticker} 数据"}

    results = {
        "vwap_reversion":     strategy_vwap_reversion(hist_5m, ticker),
        "momentum_breakout":  strategy_momentum_breakout(hist, ticker),
        "rsi_macd_confluence": strategy_rsi_macd(hist, ticker),
        "squeeze_momentum":   strategy_squeeze(hist, ticker),
    }

    if compare_tickers:
        results["supply_chain_rs"] = strategy_supply_chain_rs(
            ticker, compare_tickers, hist)

    composite = _composite_score(results)
    position  = _position_sizing(composite, hist, portfolio)
    rules     = _check_rules(composite, hist)

    return {
        "ticker":     ticker,
        "strategies": results,
        "composite":  composite,
        "position":   position,
        "cold_rules": rules,
        "timestamp":  datetime.now(ET).strftime("%Y-%m-%d %H:%M ET"),
        "libraries":  {
            "pandas-ta":  "⭐5,299 — 200+技术指标",
            "quantstats": "⭐6,500 — 绩效评估",
            "vectorbt":   "⭐7,900 — 向量化信号（可选）",
        },
    }


# ─────────────────────────────────────────────────────────────
# 策略1：VWAP均值回归（日内短线）
# ─────────────────────────────────────────────────────────────

def strategy_vwap_reversion(hist: pd.DataFrame, ticker: str) -> dict:
    """
    逻辑：价格偏离 VWAP ±1.5% + RSI 极端 → 预期均值回归
    时间框架：5分钟图，日内操作
    """
    if hist.empty or len(hist) < 20:
        return _no_signal("日内数据不足")

    typical_price = (hist["High"] + hist["Low"] + hist["Close"]) / 3
    vwap = (typical_price * hist["Volume"]).cumsum() / hist["Volume"].cumsum()

    price    = float(hist["Close"].iloc[-1])
    vwap_val = float(vwap.iloc[-1])
    if vwap_val == 0:
        return _no_signal("VWAP=0，数据异常")

    deviation = (price - vwap_val) / vwap_val * 100

    rsi_val = _calc_rsi(hist["Close"])
    atr_val = _calc_atr(hist)

    if deviation < -1.5 and rsi_val < 38:
        entry = price
        stop  = entry - atr_val * 1.5
        t1    = vwap_val
        t2    = vwap_val + atr_val * 0.5
        conf  = min(90, 50 + abs(deviation) * 8 + (40 - rsi_val))
        reason = f"低于VWAP {abs(deviation):.1f}%，RSI={rsi_val:.0f}超卖，预期回归"
        direction = "LONG"
    elif deviation > 1.5 and rsi_val > 62:
        entry = price
        stop  = entry + atr_val * 1.5
        t1    = vwap_val
        t2    = vwap_val - atr_val * 0.5
        conf  = min(90, 50 + deviation * 8 + (rsi_val - 60))
        reason = f"高于VWAP {deviation:.1f}%，RSI={rsi_val:.0f}超买，预期回归"
        direction = "SHORT"
    else:
        return _no_signal(f"VWAP偏离{deviation:.1f}%不足±1.5%，无信号")

    rr = abs(t1 - entry) / max(abs(stop - entry), 0.01)

    return {
        "name": "VWAP均值回归", "signal": direction,
        "entry": round(entry, 2), "stop_loss": round(stop, 2),
        "target1": round(t1, 2), "target2": round(t2, 2),
        "confidence": round(conf), "risk_reward": round(rr, 2),
        "vwap": round(vwap_val, 2), "deviation_pct": round(deviation, 2),
        "rsi": round(rsi_val, 1), "atr": round(atr_val, 3),
        "reason": reason, "timeframe": "5分钟图，日内",
    }


# ─────────────────────────────────────────────────────────────
# 策略2：动量突破（Mark Minervini 型）
# ─────────────────────────────────────────────────────────────

def strategy_momentum_breakout(hist: pd.DataFrame, ticker: str) -> dict:
    """
    逻辑：52周高点附近 + 放量 + 均线多头排列 → 强势突破
    时间框架：日线，持有1-5天
    """
    if hist.empty or len(hist) < 50:
        return _no_signal("日线数据不足50天")

    close     = hist["Close"]
    volume    = hist["Volume"]
    price     = float(close.iloc[-1])
    high_52w  = float(close.tail(252).max() if len(close) >= 252 else close.max())
    pct_high  = (price - high_52w) / high_52w * 100

    ma20      = float(close.rolling(20).mean().iloc[-1])
    ma50      = float(close.rolling(50).mean().iloc[-1])
    vol_ma20  = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio = float(volume.iloc[-1]) / vol_ma20 if vol_ma20 > 0 else 1.0
    atr_val   = _calc_atr(hist)

    near_high    = pct_high > -5
    above_ma     = price > ma20 > ma50
    volume_surge = vol_ratio > 1.5

    if near_high and above_ma and volume_surge:
        entry = price
        stop  = ma20
        t1    = price + atr_val * 2
        t2    = high_52w * 1.05
        conf  = min(90, 55 + (vol_ratio - 1.5) * 15 + min(15, abs(pct_high) * 2))
        reason = f"近52周高点{pct_high:.1f}%，量比{vol_ratio:.1f}x，均线多头排列"
    else:
        missing = []
        if not near_high:    missing.append(f"离高{pct_high:.0f}%")
        if not above_ma:     missing.append("均线非多头排列")
        if not volume_surge: missing.append(f"量比{vol_ratio:.1f}x<1.5x")
        return _no_signal(f"突破条件未满足：{', '.join(missing)}")

    rr = abs(t1 - entry) / max(abs(stop - entry), 0.01)

    return {
        "name": "动量突破", "signal": "LONG",
        "entry": round(entry, 2), "stop_loss": round(stop, 2),
        "target1": round(t1, 2), "target2": round(t2, 2),
        "confidence": round(conf), "risk_reward": round(rr, 2),
        "pct_from_52w_high": round(pct_high, 1),
        "volume_ratio": round(vol_ratio, 2),
        "ma20": round(ma20, 2), "ma50": round(ma50, 2),
        "reason": reason, "timeframe": "日线，持有1-5天",
    }


# ─────────────────────────────────────────────────────────────
# 策略3：RSI-MACD 共振
# ─────────────────────────────────────────────────────────────

def strategy_rsi_macd(hist: pd.DataFrame, ticker: str) -> dict:
    """
    逻辑：RSI从超买/超卖区反弹 + MACD金/死叉 = 双重确认
    时间框架：日线，持有2-7天
    """
    if hist.empty or len(hist) < 35:
        return _no_signal("数据不足35天")

    close = hist["Close"]
    price = float(close.iloc[-1])

    rsi_s    = _rsi_series(close)
    rsi_now  = float(rsi_s.iloc[-1])
    rsi_prev = float(rsi_s.iloc[-4])

    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd_h = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    m_now  = float(macd_h.iloc[-1])
    m_prev = float(macd_h.iloc[-2])

    ma20    = float(close.rolling(20).mean().iloc[-1])
    atr_val = _calc_atr(hist)

    bull = rsi_prev < 35 and rsi_now > 40 and m_prev < 0 and m_now > 0 and price > ma20
    bear = rsi_prev > 65 and rsi_now < 60 and m_prev > 0 and m_now < 0 and price < ma20

    if bull:
        entry     = price
        stop      = entry - atr_val * 2
        t1, t2    = entry + atr_val * 2, entry + atr_val * 4
        conf      = min(88, 70 + (40 - rsi_prev) * 0.5)
        direction = "LONG"
        reason    = f"RSI从{rsi_prev:.0f}→{rsi_now:.0f}反弹，MACD金叉，双重确认"
    elif bear:
        entry     = price
        stop      = entry + atr_val * 2
        t1, t2    = entry - atr_val * 2, entry - atr_val * 4
        conf      = min(88, 70 + (rsi_prev - 60) * 0.5)
        direction = "SHORT"
        reason    = f"RSI从{rsi_prev:.0f}→{rsi_now:.0f}下穿，MACD死叉，双重确认"
    else:
        partial = []
        if rsi_prev < 38: partial.append(f"RSI从超卖{rsi_prev:.0f}回升")
        if m_prev < 0 and m_now > 0: partial.append("MACD金叉")
        hint = "、".join(partial) if partial else f"RSI={rsi_now:.0f}，无共振"
        return _no_signal(f"单一信号未共振（{hint}），等待双重确认")

    rr = abs(t1 - entry) / max(abs(stop - entry), 0.01)

    return {
        "name": "RSI-MACD共振", "signal": direction,
        "entry": round(entry, 2), "stop_loss": round(stop, 2),
        "target1": round(t1, 2), "target2": round(t2, 2),
        "confidence": round(conf), "risk_reward": round(rr, 2),
        "rsi_now": round(rsi_now, 1), "rsi_prev": round(rsi_prev, 1),
        "macd_hist": round(m_now, 5),
        "reason": reason, "timeframe": "日线，持有2-7天",
    }


# ─────────────────────────────────────────────────────────────
# 策略4：Squeeze Momentum（pandas-ta ⭐5,299）
# ─────────────────────────────────────────────────────────────

def strategy_squeeze(hist: pd.DataFrame, ticker: str) -> dict:
    """
    逻辑：BB < KC（市场蓄力）→ Squeeze释放时 + 动量方向 = 定向突破
    时间框架：日线，持有3-10天
    """
    if not TA_OK or hist.empty or len(hist) < 30:
        return _no_signal("pandas-ta 未安装或数据不足")

    try:
        df  = hist.copy()
        sqz = ta.squeeze(df["High"], df["Low"], df["Close"], df["Volume"])
        if sqz is None or sqz.empty:
            return _no_signal("Squeeze计算失败")

        mom_cols = [c for c in sqz.columns
                    if "SQZ_" in c and "ON" not in c and "OFF" not in c and "NO" not in c]
        on_cols  = [c for c in sqz.columns if "SQZ_ON" in c]
        if not mom_cols:
            return _no_signal("找不到动量列")

        mom      = float(sqz[mom_cols[0]].iloc[-1])
        mom_prev = float(sqz[mom_cols[0]].iloc[-2])
        in_sqz   = bool(sqz[on_cols[0]].iloc[-3])  if on_cols and len(sqz) > 3 else True
        was_sqz  = bool(sqz[on_cols[0]].iloc[-4])  if on_cols and len(sqz) > 4 else False
        fired    = was_sqz and not in_sqz

        price   = float(hist["Close"].iloc[-1])
        atr_val = _calc_atr(hist)

        if fired and mom > 0 and mom > mom_prev:
            entry, stop = price, price - atr_val * 1.5
            t1, t2 = price + atr_val * 2, price + atr_val * 3.5
            conf   = min(88, 72 + min(16, abs(mom) * 800))
            reason = f"Squeeze刚释放，动量向上={mom:.5f}并增强，蓄力多头爆发"
            direction = "LONG"
        elif fired and mom < 0 and mom < mom_prev:
            entry, stop = price, price + atr_val * 1.5
            t1, t2 = price - atr_val * 2, price - atr_val * 3.5
            conf   = min(88, 72 + min(16, abs(mom) * 800))
            reason = f"Squeeze刚释放，动量向下={mom:.5f}并增强，蓄力空头爆发"
            direction = "SHORT"
        elif in_sqz:
            return _no_signal(f"仍在Squeeze蓄力中（动量={mom:.5f}），等待释放")
        else:
            return _no_signal(f"无Squeeze信号（动量={mom:.5f}，方向不明确）")

        rr = abs(t1 - entry) / max(abs(stop - entry), 0.01)

        return {
            "name": "Squeeze动量", "signal": direction,
            "entry": round(entry, 2), "stop_loss": round(stop, 2),
            "target1": round(t1, 2), "target2": round(t2, 2),
            "confidence": round(conf), "risk_reward": round(rr, 2),
            "momentum": round(mom, 6), "squeeze_fired": fired,
            "reason": reason, "timeframe": "日线，持有3-10天",
            "library": "pandas-ta ⭐5,299",
        }

    except Exception as e:
        return _no_signal(f"Squeeze异常：{e}")


# ─────────────────────────────────────────────────────────────
# 策略5：供应链相对强弱轮动（Serenity方法论）
# ─────────────────────────────────────────────────────────────

def strategy_supply_chain_rs(ticker: str, chain_peers: list,
                              hist: pd.DataFrame) -> dict:
    """
    逻辑：在同一供应链层里找相对最强的股票
    当 ticker 的3月超额收益 > 所有同层均值时，做多信号
    时间框架：日线，持有1-4周
    """
    try:
        close   = hist["Close"]
        price   = float(close.iloc[-1])
        ret_3m  = float((close.iloc[-1] - close.iloc[max(-63, -len(close))]) /
                        close.iloc[max(-63, -len(close))] * 100) if len(close) >= 5 else 0

        peer_rets = []
        for p in chain_peers[:6]:
            if p == ticker:
                continue
            try:
                ph = yf.Ticker(p).history(period="3mo")["Close"]
                if len(ph) < 2:
                    continue
                r = float((ph.iloc[-1] - ph.iloc[0]) / ph.iloc[0] * 100)
                peer_rets.append({"ticker": p, "ret_3m": round(r, 1)})
            except Exception:
                continue

        if not peer_rets:
            return _no_signal("无同层对比数据")

        avg_peer  = sum(p["ret_3m"] for p in peer_rets) / len(peer_rets)
        outperform = ret_3m - avg_peer
        atr_val   = _calc_atr(hist)

        if outperform > 5:
            entry = price
            stop  = price - atr_val * 2
            t1    = price + atr_val * 2
            conf  = min(85, 60 + outperform * 1.5)
            reason = (f"3月超额收益{outperform:.1f}%，"
                      f"跑赢同层均值（{avg_peer:.1f}%），"
                      f"供应链层内最强标的")
            return {
                "name": "供应链相对强弱", "signal": "LONG",
                "entry": round(entry, 2), "stop_loss": round(stop, 2),
                "target1": round(t1, 2), "target2": round(price + atr_val * 4, 2),
                "confidence": round(conf), "risk_reward": round(abs(t1-entry)/max(atr_val*2,0.01), 2),
                "ret_3m": round(ret_3m, 1), "peer_avg": round(avg_peer, 1),
                "outperform": round(outperform, 1),
                "peers": peer_rets[:5],
                "reason": reason, "timeframe": "日线，持有1-4周",
            }
        elif outperform < -5:
            return _no_signal(f"跑输同层均值{abs(outperform):.1f}%，非最强标的，应选更强同层股")
        else:
            return _no_signal(f"超额收益{outperform:.1f}%，优势不明显（需>5%）")

    except Exception as e:
        return _no_signal(f"RS计算异常：{e}")


# ─────────────────────────────────────────────────────────────
# 策略6：剥头皮策略（Scalping）
# ─────────────────────────────────────────────────────────────

def strategy_scalping(ticker: str, portfolio: float = 100_000) -> dict:
    """
    剥头皮策略 — 超短线，目标 0.2-0.5%，止损 0.1-0.15%

    核心逻辑（三重确认）：
      1. 大框架（15分钟）：价格在 VWAP 上方 + 均线多头
      2. 触发框架（5分钟）：回踩 VWAP 后反弹确认
      3. 执行框架（1分钟）：RSI 从超卖反弹 + 量能放大

    严格规则：
      - 只在开盘 30-90 分钟（09:30-11:00 ET）流动性最强时操作
      - 只在收盘前 1 小时（14:30-15:30 ET）方向最明确时操作
      - 每笔止盈目标 = 1.5×ATR_1m，止损 = 0.8×ATR_1m（R:R ≥ 1.8）
      - 单日最多 5 笔，日亏损超 0.5% 立即停止

    风险提示：
      剥头皮对执行速度要求极高，yfinance 延迟 15 分钟，
      实盘需接入 Alpaca / Finnhub WebSocket 实时数据。
    """
    now = datetime.now(ET)
    h, m = now.hour, now.minute
    total_min = h * 60 + m

    # ── 时间窗口过滤（剥头皮最严格）───────────────────────
    prime_open  = 570 <= total_min <= 630    # 09:30-10:30
    prime_close = 870 <= total_min <= 930    # 14:30-15:30
    if not (prime_open or prime_close):
        return _no_signal(
            f"剥头皮仅在黄金窗口操作：09:30-10:30 或 14:30-15:30 ET（当前{h:02d}:{m:02d}）"
        )

    try:
        tk      = yf.Ticker(ticker)
        h15     = tk.history(period="5d",  interval="15m")
        h5      = tk.history(period="5d",  interval="5m")
        h1      = tk.history(period="2d",  interval="1m")
        h1d     = tk.history(period="60d", interval="1d")

        if h5.empty or h1.empty:
            return _no_signal("分钟级数据不可用（需实盘 API）")

        # ── 大框架：15分钟 VWAP 方向 ─────────────────────
        tp15  = (h15["High"] + h15["Low"] + h15["Close"]) / 3
        vwap15 = (tp15 * h15["Volume"]).cumsum() / h15["Volume"].cumsum()
        price15 = float(h15["Close"].iloc[-1])
        vwap15_v = float(vwap15.iloc[-1])
        above_vwap15 = price15 > vwap15_v

        ma9_15 = float(h15["Close"].rolling(9).mean().iloc[-1])
        ma21_15 = float(h15["Close"].rolling(21).mean().iloc[-1])
        trend15 = ma9_15 > ma21_15

        # ── 触发框架：5分钟回踩 VWAP 后反弹 ─────────────
        tp5   = (h5["High"] + h5["Low"] + h5["Close"]) / 3
        vwap5  = (tp5 * h5["Volume"]).cumsum() / h5["Volume"].cumsum()
        c5     = h5["Close"]
        price5 = float(c5.iloc[-1])
        vwap5_v = float(vwap5.iloc[-1])

        # 检测最近3根K线是否完成了回踩+反弹
        prev3 = c5.iloc[-4:-1].values
        prev_vwap3 = vwap5.iloc[-4:-1].values
        touched_vwap = any(abs(prev3[i] - prev_vwap3[i]) / prev_vwap3[i] < 0.003
                           for i in range(len(prev3)))
        bounced = float(c5.iloc[-1]) > float(c5.iloc[-2])

        # ── 执行框架：1分钟 RSI + 量能 ───────────────────
        rsi1_s = _rsi_series(h1["Close"], 9)
        rsi1   = float(rsi1_s.iloc[-1])

        vol1   = h1["Volume"]
        vol_ma = float(vol1.tail(20).mean())
        vol_now = float(vol1.iloc[-1])
        vol_surge = vol_now > vol_ma * 1.3

        # ── ATR 计算（1分钟级别）────────────────────────
        atr1m = _calc_atr(h1)
        price  = float(h1["Close"].iloc[-1])

        # ── 信号判断 ─────────────────────────────────────
        long_cond = (
            above_vwap15 and trend15           # 大框架看多
            and price5 > vwap5_v               # 中框架价格在VWAP上
            and touched_vwap and bounced       # 回踩后反弹
            and rsi1 > 45 and rsi1 < 70        # 1分钟RSI健康区
            and vol_surge                      # 量能确认
        )

        short_cond = (
            not above_vwap15 and not trend15   # 大框架看空
            and price5 < vwap5_v               # 中框架价格在VWAP下
            and touched_vwap and not bounced   # 回踩后继续下
            and rsi1 < 55 and rsi1 > 30
            and vol_surge
        )

        if long_cond:
            direction = "LONG"
            entry = price
            stop  = price - atr1m * 0.8
            t1    = price + atr1m * 1.5
            t2    = price + atr1m * 2.5
            conf  = 65 + (10 if touched_vwap else 0) + (10 if vol_surge else 0)
            reason = (f"三重确认做多：15m VWAP上方+均线多头 / "
                      f"5m 回踩VWAP后反弹 / 1m RSI={rsi1:.0f}+量能{vol_now/vol_ma:.1f}x")
        elif short_cond:
            direction = "SHORT"
            entry = price
            stop  = price + atr1m * 0.8
            t1    = price - atr1m * 1.5
            t2    = price - atr1m * 2.5
            conf  = 65 + (10 if touched_vwap else 0) + (10 if vol_surge else 0)
            reason = (f"三重确认做空：15m VWAP下方+均线空头 / "
                      f"5m 回踩VWAP后继续下 / 1m RSI={rsi1:.0f}+量能{vol_now/vol_ma:.1f}x")
        else:
            missing = []
            if not above_vwap15:  missing.append("大框架VWAP不支持多")
            if not trend15:       missing.append("均线不多头")
            if not touched_vwap:  missing.append("未回踩VWAP")
            if not vol_surge:     missing.append(f"量能不足({vol_now/vol_ma:.1f}x<1.3x)")
            if rsi1 >= 70:        missing.append(f"RSI={rsi1:.0f}超买")
            return _no_signal(f"三重确认未满足：{' / '.join(missing[:3])}")

        rr = abs(t1 - entry) / max(abs(stop - entry), 0.001)

        # 剥头皮仓位（固定亏损额更小：0.3%）
        max_risk  = portfolio * 0.003
        stop_dist = abs(stop - entry)
        shares    = max(1, int(max_risk / stop_dist)) if stop_dist > 0 else 1

        return {
            "name":       "剥头皮策略（三重时间框架）",
            "signal":     direction,
            "entry":      round(entry, 2),
            "stop_loss":  round(stop, 2),
            "target1":    round(t1, 2),
            "target2":    round(t2, 2),
            "confidence": min(90, round(conf)),
            "risk_reward": round(rr, 2),
            "shares":     shares,
            "atr_1m":     round(atr1m, 4),
            "vwap_15m":   round(vwap15_v, 2),
            "vwap_5m":    round(vwap5_v, 2),
            "rsi_1m":     round(rsi1, 1),
            "vol_ratio":  round(vol_now / vol_ma, 2),
            "reason":     reason,
            "timeframe":  "三重：15m大框架 + 5m触发 + 1m执行",
            "session":    "开盘黄金期" if prime_open else "收盘黄金期",
            "warning":    "实盘需实时数据（Alpaca/Finnhub），yfinance 15分钟延迟仅供参考",
            "scalp_rules": [
                "目标达到50%时减半仓，止损移至成本价",
                "持仓超过5分钟无盈利主动离场",
                "单日5笔上限，日亏0.5%停止",
                "只在09:30-10:30和14:30-15:30操作",
                "不在财报/FOMC/CPI当天剥头皮",
            ],
        }

    except Exception as e:
        return _no_signal(f"剥头皮数据异常：{e}")


# ─────────────────────────────────────────────────────────────
# 综合评分 + 仓位管理
# ─────────────────────────────────────────────────────────────

def _composite_score(results: dict) -> dict:
    active = [
        {"name": k, "direction": v["signal"],
         "confidence": v.get("confidence", 50),
         "rr": v.get("risk_reward", 1)}
        for k, v in results.items()
        if v.get("signal") in ("LONG", "SHORT")
    ]

    if not active:
        return {"direction": "NO_TRADE", "consensus": 0,
                "confidence": 0, "avg_rr": 0,
                "reason": "所有策略均无触发信号，保持空仓",
                "active_signals": 0}

    longs  = [s for s in active if s["direction"] == "LONG"]
    shorts = [s for s in active if s["direction"] == "SHORT"]

    if longs and shorts:
        return {"direction": "CONFLICTED", "consensus": 0,
                "confidence": 0, "avg_rr": 0,
                "reason": f"多空冲突（{len(longs)}多/{len(shorts)}空），建议观望",
                "active_signals": len(active)}

    dominant  = longs or shorts
    direction = "LONG" if longs else "SHORT"
    avg_conf  = sum(s["confidence"] for s in dominant) / len(dominant)
    avg_rr    = sum(s["rr"] for s in dominant) / len(dominant)
    bonus     = (len(dominant) - 1) * 5        # 多信号共振加分
    final_conf = min(95, avg_conf + bonus)

    return {
        "direction":     direction,
        "consensus":     len(dominant),
        "confidence":    round(final_conf),
        "avg_rr":        round(avg_rr, 2),
        "reason":        f"{len(dominant)}策略共振{direction}，置信度{final_conf:.0f}%",
        "active_signals": len(active),
    }


def _position_sizing(composite: dict, hist: pd.DataFrame,
                     portfolio: float) -> dict:
    direction = composite.get("direction", "NO_TRADE")

    if direction in ("NO_TRADE", "CONFLICTED"):
        return {"shares": 0, "dollar_risk": 0, "position_size": 0,
                "pct_portfolio": 0, "rule": "无信号，保持空仓"}

    price   = float(hist["Close"].iloc[-1])
    atr_val = _calc_atr(hist)
    stop_d  = max(atr_val * 1.5, price * 0.005)  # 最少0.5%

    max_risk = portfolio * 0.01          # 1%风险规则
    shares   = int(max_risk / stop_d)
    pos_size = shares * price
    pct_port = pos_size / portfolio * 100

    # 低置信度缩仓
    conf = composite.get("confidence", 50)
    if conf < 65:
        shares   = int(shares * 0.5)
        pos_size = shares * price
        pct_port = pos_size / portfolio * 100

    return {
        "shares":         shares,
        "dollar_risk":    round(max_risk),
        "position_size":  round(pos_size),
        "pct_portfolio":  round(pct_port, 1),
        "stop_distance":  round(stop_d, 2),
        "rule":           f"1%风险规则：最多亏${max_risk:.0f}，止损{stop_d:.2f}（1.5ATR）",
    }


def _check_rules(composite: dict, hist: pd.DataFrame) -> list:
    rules = []
    if composite.get("direction") == "NO_TRADE":
        rules.append("信号分不足，绝对不入场")

    try:
        vix = float(yf.Ticker("^VIX").history(period="1d")["Close"].iloc[-1])
        if vix > 35:
            rules.append(f"VIX={vix:.0f}>35，所有仓位减半或停止交易")
    except Exception:
        pass

    now = datetime.now(ET)
    total_min = now.hour * 60 + now.minute
    if 570 <= total_min <= 585:
        rules.append("开盘前15分钟（09:30-09:45 ET）禁止入场")
    if total_min >= 940:
        rules.append("收盘前20分钟（15:40-16:00 ET）禁止开新仓")

    return rules


# ─────────────────────────────────────────────────────────────
# 共用辅助函数
# ─────────────────────────────────────────────────────────────

def _calc_rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    s    = _rsi_series(close)
    return float(s.iloc[-1])


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs    = gain / loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def _calc_atr(hist: pd.DataFrame, period: int = 14) -> float:
    tr = pd.concat([
        hist["High"] - hist["Low"],
        (hist["High"] - hist["Close"].shift()).abs(),
        (hist["Low"]  - hist["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_s = tr.ewm(span=period, adjust=False).mean()
    val   = float(atr_s.iloc[-1])
    return val if val == val else float(hist["Close"].iloc[-1]) * 0.015


def _no_signal(reason: str) -> dict:
    return {
        "signal": "NO_TRADE", "confidence": 0, "reason": reason,
        "entry": None, "stop_loss": None,
        "target1": None, "target2": None, "risk_reward": None,
    }
