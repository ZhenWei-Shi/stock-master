"""
价格形态识别引擎

VCP（Volatility Contraction Pattern）— Mark Minervini
  逻辑：每次回调幅度越来越小 + 成交量越来越萎缩 = 筹码洗盘完毕
  突破时放量确认 = 高胜率入场点（Minervini 胜率 >90% 的形态）

Cup & Handle（杯柄形态）— William O'Neil
  逻辑：圆弧底 + 小幅整理柄 + 突破前高
  是 CANSLIM 中最重要的买点形态

Flat Base（平台整理）— O'Neil
  逻辑：短期在前高附近窄幅整理 ≤ 15%，等待突破
"""

import yfinance as yf
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────

def analyze_patterns(ticker: str) -> dict:
    """综合形态分析：VCP + Cup&Handle + Flat Base"""
    hist = yf.Ticker(ticker).history(period="1y", interval="1d")
    if hist.empty or len(hist) < 60:
        return {"error": "数据不足60天"}

    vcp    = detect_vcp(hist)
    cup    = detect_cup_handle(hist)
    flat   = detect_flat_base(hist)

    # 综合最强形态
    active = [p for p in [vcp, cup, flat] if p.get("detected")]
    best   = max(active, key=lambda x: x.get("confidence", 0)) if active else None

    return {
        "ticker":   ticker,
        "vcp":      vcp,
        "cup_handle": cup,
        "flat_base":  flat,
        "best_pattern": best,
        "action_summary": _action_summary(best, vcp, cup, flat),
    }


# ─────────────────────────────────────────────────────────────
# VCP（波动率收缩形态）
# ─────────────────────────────────────────────────────────────

def detect_vcp(hist: pd.DataFrame) -> dict:
    """
    VCP 检测算法：
    1. 在过去 60-120 天内找出至少 2 个明显回调（> 5%）
    2. 后一次回调的幅度 < 前一次（收缩）
    3. 后一次回调的成交量 < 前一次（筹码减少）
    4. 最近价格距前高 < 10%（临近突破点）
    5. 最近 5 天成交量萎缩至近20日均量的 50-80%（蓄力）

    Minervini 标准：至少 2-4 次收缩（T形态），最后一次 < 5%
    """
    close  = hist["Close"]
    volume = hist["Volume"]
    high   = hist["High"]
    price  = float(close.iloc[-1])

    # 用近 80 天数据
    window = min(80, len(hist))
    c = close.tail(window).reset_index(drop=True)
    v = volume.tail(window).reset_index(drop=True)
    h = high.tail(window).reset_index(drop=True)

    # ── 找局部高点和低点（极值检测）────────────────────────
    # order=7：在±7根K线范围内是最高/最低点才算主要高低点
    # order=5 会将短期噪音误认为支撑/阻力，导致假VCP信号
    pivot_highs = _find_pivots(c, order=7, mode="high")
    pivot_lows  = _find_pivots(c, order=7, mode="low")

    if len(pivot_highs) < 2 or len(pivot_lows) < 1:
        return {"detected": False, "reason": "形态不足：需至少2个高点和1个低点"}

    # ── 计算每次回调幅度 ─────────────────────────────────
    contractions = []
    for i in range(1, len(pivot_highs)):
        ph_idx  = pivot_highs[i - 1]
        ph2_idx = pivot_highs[i]
        # 两个高点之间的最低点
        seg_lows  = [float(c.iloc[j]) for j in range(ph_idx, ph2_idx + 1)]
        low_val   = min(seg_lows)
        high_val  = float(c.iloc[ph_idx])
        drawdown  = (high_val - low_val) / high_val * 100

        # 对应成交量均值
        seg_vols  = [float(v.iloc[j]) for j in range(ph_idx, ph2_idx + 1)]
        avg_vol   = float(np.mean(seg_vols)) if seg_vols else 0

        contractions.append({
            "drawdown_pct": round(drawdown, 1),
            "avg_vol":      round(avg_vol),
            "start_idx":    ph_idx,
            "end_idx":      ph2_idx,
        })

    if len(contractions) < 2:
        return {"detected": False, "reason": "收缩次数不足（需≥2次回调）"}

    # ── 验证收缩性（每次回调比上次小）──────────────────
    dd_list  = [ct["drawdown_pct"] for ct in contractions[-3:]]
    vol_list = [ct["avg_vol"]      for ct in contractions[-3:]]

    dd_contracting  = all(dd_list[i] < dd_list[i-1] for i in range(1, len(dd_list)))
    vol_contracting = all(vol_list[i] < vol_list[i-1] for i in range(1, len(vol_list)))

    # ── 当前位置检查 ─────────────────────────────────────
    recent_high   = float(c.tail(30).max())
    pct_from_high = (price - recent_high) / recent_high * 100

    # 最近 5 天成交量萎缩
    vol_recent5 = float(v.tail(5).mean())
    vol_ma20    = float(v.tail(20).mean())
    vol_dry     = 0.4 <= (vol_recent5 / vol_ma20) <= 0.85 if vol_ma20 > 0 else False

    latest_dd = dd_list[-1] if dd_list else 99

    detected = (
        dd_contracting
        and len(contractions) >= 2
        and pct_from_high > -12
        and latest_dd < 15
    )

    conf = 0
    if detected:
        conf += 40
        if vol_contracting: conf += 20
        if vol_dry:         conf += 15
        if latest_dd < 8:   conf += 15
        if pct_from_high > -5: conf += 10
        conf = min(95, conf)

    pivot_price = recent_high * 1.005  # 突破点 = 前高 + 0.5%

    return {
        "detected":       detected,
        "confidence":     conf if detected else 0,
        "contractions":   contractions,
        "drawdown_series": dd_list,
        "vol_contracting": vol_contracting,
        "vol_dry_up":      vol_dry,
        "latest_drawdown_pct": latest_dd,
        "pct_from_recent_high": round(pct_from_high, 1),
        "pivot_buy_price": round(pivot_price, 2),
        "reason": (
            f"VCP确认：{len(contractions)}次收缩，最近回调{latest_dd:.1f}%，"
            f"距前高{abs(pct_from_high):.1f}%，{'量能萎缩✅' if vol_dry else '量能未萎缩'}"
        ) if detected else (
            f"VCP未确认：{'回调未收缩' if not dd_contracting else ''}，"
            f"最近回调{latest_dd:.1f}%，距前高{abs(pct_from_high):.1f}%"
        ),
        "entry_rule": "突破 $" + str(round(pivot_price, 2)) + " 且当日量 > 均量1.5x 时买入",
    }


# ─────────────────────────────────────────────────────────────
# Cup & Handle（杯柄形态）
# ─────────────────────────────────────────────────────────────

def detect_cup_handle(hist: pd.DataFrame) -> dict:
    """
    O'Neil 杯柄检测：
    1. 杯深：15%-35%（太浅=未洗盘，太深=损伤过重）
    2. 杯宽：7-65周（短期用7-25天日线代理）
    3. 柄：形成在右侧上半部，回调≤8-12%，量缩
    4. 突破：杯柄右侧突破前高，放量
    """
    close  = hist["Close"]
    volume = hist["Volume"]
    price  = float(close.iloc[-1])

    if len(close) < 50:
        return {"detected": False, "reason": "数据不足50天"}

    # O'Neil 标准：杯宽 7-65周（日线代理：35-455天）
    # 实际用1年数据检测，取最近120天作为主窗口（覆盖多数中短期杯型）
    window = min(120, len(close))
    c = close.tail(window).values
    v = volume.tail(window).values

    # 杯左高点：用前30天的最高点（不是只看前15根）
    left_period  = min(30, window // 4)
    right_period = min(40, window // 3)
    cup_high_left  = float(np.max(c[:left_period]))
    cup_low        = float(np.min(c[left_period:-right_period])) if window > left_period + right_period else float(np.min(c))
    cup_high_right = float(np.max(c[-right_period:]))

    cup_depth = (cup_high_left - cup_low) / cup_high_left * 100
    cup_recovery = (cup_high_right - cup_low) / (cup_high_left - cup_low) * 100 if cup_high_left != cup_low else 0

    cup_ok = 12 <= cup_depth <= 40 and cup_recovery >= 80

    # 柄：最近 5-15 天
    handle_window = c[-15:]
    handle_high   = float(np.max(handle_window))
    handle_low    = float(np.min(handle_window))
    handle_depth  = (handle_high - handle_low) / handle_high * 100

    handle_vol_avg  = float(np.mean(v[-15:]))
    pre_vol_avg     = float(np.mean(v[-30:-15]))
    handle_vol_dry  = handle_vol_avg < pre_vol_avg * 0.85

    handle_ok = handle_depth <= 12 and handle_vol_dry

    pivot = cup_high_left * 1.005

    detected = cup_ok and handle_ok and price > cup_high_right * 0.93

    conf = 0
    if detected:
        conf = 55
        if handle_depth < 8:  conf += 15
        if handle_vol_dry:    conf += 15
        if cup_recovery > 90: conf += 10
        if 18 <= cup_depth <= 33: conf += 5
        conf = min(92, conf)

    return {
        "detected":      detected,
        "confidence":    conf if detected else 0,
        "cup_depth_pct": round(cup_depth, 1),
        "cup_recovery_pct": round(cup_recovery, 1),
        "handle_depth_pct": round(handle_depth, 1),
        "handle_vol_dry":   handle_vol_dry,
        "pivot_buy_price":  round(pivot, 2),
        "reason": (
            f"杯柄形态：杯深{cup_depth:.1f}%，复原{cup_recovery:.1f}%，"
            f"柄深{handle_depth:.1f}%，{'量干✅' if handle_vol_dry else '量未萎缩'}"
        ) if detected else (
            f"杯柄未确认：{'杯深' + str(round(cup_depth,1)) + '%超出范围' if not cup_ok else '柄未成形'}"
        ),
        "entry_rule": f"突破 ${round(pivot, 2)} 放量买入，止损 ${round(handle_low * 0.98, 2)}",
    }


# ─────────────────────────────────────────────────────────────
# Flat Base（平台整理）
# ─────────────────────────────────────────────────────────────

def detect_flat_base(hist: pd.DataFrame) -> dict:
    """
    平台整理：价格在前高附近 5-15% 范围内横盘至少 5 周（25 天）
    通常是上涨行情中段的蓄力，突破后往往有15-25%的快速拉升
    """
    close = hist["Close"]
    price = float(close.iloc[-1])

    if len(close) < 30:
        return {"detected": False, "reason": "数据不足30天"}

    window  = 35
    c_window = close.tail(window)
    high_w   = float(c_window.max())
    low_w    = float(c_window.min())
    range_pct = (high_w - low_w) / high_w * 100

    # 平台：近35天振幅 < 15%，且价格在过去6个月高点的 85% 以上
    high_6m   = float(close.tail(126).max()) if len(close) >= 126 else high_w
    pct_below = (price - high_6m) / high_6m * 100

    flat_ok   = range_pct < 15 and pct_below > -20

    pivot     = high_w * 1.005
    detected  = flat_ok and range_pct < 12

    conf = 0
    if detected:
        conf = 50
        if range_pct < 8:    conf += 20
        if pct_below > -10:  conf += 15
        if pct_below > -5:   conf += 10
        conf = min(88, conf)

    return {
        "detected":         detected,
        "confidence":       conf if detected else 0,
        "range_pct":        round(range_pct, 1),
        "pct_from_6m_high": round(pct_below, 1),
        "pivot_buy_price":  round(pivot, 2),
        "reason": (
            f"平台整理：近35天振幅{range_pct:.1f}%，距6月高点{abs(pct_below):.1f}%，蓄力待发"
        ) if detected else (
            f"平台未确认：振幅{range_pct:.1f}%（需<12%）"
        ),
        "entry_rule": f"突破 ${round(pivot, 2)} 放量买入，止损 ${round(low_w * 0.99, 2)}",
    }


# ─────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────

def _find_pivots(series: pd.Series, order: int = 5, mode: str = "high") -> list:
    """找局部极值点索引"""
    idx = []
    vals = series.values
    n   = len(vals)
    for i in range(order, n - order):
        window = vals[i - order: i + order + 1]
        center = vals[i]
        if mode == "high" and center == max(window):
            idx.append(i)
        elif mode == "low" and center == min(window):
            idx.append(i)
    return idx


def _action_summary(best, vcp, cup, flat) -> str:
    if best is None:
        return "当前无明确形态，等待整理完成"
    name = ("VCP" if best is vcp else "杯柄形态" if best is cup else "平台整理")
    conf = best.get("confidence", 0)
    pivot = best.get("pivot_buy_price", "?")
    if conf >= 70:
        return f"✅ {name}高置信度（{conf}%）：等待放量突破 ${pivot}，此为最高胜率入场点"
    elif conf >= 50:
        return f"⚠️ {name}形成中（{conf}%）：形态尚未完全确认，继续观察"
    else:
        return f"形态初现（{name}，{conf}%），仍需等待更多确认"
