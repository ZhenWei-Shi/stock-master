"""
假突破/冲高回落识别（2026-09-29新增，观察期信号：仅展示，不参与打分/否决）

源起：2026-09-22~09-29 ASTS在收敛三角形末端反复出现"盘中冲过阻力、收盘收回"
（9/22、9/24、9/25、9/28、9/29五次），当时全靠人工对照趋势线判断。这是做空/
put方向最常见的入场形态，而cold_model此前没有任何做空setup识别。

只看日线（最后一根可以是盘中未走完的K线），纯函数、不发网络请求。三类信号：
  1. 刺穿水平阻力后收回：最高价 > 前N日最高价，收盘 < 该价位
  2. 刺穿下降趋势线后收回：最近3个摆动高点依次走低时，用最小二乘拟合下降
     阻力线，最高价 > 线值、收盘 < 线值（与wiki里ASTS三角形上沿的画法一致）
  3. 冲高回落：较昨收最大涨幅 ≥ 3%，且收盘回吐了其中 ≥ 60%
任一信号成立，且收盘位于当日区间下半部，才算"触发"。量比只作为附加说明
（带量的失败突破更可信），不作为触发条件——样本太少，先观察再定阈值。

【2026-09-29回测结论：单独使用无预测力，未接入cold_model】
29只高beta/成长股（ASTS/RKLB/NVDA/TSLA/COIN等）3年日线、收盘价口径：
  - 触发后3日平均+0.52%（±0.57），基准+0.68%；下跌概率51%，与随机无差别
  - 限定价格<MA200：同样无差别；"刺穿水平高点后收回"之后5日反而平均+2.65%
样本有幸存者偏差（多为这几年的上涨股），但结论足够明确：只看日K形态做不了
put信号。本模块保留为描述工具，下一步是叠加伽马结构（GEX）和个股消息标签，
在前向样本里看条件组合是否有效——历史期权链免费数据拿不到，只能往前攒。
"""
import numpy as np
import pandas as pd

SWING_WINDOW        = 3      # 摆动高点：左右各3根K线都不高于它
SWING_LOOKBACK      = 90     # 在最近90根K线里找摆动高点
HORIZ_LOOKBACK      = 20     # 水平阻力：前20日最高价
SPIKE_MIN_PCT       = 3.0    # 冲高回落：较昨收最大涨幅下限(%)
SPIKE_GIVEBACK      = 0.6    # 冲高回落：收盘回吐涨幅的比例下限
CLOSE_POS_MAX       = 0.5    # 收盘须位于当日区间下半部
VOL_CONFIRM_RATIO   = 1.5    # 量比≥此值标注"带量"


def find_swing_highs(high: pd.Series, window: int = SWING_WINDOW) -> list:
    """返回摆动高点的位置下标。左侧严格更低、右侧不高于（并列高点只取第一个）。"""
    vals = high.to_numpy(dtype=float)
    out = []
    for i in range(window, len(vals) - window):
        left = vals[i - window:i]
        right = vals[i + 1:i + 1 + window]
        if (vals[i] > left).all() and (vals[i] >= right).all():
            out.append(i)
    return out


def descending_resistance(high: pd.Series, idx: int,
                          window: int = SWING_WINDOW,
                          lookback: int = SWING_LOOKBACK):
    """
    idx当天的下降阻力线价位。只用idx之前已被确认的摆动高点（右侧window根
    也在idx之前），避免用到未来数据。最近3个摆动高点必须依次走低，否则返回None。
    """
    start = max(0, idx - lookback)
    seg = high.iloc[start:idx]
    swings = [start + p for p in find_swing_highs(seg, window)]
    if len(swings) < 3:
        return None
    last3 = swings[-3:]
    ys = high.iloc[last3].to_numpy(dtype=float)
    if not (ys[0] > ys[1] > ys[2]):
        return None
    slope, intercept = np.polyfit(np.array(last3, dtype=float), ys, 1)
    return float(slope * idx + intercept)


def detect_failed_breakout(hist: pd.DataFrame, idx: int = -1,
                           elapsed_frac: float = 1.0) -> dict:
    """
    判断hist第idx根日线是否为假突破/冲高回落。
    elapsed_frac：盘中调用时当天已过交易时间比例，用于折算量比（同cold_model量比门）。
    """
    n = len(hist)
    i = idx if idx >= 0 else n + idx
    if i < HORIZ_LOOKBACK + 1:
        return {"triggered": False, "signals": [], "note": "数据不足，跳过假突破识别"}

    high, low, close, vol = hist["High"], hist["Low"], hist["Close"], hist["Volume"]
    h, l, c = float(high.iloc[i]), float(low.iloc[i]), float(close.iloc[i])
    prev_c = float(close.iloc[i - 1])
    rng = h - l
    close_pos = (c - l) / rng if rng > 0 else 0.5

    signals = []
    levels = {}

    horiz = float(high.iloc[i - HORIZ_LOOKBACK:i].max())
    levels["horizontal"] = round(horiz, 2)
    if h > horiz and c < horiz:
        signals.append(f"刺穿{HORIZ_LOOKBACK}日高点${horiz:.2f}（+{(h / horiz - 1) * 100:.1f}%）后收回")

    trend = descending_resistance(high, i)
    if trend is not None:
        levels["trendline"] = round(trend, 2)
        if h > trend and c < trend:
            signals.append(f"刺穿下降阻力线${trend:.2f}（+{(h / trend - 1) * 100:.1f}%）后收回")

    up = h - prev_c
    up_pct = up / prev_c * 100 if prev_c > 0 else 0
    if up_pct >= SPIKE_MIN_PCT:
        giveback = (h - c) / up
        if giveback >= SPIKE_GIVEBACK:
            signals.append(f"冲高回落：最高较昨收+{up_pct:.1f}%，收盘回吐{giveback * 100:.0f}%")

    v20 = float(vol.iloc[i - 20:i].mean())
    vol_ratio = float(vol.iloc[i]) / (v20 * max(elapsed_frac, 0.05)) if v20 > 0 else None

    triggered = bool(signals) and close_pos <= CLOSE_POS_MAX
    if triggered:
        vol_tag = ""
        if vol_ratio is not None:
            vol_tag = (f"，带量（量比{vol_ratio:.1f}x）" if vol_ratio >= VOL_CONFIRM_RATIO
                       else f"，未放量（量比{vol_ratio:.1f}x）")
        note = "⚠️假突破/冲高回落：" + "；".join(signals) + f"；收盘位于当日区间{close_pos * 100:.0f}%处" + vol_tag
    elif signals:
        note = "有刺穿/冲高但收盘仍在当日区间上半部，不算失败：" + "；".join(signals)
    else:
        note = "无假突破/冲高回落形态"

    return {
        "triggered": triggered,
        "signals": signals,
        "levels": levels,
        "close_pos": round(close_pos, 2),
        "vol_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
        "note": note,
    }
