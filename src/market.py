"""
市场状态检测引擎 v2

新增：
  - 分发日（Distribution Day）计数 — O'Neil 顶部信号
  - QQQ + IWM 广度对比（成长/小盘是否同步）
  - 市场广度：% 股票站上 50 日均线（用 S&P 500 ETF 代理）
"""
import yfinance as yf
import pandas as pd
import numpy as np


def detect_market_state(spy_hist: pd.DataFrame, vix: float) -> dict:
    """
    五种市场状态 + 分发日计数：
    A 上升趋势  B 横盘震荡  C 下跌趋势  D 极度恐慌  E 方向不明

    分发日（Distribution Day）：
      当日 SPY 下跌 ≥ 0.2% 且成交量高于前日 → 机构出货信号
      25 个交易日内 ≥ 4 个分发日 → 市场顶部预警
      某分发日之后 5 个交易日内 SPY 涨 ≥ 5% → 该分发日失效
    """
    if spy_hist is None or spy_hist.empty:
        return _state("E", "无法获取大盘数据", "gray", "空仓", vix)

    close  = spy_hist["Close"]
    volume = spy_hist["Volume"]
    ma20   = float(close.rolling(20).mean().iloc[-1])
    ma50   = float(close.rolling(50).mean().iloc[-1])
    price  = float(close.iloc[-1])

    recent     = close.tail(20)
    range_pct  = (recent.max() - recent.min()) / recent.min() * 100
    year_low   = float(close.min())
    year_high  = float(close.max())
    profit_r   = ((price - year_low) / (year_high - year_low)
                  if year_high != year_low else 0.5)

    # ── 分发日计数 ────────────────────────────────────────────
    dist_days = _count_distribution_days(close, volume)

    # ── QQQ / IWM 广度 ────────────────────────────────────────
    breadth = _breadth_check()

    # ── 状态判断 ──────────────────────────────────────────────
    kwargs = dict(price=price, ma20=ma20, ma50=ma50,
                  range_pct=range_pct, distribution_days=dist_days,
                  breadth=breadth)

    if vix and vix > 30 and profit_r < 0.15:
        return _state("D",
            f"极度恐慌 — VIX={vix:.1f}>30，年内获利筹码极少，历史性机会",
            "purple", "重仓抄底（暂时放宽仓位限制）", vix, **kwargs)

    if price > ma20 > ma50:
        if dist_days >= 5:
            action = "警惕出货！分发日过多，减仓等待清洗"
            color  = "orange"
        elif dist_days >= 4:
            action = "持股但减少新仓，监控分发日是否继续累积"
            color  = "yellow"
        else:
            action = "持股 / 买 Call / Bull Call Spread"
            color  = "green"
        return _state("A",
            f"上升趋势 — 均线多头排列，分发日 {dist_days}/25",
            color, action, vix, **kwargs)

    if price < ma20 < ma50:
        return _state("C",
            f"下跌趋势 — 价格跌破双均线且均线空头排列",
            "red", "空仓观望 / Bear Put Spread", vix, **kwargs)

    if range_pct < 5 and abs(price - ma20) / ma20 < 0.02:
        return _state("B",
            f"横盘震荡 — 价格围绕均线窄幅波动（近20日振幅{range_pct:.1f}%）",
            "yellow", "蝴蝶策略 / 破翼蝴蝶，做空波动率", vix, **kwargs)

    return _state("E",
        f"方向不明 — 均线纠缠，等待突破确认",
        "gray", "空仓，方向不明时保持现金最优", vix, **kwargs)


# ─────────────────────────────────────────────────────────────
# 分发日计数
# ─────────────────────────────────────────────────────────────

def _count_distribution_days(close: pd.Series, volume: pd.Series,
                               window: int = 25) -> int:
    """
    统计近 window 个交易日内有效分发日数量。

    分发日条件：
      1. 当日收盘价比前日低 ≥ 0.2%
      2. 当日成交量 > 前日成交量

    失效条件：
      该分发日之后 5 个交易日内，SPY 涨幅 ≥ 5%（市场强势收复）
    """
    if len(close) < window + 5:
        return 0

    recent_c = close.tail(window + 5).values
    recent_v = volume.tail(window + 5).values
    n        = len(recent_c)

    # 先标记所有分发日
    dist_flags = []
    for i in range(1, n):
        pct_chg   = (recent_c[i] - recent_c[i-1]) / recent_c[i-1] * 100
        vol_up    = recent_v[i] > recent_v[i-1]
        is_dist   = pct_chg <= -0.2 and vol_up
        dist_flags.append(is_dist)

    # 检查失效：分发日后 5 天内 SPY 涨 ≥ 5%
    valid_dist = 0
    for i, flag in enumerate(dist_flags):
        if not flag:
            continue
        # 分发日是 recent_c[i+1]，检查 i+1 到 i+6 的涨幅
        start_price = recent_c[i + 1]
        invalidated = False
        for j in range(i + 2, min(i + 7, n)):
            if (recent_c[j] - start_price) / start_price * 100 >= 5:
                invalidated = True
                break
        # 只统计在近 window 天内的（去掉前 5 天的缓冲区）
        if not invalidated and i >= 4:
            valid_dist += 1

    return valid_dist


# ─────────────────────────────────────────────────────────────
# 市场广度（QQQ / IWM 同步检测）
# ─────────────────────────────────────────────────────────────

def _breadth_check() -> dict:
    """
    用 SPY/QQQ/IWM 三指数判断广度：
    - 三者同涨 = 最宽幅上涨，最健康
    - 只有 QQQ 涨 SPY/IWM 不涨 = 科技独撑，不稳定
    - IWM（小盘）领涨 = 风险偏好极高，潜在顶部信号
    """
    try:
        tickers  = ["SPY", "QQQ", "IWM"]
        data     = yf.download(tickers, period="20d", interval="1d",
                               auto_adjust=True, progress=False)["Close"]
        results  = {}
        for tk in tickers:
            if tk in data.columns and len(data[tk].dropna()) >= 5:
                s     = data[tk].dropna()
                ret1m = float((s.iloc[-1] - s.iloc[-20]) / s.iloc[-20] * 100)
                above_ma20 = float(s.iloc[-1]) > float(s.rolling(20).mean().iloc[-1])
                results[tk] = {"ret_20d": round(ret1m, 1), "above_ma20": above_ma20}

        spy_ok = results.get("SPY", {}).get("above_ma20", False)
        qqq_ok = results.get("QQQ", {}).get("above_ma20", False)
        iwm_ok = results.get("IWM", {}).get("above_ma20", False)

        if spy_ok and qqq_ok and iwm_ok:
            breadth_score = "strong"
            note = "SPY/QQQ/IWM 三指数全站 MA20，广度健康"
        elif spy_ok and qqq_ok and not iwm_ok:
            breadth_score = "moderate"
            note = "大盘/科技健康，IWM（小盘）落后，广度收窄"
        elif qqq_ok and not spy_ok:
            breadth_score = "narrow"
            note = "仅科技独撑，广度极窄，FANG 以外大多弱势"
        elif not spy_ok and not qqq_ok:
            breadth_score = "weak"
            note = "三指数均跌破 MA20，全面弱势"
        else:
            breadth_score = "mixed"
            note = "指数分歧，方向混乱"

        return {
            "score": breadth_score,
            "note":  note,
            "indices": results,
        }
    except Exception as e:
        return {"score": "unknown", "note": f"广度数据获取失败：{e}", "indices": {}}


# ─────────────────────────────────────────────────────────────
# Follow-Through Day（FTD）检测 — O'Neil 底部确认信号
# ─────────────────────────────────────────────────────────────

def detect_follow_through_day(spy_hist: pd.DataFrame) -> dict:
    """
    O'Neil Follow-Through Day（FTD）底部确认算法：

    定义：
      市场经历下跌后出现"尝试性反弹"（Attempted Rally）：
        某日收盘高于近期最低点
      从反弹第1天开始计数：
        第4天或之后，SPY 单日上涨 ≥ 1.5% 且成交量 > 前一日 = FTD
        FTD = 大概率底部已确认，可以开始布局做多

    FTD 失败条件：
      在 FTD 后 2 周内 SPY 再次跌破反弹起始低点 = 失败
      需要重新等待新的 FTD

    Wall Street 重要性：
      没有 FTD 的反弹大多数是"熊市反弹"（Dead Cat Bounce）
      O'Neil 研究显示：所有主要牛市底部均有 FTD 确认
    """
    if spy_hist is None or len(spy_hist) < 30:
        return {"detected": False, "reason": "数据不足"}

    close  = spy_hist["Close"].values
    volume = spy_hist["Volume"].values
    n      = len(close)

    # 找近期最低点（过去20天）
    recent_window = min(20, n)
    recent_low_idx = int(n - recent_window + np.argmin(close[-recent_window:]))
    recent_low     = close[recent_low_idx]

    # 检查是否从低点反弹（反弹起始日）
    if recent_low_idx >= n - 1:
        return {"detected": False, "reason": "尚在低点区域，未见反弹迹象"}

    # 从低点之后开始计天数
    rally_start_idx = recent_low_idx + 1
    rally_days      = n - rally_start_idx  # 从低点反弹已经多少天

    if rally_days < 4:
        return {
            "detected":        False,
            "attempted_rally": True,
            "rally_day":       rally_days,
            "reason":          f"处于尝试性反弹第{rally_days}天，需等到第4天或之后的FTD",
            "low_price":       round(float(recent_low), 2),
            "what_to_watch":   "等待第4天+出现单日涨幅≥1.5%且放量，即为FTD底部确认",
        }

    # 检查第4天之后是否有 FTD
    ftd_found = False
    ftd_day   = None
    ftd_date  = None
    ftd_gain  = None

    for i in range(rally_start_idx + 3, n):  # 从第4天开始
        daily_gain = (close[i] - close[i-1]) / close[i-1] * 100
        vol_up     = volume[i] > volume[i-1]
        if daily_gain >= 1.5 and vol_up:
            # 确认 FTD：这天后没有跌破低点
            not_broken = close[i] > recent_low
            if not_broken:
                ftd_found = True
                ftd_day   = i - rally_start_idx + 1
                ftd_gain  = round(daily_gain, 2)
                break

    if ftd_found:
        # 检查FTD后是否有失败信号（跌破低点）
        days_since_ftd = n - (rally_start_idx + ftd_day - 1)
        failed = close[-1] < recent_low

        return {
            "detected":      True,
            "ftd_day":       ftd_day,
            "ftd_gain_pct":  ftd_gain,
            "days_since":    days_since_ftd,
            "failed":        failed,
            "low_price":     round(float(recent_low), 2),
            "current_price": round(float(close[-1]), 2),
            "signal": ("FTD失败！市场跌破低点，需等待新的底部" if failed
                       else f"FTD确认（反弹第{ftd_day}天，涨{ftd_gain}%）——可以开始布局做多"),
            "action": ("重新等待底部，不要抄底" if failed
                       else "底部已确认，可布局高RS强势股，设好止损"),
            "note": "O'Neil研究：FTD后3-5周是最佳买点窗口，过了窗口期再入场风险上升",
        }

    return {
        "detected":        False,
        "attempted_rally": True,
        "rally_day":       rally_days,
        "reason":          f"反弹第{rally_days}天，但尚未出现符合条件的FTD",
        "low_price":       round(float(recent_low), 2),
        "what_to_watch":   "需要单日涨幅≥1.5%且成交量放大，才算FTD底部确认",
    }


# ─────────────────────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────────────────────

def _state(state, desc, color, action, vix,
           price=None, ma20=None, ma50=None, range_pct=None,
           distribution_days=0, breadth=None):
    return {
        "state":             state,
        "description":       desc,
        "color":             color,
        "action":            action,
        "vix":               vix,
        "spy_price":         round(price, 2) if price else None,
        "ma20":              round(ma20, 2)  if ma20  else None,
        "ma50":              round(ma50, 2)  if ma50  else None,
        "range_pct":         round(range_pct, 1) if range_pct else None,
        "distribution_days": distribution_days,
        "dist_warning":      distribution_days >= 4,
        "breadth":           breadth or {},
    }
