"""
历史回测引擎 v1

设计原则（专业级防泄露）：
  1. 信号在 T 日收盘后计算，T+1 日开盘价入场 → 零未来数据
  2. 止损/目标判断用当日 High/Low → 捕捉真实盘中触及
  3. 入场价加滑点（0.1% 激进模式，小盘股实际更高）
  4. 最多同时持有 3 个仓位（$2k 账户标准）
  5. 最大持仓 15 个交易日后强制平仓

核心 Gate（历史可重现的部分）：
  ✅ VIX 历史数据（^VIX）
  ✅ SPY 趋势（MA50 上下）
  ✅ 价格趋势（MA20 > MA50）
  ✅ RSI 范围（激进模式 35-80）
  ✅ MACD 柱翻正
  ✅ 量能放大（> 1.2x 均量）
  ✅ 距52周高点在20%以内
  ✅ ATR 止损 < 12%（激进）
  ❌ 市场时间窗口（历史回测无意义）
  ❌ VWAP（需分钟线，不可用）
  ❌ PDT（模拟账户无需考虑）
"""

import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(_DATA, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 回测配置常量（修改参数在此处；过拟合风险：避免对单一时间段调参）
# ══════════════════════════════════════════════════════════════

# ── 技术指标参数 ──────────────────────────────────────────────
RSI_PERIOD                  = 14
RSI_AGG_MIN, RSI_AGG_MAX    = 35, 80
MACD_FAST, MACD_SLOW        = 12, 26
MACD_SIGNAL                 = 9
ATR_PERIOD                  = 14
ATR_STOP_MULT               = 2.0      # 回测用ATR×2（含更多余量）
MA_SHORT, MA_MID             = 20, 50

# ── 入场门限 ─────────────────────────────────────────────────
VIX_PANIC_HARD              = 40.0     # VIX>40：禁入
SPY_MA50_MIN_RATIO          = 0.97     # SPY至少在MA50的97%以上
VOL_RATIO_MIN               = 1.2      # 量比≥1.2倍均量
NEAR_HIGH_52W_MAX_DROP      = 0.20     # 距52周高点≤20%
MAX_STOP_PCT                = 12.0     # 最大ATR止损%
MIN_STOP_PCT                = 0.3      # 最小ATR止损%

# ── 仓位管理 ─────────────────────────────────────────────────
DEFAULT_MAX_POSITIONS       = 3        # 最多同时持N个仓位
DEFAULT_ACCOUNT             = 2_000.0  # 默认回测资金
DEFAULT_RISK_PCT            = 3.0      # 每笔风险% (3% of account)
DEFAULT_MAX_HOLD_DAYS       = 15       # 最长持仓天数
DEFAULT_SLIPPAGE_PCT        = 0.1      # 滑点%
DEFAULT_TARGET_RR           = 2.0      # 目标盈亏比
MAX_POSITION_PCT            = 0.50     # 单仓最大比例

# ── 数据预热 ─────────────────────────────────────────────────
DATA_WARMUP_DAYS            = 90       # 指标预热天数（MA50需要50天）
MIN_TRADING_DAYS            = 20       # 回测至少需要N个交易日

# ══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# 技术指标（逐日切片安全版）
# ─────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 999
    return float(100 - 100 / (1 + rs))


def _macd_hist(close: pd.Series) -> float:
    if len(close) < 30:
        return 0.0
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line  = ema12 - ema26
    sig   = line.ewm(span=9, adjust=False).mean()
    hist  = line - sig
    return float(hist.iloc[-1])


def _macd_hist_prev(close: pd.Series) -> float:
    if len(close) < 31:
        return 0.0
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line  = ema12 - ema26
    sig   = line.ewm(span=9, adjust=False).mean()
    hist  = line - sig
    return float(hist.iloc[-2])


def _atr(high, low, close, period: int = 14) -> float:
    if len(close) < period + 1:
        return float(close.iloc[-1] * 0.03)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


# ─────────────────────────────────────────────────────────────
# 入场信号计算（T 日收盘后）
# ─────────────────────────────────────────────────────────────

def _check_entry_signal(
    hist_slice: pd.DataFrame,  # T 日及之前的所有历史数据
    vix_slice:  pd.Series,
    spy_slice:  pd.DataFrame,
) -> dict:
    """
    对 hist_slice 的最后一天（T 日收盘后）计算入场信号。
    返回：{signal: bool, reason: str, stop_loss_pct: float, score: int}
    """
    if len(hist_slice) < 55:  # MA50 需要50根，再加安全边际
        return {"signal": False, "reason": "历史数据不足（<55天）", "score": 0}

    close  = hist_slice["Close"]
    high   = hist_slice["High"]
    low    = hist_slice["Low"]
    volume = hist_slice["Volume"]
    price  = float(close.iloc[-1])

    # Gate: VIX
    vix = float(vix_slice.iloc[-1]) if len(vix_slice) > 0 else 20.0
    if vix > 40:
        return {"signal": False, "reason": f"VIX={vix:.1f}极度恐慌，禁止入场", "score": 0}

    # Gate: SPY 趋势（大盘框架）
    if len(spy_slice) >= 50:
        spy_close = spy_slice["Close"]
        spy_ma50  = float(spy_close.rolling(50).mean().iloc[-1])
        spy_price = float(spy_close.iloc[-1])
        if spy_price < spy_ma50 * 0.97:
            return {"signal": False, "reason": f"SPY在MA50下方{((spy_price/spy_ma50)-1)*100:.1f}%，空头市场禁止做多", "score": 0}

    # Gate: 价格趋势
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    if price < ma50:
        return {"signal": False, "reason": f"价格${price:.2f}<MA50${ma50:.2f}，趋势不支持", "score": 0}
    if ma20 < ma50:
        return {"signal": False, "reason": f"MA20<MA50，短期趋势弱于长期", "score": 0}

    # Gate: RSI（激进 35-80）
    rsi_val = _rsi(close)
    if rsi_val < 35 or rsi_val > 80:
        return {"signal": False, "reason": f"RSI={rsi_val:.1f}，不在35-80激进区间", "score": 0}

    # Gate: MACD 柱翻正（动能转正）
    mh_now  = _macd_hist(close)
    mh_prev = _macd_hist_prev(close)
    if mh_now <= 0 or mh_now <= mh_prev:
        return {"signal": False, "reason": f"MACD柱={mh_now:.4f}未翻正或未加速", "score": 0}

    # Gate: 量能（当日成交量 > 20日均量 × 1.2）
    vol_ma20  = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio = float(volume.iloc[-1]) / vol_ma20 if vol_ma20 > 0 else 1.0
    if vol_ratio < 1.2:
        return {"signal": False, "reason": f"量能不足：{vol_ratio:.2f}x均量（需>1.2x）", "score": 0}

    # Gate: 距52周高点（近突破信号）
    high_52w = float(high.tail(252).max())
    near_high_pct = (high_52w - price) / high_52w * 100
    if near_high_pct > 20:
        return {"signal": False, "reason": f"距52周高点{near_high_pct:.1f}%，不在突破区间", "score": 0}

    # Gate: 止损大小（ATR × 2 ÷ 价格 < 12%）
    atr_val    = _atr(high, low, close)
    stop_dist  = atr_val * 2
    stop_pct   = stop_dist / price * 100
    if stop_pct > 12:
        return {"signal": False, "reason": f"ATR止损距离{stop_pct:.1f}%>12%，风险过大", "score": 0}

    # 评分（通过所有门则计算加权分）
    score = 60  # 基础分（通过全部门）
    if vix < 20:          score += 5
    if rsi_val < 60:      score += 5   # 非超买区更好
    if vol_ratio > 2.0:   score += 5   # 放量突破
    if near_high_pct < 5: score += 10  # 创新高区域（突破形态）
    if mh_now > mh_prev * 1.5: score += 5  # MACD 加速

    return {
        "signal":       True,
        "score":        min(score, 100),
        "rsi":          round(rsi_val, 1),
        "macd_hist":    round(mh_now, 4),
        "vol_ratio":    round(vol_ratio, 2),
        "near_high_pct": round(near_high_pct, 1),
        "stop_loss_pct": round(stop_pct, 2),
        "atr":          round(atr_val, 4),
        "vix":          round(vix, 1),
        "reason":       "全部门通过，入场信号有效",
    }


# ─────────────────────────────────────────────────────────────
# 主回测引擎
# ─────────────────────────────────────────────────────────────

def run_backtest(
    tickers:       list,
    start_date:    str   = "2024-01-01",
    end_date:      str   = "2025-12-31",
    account:       float = 2000.0,
    max_positions: int   = 3,
    risk_pct:      float = 3.0,      # 每笔最大风险（账户%）
    max_hold_days: int   = 15,       # 最长持仓天数
    slippage_pct:  float = 0.1,      # 滑点（小盘股用0.1%）
    target_rr:     float = 2.0,      # 目标盈亏比（止损的N倍）
    verbose:       bool  = False,
) -> dict:
    """
    历史回测主入口。

    参数：
      tickers     — 股票列表，如 ["NVDA", "AMD", "TSLA"]
      start_date  — 回测开始日期（YYYY-MM-DD）
      end_date    — 回测结束日期（YYYY-MM-DD）
      account     — 初始账户净值（美元）
      risk_pct    — 每笔最大风险（账户百分比）
      target_rr   — 目标盈亏比（2.0 = 止损1%时目标2%）
      verbose     — 是否打印每笔交易

    返回：
      完整绩效报告 + 逐笔交易记录

    ⚠️ 过拟合警告：
      回测参数（RSI范围、止损%、量比门限）均基于近期市场。
      单一时间段结果不代表未来表现。建议：
        1. 用多段时间窗口验证（Walk-Forward）
        2. 避免用回测结果反推参数（曲线拟合）
        3. 用 Out-of-Sample 数据做最终验证
    """
    print(f"[回测] 下载历史数据... 标的：{tickers} + SPY + VIX")
    print(f"[回测] 周期：{start_date} → {end_date}")

    # 多下载60天用于初始指标预热（MA50 需要50个数据点）
    fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")

    # 批量下载（一次请求减少API压力）
    all_tickers = list(set(tickers + ["SPY", "^VIX"]))
    raw = yf.download(
        all_tickers,
        start=fetch_start,
        end=end_date,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )

    # 提取各标的数据
    def _get(symbol):
        if len(all_tickers) == 1:
            return raw
        try:
            d = raw[symbol] if symbol in raw.columns.get_level_values(0) else pd.DataFrame()
            return d.dropna(how="all")
        except Exception:
            return pd.DataFrame()

    spy_data = _get("SPY")
    vix_data = _get("^VIX")
    vix_close = vix_data["Close"] if not vix_data.empty else pd.Series(dtype=float)

    # 回测交易日列表（start_date 开始）
    start_dt = pd.Timestamp(start_date)
    end_dt   = pd.Timestamp(end_date)

    # 用 SPY 的交易日作为基准（避免周末/假日）
    if spy_data.empty:
        print("[回测] 警告：无法获取 SPY 数据，回测终止")
        return {"ok": False, "error": "SPY数据获取失败"}

    trading_days = spy_data.index[(spy_data.index >= start_dt) & (spy_data.index <= end_dt)]
    if len(trading_days) < 20:
        return {"ok": False, "error": f"回测期间交易日不足（{len(trading_days)}天）"}

    print(f"[回测] 有效交易日：{len(trading_days)} 天")

    # 账户状态
    cash       = account
    peak_value = account
    positions  = {}   # trade_id → position dict
    trade_log  = []   # 所有已平仓交易

    # 主循环：逐日模拟
    for i, signal_day in enumerate(trading_days[:-1]):  # 最后一天无法入场
        entry_day = trading_days[i + 1]  # T+1 日实际入场

        # 更新持仓状态（用 entry_day 的价格检查止损/目标）
        to_close = []
        for tid, pos in positions.items():
            tk_data = _get(pos["ticker"])
            if tk_data.empty or entry_day not in tk_data.index:
                continue

            day_row  = tk_data.loc[entry_day]
            day_high = float(day_row["High"])
            day_low  = float(day_row["Low"])
            day_close = float(day_row["Close"])

            exit_price  = None
            exit_reason = None
            hold_days   = (entry_day - pos["entry_day"]).days

            if day_low <= pos["stop_loss"]:
                # 触及止损：用止损价（保守）
                exit_price  = pos["stop_loss"]
                exit_reason = "止损"
            elif day_high >= pos["target"]:
                # 触及目标：用目标价
                exit_price  = pos["target"]
                exit_reason = "目标"
            elif hold_days >= max_hold_days:
                # 超时强制平仓
                exit_price  = day_close
                exit_reason = f"超时{hold_days}天"

            if exit_price is not None:
                # 平仓含滑点（卖出价略低）
                exec_exit  = exit_price * (1 - slippage_pct / 100)
                pnl        = (exec_exit - pos["entry_price"]) * pos["shares"]
                pnl_pct    = (exec_exit - pos["entry_price"]) / pos["entry_price"] * 100
                proceeds   = exec_exit * pos["shares"]
                cash      += proceeds

                trade_log.append({
                    "ticker":       pos["ticker"],
                    "entry_day":    str(pos["entry_day"].date()),
                    "exit_day":     str(entry_day.date()),
                    "hold_days":    hold_days,
                    "entry_price":  round(pos["entry_price"], 4),
                    "exit_price":   round(exec_exit, 4),
                    "shares":       pos["shares"],
                    "pnl":          round(pnl, 2),
                    "pnl_pct":      round(pnl_pct, 2),
                    "result":       "win" if pnl > 0 else "loss",
                    "exit_reason":  exit_reason,
                    "signal_score": pos["signal_score"],
                    "stop_loss":    pos["stop_loss"],
                    "target":       pos["target"],
                })
                to_close.append(tid)

                if verbose:
                    emoji = "✅" if pnl > 0 else "❌"
                    print(f"  {emoji} 平仓 {pos['ticker']} {exit_reason}  "
                          f"P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)  "
                          f"持仓{hold_days}天")

        for tid in to_close:
            del positions[tid]

        # 更新账户总值
        pos_value = sum(
            float((_get(p["ticker"]).loc[entry_day]["Close"]
                   if not _get(p["ticker"]).empty and entry_day in _get(p["ticker"]).index
                   else p["entry_price"])) * p["shares"]
            for p in positions.values()
        )
        total_value = cash + pos_value
        peak_value  = max(peak_value, total_value)

        # 信号扫描（T 日收盘后，T+1 入场）
        if len(positions) >= max_positions:
            continue  # 仓位已满

        entry_day_data_available = {}
        for ticker in tickers:
            tk_data = _get(ticker)
            if tk_data.empty:
                continue
            # 已有仓位的标的跳过
            if any(p["ticker"] == ticker for p in positions.values()):
                continue
            # T 日及之前的切片（严格不含 T+1）
            hist_slice = tk_data[tk_data.index <= signal_day]
            if len(hist_slice) < 55:
                continue

            spy_slice = spy_data[spy_data.index <= signal_day]
            vix_slice = vix_close[vix_close.index <= signal_day].tail(5)

            sig = _check_entry_signal(hist_slice, vix_slice, spy_slice)
            if not sig.get("signal"):
                continue

            # T+1 日开盘价入场
            if entry_day not in tk_data.index:
                continue
            entry_row   = tk_data.loc[entry_day]
            entry_price = float(entry_row["Open"]) * (1 + slippage_pct / 100)

            # 仓位计算（3% 风险规则）
            stop_pct   = sig["stop_loss_pct"] / 100
            risk_dollar = total_value * (risk_pct / 100)
            stop_dist  = entry_price * stop_pct
            if stop_dist <= 0:
                continue  # 止损距离无效，跳过此信号
            shares     = max(1, int(risk_dollar / stop_dist))
            cost       = shares * entry_price
            max_pos_val = total_value * 0.50  # 单仓不超过账户 50%

            if cost > cash or cost > max_pos_val:
                shares = max(1, int(min(cash, max_pos_val) / entry_price))
                cost   = shares * entry_price
                if cost > cash:
                    continue  # 资金不足

            stop_loss = round(entry_price * (1 - stop_pct), 4)
            target    = round(entry_price * (1 + stop_pct * target_rr), 4)
            cash     -= cost

            tid = f"{ticker}_{str(signal_day.date()).replace('-', '')}"
            positions[tid] = {
                "ticker":       ticker,
                "entry_day":    entry_day,
                "entry_price":  round(entry_price, 4),
                "shares":       shares,
                "stop_loss":    stop_loss,
                "target":       target,
                "signal_score": sig["score"],
            }

            if verbose:
                print(f"  📈 开仓 {ticker} ×{shares}股 @ ${entry_price:.2f}  "
                      f"止损${stop_loss:.2f}  目标${target:.2f}  "
                      f"信号分{sig['score']}  ({str(signal_day.date())}信号)")

            if len(positions) >= max_positions:
                break

    # 强制平仓所有剩余仓位（回测结束日）
    last_day = trading_days[-1]
    for tid, pos in positions.items():
        tk_data = _get(pos["ticker"])
        if not tk_data.empty and last_day in tk_data.index:
            exit_price = float(tk_data.loc[last_day]["Close"]) * (1 - slippage_pct / 100)
        else:
            exit_price = pos["entry_price"]
        pnl     = (exit_price - pos["entry_price"]) * pos["shares"]
        pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
        cash   += exit_price * pos["shares"]
        hold    = (last_day - pos["entry_day"]).days
        trade_log.append({
            "ticker":       pos["ticker"],
            "entry_day":    str(pos["entry_day"].date()),
            "exit_day":     str(last_day.date()),
            "hold_days":    hold,
            "entry_price":  round(pos["entry_price"], 4),
            "exit_price":   round(exit_price, 4),
            "shares":       pos["shares"],
            "pnl":          round(pnl, 2),
            "pnl_pct":      round(pnl_pct, 2),
            "result":       "win" if pnl > 0 else "loss",
            "exit_reason":  "回测结束强平",
            "signal_score": pos["signal_score"],
            "stop_loss":    pos["stop_loss"],
            "target":       pos["target"],
        })

    # ── 绩效统计 ──────────────────────────────────────────────
    total_val = cash
    total_trades = len(trade_log)

    if total_trades == 0:
        return {
            "ok": True,
            "note": "回测期间没有触发任何交易信号，请检查参数或扩大标的范围",
            "params": {"tickers": tickers, "start": start_date, "end": end_date,
                       "account": account},
        }

    wins   = [t for t in trade_log if t["result"] == "win"]
    losses = [t for t in trade_log if t["result"] == "loss"]

    win_rate  = len(wins) / total_trades
    avg_win   = float(np.mean([t["pnl_pct"] for t in wins]))   if wins   else 0
    avg_loss  = float(np.mean([t["pnl_pct"] for t in losses])) if losses else 0
    rr_ratio  = abs(avg_win / avg_loss)                          if avg_loss != 0 else 0
    total_pnl = sum(t["pnl"] for t in trade_log)
    total_pct = total_pnl / account * 100

    # SPY 同期收益（基准对比）
    spy_return = 0.0
    if not spy_data.empty:
        spy_s = spy_data[(spy_data.index >= start_dt) & (spy_data.index <= end_dt)]
        if len(spy_s) > 1:
            spy_return = (float(spy_s["Close"].iloc[-1]) / float(spy_s["Close"].iloc[0]) - 1) * 100

    # Kelly Criterion
    kelly_f = 0.0
    if avg_loss != 0 and rr_ratio > 0:
        kelly_f = win_rate - (1 - win_rate) / rr_ratio

    # 最大回撤（逐笔近似）
    running_pnl = account
    peak_bt     = account
    max_dd      = 0.0
    for t in trade_log:
        running_pnl += t["pnl"]
        peak_bt      = max(peak_bt, running_pnl)
        dd           = (running_pnl - peak_bt) / peak_bt * 100
        max_dd       = min(max_dd, dd)

    # 平均持仓天数
    avg_hold = float(np.mean([t["hold_days"] for t in trade_log]))

    result = {
        "ok":           True,
        "params": {
            "tickers":      tickers,
            "start":        start_date,
            "end":          end_date,
            "account":      account,
            "risk_pct":     risk_pct,
            "target_rr":    target_rr,
            "slippage_pct": slippage_pct,
            "max_hold_days": max_hold_days,
        },
        "summary": {
            "total_trades":     total_trades,
            "wins":             len(wins),
            "losses":           len(losses),
            "win_rate_pct":     round(win_rate * 100, 1),
            "avg_win_pct":      round(avg_win, 2),
            "avg_loss_pct":     round(avg_loss, 2),
            "risk_reward":      round(rr_ratio, 2),
            "avg_hold_days":    round(avg_hold, 1),
            "total_pnl_usd":    round(total_pnl, 2),
            "total_return_pct": round(total_pct, 2),
            "final_account":    round(account + total_pnl, 2),
        },
        "vs_benchmark": {
            "strategy_return_pct": round(total_pct, 2),
            "spy_return_pct":      round(spy_return, 2),
            "alpha_pct":           round(total_pct - spy_return, 2),
            "beat_market":         total_pct > spy_return,
        },
        "risk": {
            "max_drawdown_pct": round(max_dd, 2),
            "kelly_full_pct":   round(kelly_f * 100, 1),
            "kelly_half_pct":   round(kelly_f * 50, 1),
            "negative_edge":    kelly_f < 0,
        },
        "trades": trade_log,
    }

    # 保存回测结果
    out_path = os.path.join(_DATA, "backtest_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"[回测] 结果已保存到 {out_path}")

    return result


# ─────────────────────────────────────────────────────────────
# 格式化输出（终端友好）
# ─────────────────────────────────────────────────────────────

def print_report(r: dict):
    if not r.get("ok"):
        print(f"回测失败：{r.get('error', r.get('note'))}")
        return

    p    = r["params"]
    s    = r["summary"]
    vs   = r["vs_benchmark"]
    risk = r["risk"]

    print("\n" + "="*55)
    print("  历史回测报告")
    print("="*55)
    print(f"  标的：{', '.join(p['tickers'])}")
    print(f"  周期：{p['start']} → {p['end']}")
    print(f"  初始资金：${p['account']:,.0f}")
    print("-"*55)
    print(f"  总交易：{s['total_trades']}笔  "
          f"盈{s['wins']}亏{s['losses']}  胜率{s['win_rate_pct']}%")
    print(f"  平均盈利：+{s['avg_win_pct']}%   平均亏损：{s['avg_loss_pct']}%")
    print(f"  盈亏比：{s['risk_reward']:.2f}x   平均持仓：{s['avg_hold_days']}天")
    print("-"*55)
    print(f"  策略收益：{s['total_return_pct']:+.2f}%  "
          f"(${s['total_pnl_usd']:+,.2f})")
    print(f"  SPY 同期：{vs['spy_return_pct']:+.2f}%")
    print(f"  超额收益：{vs['alpha_pct']:+.2f}%  "
          f"{'✅ 跑赢大盘' if vs['beat_market'] else '❌ 未跑赢大盘'}")
    print("-"*55)
    print(f"  最大回撤：{risk['max_drawdown_pct']:.2f}%")
    kelly_note = "🚨 负期望，策略需优化" if risk["negative_edge"] else f"半Kelly建议仓位：{risk['kelly_half_pct']}%"
    print(f"  Kelly：{kelly_note}")
    print("="*55 + "\n")

    if s["total_trades"] <= 20:
        print("⚠️  样本量较少，结论参考意义有限。建议≥30笔。")


# ─────────────────────────────────────────────────────────────
# CLI 直接运行
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="历史回测引擎")
    ap.add_argument("--tickers",  default="NVDA,AMD,TSLA,AAPL,MSFT",
                    help="股票代码，逗号分隔")
    ap.add_argument("--start",    default="2023-01-01", help="开始日期 YYYY-MM-DD")
    ap.add_argument("--end",      default="2024-12-31", help="结束日期 YYYY-MM-DD")
    ap.add_argument("--account",  type=float, default=2000.0, help="初始资金")
    ap.add_argument("--risk",     type=float, default=3.0,    help="每笔风险%")
    ap.add_argument("--rr",       type=float, default=2.0,    help="目标盈亏比")
    ap.add_argument("--verbose",  action="store_true",        help="打印每笔交易")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    result  = run_backtest(
        tickers    = tickers,
        start_date = args.start,
        end_date   = args.end,
        account    = args.account,
        risk_pct   = args.risk,
        target_rr  = args.rr,
        verbose    = args.verbose,
    )
    print_report(result)
