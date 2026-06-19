"""
日内量化交易引擎（冷静模型）

设计原则：
  - 所有决策完全规则化，无主观判断
  - 信号分 < 6 → 绝不入场
  - 5 个独立维度打分，任一过滤器触发 → 强制 NO_TRADE
  - 仓位大小由数学公式决定（1% 风险规则）
"""
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import pytz

ET = pytz.timezone("America/New_York")

# ── 铁律（不可违反的冷静规则）────────────────────────────────
COLD_RULES = [
    "信号评分 < 6 分，绝对不入场（无论感觉多强）",
    "止损触及，立即离场，不抱侥幸，不等反弹",
    "单日亏损达账户 3%，当天停止所有交易",
    "目标1达到后，立即将止损移至成本价（锁利保本）",
    "开盘前 15 分钟（09:30-09:45 ET）禁止入场",
    "收盘前 20 分钟（15:40-16:00 ET）禁止开新仓",
    "财报前 3 天内不建新仓（无论信号多强）",
    "VIX > 35 时，所有仓位减半或停止交易",
    "每笔交易最大风险 ≤ 账户的 1%（仓位由公式决定）",
    "单日最多 3 笔交易（防止情绪性过度交易）",
]


# ── 日内数据获取 ──────────────────────────────────────────────

def get_intraday_data(ticker: str, interval: str = "5m") -> dict:
    period_map = {"1m": "1d", "5m": "5d", "15m": "1mo", "30m": "1mo"}
    period = period_map.get(interval, "5d")

    hist = yf.Ticker(ticker).history(period=period, interval=interval)
    if hist.empty:
        return {"error": f"无 {interval} 数据"}

    hist = _add_vwap(hist)
    hist = _add_volume_rate(hist)

    # 过去20日平均日成交量（用于量能参考）
    hist1d = yf.Ticker(ticker).history(period="30d", interval="1d")
    avg_daily_vol = int(hist1d["Volume"].mean()) if not hist1d.empty else 0

    chart = []
    for ts, row in hist.iterrows():
        try:
            t = ts.tz_convert(ET) if ts.tzinfo else ts
            chart.append({
                "t":    t.strftime("%H:%M"),
                "date": t.strftime("%m/%d"),
                "o":    round(float(row["Open"]),   2),
                "h":    round(float(row["High"]),   2),
                "l":    round(float(row["Low"]),    2),
                "c":    round(float(row["Close"]),  2),
                "v":    int(row["Volume"]),
                "vwap": round(float(row["vwap"]), 2) if not np.isnan(float(row["vwap"])) else None,
                "vr":   round(float(row["vol_rate"]), 2) if not np.isnan(float(row["vol_rate"])) else None,
            })
        except Exception:
            continue

    last = hist.iloc[-1]
    price    = float(last["Close"])
    vwap_val = float(last["vwap"])
    vr       = float(last["vol_rate"]) if not np.isnan(float(last["vol_rate"])) else 1.0

    return {
        "ticker":         ticker,
        "interval":       interval,
        "chart":          chart[-240:],
        "current_price":  round(price, 2),
        "vwap":           round(vwap_val, 2) if not np.isnan(vwap_val) else None,
        "vwap_diff_pct":  round((price - vwap_val) / vwap_val * 100, 3) if not np.isnan(vwap_val) and vwap_val else None,
        "volume_rate":    round(vr, 2),
        "avg_daily_vol":  avg_daily_vol,
    }


def get_premarket_data(ticker: str) -> dict:
    """盘前 / 盘后数据 + 跳空策略"""
    try:
        pre  = yf.Ticker(ticker).history(period="2d", interval="5m", prepost=True)
        daily = yf.Ticker(ticker).history(period="5d", interval="1d")

        if pre.empty or daily.empty:
            return {"error": "无盘前数据"}

        prev_close = float(daily["Close"].iloc[-2]) if len(daily) >= 2 else None

        pre_bars = []
        for ts, row in pre.iterrows():
            t = ts.tz_convert(ET) if ts.tzinfo else ts
            h, m = t.hour, t.minute
            if (4 <= h < 9) or (h == 9 and m < 30):
                pre_bars.append({
                    "t": t.strftime("%H:%M"), "date": t.strftime("%m/%d"),
                    "c": round(float(row["Close"]), 2), "v": int(row["Volume"])
                })

        pre_price = pre_bars[-1]["c"] if pre_bars else None
        gap_pct   = round((pre_price - prev_close) / prev_close * 100, 2) if (pre_price and prev_close) else None

        if not gap_pct:
            strat, desc = "normal", "正常开盘"
        elif gap_pct > 5:
            strat, desc = "gap_up_fade",      f"高开跳空 +{gap_pct}%，策略：等开盘稳定后反手做空（Gap Fade）"
        elif gap_pct < -5:
            strat, desc = "gap_down_fade",    f"低开跳空 {gap_pct}%，策略：等开盘稳定后抄底做多（Gap Fade）"
        elif gap_pct > 1.5:
            strat, desc = "gap_up_continue",  f"小幅高开 +{gap_pct}%，顺势做多，等 VWAP 确认"
        elif gap_pct < -1.5:
            strat, desc = "gap_down_continue",f"小幅低开 {gap_pct}%，顺势做空，等 VWAP 确认"
        else:
            strat, desc = "normal", f"平开（{gap_pct:+.2f}%），按 VWAP 方向操作"

        return {
            "prev_close": prev_close,
            "pre_price":  pre_price,
            "gap_pct":    gap_pct,
            "pre_bars":   pre_bars[-16:],
            "strategy":   strat,
            "strategy_desc": desc,
        }
    except Exception as e:
        return {"error": str(e)}


# ── 冷静量化模型主入口 ─────────────────────────────────────────

def run_quant_model(ticker: str, portfolio_size: float = 100_000,
                    max_risk_pct: float = 0.01) -> dict:
    """
    五维评分量化模型（满分 10 分）
    ≥8 → 强信号  ≥6 → 普通信号  <6 → 强制 NO_TRADE

    维度：VWAP位置(3) + 成交量(2) + 大盘对齐(2) + 动量(2) + 期权流(1)
    """
    out = {
        "ticker": ticker, "signal": "NO_TRADE", "confidence": 0,
        "score": 0, "max_score": 10,
        "scores": {}, "filters": {},
        "entry": None, "stop": None, "target1": None, "target2": None,
        "position_size": None, "shares": None, "risk_reward": None,
        "atr5m": None, "rules_violated": [], "reasoning": [],
        "timestamp": datetime.now(ET).strftime("%Y-%m-%d %H:%M ET"),
        "cold_rules": COLD_RULES,
    }

    try:
        t      = yf.Ticker(ticker)
        h5     = t.history(period="5d",  interval="5m")
        h1d    = t.history(period="60d", interval="1d")
        spy5   = yf.Ticker("SPY").history(period="2d", interval="5m")

        if h5.empty:
            out["reasoning"].append("5分钟数据不可用")
            return out

        h5 = _add_vwap(h5)
        h5 = _add_volume_rate(h5)

        cur   = h5.iloc[-1]
        price = float(cur["Close"])
        vwap  = float(cur["vwap"])  if not np.isnan(float(cur["vwap"]))     else price
        vr    = float(cur["vol_rate"]) if not np.isnan(float(cur["vol_rate"])) else 1.0

        # ── 过滤器（任一 pass=False → 强制 NO_TRADE）─────────────
        filters = {}
        now = datetime.now(ET)
        hh, mm = now.hour, now.minute
        pre_open  = (hh < 9) or (hh == 9 and mm < 30)
        post_open = hh >= 16
        too_early = hh == 9 and mm < 45
        too_late  = hh == 15 and mm >= 40

        if pre_open or post_open:
            filters["time"] = {"pass": False, "reason": f"非交易时段（{hh:02d}:{mm:02d} ET）"}
        elif too_early:
            filters["time"] = {"pass": False, "reason": f"开盘前 15 分钟（{hh:02d}:{mm:02d} ET），等价格稳定"}
        elif too_late:
            filters["time"] = {"pass": False, "reason": f"收盘前 20 分钟（{hh:02d}:{mm:02d} ET），不开新仓"}
        else:
            filters["time"] = {"pass": True,  "reason": f"交易时段 {hh:02d}:{mm:02d} ET ✅"}

        try:
            vix = float(yf.Ticker("^VIX").history(period="1d")["Close"].iloc[-1])
            if vix > 35:
                filters["vix"] = {"pass": False, "reason": f"VIX={vix:.1f} > 35，极端波动，停止交易"}
            else:
                filters["vix"] = {"pass": True,  "reason": f"VIX={vix:.1f} 正常 ✅"}
        except Exception:
            vix = 20.0
            filters["vix"] = {"pass": "warn", "reason": "VIX 数据暂不可用，建议人工确认后再入场"}

        if vr < 0.4:
            filters["volume"] = {"pass": False, "reason": f"成交量极低（{vr:.0%} 均值），流动性不足"}
        else:
            filters["volume"] = {"pass": True,  "reason": f"成交量 {vr:.1f}x 均值 ✅"}

        try:
            cal = t.calendar
            ne  = None
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                ne = (list(ed)[0] if hasattr(ed,'__iter__') and not isinstance(ed,str) else ed) if ed else None
            elif cal is not None and hasattr(cal,'columns') and "Earnings Date" in cal.columns:
                ne = cal["Earnings Date"].iloc[0]
            if ne:
                days = (pd.Timestamp(ne).tz_localize(None) - pd.Timestamp(now.replace(tzinfo=None))).days
                if 0 <= days <= 3:
                    filters["earnings"] = {"pass": False, "reason": f"财报在 {days} 天后，黑名单期"}
                else:
                    filters["earnings"] = {"pass": True,  "reason": f"距财报 {days} 天 ✅"}
            else:
                filters["earnings"] = {"pass": True, "reason": "无近期财报 ✅"}
        except Exception:
            filters["earnings"] = {"pass": True, "reason": "财报日期未知，默认通过"}

        out["filters"] = filters
        failed = [v["reason"] for v in filters.values() if not v["pass"]]
        if failed:
            out["rules_violated"] = failed
            out["reasoning"].append("过滤器阻断：" + " | ".join(failed))
            return out

        # ── 五维评分 ─────────────────────────────────────────────
        sc = {}

        # 1. VWAP 位置（3分）
        vd = (price - vwap) / vwap * 100
        recent4 = h5.tail(4)
        crossed_up = (len(recent4) >= 4 and
                      float(recent4["Close"].iloc[0])  < float(recent4["vwap"].iloc[0]) and
                      float(recent4["Close"].iloc[-1]) > float(recent4["vwap"].iloc[-1]))
        crossed_dn = (len(recent4) >= 4 and
                      float(recent4["Close"].iloc[0])  > float(recent4["vwap"].iloc[0]) and
                      float(recent4["Close"].iloc[-1]) < float(recent4["vwap"].iloc[-1]))

        if price > vwap:
            pts = 3 if crossed_up else 2
            sc["vwap"] = {"score": pts, "max": 3, "label": "VWAP位置",
                          "detail": f"{'刚突破VWAP🚀' if crossed_up else '位于VWAP上方'} +{vd:.2f}%",
                          "direction": "long"}
        elif abs(vd) < 0.1:
            sc["vwap"] = {"score": 1, "max": 3, "label": "VWAP位置",
                          "detail": f"贴近VWAP（{vd:+.2f}%），方向待定", "direction": "neutral"}
        else:
            pts = 3 if crossed_dn else 2
            sc["vwap"] = {"score": pts, "max": 3, "label": "VWAP位置",
                          "detail": f"{'刚跌破VWAP🔻' if crossed_dn else '位于VWAP下方'} {vd:.2f}%",
                          "direction": "short"}

        vwap_dir = sc["vwap"]["direction"]  # long / short / neutral

        # 2. 成交量确认（2分）
        if vr >= 2.0:
            sc["volume"] = {"score": 2, "max": 2, "label": "成交量", "detail": f"爆量 {vr:.1f}x 🔥"}
        elif vr >= 1.3:
            sc["volume"] = {"score": 1, "max": 2, "label": "成交量", "detail": f"放量 {vr:.1f}x"}
        else:
            sc["volume"] = {"score": 0, "max": 2, "label": "成交量", "detail": f"量能不足 {vr:.1f}x"}

        # 3. 大盘对齐（2分）
        spy_pts, spy_desc = 0, "SPY 数据不可用"
        if not spy5.empty:
            spy5v = _add_vwap(spy5)
            sc_p  = float(spy5v["Close"].iloc[-1])
            sc_vw = float(spy5v["vwap"].iloc[-1])
            sc_mom= float(spy5v["Close"].pct_change(6).iloc[-1] * 100)
            spy_above = sc_p > sc_vw
            if vwap_dir == "long":
                spy_pts  = 2 if (spy_above and sc_mom > 0.1) else (1 if spy_above or sc_mom > 0 else 0)
                spy_desc = f"SPY {'顺势✅' if spy_pts > 0 else '逆势⚠️'} ({sc_mom:+.2f}%)"
            elif vwap_dir == "short":
                spy_pts  = 2 if (not spy_above and sc_mom < -0.1) else (1 if not spy_above or sc_mom < 0 else 0)
                spy_desc = f"SPY {'配合做空✅' if spy_pts > 0 else '逆势⚠️'} ({sc_mom:+.2f}%)"
            else:
                spy_pts = 1; spy_desc = f"SPY 中性 ({sc_mom:+.2f}%)"
        sc["market"] = {"score": spy_pts, "max": 2, "label": "大盘对齐", "detail": spy_desc}

        # 4. 技术动量 RSI+MACD+Supertrend（2分）
        from src.technical import rsi as calc_rsi, macd as calc_macd, get_advanced_indicators
        cl = h5["Close"].tail(60)
        rsi_v = float(calc_rsi(cl).iloc[-1])
        _, _, mhist = calc_macd(cl)
        macd_bull = float(mhist.iloc[-1]) > 0

        adv    = get_advanced_indicators(h5.tail(80))
        st_dir = adv.get("supertrend_dir")
        sqz    = adv.get("squeeze_dir")

        if vwap_dir == "long":
            confirm = sum([rsi_v > 50, macd_bull, st_dir == 1, sqz == "up"])
            if confirm >= 3:
                mom_pts, mom_desc = 2, f"RSI={rsi_v:.0f} + MACD多 + Supertrend多 强确认 ✅"
            elif confirm >= 2:
                mom_pts, mom_desc = 1, f"RSI={rsi_v:.0f}，{confirm}/4项指标确认多"
            else:
                mom_pts, mom_desc = 0, f"RSI={rsi_v:.0f}，动量确认不足（{confirm}/4）"
        elif vwap_dir == "short":
            confirm = sum([rsi_v < 50, not macd_bull, st_dir == -1, sqz == "down"])
            if confirm >= 3:
                mom_pts, mom_desc = 2, f"RSI={rsi_v:.0f} + MACD空 + Supertrend空 强确认 ✅"
            elif confirm >= 2:
                mom_pts, mom_desc = 1, f"RSI={rsi_v:.0f}，{confirm}/4项指标确认空"
            else:
                mom_pts, mom_desc = 0, f"RSI={rsi_v:.0f}，空头确认不足（{confirm}/4）"

        else:
            mom_pts = 1; mom_desc = f"RSI={rsi_v:.0f}，Supertrend={'多' if st_dir==1 else '空' if st_dir==-1 else '中性'}"
        sc["momentum"] = {"score": mom_pts, "max": 2, "label": "技术动量", "detail": mom_desc,
                          "rsi": round(rsi_v, 1), "supertrend_dir": st_dir, "squeeze_dir": sqz}

        # 5. 期权流（1分，可为-1惩罚）
        opt_adj, opt_desc = 0, "无期权数据"
        try:
            exps = t.options
            if exps:
                chain = t.option_chain(exps[0])
                calls_v = int(chain.calls["volume"].fillna(0).sum())
                puts_v  = int(chain.puts["volume"].fillna(0).sum())
                from src.options_sources import detect_unusual_activity
                ua = detect_unusual_activity(chain.calls, chain.puts, price)
                if ua["alert_level"] in ("high","medium"):
                    if vwap_dir == "long" and calls_v > puts_v:
                        opt_adj  = 1;  opt_desc = f"异常 Call 买入 🔥（C/P={calls_v//max(puts_v,1):.1f}x）"
                    elif vwap_dir == "short" and puts_v > calls_v:
                        opt_adj  = 1;  opt_desc = f"异常 Put 买入 🔻（P/C={puts_v//max(calls_v,1):.1f}x）"
                    else:
                        opt_adj  = -1; opt_desc = f"期权流与方向相反 ⚠️（C={calls_v:,} P={puts_v:,}）"
                else:
                    opt_desc = f"期权成交正常（C={calls_v:,} P={puts_v:,}）"
        except Exception:
            pass
        sc["options_flow"] = {"score": max(0, opt_adj), "max": 1,
                              "label": "期权订单流", "detail": opt_desc,
                              "penalty": opt_adj < 0}

        # 汇总
        base  = sum(s["score"] for s in sc.values())
        penalty = sum(1 for s in sc.values() if s.get("penalty"))
        total = max(0, base - penalty)

        out["score"]  = total
        out["scores"] = sc

        # ATR（5分钟级别，14期）
        hi  = h5["High"].tail(20)
        lo  = h5["Low"].tail(20)
        cl2 = h5["Close"].tail(20)
        tr  = pd.concat([(hi-lo), (hi-cl2.shift()).abs(), (lo-cl2.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])
        out["atr5m"] = round(atr, 3)

        # 信号判断
        if total >= 8 and vwap_dir != "neutral":
            sig = "STRONG_LONG" if vwap_dir == "long" else "STRONG_SHORT"
            conf = min(95, 65 + total * 3)
        elif total >= 6 and vwap_dir != "neutral":
            sig = "LONG" if vwap_dir == "long" else "SHORT"
            conf = 50 + total * 4
        else:
            sig  = "NO_TRADE"
            conf = total * 7

        out["signal"]     = sig
        out["confidence"] = int(conf)

        # 交易参数（仅当有信号）
        if sig != "NO_TRADE":
            d       = 1 if "LONG" in sig else -1
            stop_d  = atr * 1.0
            entry   = price
            stop    = round(entry - d * stop_d,       2)
            tgt1    = round(entry + d * stop_d * 1.5, 2)
            tgt2    = round(entry + d * stop_d * 2.5, 2)
            rr1     = round(abs(tgt1-entry) / max(abs(entry-stop), 0.01), 2)
            risk_amt = portfolio_size * max_risk_pct
            shares   = max(1, int(risk_amt / max(abs(entry-stop), 0.01)))
            pos_size = round(shares * entry, 2)

            from src.short_term import LEVERAGE_MAP
            lev = LEVERAGE_MAP.get(ticker, {})
            lev_ticker = lev.get("long" if d > 0 else "short", "")

            out.update({
                "entry": entry, "stop": stop,
                "target1": tgt1, "target2": tgt2,
                "stop_dist": round(stop_d, 3),
                "risk_reward": rr1,
                "shares": shares,
                "position_size": pos_size,
                "risk_per_trade": round(risk_amt, 2),
                "leverage_etf": lev_ticker,
            })

            direction_label = "做多" if d > 0 else "做空"
            out["reasoning"].append(
                f"{direction_label}信号（{total}/10，置信度{conf}%）| "
                f"VWAP {vd:+.2f}% | 量能 {vr:.1f}x | RSI {rsi_v:.0f}"
            )
        else:
            out["reasoning"].append(f"评分 {total}/10，低于入场门槛（6分），当前策略：空仓等待")

        return out

    except Exception as e:
        import traceback
        out["reasoning"].append(f"模型异常: {e}")
        return out


# ── 内部工具函数 ──────────────────────────────────────────────

def _add_vwap(hist: pd.DataFrame) -> pd.DataFrame:
    hist = hist.copy()
    hist["tp"]     = (hist["High"] + hist["Low"] + hist["Close"]) / 3
    hist["tp_vol"] = hist["tp"] * hist["Volume"]
    vwap_vals, cum_tv, cum_v, prev_date = [], 0.0, 0.0, None

    for ts, row in hist.iterrows():
        try:
            d = ts.tz_convert(ET).date() if ts.tzinfo else ts.date()
        except Exception:
            d = str(ts)[:10]
        if d != prev_date:
            cum_tv = cum_v = 0.0
            prev_date = d
        v = float(row["Volume"])
        if v > 0:
            cum_tv += float(row["tp_vol"])
            cum_v  += v
        vwap_vals.append(cum_tv / cum_v if cum_v > 0 else np.nan)

    hist["vwap"] = vwap_vals
    return hist


def _add_volume_rate(hist: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    hist = hist.copy()
    avg  = hist["Volume"].rolling(window, min_periods=5).mean()
    hist["vol_rate"] = hist["Volume"] / avg.replace(0, np.nan)
    return hist
