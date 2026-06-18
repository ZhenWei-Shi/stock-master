"""
机构资金追踪引擎（Smart Money Tracker）

个人投资者能做到的极限：

1. 异常期权活动（UOA）——机构在期权市场下注，散户可以跟踪
   原理：大机构买入大量 OTM call/put 时，期权量 >> 未平仓量，几天后往往有大动作

2. 做市商 Gamma 敞口（GEX）——理解做市商的对冲逻辑
   正 GEX：做市商多 Gamma → 逢跌买入、逢涨卖出 → 价格被钉住（Pinning）
   负 GEX：做市商空 Gamma → 逢跌卖出、逢涨买入 → 价格趋势加速（Explosive Move）
   关键价位：Gamma 墙（Pin 风险）、Gamma 悬崖（爆发风险）

3. 空头挤压探测器（Short Squeeze Radar）
   高空仓 + 价格上涨 + 放量 = 逼空（散户可跟进做多）

4. 智能资金流向（Smart Money Flow）
   开盘30分钟 = 散户情绪化操作（噪音）
   收盘60分钟 = 机构主力建仓/减仓（信号）
   差值 = 机构意图

5. 机构持仓动量（13F 趋势分析）
   季度增持 + 新进机构数量 = 聪明钱集中度提升

用法：
  from src.smart_money import full_smart_money_scan
  result = full_smart_money_scan("NVDA")
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz

ET = pytz.timezone("America/New_York")


# ─────────────────────────────────────────────────────────────
# 1. 异常期权活动（Unusual Options Activity）
# ─────────────────────────────────────────────────────────────

def detect_unusual_options(ticker: str) -> dict:
    """
    扫描期权链，识别机构方向性押注。

    判定标准（参考 Unusual Whales / Market Chameleon）：
      强烈信号：volume > 3× open_interest AND volume > 500 AND OTM
      普通信号：volume > 2× open_interest AND volume > 200

    返回：
      uoa_calls   — 看涨异常（机构做多信号）
      uoa_puts    — 看跌异常（机构对冲或做空信号）
      bias        — 综合偏向（bullish / bearish / neutral）
      call_put_ratio — 当日期权成交量比值
      key_strikes — 最活跃行权价（机构关注的价位）
    """
    try:
        tk    = yf.Ticker(ticker)
        price = float(tk.history(period="1d")["Close"].iloc[-1])
        exps  = tk.options  # 所有到期日

        if not exps:
            return {"ok": False, "reason": "无期权数据"}

        # 只看最近2个到期日（机构近期押注）
        check_exps = exps[:min(3, len(exps))]

        uoa_calls, uoa_puts = [], []
        total_call_vol = total_put_vol = 0

        for exp in check_exps:
            try:
                chain = tk.option_chain(exp)
                calls = chain.calls.copy()
                puts  = chain.puts.copy()

                for df, side in [(calls, "call"), (puts, "put")]:
                    if df.empty:
                        continue

                    if side == "call":
                        total_call_vol += int(df["volume"].fillna(0).sum())
                        # OTM calls = strike > price（看涨押注）
                        otm = df[df["strike"] > price * 1.01]
                    else:
                        total_put_vol += int(df["volume"].fillna(0).sum())
                        # OTM puts = strike < price（看跌押注）
                        otm = df[df["strike"] < price * 0.99]

                    for _, row in otm.iterrows():
                        vol = int(row.get("volume") or 0)
                        oi  = int(row.get("openInterest") or 1)
                        if oi < 1:
                            oi = 1
                        ratio = vol / oi

                        if vol < 100:
                            continue

                        strength = (
                            "🔥 极强" if (ratio >= 3 and vol >= 500) else
                            "强"    if (ratio >= 2 and vol >= 200) else
                            "普通"
                        )

                        if ratio < 1.5:
                            continue  # 过滤正常波动

                        premium = float(row.get("lastPrice") or 0) * vol * 100
                        entry = {
                            "expiry":    exp,
                            "strike":    float(row["strike"]),
                            "type":      side,
                            "volume":    vol,
                            "open_interest": oi,
                            "vol_oi_ratio":  round(ratio, 1),
                            "last_price":    round(float(row.get("lastPrice") or 0), 2),
                            "premium_total": round(premium, 0),
                            "iv":        round(float(row.get("impliedVolatility") or 0) * 100, 1),
                            "strike_dist_pct": round((float(row["strike"]) - price) / price * 100, 1),
                            "strength":  strength,
                        }
                        if side == "call":
                            uoa_calls.append(entry)
                        else:
                            uoa_puts.append(entry)
            except Exception:
                continue

        # 按 premium 排序（最大的赌注排前面）
        uoa_calls.sort(key=lambda x: x["premium_total"], reverse=True)
        uoa_puts.sort(key=lambda x: x["premium_total"], reverse=True)

        call_put_ratio = round(total_call_vol / max(total_put_vol, 1), 2)

        # 综合偏向判断
        call_premium = sum(x["premium_total"] for x in uoa_calls)
        put_premium  = sum(x["premium_total"] for x in uoa_puts)
        total_prem   = call_premium + put_premium

        if total_prem == 0:
            bias = "neutral"
        elif call_premium / total_prem > 0.65:
            bias = "bullish"
        elif put_premium / total_prem > 0.65:
            bias = "bearish"
        else:
            bias = "neutral"

        # 最活跃行权价（机构关注的支撑/阻力位）
        all_rows = uoa_calls + uoa_puts
        key_strikes = sorted(
            set(x["strike"] for x in all_rows if x["vol_oi_ratio"] >= 2),
            key=lambda s: abs(s - price)
        )[:5]

        signal_strength = "无异常"
        if uoa_calls or uoa_puts:
            max_ratio = max((x["vol_oi_ratio"] for x in all_rows), default=0)
            if max_ratio >= 3:
                signal_strength = "极强机构信号"
            elif max_ratio >= 2:
                signal_strength = "明确机构方向"
            else:
                signal_strength = "轻微异常"

        return {
            "ok":             True,
            "ticker":         ticker,
            "price":          round(price, 2),
            "uoa_calls":      uoa_calls[:5],
            "uoa_puts":       uoa_puts[:5],
            "bias":           bias,
            "bias_label": {
                "bullish": "看涨（机构押注上涨）",
                "bearish": "看跌（机构对冲或押注下跌）",
                "neutral": "中性（无明确方向）",
            }.get(bias, "未知"),
            "call_put_ratio": call_put_ratio,
            "call_premium_usd": round(call_premium, 0),
            "put_premium_usd":  round(put_premium, 0),
            "key_strikes":    key_strikes,
            "signal_strength": signal_strength,
            "note": (
                "📌 机构期权活动是最直接的跟庄信号之一，"
                "但注意：大量 put 也可能是机构对冲现货多头，而非做空。"
            ),
        }

    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────
# 2. 做市商 Gamma 敞口（GEX）
# ─────────────────────────────────────────────────────────────

def calculate_gex(ticker: str) -> dict:
    """
    计算做市商净 Gamma 敞口。

    公式（SpotGamma 方法）：
      Call GEX = call_OI × gamma × 100 × spot²  （做市商空 call → 需买股对冲）
      Put  GEX = put_OI  × gamma × 100 × spot²  （做市商多 put → 需卖股对冲）
      Net  GEX = Call_GEX - Put_GEX

    正 GEX → 做市商净多 Gamma → 价格稳定（做市商反向对冲）
    负 GEX → 做市商净空 Gamma → 价格加速（做市商顺向对冲，放大波动）
    GEX = 0 的行权价 = 翻转点（Flip Point），跌破则进入负 Gamma 区域

    注意：yfinance 不提供 Greeks，用 Black-Scholes 简化估算。
    """
    try:
        import math

        tk    = yf.Ticker(ticker)
        hist  = tk.history(period="5d", interval="1d")
        price = float(hist["Close"].iloc[-1])
        exps  = tk.options

        if not exps:
            return {"ok": False, "reason": "无期权数据"}

        # 年化波动率（30日历史波动率）
        hist_1mo = tk.history(period="1mo", interval="1d")
        if len(hist_1mo) >= 10:
            returns = hist_1mo["Close"].pct_change().dropna()
            hv30    = float(returns.std() * np.sqrt(252))
        else:
            hv30 = 0.30

        def bs_gamma(S, K, T, sigma, r=0.05):
            """Black-Scholes Gamma（在 S 时刻，行权价 K 的 Gamma 值）"""
            if T <= 0 or sigma <= 0:
                return 0.0
            try:
                d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
                return math.exp(-0.5 * d1**2) / (S * sigma * math.sqrt(T) * math.sqrt(2 * math.pi))
            except Exception:
                return 0.0

        gex_by_strike = {}
        call_gex_total = put_gex_total = 0.0

        for exp in exps[:3]:  # 只看最近3个到期日
            try:
                # 最小0.01年（≈3.6天），防止 T→0 时 Gamma 极值爆炸
                T = max((datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days / 365, 0.01)
                chain = tk.option_chain(exp)

                for row in chain.calls.itertuples():
                    oi = int(getattr(row, "openInterest", 0) or 0)
                    if oi < 10:
                        continue
                    g = bs_gamma(price, row.strike, T, hv30)
                    gex = g * oi * 100 * price ** 2  # 美元 Gamma
                    call_gex_total += gex
                    strike = float(row.strike)
                    gex_by_strike[strike] = gex_by_strike.get(strike, 0) + gex

                for row in chain.puts.itertuples():
                    oi = int(getattr(row, "openInterest", 0) or 0)
                    if oi < 10:
                        continue
                    g = bs_gamma(price, row.strike, T, hv30)
                    gex = g * oi * 100 * price ** 2
                    put_gex_total += gex
                    strike = float(row.strike)
                    gex_by_strike[strike] = gex_by_strike.get(strike, 0) - gex  # put 方向相反

            except Exception:
                continue

        net_gex = call_gex_total - put_gex_total

        # Gamma 墙（最大正 GEX 的行权价 = 最强 Pin 风险）
        gamma_wall = max(gex_by_strike, key=gex_by_strike.get) if gex_by_strike else price
        # 翻转点（GEX 从正变负的最近行权价）
        sorted_strikes = sorted(gex_by_strike.keys())
        flip_point = None
        for i in range(len(sorted_strikes) - 1):
            if gex_by_strike[sorted_strikes[i]] > 0 and gex_by_strike[sorted_strikes[i+1]] < 0:
                flip_point = sorted_strikes[i+1]
                break

        # 重要 GEX 水平（做市商关注的价位）
        top_levels = sorted(gex_by_strike.items(), key=lambda x: abs(x[1]), reverse=True)[:6]

        regime = "稳定区（正Gamma）" if net_gex > 0 else "波动区（负Gamma）"
        action = (
            "做市商会逢跌买入、逢涨卖出，价格趋于稳定 → 等待突破后跟进" if net_gex > 0 else
            "做市商顺势对冲，价格趋势会加速 → 突破后动量更强，止损要宽"
        )

        return {
            "ok":              True,
            "ticker":          ticker,
            "price":           round(price, 2),
            "net_gex":         round(net_gex, 0),
            "net_gex_bn":      round(net_gex / 1e9, 3),
            "call_gex":        round(call_gex_total, 0),
            "put_gex":         round(put_gex_total, 0),
            "gamma_regime":    regime,
            "action_note":     action,
            "gamma_wall":      round(gamma_wall, 2),
            "flip_point":      round(flip_point, 2) if flip_point else None,
            "key_levels":      [
                {"strike": round(s, 2), "gex": round(g, 0),
                 "type": "阻力（Pin）" if g > 0 else "支撑→爆发位"}
                for s, g in top_levels
            ],
            "hv30_used":       round(hv30 * 100, 1),
            "note": (
                "⚠️ GEX 数据用 B-S 模型估算，方向可信，绝对数值仅供参考。"
                "专业版使用 SpotGamma / Unusual Whales（付费）数据。"
            ),
        }

    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────
# 3. 空头挤压探测器（Short Squeeze Radar）
# ─────────────────────────────────────────────────────────────

def detect_short_squeeze(ticker: str) -> dict:
    """
    空头挤压潜力评分（0-100）

    判定因素：
      - 空仓比例（Short Float %）：越高越危险，一旦股价上涨，空头被迫买入
      - 到期天数（Days to Cover）：多少天成交量才能覆盖全部空仓
      - 价格动量：空仓高但股价还在涨 = 逼空正在进行
      - 量能：成交量 > 3× 日均 = 多头攻击信号

    参考：GameStop 事件（GME）2021年1月：空仓130%+，3天内+2400%
    """
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info
        hist = tk.history(period="1mo", interval="1d")

        if hist.empty:
            return {"ok": False, "reason": "无历史数据"}

        price     = float(hist["Close"].iloc[-1])
        short_pct = float(info.get("shortPercentOfFloat") or 0)
        dtc       = float(info.get("shortRatio") or 0)      # Days to Cover
        shares_short = int(info.get("sharesShort") or 0)
        float_shares = int(info.get("floatShares") or 1)

        # 近10日价格动量
        ret_10d = (hist["Close"].iloc[-1] - hist["Close"].iloc[-10]) / hist["Close"].iloc[-10] if len(hist) >= 10 else 0
        # 量能
        vol_avg = float(hist["Volume"].rolling(20).mean().iloc[-1])
        vol_now = float(hist["Volume"].iloc[-1])
        vol_ratio = vol_now / max(vol_avg, 1)

        # ── 评分系统 ──────────────────────────────────────────
        score = 0
        reasons = []

        if short_pct >= 0.30:
            score += 30
            reasons.append(f"空仓极高：{short_pct*100:.1f}%（>30%，GME级别逼空风险）")
        elif short_pct >= 0.20:
            score += 20
            reasons.append(f"空仓高：{short_pct*100:.1f}%（>20%，挤压潜力大）")
        elif short_pct >= 0.10:
            score += 10
            reasons.append(f"空仓中等：{short_pct*100:.1f}%（10-20%）")
        else:
            reasons.append(f"空仓偏低：{short_pct*100:.1f}%（<10%，挤压潜力小）")

        if dtc >= 5:
            score += 20
            reasons.append(f"覆盖天数{dtc:.1f}天（>5天，空头平仓压力大）")
        elif dtc >= 3:
            score += 10
            reasons.append(f"覆盖天数{dtc:.1f}天（中等压力）")

        if ret_10d > 0.10:
            score += 25
            reasons.append(f"近10日涨{ret_10d*100:.1f}%，空头正在被挤压")
        elif ret_10d > 0.05:
            score += 15
            reasons.append(f"近10日涨{ret_10d*100:.1f}%，逼空信号开始")
        elif ret_10d < -0.05:
            score -= 10
            reasons.append(f"近10日跌{abs(ret_10d)*100:.1f}%，空头暂时占优")

        if vol_ratio > 3:
            score += 20
            reasons.append(f"成交量{vol_ratio:.1f}×均量（多头攻击信号）")
        elif vol_ratio > 2:
            score += 10
            reasons.append(f"成交量{vol_ratio:.1f}×均量（量能放大）")

        score = max(0, min(100, score))

        squeeze_level = (
            "🚀 极高（逼空行情可能进行中）" if score >= 70 else
            "🟡 中等（值得关注）"           if score >= 40 else
            "🟢 较低（暂无挤压信号）"
        )

        action = (
            "考虑跟多：空头被迫平仓会持续推高股价，但注意动量消退信号（量能萎缩）"
            if score >= 70 else
            "观察：如果价格继续上涨 + 量能持续，可能发展为逼空行情"
            if score >= 40 else
            "空仓比例不高，暂无逼空机会"
        )

        return {
            "ok":             True,
            "ticker":         ticker,
            "price":          round(price, 2),
            "short_float_pct": round(short_pct * 100, 1),
            "days_to_cover":  round(dtc, 1),
            "shares_short":   shares_short,
            "squeeze_score":  score,
            "squeeze_level":  squeeze_level,
            "price_momentum_10d": round(ret_10d * 100, 1),
            "volume_ratio":   round(vol_ratio, 2),
            "reasons":        reasons,
            "action":         action,
            "risk_note": (
                "⚠️ 高空仓也可能是正确的（基本面差的股票应该被做空）。"
                "逼空只是短期动量，不代表基本面改善。"
            ),
        }

    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────
# 4. 智能资金流向（Smart Money Flow Index）
# ─────────────────────────────────────────────────────────────

def smart_money_flow(ticker: str) -> dict:
    """
    区分机构"智能资金"和散户"情绪化噪音"。

    理论依据：
      - 开盘30分钟：散户情绪化操作（恐慌买/卖，追涨杀跌）
      - 收盘60分钟：机构资金建仓/减仓（这才是真实方向）

    Smart Money Flow Index = 收盘段加权价格变化 / 开盘段加权价格变化

    SMF > 1 且为正 = 机构在暗中买入（即使盘面看起来很弱）
    SMF 强 且开盘弱 = 经典机构吸筹（散户恐慌卖，机构接盘）
    """
    try:
        tk = yf.Ticker(ticker)

        # 获取最近1天的5分钟数据
        hist_5m = tk.history(period="1d", interval="5m")

        if hist_5m.empty or len(hist_5m) < 15:
            return {"ok": False, "reason": "盘中数据不足（需盘中或当日）"}

        close_all = hist_5m["Close"]
        vol_all   = hist_5m["Volume"]

        # 开盘段：前6根（30分钟）
        open_bars   = hist_5m.iloc[:6]
        # 收盘段：后12根（60分钟）
        close_bars  = hist_5m.iloc[-12:]

        def _vwap_segment(bars):
            """成交量加权平均价"""
            tv = (bars["Close"] * bars["Volume"]).sum()
            v  = bars["Volume"].sum()
            return tv / max(v, 1)

        open_vwap  = _vwap_segment(open_bars)
        close_vwap = _vwap_segment(close_bars)

        prev_close_hist = tk.history(period="2d", interval="1d")
        prev_close = float(prev_close_hist["Close"].iloc[-2]) if len(prev_close_hist) >= 2 else float(close_all.iloc[0])
        today_close = float(close_all.iloc[-1])

        # 开盘段变化（情绪化噪音）
        open_move  = (open_vwap - prev_close) / prev_close * 100
        # 收盘段变化（机构动作）
        close_move = (close_vwap - open_vwap) / open_vwap * 100
        # 全天净变化
        day_return = (today_close - prev_close) / prev_close * 100

        # SMF 信号判断
        if close_move > 0.3 and open_move < 0:
            smf_signal = "🔵 强力吸筹：散户开盘恐慌卖，机构收盘积极买"
            smf_bias   = "bullish"
        elif close_move > 0.3 and open_move > 0:
            smf_signal = "🟢 持续买入：开盘和收盘均有资金流入"
            smf_bias   = "bullish"
        elif close_move < -0.3 and open_move > 0:
            smf_signal = "🔴 机构悄然出货：开盘散户追涨，收盘机构卖出"
            smf_bias   = "bearish"
        elif close_move < -0.3 and open_move < 0:
            smf_signal = "⚫ 持续卖出：全天资金净流出"
            smf_bias   = "bearish"
        else:
            smf_signal = "⚪ 中性：无明显机构方向"
            smf_bias   = "neutral"

        # 量能分布（收盘量 vs 开盘量）
        close_vol_pct = close_bars["Volume"].sum() / max(vol_all.sum(), 1) * 100
        open_vol_pct  = open_bars["Volume"].sum()  / max(vol_all.sum(), 1) * 100

        return {
            "ok":           True,
            "ticker":       ticker,
            "current_price": round(today_close, 2),
            "day_return_pct": round(day_return, 2),
            "open_move_pct":  round(open_move, 2),
            "close_move_pct": round(close_move, 2),
            "close_vol_share_pct": round(close_vol_pct, 1),
            "open_vol_share_pct":  round(open_vol_pct, 1),
            "smf_signal":   smf_signal,
            "smf_bias":     smf_bias,
            "note": (
                "收盘量占全天成交量的比例越高，机构意图越明确。"
                "最后30分钟的价格方向是最可信的机构信号。"
            ),
        }

    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────
# 5. 机构持仓动量（13F 趋势）
# ─────────────────────────────────────────────────────────────

def institutional_momentum(ticker: str) -> dict:
    """
    分析机构13F持仓的季度趋势。

    逻辑：
      - 3+ 大机构同时增持 + 股价创新高 = 最强的跟庄信号
      - 机构净减持 + 股价还在高位 = 危险（聪明钱在撤退）

    数据来源：yfinance institutional_holders（延迟45天，但方向正确）
    """
    try:
        tk    = yf.Ticker(ticker)
        info  = tk.info
        price = float(tk.history(period="1d")["Close"].iloc[-1])

        inst_holders = tk.institutional_holders
        major_holders= tk.major_holders

        if inst_holders is None or inst_holders.empty:
            return {"ok": False, "reason": "无机构持仓数据"}

        # 机构持仓比例
        inst_pct  = float(info.get("heldPercentInstitutions", 0)) * 100
        insider_pct = float(info.get("heldPercentInsiders", 0)) * 100

        # 分析最大机构持仓变化
        top_holders = inst_holders.head(10).copy()
        holders_info = []
        for _, row in top_holders.iterrows():
            holders_info.append({
                "holder":    str(row.get("Holder", "")),
                "shares":    int(row.get("Shares", 0)),
                "date":      str(row.get("Date Reported", "")),
                "pct_held":  round(float(row.get("% Out", 0)) * 100, 2),
            })

        # 13F 延迟分析：如果机构最新持仓日期是上季度，股价相对该日期的涨幅
        if holders_info:
            try:
                report_dates = [h["date"] for h in holders_info if h["date"] and h["date"] != "nan"]
                latest_13f   = max(report_dates) if report_dates else None

                since_13f_return = 0.0
                if latest_13f and latest_13f != "nan":
                    hist_long = tk.history(start=latest_13f[:10], interval="1d")
                    if len(hist_long) > 1:
                        since_13f_return = (price / float(hist_long["Close"].iloc[0]) - 1) * 100
            except Exception:
                latest_13f = None
                since_13f_return = 0.0
        else:
            latest_13f = None
            since_13f_return = 0.0

        # 综合评估
        signal = "neutral"
        analysis = []

        if inst_pct > 70:
            analysis.append(f"机构持仓占{inst_pct:.1f}%（高机构化，波动受机构主导）")
        elif inst_pct > 50:
            analysis.append(f"机构持仓占{inst_pct:.1f}%（中等机构持仓）")
        else:
            analysis.append(f"机构持仓仅{inst_pct:.1f}%（散户为主，波动大）")

        if since_13f_return > 15:
            signal = "bullish"
            analysis.append(f"13F披露以来股价涨{since_13f_return:.1f}%：机构建仓后价格验证✅")
        elif since_13f_return > 0:
            signal = "bullish"
            analysis.append(f"13F披露以来涨{since_13f_return:.1f}%：机构建仓后温和上涨")
        elif since_13f_return < -10:
            signal = "bearish"
            analysis.append(f"13F披露以来跌{abs(since_13f_return):.1f}%⚠️：机构建仓后股价下跌，注意")

        if insider_pct > 5:
            analysis.append(f"内部人持仓{insider_pct:.1f}%（管理层有较大利益绑定）")

        return {
            "ok":              True,
            "ticker":          ticker,
            "price":           round(price, 2),
            "inst_held_pct":   round(inst_pct, 1),
            "insider_held_pct": round(insider_pct, 1),
            "latest_13f_date": latest_13f,
            "since_13f_return_pct": round(since_13f_return, 1),
            "top_holders":     holders_info[:5],
            "signal":          signal,
            "analysis":        analysis,
            "note": (
                "13F数据延迟45天，反映上季度末持仓。"
                "机构建仓后股价上涨 = 机构判断正确，可以跟随。"
                "但机构也会错，不能盲从。"
            ),
        }

    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────
# 6. 综合机构追踪报告（主入口）
# ─────────────────────────────────────────────────────────────

def full_smart_money_scan(ticker: str) -> dict:
    """
    一键生成完整机构资金追踪报告。

    综合评分说明：
      80-100 = 机构信号极强，与技术信号共振则果断跟进
      60-79  = 有机构参与，注意方向确认
      40-59  = 机构信号模糊，以技术信号为主
      0-39   = 无明显机构参与，谨慎
    """
    uoa  = detect_unusual_options(ticker)
    gex  = calculate_gex(ticker)
    sqz  = detect_short_squeeze(ticker)
    smf  = smart_money_flow(ticker)
    inst = institutional_momentum(ticker)

    # 综合评分（满分100）
    score = 0
    signals = []

    # UOA 分
    if uoa.get("ok"):
        if uoa["bias"] == "bullish":
            uoa_score = 25 if "极强" in uoa.get("signal_strength", "") else 15
            score += uoa_score
            signals.append(f"期权异常：看涨押注（+{uoa_score}）")
        elif uoa["bias"] == "bearish":
            score -= 10
            signals.append("期权异常：看跌押注（-10，谨慎做多）")

    # GEX 分（负 Gamma 时动量更强）
    if gex.get("ok"):
        if gex["net_gex"] < 0:
            score += 10
            signals.append("负Gamma区：价格趋势会加速（+10）")
        else:
            score += 5
            signals.append("正Gamma区：价格趋于稳定（+5）")

    # 空头挤压分
    if sqz.get("ok"):
        sq_score = min(25, int(sqz["squeeze_score"] * 0.25))
        score += sq_score
        if sqz["squeeze_score"] >= 50:
            signals.append(f"空头挤压：潜力{sqz['squeeze_score']}分（+{sq_score}）")

    # SMF 分
    if smf.get("ok"):
        if smf["smf_bias"] == "bullish":
            score += 20
            signals.append("智能资金：收盘段净流入（+20）")
        elif smf["smf_bias"] == "bearish":
            score -= 15
            signals.append("智能资金：收盘段净流出（-15，机构出货）")

    # 机构持仓分
    if inst.get("ok"):
        if inst["signal"] == "bullish":
            score += 20
            signals.append(f"机构持仓：13F后涨{inst['since_13f_return_pct']}%（+20）")
        elif inst["signal"] == "bearish":
            score -= 10
            signals.append("机构持仓：13F后下跌（-10）")

    score = max(0, min(100, score))

    verdict = (
        "🔥 极强机构共振，果断跟进（结合技术信号）" if score >= 80 else
        "✅ 机构参与明确，方向优先"              if score >= 60 else
        "🟡 机构信号模糊，以技术为主"             if score >= 40 else
        "⚪ 无明显机构参与，纯技术操作即可"
    )

    return {
        "ok":               True,
        "ticker":           ticker,
        "smart_money_score": score,
        "verdict":          verdict,
        "signals":          signals,
        "details": {
            "unusual_options":    uoa,
            "gamma_exposure":     gex,
            "short_squeeze":      sqz,
            "smart_money_flow":   smf,
            "institutional_13f":  inst,
        },
        "strategy_note": (
            "个人投资者的优势：灵活快速。机构资金体量大，建仓需要数周甚至数月，"
            "你只需要在他们建仓早期发现信号并跟进，在目标达到前退出。"
            "不要和机构对着干——顺势喝汤，不抢肉。"
        ),
    }


# ─────────────────────────────────────────────────────────────
# 快速格式化输出（Telegram 友好）
# ─────────────────────────────────────────────────────────────

def format_smart_money_telegram(result: dict) -> str:
    if not result.get("ok"):
        return f"❌ 扫描失败：{result.get('reason')}"

    t  = result["ticker"]
    sc = result["smart_money_score"]
    v  = result["verdict"]
    lines = [f"🔍 <b>{t} 机构追踪报告</b>", f"综合评分：{sc}/100", v, ""]

    sigs = result.get("signals", [])
    if sigs:
        lines.append("📊 <b>信号详情</b>")
        for s in sigs:
            lines.append(f"  · {s}")
        lines.append("")

    d = result.get("details", {})

    # GEX
    gex = d.get("gamma_exposure", {})
    if gex.get("ok"):
        lines.append(f"🎯 <b>做市商Gamma</b>：{gex['gamma_regime']}")
        lines.append(f"  Gamma墙：${gex['gamma_wall']}（价格Pin位）")
        if gex.get("flip_point"):
            lines.append(f"  翻转点：${gex['flip_point']}（跌破进入爆发区）")

    # UOA
    uoa = d.get("unusual_options", {})
    if uoa.get("ok") and uoa.get("bias") != "neutral":
        lines.append(f"\n📈 <b>期权异常</b>：{uoa['bias_label']}")
        lines.append(f"  C/P比：{uoa['call_put_ratio']}  信号强度：{uoa['signal_strength']}")

    # 空头挤压
    sqz = d.get("short_squeeze", {})
    if sqz.get("ok") and sqz.get("squeeze_score", 0) >= 40:
        lines.append(f"\n🐻 <b>空头挤压</b>：{sqz['squeeze_level']}")
        lines.append(f"  空仓：{sqz['short_float_pct']}%  覆盖天数：{sqz['days_to_cover']}天")

    # SMF
    smf = d.get("smart_money_flow", {})
    if smf.get("ok"):
        lines.append(f"\n💰 <b>智能资金流向</b>")
        lines.append(f"  {smf['smf_signal']}")
        lines.append(f"  开盘段：{smf['open_move_pct']:+.2f}%  收盘段：{smf['close_move_pct']:+.2f}%")

    return "\n".join(lines)
