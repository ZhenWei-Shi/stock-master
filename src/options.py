"""
期权分析引擎 v2

新增：
  calculate_dealer_delta(calls, puts, price) — 做市商净 Delta 暴露
  Dealer Net Delta > 0 → 做市商净做多 → 价格上涨时做市商卖出（压制）
  Dealer Net Delta < 0 → 做市商净做空 → 价格上涨时做市商买入（加速）
"""
import pandas as pd
import numpy as np
import math


def calculate_max_pain(calls: pd.DataFrame, puts: pd.DataFrame, current_price: float) -> dict:
    """
    Max Pain = 让期权买家亏损最多（做市商获利最多）的价格点。
    遍历所有行权价，计算该价格下所有实值期权的总价值，最小值即Max Pain。
    """
    try:
        all_strikes = sorted(set(list(calls["strike"]) + list(puts["strike"])))
        # 只看当前价格 50%~200% 范围内的行权价
        strikes = [s for s in all_strikes if current_price * 0.5 <= s <= current_price * 2.0] or all_strikes

        pain_map = {}
        for p in strikes:
            call_pain = sum(
                max(0, p - row.strike) * row.openInterest * 100
                for row in calls.itertuples()
            )
            put_pain = sum(
                max(0, row.strike - p) * row.openInterest * 100
                for row in puts.itertuples()
            )
            pain_map[p] = call_pain + put_pain

        max_pain_price = min(pain_map, key=pain_map.get)
        distance_pct = (max_pain_price - current_price) / current_price * 100

        return {
            "max_pain": max_pain_price,
            "distance_pct": round(distance_pct, 2),
            "pain_map": {str(k): v for k, v in pain_map.items()},
            "current_price": current_price,
        }
    except Exception as e:
        return {"max_pain": None, "error": str(e)}


def analyze_gamma_squeeze(calls: pd.DataFrame, puts: pd.DataFrame, current_price: float) -> dict:
    """
    检测 Gamma Squeeze 前兆。
    评分满分10分，≥6分触发信号。
    """
    try:
        signals = []
        score = 0

        total_call_oi = int(calls["openInterest"].sum())
        total_put_oi = int(puts["openInterest"].sum())
        pc_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

        # 信号1：Put/Call OI比值
        if pc_ratio < 0.5:
            score += 3
            signals.append({"ok": True, "text": f"Call OI主导（P/C比={pc_ratio:.2f}，远低于1）"})
        elif pc_ratio < 0.8:
            score += 1
            signals.append({"ok": None, "text": f"Call偏多（P/C比={pc_ratio:.2f}）"})
        else:
            signals.append({"ok": False, "text": f"P/C比={pc_ratio:.2f}，Call优势不明显"})

        # 信号2：价格是否接近高OI Call行权价（上方5%内）
        nearby = calls[(calls["strike"] >= current_price) & (calls["strike"] <= current_price * 1.1)]
        if not nearby.empty:
            top_strike = float(nearby.nlargest(1, "openInterest")["strike"].iloc[0])
            gap_pct = (top_strike - current_price) / current_price * 100
            if gap_pct < 3:
                score += 3
                signals.append({"ok": True, "text": f"价格距最高OI Call仅{gap_pct:.1f}%（${top_strike}），即将触发Gamma"})
            elif gap_pct < 7:
                score += 1
                signals.append({"ok": None, "text": f"接近高OI Call ${top_strike}（距{gap_pct:.1f}%），需突破"})
            else:
                signals.append({"ok": False, "text": f"高OI Call ${top_strike} 距当前价{gap_pct:.1f}%，偏远"})
        else:
            signals.append({"ok": False, "text": "当前价格上方无高OI Call"})

        # 信号3：IV水平
        if "impliedVolatility" in calls.columns:
            atm = calls[abs(calls["strike"] - current_price) / current_price < 0.05]
            if not atm.empty:
                avg_iv = float(atm["impliedVolatility"].mean())
                if avg_iv < 0.4:
                    score += 2
                    signals.append({"ok": True, "text": f"IV较低（{avg_iv:.0%}），期权便宜，上行空间大"})
                elif avg_iv < 0.7:
                    score += 1
                    signals.append({"ok": None, "text": f"IV中等（{avg_iv:.0%}）"})
                else:
                    signals.append({"ok": False, "text": f"IV偏高（{avg_iv:.0%}），期权偏贵"})

        # 信号4：近价格±20%范围内，前6大OI是否以Call为主
        lo, hi = current_price * 0.8, current_price * 1.3
        nearby_calls = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)][["strike", "openInterest"]].assign(type="call")
        nearby_puts = puts[(puts["strike"] >= lo) & (puts["strike"] <= hi)][["strike", "openInterest"]].assign(type="put")
        combined = pd.concat([nearby_calls, nearby_puts]).nlargest(6, "openInterest")
        call_count = int((combined["type"] == "call").sum())
        if call_count >= 5:
            score += 2
            signals.append({"ok": True, "text": f"近价格范围前6大OI中{call_count}档是Call，Gamma结构极度看多"})
        elif call_count >= 4:
            score += 1
            signals.append({"ok": None, "text": f"前6大OI中{call_count}档是Call，偏多但不强"})
        else:
            signals.append({"ok": False, "text": f"前6大OI中仅{call_count}档是Call，Put压制明显"})

        # OI分布图数据
        oi_chart = []
        all_strikes = sorted(set(list(calls["strike"]) + list(puts["strike"])))
        for s in all_strikes:
            if current_price * 0.7 <= s <= current_price * 1.5:
                c_oi = int(calls[calls["strike"] == s]["openInterest"].sum())
                p_oi = int(puts[puts["strike"] == s]["openInterest"].sum())
                oi_chart.append({"strike": s, "call_oi": c_oi, "put_oi": p_oi})

        return {
            "signal": score >= 6,
            "score": score,
            "max_score": 10,
            "signals": signals,
            "pc_ratio": round(pc_ratio, 3),
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "oi_chart": oi_chart,
        }
    except Exception as e:
        return {"signal": False, "score": 0, "max_score": 10,
                "signals": [{"ok": False, "text": f"分析出错: {e}"}], "oi_chart": []}


def calculate_dealer_delta(calls: pd.DataFrame, puts: pd.DataFrame,
                           current_price: float) -> dict:
    """
    做市商净 Delta 暴露（Dealer Net Delta Exposure）

    原理：
      做市商（Market Maker）卖出期权给散户，本身持有反向头寸，
      必须通过买卖标的来对冲 delta，形成机械性市场力量。

      Call OI：散户买 Call → 做市商卖 Call → 做市商 delta < 0（做空风险）
               → 做市商需要买入股票对冲 → 净做多标的
      Put OI ：散户买 Put  → 做市商卖 Put  → 做市商 delta > 0（做多风险）
               → 做市商需要卖出股票对冲 → 净做空标的

    结论：
      dealer_net_delta > 0 → 做市商持有多头标的 → 价格上涨时做市商减仓 → 压制上涨
      dealer_net_delta < 0 → 做市商持有空头标的 → 价格上涨时做市商买入 → 加速上涨（Gamma Squeeze）

    Delta 近似（BSM ATM approx）：
      ATM Call delta ≈ 0.5，OTM Call 越远 delta 越小
      使用简化线性近似：delta = 0.5 × exp(-|log(S/K)| / 0.3)
    """
    try:
        S = current_price

        def approx_delta_call(strike):
            moneyness = math.log(S / strike) if strike > 0 else 0
            return 0.5 * math.exp(moneyness / 0.3) if moneyness < 0 else min(0.99, 0.5 * math.exp(moneyness / 0.2))

        def approx_delta_put(strike):
            return approx_delta_call(strike) - 1.0

        # 只看当前价格 ±30% 范围（流动性聚集区）
        lo, hi = S * 0.70, S * 1.30

        call_delta_exp = 0.0
        for _, row in calls.iterrows():
            k   = float(row.get("strike", 0))
            oi  = float(row.get("openInterest", 0) or 0)
            if lo <= k <= hi and oi > 0:
                d = approx_delta_call(k)
                call_delta_exp += d * oi * 100   # 每份合约 100 股

        put_delta_exp = 0.0
        for _, row in puts.iterrows():
            k   = float(row.get("strike", 0))
            oi  = float(row.get("openInterest", 0) or 0)
            if lo <= k <= hi and oi > 0:
                d = approx_delta_put(k)
                put_delta_exp += d * oi * 100    # put delta 是负数

        # 做市商角度：持有与散户相反的仓位
        # 散户买 call → 做市商空 call → 做市商需要买股对冲 → dealer long
        dealer_net_delta = call_delta_exp + put_delta_exp
        # call_delta > 0，put_delta < 0，dealer = call - |put|

        dollar_exposure = dealer_net_delta * S

        if dealer_net_delta < -500_000:
            regime = "NEGATIVE_GAMMA"
            desc   = f"做市商净空仓{abs(dealer_net_delta/1e6):.1f}M delta，价格上涨时被迫买入，形成正反馈（Gamma Squeeze 燃料）"
            bias   = "BULLISH_AMPLIFIER"
        elif dealer_net_delta < 0:
            regime = "SLIGHT_NEGATIVE"
            desc   = f"做市商轻微净空仓，上涨有一定助力但不强"
            bias   = "MILD_BULLISH"
        elif dealer_net_delta < 500_000:
            regime = "SLIGHT_POSITIVE"
            desc   = f"做市商轻微净多仓，上涨会有轻微压制"
            bias   = "MILD_BEARISH"
        else:
            regime = "POSITIVE_GAMMA"
            desc   = f"做市商净多仓{dealer_net_delta/1e6:.1f}M delta，价格上涨时做市商卖出对冲，压制涨幅（Pin Risk 区间）"
            bias   = "BEARISH_SUPPRESSOR"

        # 找 gamma flip 价格（dealer delta 从负变正的价格点）
        gamma_flip = _find_gamma_flip(calls, puts, S)

        return {
            "dealer_net_delta":     round(dealer_net_delta),
            "dollar_exposure":      round(dollar_exposure),
            "call_delta_exposure":  round(call_delta_exp),
            "put_delta_exposure":   round(put_delta_exp),
            "regime":               regime,
            "bias":                 bias,
            "description":          desc,
            "gamma_flip_price":     gamma_flip,
            "interpretation": (
                f"Gamma Flip 价格：${gamma_flip}。"
                f"价格 {'高于' if S > (gamma_flip or S) else '低于'} Gamma Flip → "
                f"{'正Gamma区（做市商压制波动）' if S > (gamma_flip or S) else '负Gamma区（做市商放大波动）'}"
            ) if gamma_flip else "无法计算 Gamma Flip 价格",
        }
    except Exception as e:
        return {"error": str(e), "dealer_net_delta": None}


def _find_gamma_flip(calls: pd.DataFrame, puts: pd.DataFrame,
                     current_price: float) -> float | None:
    """
    扫描价格区间，找使 dealer_net_delta = 0 的价格（Gamma Flip Point）
    """
    try:
        lo = current_price * 0.80
        hi = current_price * 1.20
        prices = [lo + (hi - lo) * i / 40 for i in range(41)]

        def net_delta_at(S):
            total = 0.0
            for _, row in calls.iterrows():
                k  = float(row.get("strike", 0))
                oi = float(row.get("openInterest", 0) or 0)
                if 0 < k and oi > 0:
                    mon = math.log(S / k)
                    d   = 0.5 * math.exp(mon / 0.3) if mon < 0 else min(0.99, 0.5 * math.exp(mon / 0.2))
                    total += d * oi * 100
            for _, row in puts.iterrows():
                k  = float(row.get("strike", 0))
                oi = float(row.get("openInterest", 0) or 0)
                if 0 < k and oi > 0:
                    mon = math.log(S / k)
                    d_c = 0.5 * math.exp(mon / 0.3) if mon < 0 else min(0.99, 0.5 * math.exp(mon / 0.2))
                    total += (d_c - 1.0) * oi * 100
            return total

        # 找符号变化点
        prev_sign = None
        for p in prices:
            nd = net_delta_at(p)
            sign = 1 if nd >= 0 else -1
            if prev_sign is not None and sign != prev_sign:
                return round(p, 2)
            prev_sign = sign
        return None
    except Exception:
        return None
