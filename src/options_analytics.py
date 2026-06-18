"""
期权深度分析：
  - 预期移动幅度 (Expected Move) = ATM straddle 价格
  - IV Skew = OTM Put IV vs OTM Call IV
  - IV vs HV 比较（期权贵不贵）
  - VIX 期限结构（Contango / Backwardation）
"""
import yfinance as yf
import numpy as np
from datetime import datetime


def get_iv_analytics(calls, puts, current_price: float,
                     expiry_str: str = None, price_hist=None) -> dict:
    return {
        "expected_move": _expected_move(calls, puts, current_price, expiry_str),
        "skew":          _iv_skew(calls, puts, current_price),
        "iv_hv":         _iv_vs_hv(calls, current_price, price_hist),
    }


def _expected_move(calls, puts, current_price, expiry_str):
    if calls is None or puts is None or calls.empty or puts.empty:
        return None
    try:
        calls = calls.copy(); calls["_d"] = (calls["strike"] - current_price).abs()
        puts  = puts.copy();  puts["_d"]  = (puts["strike"]  - current_price).abs()
        ac = calls.nsmallest(1, "_d").iloc[0]
        ap = puts.nsmallest(1,  "_d").iloc[0]

        def mid(row):
            b, a = float(row.get("bid") or 0), float(row.get("ask") or 0)
            return (b + a) / 2 if b and a else float(row.get("lastPrice") or 0)

        c_mid, p_mid = mid(ac), mid(ap)
        em      = c_mid + p_mid
        em_pct  = em / current_price * 100

        days = None
        if expiry_str:
            try:
                days = (datetime.strptime(expiry_str, "%Y-%m-%d") - datetime.now()).days
            except Exception:
                pass

        return {
            "expected_move":     round(em, 2),
            "expected_move_pct": round(em_pct, 2),
            "upside":   round(current_price + em, 2),
            "downside": round(current_price - em, 2),
            "atm_strike":    float(ac["strike"]),
            "atm_call_mid":  round(c_mid, 2),
            "atm_put_mid":   round(p_mid, 2),
            "days_to_expiry": days,
        }
    except Exception as e:
        return {"error": str(e)}


def _iv_skew(calls, puts, current_price):
    if calls is None or puts is None or calls.empty or puts.empty:
        return None
    try:
        # 10% OTM put vs 10% OTM call
        pc = puts.copy();  pc["_d"] = (pc["strike"] - current_price * 0.90).abs()
        cc = calls.copy(); cc["_d"] = (cc["strike"] - current_price * 1.10).abs()
        atm = calls.copy(); atm["_d"] = (atm["strike"] - current_price).abs()

        put_iv  = float(pc.nsmallest(1, "_d").iloc[0].get("impliedVolatility") or 0) * 100
        call_iv = float(cc.nsmallest(1, "_d").iloc[0].get("impliedVolatility") or 0) * 100
        atm_iv  = float(atm.nsmallest(1, "_d").iloc[0].get("impliedVolatility") or 0) * 100
        skew    = put_iv - call_iv

        return {
            "put_iv":  round(put_iv, 1),
            "call_iv": round(call_iv, 1),
            "atm_iv":  round(atm_iv, 1),
            "skew":    round(skew, 1),
            "skew_type": "put_skew" if skew > 3 else "call_skew" if skew < -3 else "flat",
            "interpretation": (
                "Put 溢价高，市场在买保险（对冲 / 看跌情绪）" if skew > 5 else
                "轻微 Put 溢价，市场正常对冲" if skew > 2 else
                "Call 溢价高，市场期待上涨（乐观情绪）" if skew < -3 else
                "斜率平坦，市场方向中性"
            ),
        }
    except Exception as e:
        return {"error": str(e)}


def _iv_vs_hv(calls, current_price, price_hist):
    result = {}
    try:
        if calls is not None and not calls.empty:
            c = calls.copy(); c["_d"] = (c["strike"] - current_price).abs()
            result["atm_iv"] = round(
                float(c.nsmallest(1, "_d").iloc[0].get("impliedVolatility") or 0) * 100, 1)
    except Exception:
        pass

    if price_hist is not None and not price_hist.empty:
        try:
            lr = np.log(price_hist["Close"] / price_hist["Close"].shift(1)).dropna()
            result["hv20"] = round(float(lr.tail(20).std() * np.sqrt(252) * 100), 1)
            result["hv60"] = round(float(lr.tail(60).std() * np.sqrt(252) * 100), 1)
            if result.get("atm_iv") and result.get("hv20"):
                ratio = result["atm_iv"] / result["hv20"]
                result["iv_hv_ratio"] = round(ratio, 2)
                result["assessment"] = (
                    "期权价格偏贵（IV >> HV），卖方策略占优（备兑买权 / 卖 Put）" if ratio > 1.3 else
                    "期权价格偏便宜（IV << HV），买方策略占优（买 Call / Put）" if ratio < 0.8 else
                    "期权定价合理（IV ≈ HV）"
                )
        except Exception:
            pass

    return result


def get_vix_term_structure() -> dict:
    """VIX9D / VIX / VIX3M 期限结构"""
    syms = {"vix9d": "^VIX9D", "vix": "^VIX", "vix3m": "^VIX3M"}
    vals = {}
    for k, s in syms.items():
        try:
            h = yf.Ticker(s).history(period="5d")
            if not h.empty:
                vals[k] = round(float(h["Close"].iloc[-1]), 2)
        except Exception:
            vals[k] = None

    v9, v30, v93 = vals.get("vix9d"), vals.get("vix"), vals.get("vix3m")
    if v9 and v30 and v93:
        if v9 < v30 < v93:
            struct, interp = "contango",         "正常升水（近低远高），市场平静，波动率卖方占优"
        elif v9 > v30 > v93:
            struct, interp = "backwardation",    "⚠️ 期限结构倒挂（近高远低），市场恐慌！注意风险"
        elif v9 > v30:
            struct, interp = "partial_inversion","短端倒挂，近期波动率风险上升"
        else:
            struct, interp = "flat",             "期限结构平坦，方向不明"
    else:
        struct, interp = "unknown", "数据不足"

    return {
        **vals,
        "structure":      struct,
        "interpretation": interp,
        "term_chart": [
            {"label": "VIX9D",  "value": v9},
            {"label": "VIX",    "value": v30},
            {"label": "VIX3M",  "value": v93},
        ],
    }
