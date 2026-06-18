"""
技术指标计算引擎

引用项目：
  pandas-ta ⭐5,299 (twopirllc/pandas-ta) — Supertrend, Squeeze, Stochastic,
                                             Ichimoku, Williams%R, OBV, CCI
"""
import pandas as pd
import numpy as np

try:
    import ta as _ta
    TA_OK = True
except ImportError:
    TA_OK = False


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist


def bollinger(series: pd.Series, period=20, std_dev=2):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    pct_b = (series - lower) / (upper - lower)  # 0=下轨, 0.5=中轨, 1=上轨
    return upper, mid, lower, pct_b


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def get_indicators(hist: pd.DataFrame) -> dict:
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    volume = hist["Volume"]

    ma5  = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    rsi_s = rsi(close)
    macd_line, macd_sig, macd_hist = macd(close)
    bb_up, bb_mid, bb_lo, pct_b = bollinger(close)
    atr_s = atr(high, low, close)

    vol_ma20  = volume.rolling(20).mean()
    vol_ratio = volume / vol_ma20

    def v(s):
        x = s.iloc[-1]
        return None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)

    curr = {
        "price":         v(close),
        "ma5":           v(ma5),
        "ma10":          v(ma10),
        "ma20":          v(ma20),
        "ma50":          v(ma50),
        "rsi":           v(rsi_s),
        "macd_line":     v(macd_line),
        "macd_signal":   v(macd_sig),
        "macd_hist":     v(macd_hist),
        "macd_hist_prev": float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else 0,
        "bb_upper":      v(bb_up),
        "bb_middle":     v(bb_mid),
        "bb_lower":      v(bb_lo),
        "pct_b":         v(pct_b),
        "atr":           v(atr_s),
        "atr_pct":       round(float(atr_s.iloc[-1] / close.iloc[-1] * 100), 2) if v(atr_s) else None,
        "vol_ratio":     v(vol_ratio),
    }

    # 图表数据（最近60天，带指标叠加）
    n = min(60, len(hist))
    chart = []
    for i in range(n):
        idx = -(n - i)
        def cv(s):
            x = s.iloc[idx]
            return None if (x is None or (isinstance(x, float) and np.isnan(x))) else round(float(x), 3)
        chart.append({
            "d":      hist.index[idx].strftime("%m/%d"),
            "c":      cv(close),
            "v":      int(volume.iloc[idx]),
            "ma20":   cv(ma20),
            "ma50":   cv(ma50),
            "rsi":    cv(rsi_s),
            "macd_h": cv(macd_hist),
            "bb_u":   cv(bb_up),
            "bb_l":   cv(bb_lo),
        })

    return {"current": curr, "chart": chart}


def get_extended_indicators(hist: pd.DataFrame) -> dict:
    """
    扩展指标组 — pandas-ta ⭐5,299
    新增：Stochastic, OBV, Williams%R, Ichimoku, CCI
    综合输出 net_bias 方向建议
    """
    if not TA_OK:
        return {"error": "pandas-ta 未安装，运行 pip install pandas-ta"}
    try:
        df    = hist.copy()
        close = df["Close"]
        price = float(close.iloc[-1])

        # Stochastic %K/%D
        stoch  = ta.stoch(df["High"], df["Low"], close)
        stk    = round(float(stoch.iloc[-1, 0]), 1) if stoch is not None and not stoch.empty else None
        std    = round(float(stoch.iloc[-1, 1]), 1) if stoch is not None and not stoch.empty else None

        # OBV（量价方向）
        obv      = ta.obv(close, df["Volume"])
        obv_val  = float(obv.iloc[-1]) if obv is not None else None
        obv_trend = None
        if obv is not None and len(obv) >= 5:
            obv_trend = "rising" if float(obv.iloc[-1]) > float(obv.iloc[-5]) else "falling"

        # Williams %R
        willr = ta.willr(df["High"], df["Low"], close)
        wr    = round(float(willr.iloc[-1]), 1) if willr is not None else None

        # CCI
        cci     = ta.cci(df["High"], df["Low"], close)
        cci_val = round(float(cci.iloc[-1]), 1) if cci is not None else None

        # Ichimoku（仅 Tenkan/Kijun，不传 lookahead 避免未来数据）
        tenkan, kijun = None, None
        try:
            ichi = ta.ichimoku(df["High"], df["Low"], close, lookahead=False)
            if ichi is not None:
                ich_df = ichi[0] if isinstance(ichi, tuple) else ichi
                if not ich_df.empty and ich_df.shape[1] >= 2:
                    tenkan = round(float(ich_df.iloc[-1, 0]), 2)
                    kijun  = round(float(ich_df.iloc[-1, 1]), 2)
        except Exception:
            pass

        # 综合偏向评分
        signals = []
        if stk is not None:
            if stk < 20: signals.append({"d": "LONG",  "src": f"Stoch%K={stk}超卖", "w": 2})
            elif stk > 80: signals.append({"d": "SHORT", "src": f"Stoch%K={stk}超买", "w": 2})
        if wr is not None:
            if wr < -80: signals.append({"d": "LONG",  "src": f"W%R={wr}极端超卖", "w": 2})
            elif wr > -20: signals.append({"d": "SHORT", "src": f"W%R={wr}极端超买", "w": 2})
        if obv_trend:
            signals.append({"d": "LONG" if obv_trend == "rising" else "SHORT",
                            "src": f"OBV{obv_trend}", "w": 1})
        if cci_val is not None:
            if cci_val < -100: signals.append({"d": "LONG",  "src": f"CCI={cci_val}超卖", "w": 1})
            elif cci_val > 100: signals.append({"d": "SHORT", "src": f"CCI={cci_val}超买", "w": 1})
        if tenkan and kijun:
            if price > tenkan > kijun:
                signals.append({"d": "LONG",  "src": "价格在云图上方，多头结构", "w": 1})
            elif price < tenkan < kijun:
                signals.append({"d": "SHORT", "src": "价格在云图下方，空头结构", "w": 1})

        longs  = sum(s["w"] for s in signals if s["d"] == "LONG")
        shorts = sum(s["w"] for s in signals if s["d"] == "SHORT")

        return {
            "stochastic": {"k": stk, "d": std,
                           "signal": ("超卖" if stk and stk < 20 else
                                      "超买" if stk and stk > 80 else "中性")},
            "obv":         {"trend": obv_trend},
            "williams_r":  {"value": wr,
                            "signal": ("极端超卖" if wr and wr < -80 else
                                       "极端超买" if wr and wr > -20 else "中性")},
            "ichimoku":    {"tenkan": tenkan, "kijun": kijun,
                            "above_cloud": price > tenkan if tenkan else None},
            "cci":         {"value": cci_val,
                            "signal": ("超卖" if cci_val and cci_val < -100 else
                                       "超买" if cci_val and cci_val > 100 else "中性")},
            "signals":     signals,
            "net_bias":    ("LONG" if longs > shorts else
                            "SHORT" if shorts > longs else "NEUTRAL"),
            "library":     "pandas-ta ⭐5,299",
        }
    except Exception as e:
        return {"error": str(e)}


def get_advanced_indicators(hist: pd.DataFrame) -> dict:
    """pandas-ta 高级指标：Supertrend + Squeeze Momentum"""
    try:
        df = hist.copy()

        # Supertrend (7, 3.0) — 日内追踪止损方向基准
        st = ta.supertrend(df["High"], df["Low"], df["Close"], length=7, multiplier=3.0)
        supertrend_val, supertrend_dir = None, None
        if st is not None:
            val_col = [c for c in st.columns if c.startswith("SUPERT_7") and "d" not in c and "s" not in c]
            dir_col = [c for c in st.columns if "SUPERTd" in c]
            supertrend_val = round(float(st[val_col[0]].iloc[-1]), 3) if val_col else None
            supertrend_dir = int(st[dir_col[0]].iloc[-1]) if dir_col else None  # 1=多头, -1=空头

        # Squeeze Momentum (LazyBear)
        sqz = ta.squeeze(df["High"], df["Low"], df["Close"], df["Volume"])
        sqz_val = None
        if sqz is not None:
            h_col = [c for c in sqz.columns if "SQZ_" in c
                     and "ON" not in c and "OFF" not in c and "NO" not in c]
            sqz_val = round(float(sqz[h_col[0]].iloc[-1]), 4) if h_col else None

        return {
            "supertrend":       supertrend_val,
            "supertrend_dir":   supertrend_dir,   # 1=看多, -1=看空
            "squeeze_momentum": sqz_val,
            "squeeze_dir":      ("up" if sqz_val and sqz_val > 0 else
                                 "down" if sqz_val else None),
        }
    except Exception:
        return {"supertrend": None, "supertrend_dir": None,
                "squeeze_momentum": None, "squeeze_dir": None}
