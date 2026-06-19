"""
多源期权数据聚合器

免费来源（无需Key）：
  - yfinance          （已有，15分钟延迟）
  - Nasdaq官方API     （免费JSON，15分钟延迟）
  - CBOE P/C比率      （市场整体情绪）
  - 异常期权活动检测  （成交量/OI异常 = 机构布局信号）

可选配置（需API Key）：
  - Tradier           （免费账户即可，实时数据）
  - Alpaca            （免费账户，实时数据）
"""
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

_NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nasdaq.com",
}

# ─────────────────────────────────────────────────────────
# Nasdaq 官方 API（免费，无需Key）
# ─────────────────────────────────────────────────────────

def get_nasdaq_options(ticker: str, expiry: str = None) -> dict:
    """
    Nasdaq 官方期权链 API
    返回 calls/puts DataFrame 格式（与 yfinance 保持一致）
    """
    try:
        url = f"https://api.nasdaq.com/api/quote/{ticker.upper()}/option-chain"
        params = {
            "assetclass": "stocks",
            "limit": 500,
            "type": "",
            "money": "all",
            "expirymovement": "all",
            "expiration": expiry or "",
        }
        r = requests.get(url, headers=_NASDAQ_HEADERS, params=params, timeout=10)
        if r.status_code != 200:
            return {"error": f"Nasdaq API HTTP {r.status_code}", "calls": None, "puts": None}

        data = r.json().get("data", {})
        rows = data.get("optionChainList", {}).get("rows", [])
        expirations = data.get("expiryList", [])

        calls_list, puts_list = [], []
        for row in rows:
            strike_raw = row.get("drillDownURL", "").split("=")[-1]
            try:
                strike = float(strike_raw)
            except Exception:
                strike = None

            def parse_side(side_data, side):
                if not side_data:
                    return None
                try:
                    return {
                        "strike": strike,
                        "lastPrice": _f(side_data.get("lastPrice")),
                        "bid": _f(side_data.get("bid")),
                        "ask": _f(side_data.get("ask")),
                        "openInterest": _i(side_data.get("openInterest")),
                        "volume": _i(side_data.get("volume")),
                        "impliedVolatility": _f(side_data.get("impliedVolatility")),
                        "inTheMoney": side_data.get("inTheMoney", False),
                    }
                except Exception:
                    return None

            c = parse_side(row.get("call"), "call")
            p = parse_side(row.get("put"), "put")
            if c:
                calls_list.append(c)
            if p:
                puts_list.append(p)

        calls_df = pd.DataFrame(calls_list) if calls_list else None
        puts_df  = pd.DataFrame(puts_list)  if puts_list  else None

        return {
            "source": "Nasdaq",
            "expiry": expiry,
            "expirations": expirations[:12],
            "calls": calls_df,
            "puts":  puts_df,
            "row_count": len(rows),
        }
    except Exception as e:
        return {"error": str(e), "calls": None, "puts": None}


# ─────────────────────────────────────────────────────────
# CBOE Put/Call 比率（市场整体情绪，免费）
# ─────────────────────────────────────────────────────────

def get_cboe_pcr() -> dict:
    """
    CBOE 每日 Put/Call 比率数据
    Total PCR > 1.0 = 市场恐慌 / PCR < 0.7 = 市场贪婪
    """
    try:
        # CBOE 公开数据文件
        url = "https://www.cboe.com/data/multipage-options-chart/chart-data/options-chart-data.json"
        r = requests.get(url, timeout=8,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            data = r.json()
            # 尝试解析
            pcr_total = data.get("Total P/C Ratio") or data.get("TOTAL") or None
            return {"source": "CBOE", "pcr_total": pcr_total, "raw": list(data.keys())[:5]}

        # 备用：从 CBOE 统计页面抓取
        url2 = "https://www.cboe.com/us/options/market_statistics/daily/"
        r2 = requests.get(url2, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        # 简单文本搜索
        text = r2.text
        pcr = None
        if "Put/Call" in text or "put/call" in text:
            import re
            matches = re.findall(r'(\d+\.\d+)', text[text.find("Put"):text.find("Put")+200])
            if matches:
                pcr = float(matches[0])

        return {
            "source": "CBOE",
            "pcr_total": pcr,
            "note": "市场整体 PCR：>1.0 恐慌 / <0.7 贪婪",
        }
    except Exception as e:
        return {"source": "CBOE", "error": str(e), "pcr_total": None}


# ─────────────────────────────────────────────────────────
# Tradier API（免费sandbox账户 / 付费实盘账户）
# ─────────────────────────────────────────────────────────

def get_tradier_options(ticker: str, expiry: str = None) -> dict:
    """
    Tradier API 期权链
    需在 config.py 或环境变量设置 TRADIER_TOKEN
    Sandbox（纸交易）免费，实盘需付费账户
    """
    token = os.environ.get("TRADIER_TOKEN", "")
    if not token:
        return {"source": "Tradier", "error": "未配置 TRADIER_TOKEN 环境变量",
                "setup": "在 tradier.com 注册免费账户后，设置环境变量 TRADIER_TOKEN=your_token"}

    base = "https://sandbox.tradier.com/v1"   # sandbox
    # base = "https://api.tradier.com/v1"     # 实盘（取消注释）
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    try:
        # 先获取到期日列表
        r = requests.get(f"{base}/markets/options/expirations",
                         params={"symbol": ticker, "includeAllRoots": "true"},
                         headers=headers, timeout=10)
        if r.status_code != 200:
            return {"source": "Tradier", "error": f"HTTP {r.status_code}"}

        exps = r.json().get("expirations", {}).get("date", [])
        target = expiry if (expiry and expiry in exps) else (exps[0] if exps else None)
        if not target:
            return {"source": "Tradier", "error": "无可用到期日"}

        # 获取期权链
        r2 = requests.get(f"{base}/markets/options/chains",
                          params={"symbol": ticker, "expiration": target, "greeks": "true"},
                          headers=headers, timeout=10)
        options = r2.json().get("options", {}).get("option", [])

        calls = [o for o in options if o.get("option_type") == "call"]
        puts  = [o for o in options if o.get("option_type") == "put"]

        def to_df(lst):
            if not lst:
                return None
            rows = []
            for o in lst:
                rows.append({
                    "strike": o.get("strike"),
                    "lastPrice": o.get("last"),
                    "bid": o.get("bid"),
                    "ask": o.get("ask"),
                    "openInterest": o.get("open_interest"),
                    "volume": o.get("volume"),
                    "impliedVolatility": o.get("greeks", {}).get("smv_vol"),
                    "delta": o.get("greeks", {}).get("delta"),
                    "gamma": o.get("greeks", {}).get("gamma"),
                    "theta": o.get("greeks", {}).get("theta"),
                    "vega":  o.get("greeks", {}).get("vega"),
                })
            return pd.DataFrame(rows)

        return {
            "source": "Tradier",
            "expiry": target,
            "expirations": exps[:12],
            "calls": to_df(calls),
            "puts":  to_df(puts),
            "has_greeks": True,
        }
    except Exception as e:
        return {"source": "Tradier", "error": str(e)}


# ─────────────────────────────────────────────────────────
# Alpaca API（免费账户，实时数据）
# ─────────────────────────────────────────────────────────

def get_alpaca_options(ticker: str, expiry: str = None) -> dict:
    """
    Alpaca Markets API 期权数据
    需配置 ALPACA_KEY 和 ALPACA_SECRET 环境变量
    免费账户可获取延迟数据，订阅后实时
    """
    key    = os.environ.get("ALPACA_KEY", "")
    secret = os.environ.get("ALPACA_SECRET", "")
    if not key or not secret:
        return {"source": "Alpaca",
                "error": "未配置 ALPACA_KEY / ALPACA_SECRET",
                "setup": "在 alpaca.markets 注册免费账户，获取 API Key"}
    try:
        headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        }
        params = {"feed": "indicative", "limit": 1000}
        if expiry:
            params["expiration_date"] = expiry

        r = requests.get(
            f"https://data.alpaca.markets/v1beta1/options/snapshots/{ticker.upper()}",
            headers=headers, params=params, timeout=10
        )
        if r.status_code != 200:
            return {"source": "Alpaca", "error": f"HTTP {r.status_code}: {r.text[:100]}"}

        data = r.json()
        snapshots = data.get("snapshots", {})
        calls_list, puts_list = [], []
        for contract_sym, snap in snapshots.items():
            details = snap.get("latestQuote") or {}
            greeks  = snap.get("greeks") or {}
            # 解析合约符号获取strike和类型
            # 格式: NVDA250620C00130000
            try:
                typ = "call" if "C" in contract_sym[-15:] else "put"
                strike_str = contract_sym[-8:]
                strike = int(strike_str) / 1000
            except Exception:
                continue

            row = {
                "strike": strike,
                "bid": details.get("bp"),
                "ask": details.get("ap"),
                "openInterest": snap.get("openInterest") or 0,  # dailyBar.o是开盘价非OI
                "volume": snap.get("dailyBar", {}).get("v"),
                "impliedVolatility": snap.get("impliedVolatility"),
                "delta": greeks.get("delta"),
                "gamma": greeks.get("gamma"),
                "theta": greeks.get("theta"),
            }
            if typ == "call":
                calls_list.append(row)
            else:
                puts_list.append(row)

        return {
            "source": "Alpaca",
            "calls": pd.DataFrame(calls_list) if calls_list else None,
            "puts":  pd.DataFrame(puts_list)  if puts_list  else None,
            "has_greeks": True,
        }
    except Exception as e:
        return {"source": "Alpaca", "error": str(e)}


# ─────────────────────────────────────────────────────────
# 异常期权活动检测（Smart Money 信号）
# ─────────────────────────────────────────────────────────

def detect_unusual_activity(calls: pd.DataFrame, puts: pd.DataFrame,
                             current_price: float) -> dict:
    """
    检测异常期权活动：
    成交量 >> OI → 今日大量新开仓 → 机构押注信号
    Put大单 + 远OTM → 对冲/做空布局
    Call大单 + 近ATM + 短期 → 方向性押注做多
    """
    if calls is None or puts is None:
        return {"signals": [], "alert_level": "none"}

    signals = []
    alert_level = "none"

    def scan(df, side):
        if df is None or df.empty:
            return
        for _, row in df.iterrows():
            try:
                vol = int(row.get("volume") or 0)
                oi  = int(row.get("openInterest") or 1)
                strike = float(row.get("strike") or 0)
                iv = float(row.get("impliedVolatility") or 0)
                if vol == 0 or oi == 0:
                    continue

                vol_oi_ratio = vol / oi
                moneyness = (strike - current_price) / current_price * 100
                is_atm = abs(moneyness) < 5

                # 大单检测：成交量超过 OI 的 3 倍 AND 成交量 > 1000
                if vol_oi_ratio > 3 and vol > 1000:
                    signal_type = "UNUSUAL_CALL" if side == "call" else "UNUSUAL_PUT"
                    direction = "做多押注" if side == "call" else ("保护性对冲" if moneyness < -10 else "做空押注")
                    signals.append({
                        "type": signal_type,
                        "strike": strike,
                        "volume": vol,
                        "oi": oi,
                        "ratio": round(vol_oi_ratio, 1),
                        "moneyness_pct": round(moneyness, 1),
                        "iv": round(iv * 100, 1) if iv else None,
                        "direction": direction,
                        "desc": f"${strike} {side.upper()} 成交量{vol:,}为OI({oi:,})的{vol_oi_ratio:.1f}倍 — 疑似机构{direction}",
                    })

                # 超大单（成交量 > 5000）
                if vol > 5000 and is_atm:
                    signals.append({
                        "type": "MASSIVE_FLOW",
                        "strike": strike,
                        "volume": vol,
                        "oi": oi,
                        "direction": "ATM大资金押注",
                        "desc": f"⚡ ${strike} {side.upper()} 单日成交{vol:,}，ATM超大资金定向押注",
                    })
            except Exception:
                continue

    scan(calls, "call")
    scan(puts,  "put")

    # 去重并排序
    seen = set()
    unique = []
    for s in sorted(signals, key=lambda x: x.get("volume", 0), reverse=True):
        k = (s["strike"], s["type"])
        if k not in seen:
            seen.add(k)
            unique.append(s)

    unique = unique[:8]

    if any(s["type"] == "MASSIVE_FLOW" for s in unique):
        alert_level = "high"
    elif len(unique) >= 3:
        alert_level = "medium"
    elif unique:
        alert_level = "low"

    return {
        "signals": unique,
        "alert_level": alert_level,
        "total_unusual": len(unique),
    }


# ─────────────────────────────────────────────────────────
# 聚合入口：合并所有来源
# ─────────────────────────────────────────────────────────

def aggregate_options(ticker: str, yf_calls, yf_puts, current_price: float,
                       expiry: str = None) -> dict:
    """
    聚合所有期权数据源，返回统一格式
    """
    sources = {}

    # yfinance（已有，作为基准）
    if yf_calls is not None:
        sources["yfinance"] = {
            "source": "Yahoo Finance",
            "available": True,
            "calls_rows": len(yf_calls),
            "puts_rows": len(yf_puts),
        }

    # Nasdaq API
    nasdaq = get_nasdaq_options(ticker, expiry)
    if "error" not in nasdaq:
        sources["nasdaq"] = {
            "source": "Nasdaq",
            "available": True,
            "calls_rows": len(nasdaq["calls"]) if nasdaq.get("calls") is not None else 0,
            "puts_rows":  len(nasdaq["puts"])  if nasdaq.get("puts")  is not None else 0,
        }
        # 如果 Nasdaq 数据更丰富，补充进来
        if nasdaq.get("calls") is not None and (yf_calls is None or len(nasdaq["calls"]) > len(yf_calls)):
            yf_calls = nasdaq["calls"]
            yf_puts  = nasdaq["puts"]
    else:
        sources["nasdaq"] = {"source": "Nasdaq", "available": False, "error": nasdaq.get("error")}

    # Tradier（可选）
    tradier_token = os.environ.get("TRADIER_TOKEN", "")
    if tradier_token:
        tr = get_tradier_options(ticker, expiry)
        sources["tradier"] = {
            "source": "Tradier",
            "available": "error" not in tr,
            "has_greeks": tr.get("has_greeks", False),
        }
    else:
        sources["tradier"] = {"source": "Tradier", "available": False,
                               "setup": "设置 TRADIER_TOKEN 环境变量即可启用"}

    # Alpaca（可选）
    alpaca_key = os.environ.get("ALPACA_KEY", "")
    if alpaca_key:
        ap = get_alpaca_options(ticker, expiry)
        sources["alpaca"] = {"source": "Alpaca", "available": "error" not in ap,
                              "has_greeks": ap.get("has_greeks", False)}
    else:
        sources["alpaca"] = {"source": "Alpaca", "available": False,
                              "setup": "设置 ALPACA_KEY / ALPACA_SECRET 环境变量即可启用"}

    # 异常活动检测
    unusual = detect_unusual_activity(yf_calls, yf_puts, current_price)

    # CBOE 市场整体 P/C
    cboe = get_cboe_pcr()

    return {
        "sources": sources,
        "unusual_activity": unusual,
        "cboe": cboe,
        "best_calls": yf_calls,
        "best_puts":  yf_puts,
    }


# ─────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────

def _f(val):
    try:
        s = str(val).replace(",", "").replace("%", "")
        return float(s)
    except Exception:
        return None

def _i(val):
    try:
        s = str(val).replace(",", "")
        return int(float(s))
    except Exception:
        return 0
