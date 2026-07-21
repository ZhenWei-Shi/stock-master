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

import os
import json
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz

ET = pytz.timezone("America/New_York")
_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
_UOA_WATCHLIST_PATH = os.path.join(_DATA, "uoa_watchlist.json")
_UOA_ALERTED_PATH   = os.path.join(_DATA, "uoa_alerted.json")
_UOA_ALERTED_KEEP_DAYS = 2   # 去重记录保留天数，避免文件无限增长

# ══════════════════════════════════════════════════════════════
# 机构追踪配置常量（修改参数在此处）
# ══════════════════════════════════════════════════════════════

# ── UOA（异常期权活动）─────────────────────────────────────────
UOA_EXTREME_RATIO           = 3.0      # Vol/OI ≥3 = 极强信号
UOA_EXTREME_VOL             = 500      # 极强需配合量≥500
UOA_NORMAL_RATIO            = 2.0      # Vol/OI ≥2 = 普通异常
UOA_NORMAL_VOL              = 200      # 普通异常需配合量≥200
UOA_MIN_VOLUME              = 100      # 低于此量不计入
OTM_CALL_BUFFER             = 1.01     # OTM call = strike > price*1.01
OTM_PUT_BUFFER              = 0.99     # OTM put  = strike < price*0.99
MAX_EXPIRATIONS             = 3        # 只看最近N个到期日

# ── GEX（做市商Gamma敞口）────────────────────────────────────
GEX_OI_MIN                  = 10       # 最低未平仓量才计入GEX
GEX_T_MIN                   = 0.01     # 最小到期时间（年），防Gamma极值
GEX_BS_RATE                 = 0.05     # Black-Scholes无风险利率
HV_DEFAULT                  = 0.30     # 历史波动率默认值
HV_MIN_DAYS                 = 10       # 计算历史波动率最少需要N天

# ── 空头挤压 ─────────────────────────────────────────────────
SQUEEZE_EXTREME_FLOAT       = 0.30     # 空头>30% = GME级别
SQUEEZE_HIGH_FLOAT          = 0.20     # 空头>20%
SQUEEZE_MED_FLOAT           = 0.10     # 空头>10%
SQUEEZE_DTC_HIGH            = 5        # Days-to-cover >5
SQUEEZE_DTC_MED             = 3        # Days-to-cover >3
SQUEEZE_MOM_STRONG          = 0.10     # 10日涨幅>10%
SQUEEZE_MOM_MOD             = 0.05     # 10日涨幅>5%
SQUEEZE_VOL_SPIKE           = 3.0      # 量比>3倍
SQUEEZE_VOL_ELEVATED        = 2.0      # 量比>2倍
SQUEEZE_SCORE_MIN           = 0
SQUEEZE_SCORE_MAX           = 100
SQUEEZE_THRESHOLD_HIGH      = 70       # ≥70 = 极高挤压风险
SQUEEZE_THRESHOLD_MED       = 40       # ≥40 = 中等

# ── SMF（智能资金流向）────────────────────────────────────────
SMF_OPEN_BARS               = 6        # 开盘30分钟 = 6根5分钟K线
SMF_CLOSE_BARS              = 12       # 收盘60分钟 = 12根5分钟K线
SMF_MIN_BARS                = 15       # 计算SMF最少需要N根K线
SMF_MOVE_THRESHOLD          = 0.3      # ±0.3%以上视为有效信号

# ── 13F机构持仓 ────────────────────────────────────────────
INST_HIGH_HELD              = 0.70     # ≥70% = 高机构化
INST_MED_HELD               = 0.50     # ≥50%
INST_LOW_HELD               = 0.15     # ≥15% = 有机构参与
INSIDER_SKIN_THRESHOLD      = 0.05     # 内部人持股>5% = 利益绑定

# ── 综合评分 ─────────────────────────────────────────────────
SMART_SCORE_MIN             = 0
SMART_SCORE_MAX             = 100
SMART_THRESHOLD_EXTREME     = 80
SMART_THRESHOLD_GOOD        = 60
SMART_THRESHOLD_MODERATE    = 40

# ══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# 1. 异常期权活动（Unusual Options Activity）
# ─────────────────────────────────────────────────────────────

def detect_unusual_options(ticker: str) -> dict:
    """
    期权流向分析 v2（Bid/Ask 方向推断）

    改进：使用 Lee-Ready 规则推断每笔成交的主动方向
      proximity_to_ask = (lastPrice - bid) / (ask - bid)
      > 0.6 → 主动买入（买方付 ask，看涨信号）
      < 0.4 → 主动卖出（卖方收 bid，方向相反）

    相比 v1（纯 volume/OI）的关键优势：
      - 区分"有人买 call"vs"有人卖 covered call"（数量相同，方向相反）
      - 仍依赖 yfinance，但比随机猜测更可信（R² 约 0.6）
      - 不替代真实 tape（仍有局限），但已从噪音升级为弱信号

    返回：
      bias:             bullish / bearish / neutral
      net_call_flow:    加权净买方 call 量（正=主动买入）
      net_put_flow:     加权净买方 put 量
      call_put_ratio:   总 call/put 成交量比
      key_strikes:      最活跃行权价位
      signal_strength:  信号强度描述
      data_quality:     始终标注数据局限性
    """
    try:
        tk    = yf.Ticker(ticker)
        _hist1 = tk.history(period="5d")
        if _hist1.empty:
            return {"ok": False, "reason": f"{ticker} 无价格数据"}
        price = float(_hist1["Close"].iloc[-1])
        exps  = tk.options

        if not exps:
            return {"ok": False, "reason": "无期权数据"}

        check_exps = exps[:min(MAX_EXPIRATIONS, len(exps))]

        net_call_flow = 0.0   # 正 = 主动买 call，负 = 主动卖 call
        net_put_flow  = 0.0   # 正 = 主动买 put，负 = 主动卖 put
        total_call_vol = total_put_vol = 0
        key_strikes_raw = {}  # strike → abs(net_flow)

        for exp in check_exps:
            try:
                chain = tk.option_chain(exp)
                for df, side in [(chain.calls, "call"), (chain.puts, "put")]:
                    if df.empty:
                        continue

                    for _, row in df.iterrows():
                        strike = float(row.get("strike") or 0)
                        vol    = int(row.get("volume")       or 0)
                        oi     = int(row.get("openInterest") or 0)
                        bid    = float(row.get("bid")        or 0)
                        ask    = float(row.get("ask")        or 0)
                        last   = float(row.get("lastPrice")  or 0)

                        if vol < UOA_MIN_VOLUME or oi < 10:
                            continue
                        if abs(strike - price) / price > 0.10:
                            continue  # 只看 ±10% 范围内，远 OTM 通常是噪音

                        # ── Bid/Ask 方向推断（Lee-Ready 规则简化版）──
                        spread = ask - bid
                        # lastPrice 必须落在合理区间内才可信（陈旧数据会漂移到 bid/ask 之外）
                        if spread > 0 and last > 0 and bid * 0.8 <= last <= ask * 1.5:
                            proximity = (last - bid) / spread   # 0=bid, 1=ask
                            # 钳位到 [-1, +1]，防止陈旧 lastPrice 产生无界权重
                            direction_weight = max(-1.0, min(1.0, (proximity - 0.5) * 2))
                        else:
                            direction_weight = 0.0   # 数据陈旧或无价差，无法判断方向

                        net_flow = vol * direction_weight   # 正=主动买入

                        if side == "call":
                            total_call_vol += vol
                            net_call_flow  += net_flow
                        else:
                            total_put_vol += vol
                            net_put_flow  += net_flow

                        key_strikes_raw[strike] = (
                            key_strikes_raw.get(strike, 0) + abs(net_flow)
                        )

            except Exception:
                continue

        # ── 偏向判断 ────────────────────────────────────────────
        call_pt = max(abs(net_call_flow), 1)
        put_pt  = max(abs(net_put_flow),  1)
        total_pt = call_pt + put_pt

        # 只有当流向显著时才判断方向（否则 neutral）
        MIN_FLOW = 200   # 净流向绝对值需超过 200 手才算信号

        if net_call_flow > MIN_FLOW and net_call_flow / total_pt > 0.55:
            bias, strength = "bullish", "Call 主动买入主导"
        elif net_put_flow > MIN_FLOW and net_put_flow / total_pt > 0.55:
            bias, strength = "bearish", "Put 主动买入主导（可能对冲，非纯做空）"
        else:
            bias, strength = "neutral", "流向均衡或样本不足"

        call_put_ratio = round(total_call_vol / max(total_put_vol, 1), 2)

        # 最活跃行权价（按净流向绝对值排序）
        key_strikes = sorted(key_strikes_raw, key=key_strikes_raw.get, reverse=True)[:5]

        return {
            "ok":             True,
            "ticker":         ticker,
            "price":          round(price, 2),
            "bias":           bias,
            "net_call_flow":  round(net_call_flow),
            "net_put_flow":   round(net_put_flow),
            "call_put_ratio": call_put_ratio,
            "key_strikes":    key_strikes,
            "signal_strength": strength,
            "data_quality":   (
                "⚠️ 基于 yfinance bid/ask 位置推断（Lee-Ready简化），"
                "比纯 Vol/OI 更可信但仍非 tape 数据。"
                "大型机构常在盘前/AH 执行，此信号有盲区。"
            ),
        }

    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────
# 1b. 异常大单检测（2026-07-21新增，Telegram /uoa 指令）
# ─────────────────────────────────────────────────────────────
# 跟上面 detect_unusual_options() 不同：那个算的是整条期权链的净流向
# 偏向（一个汇总方向），这里是找具体哪个行权价出现了"当日成交量远超
# 现有未平仓量"的异常大单——Vol/OI比值高，说明是新开的大仓位，不是老
# 仓位换手。用的正是本文件顶部一直定义着但从未被调用过的
# UOA_EXTREME_RATIO/UOA_NORMAL_RATIO等阈值常量。

def detect_large_orders(ticker: str) -> dict:
    """
    扫描个股期权链（近MAX_EXPIRATIONS个到期日，spot±10%范围内），找出
    Vol/OI比值异常的行权价，按新增权利金金额从高到低排序。

    两档severity：
      EXTREME：Vol/OI≥UOA_EXTREME_RATIO(3.0) 且 Vol≥UOA_EXTREME_VOL(500)
      NORMAL： Vol/OI≥UOA_NORMAL_RATIO(2.0)  且 Vol≥UOA_NORMAL_VOL(200)
    """
    try:
        tk   = yf.Ticker(ticker)
        # 用分钟级数据拿精确报价时间戳；日线数据收盘后时间戳固定是00:00:00，
        # 不够精确。盘中能拿到真实到分钟的最新价，收盘后则是当天最后一分钟。
        hist = tk.history(period="1d", interval="1m")
        if hist.empty:
            hist = tk.history(period="1d")   # 分钟级数据缺失时兜底用日线
        if hist.empty:
            return {"ok": False, "reason": f"{ticker} 无价格数据"}
        price      = float(hist["Close"].iloc[-1])
        quote_time = hist.index[-1]

        exps = tk.options
        if not exps:
            return {"ok": False, "reason": "无期权数据"}

        alerts = []
        for exp in exps[:MAX_EXPIRATIONS]:
            try:
                chain = tk.option_chain(exp)
                for df, opt_type in [(chain.calls, "CALL"), (chain.puts, "PUT")]:
                    if df.empty:
                        continue
                    for _, row in df.iterrows():
                        strike = float(row.get("strike") or 0)
                        vol    = int(row.get("volume") or 0)
                        oi     = int(row.get("openInterest") or 0)
                        if strike <= 0 or oi <= 0 or vol < UOA_NORMAL_VOL:
                            continue
                        if abs(strike - price) / price > 0.10:
                            continue  # 只看spot±10%，远OTM通常是噪音

                        ratio = vol / oi
                        if ratio >= UOA_EXTREME_RATIO and vol >= UOA_EXTREME_VOL:
                            severity = "EXTREME"
                        elif ratio >= UOA_NORMAL_RATIO and vol >= UOA_NORMAL_VOL:
                            severity = "NORMAL"
                        else:
                            continue

                        last = float(row.get("lastPrice") or 0)
                        bid  = float(row.get("bid") or 0)
                        ask  = float(row.get("ask") or 0)
                        iv   = float(row.get("impliedVolatility") or 0)

                        alerts.append({
                            "expiry":   exp,
                            "type":     opt_type,
                            "strike":   strike,
                            "volume":   vol,
                            "oi":       oi,
                            "ratio":    round(ratio, 1),
                            "notional": vol * last * 100,
                            "bid":      bid,
                            "ask":      ask,
                            "iv":       round(iv * 100, 1),
                            "severity": severity,
                        })
            except Exception:
                continue

        alerts.sort(key=lambda a: a["notional"], reverse=True)
        return {
            "ok":         True,
            "ticker":     ticker,
            "price":      round(price, 2),
            "quote_time": str(quote_time),
            "alerts":     alerts,
        }
    except Exception as e:
        return {"ok": False, "reason": str(e)}


def format_large_orders_telegram(result: dict) -> str:
    """把 detect_large_orders() 的结果格式化为Telegram警报文本。"""
    if not result.get("ok"):
        return f"⚠️ {result.get('ticker', '?')}：{result.get('reason', '检测失败')}"

    ticker     = result["ticker"]
    alerts     = result["alerts"]
    now_et     = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    quote_time = result["quote_time"]
    price      = result["price"]

    if not alerts:
        return (
            f"✅ <b>{ticker}期权检查</b>\n"
            f"时间：{now_et}\n"
            f"股价：${price}（行情时间 {quote_time}）\n"
            f"未发现异常大单（Vol/OI均低于{UOA_NORMAL_RATIO}x或成交量不足{UOA_NORMAL_VOL}张）"
        )

    lines = [
        f"🚨 <b>{ticker}期权异动</b>",
        f"时间：{now_et}",
        f"股价：${price}（行情时间 {quote_time}）",
        "",
    ]
    for a in alerts[:8]:
        icon = "🔥" if a["severity"] == "EXTREME" else "⚡"
        lines.append(
            f"{icon} {a['expiry']} {a['type']} ${a['strike']:g}: "
            f"新增{a['volume']}张，增量权利金${a['notional']/1000:.1f}K，"
            f"当日量/OI={a['ratio']}x，Bid/Ask ${a['bid']:.2f}/${a['ask']:.2f}，"
            f"IV {a['iv']}%"
        )
    if len(alerts) > 8:
        lines.append(f"\n...另有{len(alerts)-8}条未显示")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 1c. UOA 监控列表 + 自动推送（2026-07-21新增，/uoa指令升级用）
# ─────────────────────────────────────────────────────────────
# 用户用 /uoa TICKER 查询后，该ticker自动加入本监控列表，scheduler每小时
# （跟盘中止损检查同频）扫描一遍，只推送"今天第一次出现"或"严重程度从
# NORMAL升级到EXTREME"的大单——同一个还在场上的大单不会每小时重复刷屏。

def _load_uoa_watchlist() -> list:
    if not os.path.exists(_UOA_WATCHLIST_PATH):
        return []
    try:
        with open(_UOA_WATCHLIST_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_uoa_watchlist(tickers: list):
    os.makedirs(_DATA, exist_ok=True)
    tmp = _UOA_WATCHLIST_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tickers, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _UOA_WATCHLIST_PATH)


def add_uoa_watch(ticker: str) -> list:
    """把ticker加入UOA自动监控列表，已在列表里则不重复添加。返回最新列表。"""
    wl = _load_uoa_watchlist()
    ticker = ticker.upper()
    if ticker not in wl:
        wl.append(ticker)
        _save_uoa_watchlist(wl)
    return wl


def remove_uoa_watch(ticker: str) -> list:
    """把ticker移出UOA自动监控列表。返回最新列表。"""
    wl = _load_uoa_watchlist()
    ticker = ticker.upper()
    wl = [t for t in wl if t != ticker]
    _save_uoa_watchlist(wl)
    return wl


def get_uoa_watchlist() -> list:
    return _load_uoa_watchlist()


def _load_uoa_alerted() -> dict:
    if not os.path.exists(_UOA_ALERTED_PATH):
        return {}
    try:
        with open(_UOA_ALERTED_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_uoa_alerted(data: dict):
    os.makedirs(_DATA, exist_ok=True)
    tmp = _UOA_ALERTED_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _UOA_ALERTED_PATH)


def check_uoa_watchlist_for_new_alerts() -> dict:
    """
    扫描UOA监控列表全部ticker，只返回"今天首次出现"或"严重程度升级"的
    异常大单（NORMAL已警告过、且这次还是NORMAL的不重复返回；NORMAL升级
    到EXTREME会再警告一次）。

    返回：{ticker: {"price":.., "quote_time":.., "alerts":[本次新增的alert...]}}
    只包含确实有新内容要推送的ticker，供scheduler直接遍历发送。
    """
    watchlist = get_uoa_watchlist()
    if not watchlist:
        return {}

    today_str = datetime.now(ET).strftime("%Y-%m-%d")
    alerted = _load_uoa_alerted()
    today_alerted = alerted.get(today_str, {})
    severity_rank = {"NORMAL": 1, "EXTREME": 2}

    results = {}
    for ticker in watchlist:
        try:
            result = detect_large_orders(ticker)
        except Exception:
            continue
        if not result.get("ok"):
            continue

        new_alerts = []
        for a in result["alerts"]:
            key = f"{ticker}|{a['expiry']}|{a['type']}|{a['strike']}"
            prev = today_alerted.get(key)
            if prev is None or severity_rank[a["severity"]] > severity_rank.get(prev, 0):
                new_alerts.append(a)
                today_alerted[key] = a["severity"]

        if new_alerts:
            results[ticker] = {
                "price": result["price"],
                "quote_time": result["quote_time"],
                "alerts": new_alerts,
            }

    # 保存去重状态，顺带清掉超过_UOA_ALERTED_KEEP_DAYS天的旧记录避免文件无限增长
    alerted[today_str] = today_alerted
    cutoff = (datetime.now(ET) - timedelta(days=_UOA_ALERTED_KEEP_DAYS)).strftime("%Y-%m-%d")
    alerted = {d: v for d, v in alerted.items() if d >= cutoff}
    _save_uoa_alerted(alerted)

    return results


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
            hv_note = f"HV30={hv30*100:.1f}%（实测）"
        else:
            hv30    = 0.30
            hv_note = "⚠️ 数据不足(<10天)，HV30使用默认值30%，Gamma估算精度低"

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

        import pytz as _pytz
        _ET = _pytz.timezone("America/New_York")

        for exp in exps[:3]:  # 只看最近3个到期日
            try:
                exp_dt = _ET.localize(datetime.strptime(exp, "%Y-%m-%d").replace(hour=16))
                T = max((exp_dt - datetime.now(_ET)).total_seconds() / (365 * 86400), 0.01)
                chain = tk.option_chain(exp)

                for row in chain.calls.itertuples():
                    oi = int(getattr(row, "openInterest", 0) or 0)
                    if oi < 10:
                        continue
                    iv = float(getattr(row, "impliedVolatility", 0) or 0)
                    sigma = iv if iv > 0.01 else hv30  # 优先用逐档IV，缺失时fallback HV30
                    g = bs_gamma(price, row.strike, T, sigma)
                    gex = g * oi * 100 * price ** 2  # 美元 Gamma
                    call_gex_total += gex
                    strike = float(row.strike)
                    gex_by_strike[strike] = gex_by_strike.get(strike, 0) + gex

                for row in chain.puts.itertuples():
                    oi = int(getattr(row, "openInterest", 0) or 0)
                    if oi < 10:
                        continue
                    iv = float(getattr(row, "impliedVolatility", 0) or 0)
                    sigma = iv if iv > 0.01 else hv30
                    g = bs_gamma(price, row.strike, T, sigma)
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
            "hv_note":         hv_note,
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

        prev_close_hist = tk.history(period="5d", interval="1d")  # 周一只有1日数据时period="2d"不够
        prev_close = float(prev_close_hist["Close"].iloc[-2]) if len(prev_close_hist) >= 2 else float(prev_close_hist["Close"].iloc[-1])
        today_close = float(close_all.iloc[-1])

        # 开盘段变化（情绪化噪音）
        open_move  = (open_vwap - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
        # 收盘段变化（机构动作）
        close_move = (close_vwap - open_vwap) / open_vwap * 100 if open_vwap > 0 else 0.0
        # 全天净变化
        day_return = (today_close - prev_close) / prev_close * 100 if prev_close > 0 else 0.0

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
        _hist_inst = tk.history(period="5d")
        if _hist_inst.empty:
            return {"ok": False, "reason": f"{ticker} 无价格数据"}
        price = float(_hist_inst["Close"].iloc[-1])

        inst_holders = tk.institutional_holders
        major_holders= tk.major_holders

        if inst_holders is None or inst_holders.empty:
            return {"ok": False, "reason": "无机构持仓数据"}

        # 机构持仓比例
        inst_pct  = float(info.get("institutionPercentHeld", 0)) * 100  # yfinance正确字段名
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
        lines.append(f"\n📈 <b>期权异常</b>：{uoa.get('bias', 'neutral')}")
        lines.append(f"  C/P比：{uoa.get('call_put_ratio', '—')}  信号强度：{uoa.get('signal_strength', '—')}")

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
