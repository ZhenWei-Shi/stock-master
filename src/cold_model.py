"""
冷静交易决策引擎（Cold Mind Model v3）

原则：一票否决制
- 任何一条红灯 → 整个信号无效，保持空仓
- 所有绿灯 → 才输出入场建议

两种模式：
  标准模式（standard）：保守，适合 $25k+ 账户，1%风险规则
  激进模式（aggressive）：适合 $2k-$10k 小账户，3%风险，动量突破优化

核心评分：0-100分
  标准模式：≥75 = GO
  激进模式：≥65 = GO（门槛略低，换取更多机会）

第三次审核修复（2026-06-17）：
  ✅ Gate H 方向修正：近新高 = 突破信号（原为"警告"，与Minervini理论相反）
  ✅ Gate D RSI 激进扩展：35-80（原40-70，封锁了动量黄金区间）
  ✅ Gate F VWAP：摆动/激进模式下跳过（VWAP跨天累积无统计意义）
  ✅ Gate G 止损上限：激进模式放宽至12%（原8%，过滤高波动品种）
  ✅ Gate A 时间窗口：激进摆动模式放宽至收盘前5分钟
  ✅ 仓位计算：激进模式3%风险，单仓上限50%（原1%+20%）
  ✅ 评分权重：修正 near_high / time_window / vwap 权重误导
  ✅ 激进模式加分项：RS>85, VCP形态, 财报加速各加分
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from .pdt_guard import check_pdt_risk, will_trigger_pdt

ET = pytz.timezone("America/New_York")

# ══════════════════════════════════════════════════════════════
# 全局配置常量（修改参数在此处，无需深入函数体）
# ══════════════════════════════════════════════════════════════

# ── 账户规模门限 ──────────────────────────────────────────────
PORTFOLIO_FORCE_AGGRESSIVE  = 5_000    # ≤$5k 自动激活激进模式
PORTFOLIO_SMALL_THRESHOLD   = 10_000   # <$10k 使用小账户风险比例
PDT_ACCOUNT_MINIMUM         = 25_000   # 联邦PDT最低保证金

# ── 市场时间窗口（美东时间分钟数）────────────────────────────
MARKET_OPEN_MIN             = 570      # 09:30
MARKET_CLOSE_MIN            = 960      # 16:00
OPEN_BUFFER_STD             = 15       # 标准模式：开盘后15分钟禁入
CLOSE_BUFFER_STD            = 20       # 标准模式：收盘前20分钟禁入
OPEN_BUFFER_AGG             = 5        # 激进/摆动：5分钟
CLOSE_BUFFER_AGG            = 5        # 激进/摆动：5分钟

# ── VIX 恐慌指数阈值 ─────────────────────────────────────────
VIX_PANIC_HARD              = 40.0     # 硬性禁入
VIX_ELEVATED                = 28.0     # 预警（减半仓位）
VIX_DEFAULT_FALLBACK        = 20.0     # 获取失败时的默认值

# ── 移动均线周期 ─────────────────────────────────────────────
MA_SHORT                    = 20
MA_MID                      = 50
MA_LONG                     = 200

# ── RSI 配置 ─────────────────────────────────────────────────
RSI_PERIOD                  = 14
RSI_STD_MIN, RSI_STD_MAX    = 40, 70   # 标准模式
RSI_AGG_MIN, RSI_AGG_MAX    = 35, 80   # 激进模式

# ── ATR 止损空间 ─────────────────────────────────────────────
ATR_PERIOD                  = 14
ATR_STOP_MULT               = 1.5      # ATR × 1.5 = 止损距离
MAX_STOP_PCT_STD            = 8.0      # 标准模式最大止损%
MAX_STOP_PCT_AGG            = 12.0     # 激进模式最大止损%
MIN_STOP_PCT                = 0.3      # 最小止损%（防信号失真）

# ── 量能配置 ─────────────────────────────────────────────────
VOL_MA_PERIOD               = 20
VOL_RATIO_MIN               = 0.8      # 最低量比门槛
PRICE_DROP_WARN_PCT         = -2.0     # 价格跌幅触发放量预警
SELLOFF_VOL_RATIO           = 2.0      # 高量比+下跌 = 抛售

# ── VWAP 配置 ─────────────────────────────────────────────────
VWAP_MIN_BARS               = 5        # 日内最少K线数
VWAP_LONG_PREMIUM_MAX       = 5.0      # 多头：最大溢价VWAP%
VWAP_LONG_DISCOUNT_MIN      = -3.0     # 多头：最大折价VWAP%

# ── 入场评分门限 ─────────────────────────────────────────────
GO_THRESHOLD_STD            = 75       # 标准模式入场分
GO_THRESHOLD_AGG            = 65       # 激进模式入场分
WAIT_THRESHOLD_STD          = 55       # 标准模式等待分
WAIT_THRESHOLD_AGG          = 45       # 激进模式等待分

# ── 仓位风险比例 ─────────────────────────────────────────────
RISK_PCT_AGG                = 0.03     # 激进模式：账户3%
RISK_PCT_SMALL              = 0.015    # 小账户：1.5%
RISK_PCT_STD                = 0.01     # 标准模式：1%
MAX_POS_PCT_AGG             = 0.50     # 激进单仓上限：50%
MAX_POS_PCT_STD             = 0.25     # 标准单仓上限：25%

# ── 出场倍率 ─────────────────────────────────────────────────
TARGET_1_ATR_MULT           = 2.0      # 目标1：ATR×2
TARGET_2_ATR_MULT           = 3.5      # 目标2：ATR×3.5

# ── 期望值估算（激进模式） ───────────────────────────────────
EV_WIN_RATE                 = 0.60     # 假设胜率
EV_RR_RATIO                 = 3.0      # 假设盈亏比

# ── 加分/扣分配置 ────────────────────────────────────────────
CANSLIM_A_BONUS             = 15       # CANSLIM A级加分
CANSLIM_B_BONUS             = 8        # CANSLIM B级加分
CANSLIM_D_PENALTY           = 12       # CANSLIM D级扣分
PEAD_BONUS                  = 8        # PEAD漂移信号加分
QUALITY_A_BONUS             = 5        # 质量因子A加分
UOA_BULL_BONUS              = 0        # 已下架：yfinance期权量无法区分买方方向，数据无效
UOA_BEAR_PENALTY            = 0        # 已下架：同上
SMF_BULL_BONUS              = 10       # 机构资金流入加分
SMF_BEAR_PENALTY            = 8        # 机构资金流出扣分
SQUEEZE_BONUS               = 5        # 逼空信号加分
SQUEEZE_SCORE_MIN           = 60       # 触发加分的最低逼空分
SECTOR_TOP_BONUS            = 5        # 板块前3名加分
SECTOR_ACCEL_BONUS          = 3        # 板块加速流入额外加分

# ── 回测/扫描限制 ─────────────────────────────────────────────
SCAN_TICKER_LIMIT           = 20       # 单次批量扫描最多20只

# ══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────

def cold_decision(ticker: str, portfolio: float = 100_000,
                  direction: str = "LONG",
                  account_type: str = "margin",
                  day_trades_used: int = 0,
                  is_intraday: bool = True,
                  aggressive_mode: bool = False) -> dict:
    """
    最冷静的入场决策（v3）

    参数：
      ticker          — 股票代码
      portfolio       — 账户净值（美元）
      direction       — LONG / SHORT
      account_type    — "margin" 或 "cash"
      day_trades_used — 本周已用日内交易次数
      is_intraday     — 是否日内交易（影响PDT）
      aggressive_mode — 激进模式：为 $2k-$10k 小账户优化
                        * RSI区间扩展至35-80（含动量突破信号）
                        * 近新高 = 突破信号（不再是警告）
                        * 止损上限放宽至12%
                        * 风险比例3%（非1%）
                        * 单仓上限50%
                        * VWAP门限跳过（摆动交易无关）
                        * 入场门槛降至65分（非75分）

    返回：
      verdict: GO / WAIT / ABORT
      score: 0-100
      gates, entry_plan, pdt_status, expected_value
    """
    # 账户 < $5k 自动激活激进模式
    if portfolio <= 0:
        return {"verdict": "ABORT", "reason": "账户净值无效（≤0）",
                "score": 0, "gates": {}, "entry_plan": None}
    if not ticker or not isinstance(ticker, str):
        return {"verdict": "ABORT", "reason": "代码无效",
                "score": 0, "gates": {}, "entry_plan": None}
    if portfolio <= PORTFOLIO_FORCE_AGGRESSIVE:
        aggressive_mode = True
    now = datetime.now(ET)
    gates = {}

    # ── 数据获取 ─────────────────────────────────────────
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info
        hist_1y  = tk.history(period="1y",  interval="1d")
        hist_5d  = tk.history(period="5d",  interval="5m")
        hist_3mo = tk.history(period="3mo", interval="1d")
    except Exception as e:
        return {"verdict": "ABORT", "reason": f"数据获取失败：{e}",
                "score": 0, "gates": {}, "entry_plan": None}

    if hist_1y.empty:
        return {"verdict": "ABORT", "reason": "无历史数据",
                "score": 0, "gates": {}, "entry_plan": None}

    close = hist_1y["Close"]
    price = float(close.iloc[-1])

    # ── Gate A：市场时间窗口 ──────────────────────────────
    total_min  = now.hour * 60 + now.minute
    is_weekend = now.weekday() >= 5

    if aggressive_mode or not is_intraday:
        no_entry_open  = total_min < MARKET_OPEN_MIN + OPEN_BUFFER_AGG
        no_entry_close = total_min > MARKET_CLOSE_MIN - CLOSE_BUFFER_AGG
        open_note  = "开盘前5分钟（流动性陷阱），禁止入场"
        close_note = "收盘前5分钟，市价单滑点过大"
    else:
        no_entry_open  = total_min < MARKET_OPEN_MIN + OPEN_BUFFER_STD
        no_entry_close = total_min > MARKET_CLOSE_MIN - CLOSE_BUFFER_STD
        open_note  = f"开盘前{OPEN_BUFFER_STD}分钟（流动性陷阱），禁止入场"
        close_note = f"收盘前{CLOSE_BUFFER_STD}分钟，禁止日内新仓（隔夜敞口）"

    if is_weekend:
        gates["time_window"] = {"pass": False, "note": "周末市场关闭，无法交易"}
    elif total_min < MARKET_OPEN_MIN or total_min >= MARKET_CLOSE_MIN:
        gates["time_window"] = {"pass": False, "note": "市场已关闭，做好计划等待开盘"}
    elif no_entry_open:
        gates["time_window"] = {"pass": False, "note": open_note}
    elif no_entry_close:
        gates["time_window"] = {"pass": False, "note": close_note}
    else:
        gates["time_window"] = {"pass": True, "note": f"交易时间正常 {now.strftime('%H:%M')} ET"}

    # ── Gate B：VIX 恐慌指数 ─────────────────────────────
    try:
        vix_val = float(yf.Ticker("^VIX").history(period="1d")["Close"].iloc[-1])
        if vix_val > 40:
            gates["vix"] = {"pass": False, "note": f"VIX={vix_val:.1f}，市场极度恐慌，禁止所有交易"}
        elif vix_val > 28:
            gates["vix"] = {"pass": "warn", "note": f"VIX={vix_val:.1f}，波动偏高，仓位减半"}
        else:
            gates["vix"] = {"pass": True, "note": f"VIX={vix_val:.1f}，波动正常"}
    except Exception:
        vix_val = 20
        gates["vix"] = {"pass": "warn", "note": "无法获取VIX，按正常处理"}

    # ── Gate C：趋势方向确认（大框架） ───────────────────
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

    if direction == "LONG":
        if ma200 is not None:
            # 完整多头排列：价格 > MA20 > MA50 > MA200（防死叉陷阱）
            trend_pass = price > ma20 > ma50 > ma200
            if not trend_pass:
                if price > ma20 > ma50 and ma50 <= ma200:
                    fail_reason = f"MA50({ma50:.2f})≤MA200({ma200:.2f})，中期均线死叉，禁止做多"
                elif price <= ma200:
                    fail_reason = f"价格低于MA200({ma200:.2f})，长期趋势向下"
                else:
                    fail_reason = f"趋势不支持做多：{'价格<MA20' if price < ma20 else 'MA20<MA50'}"
            else:
                fail_reason = ""
        else:
            trend_pass  = price > ma20 > ma50
            fail_reason = f"趋势不支持做多：{'价格<MA20' if price < ma20 else 'MA20<MA50'}"
        gates["trend"] = {
            "pass": trend_pass,
            "note": (f"多头排列：{price:.2f} > MA20({ma20:.2f}) > MA50({ma50:.2f})"
                     + (f" > MA200({ma200:.2f})" if ma200 is not None else "")
                     if trend_pass else fail_reason),
        }
    else:
        trend_pass = price < ma20 < ma50
        gates["trend"] = {
            "pass": trend_pass,
            "note": (f"空头排列：价格{price:.2f} < MA20({ma20:.2f}) < MA50({ma50:.2f})"
                     if trend_pass else "趋势不支持做空"),
        }

    # ── Gate D：RSI 区间 ─────────────────────────────────
    rsi_s   = _rsi_series(close)
    rsi_now = float(rsi_s.iloc[-1])

    if direction == "LONG":
        if aggressive_mode:
            # 激进模式：35-80（含动量突破黄金区间）
            # Minervini数据：最强突破发生在RSI从65推向80+时，不应屏蔽
            rsi_zone_ok  = 35 <= rsi_now <= 80
            rsi_momentum = 60 <= rsi_now <= 80   # 动量强烈
            rsi_note = (
                f"RSI={rsi_now:.1f}，{'动量突破区间(60-80)，Minervini强势信号' if rsi_momentum else '合理做多区间(35-80)'}"
                if rsi_zone_ok else
                f"RSI={rsi_now:.1f}，{'极度超买>80，短期追高风险大' if rsi_now > 80 else '超卖<35，等待底部企稳'}"
            )
        else:
            rsi_zone_ok = 40 <= rsi_now <= 70
            rsi_note = (f"RSI={rsi_now:.1f}，处于合理做多区间(40-70)"
                        if rsi_zone_ok else
                        f"RSI={rsi_now:.1f}，{'超买>70，标准模式避免追高' if rsi_now > 70 else '过度超卖<40，等待企稳'}")
        gates["rsi"] = {"pass": rsi_zone_ok, "note": rsi_note}
    else:
        rsi_zone_ok = 20 <= rsi_now <= 55 if aggressive_mode else 30 <= rsi_now <= 60
        gates["rsi"] = {
            "pass": rsi_zone_ok,
            "note": (f"RSI={rsi_now:.1f}，处于合理做空区间"
                     if rsi_zone_ok else f"RSI={rsi_now:.1f}，不适合做空"),
        }

    # ── Gate E：成交量确认（量价配合） ───────────────────
    vol   = hist_1y["Volume"]
    vol_m20 = float(vol.rolling(20).mean().iloc[-1])
    vol_today = float(vol.iloc[-1])
    vol_ratio = vol_today / vol_m20 if vol_m20 > 0 else 1.0

    price_chg = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) >= 2 else 0

    if direction == "LONG":
        vol_ok = vol_ratio >= 0.8 and not (price_chg < -2 and vol_ratio > 2)
        gates["volume"] = {
            "pass": vol_ok,
            "note": (f"量比{vol_ratio:.1f}x，{'放量上涨' if price_chg > 0 else '量比正常'}"
                     if vol_ok else
                     f"量比{vol_ratio:.1f}x + 价格跌{price_chg:.1f}%，放量下跌，禁止做多"),
        }
    else:
        vol_ok = vol_ratio >= 0.8
        gates["volume"] = {
            "pass": vol_ok,
            "note": f"量比{vol_ratio:.1f}x",
        }

    # ── Gate F：VWAP 位置 ────────────────────────────────
    # VWAP 是每日09:30重置的日内指标。跨多天的累积VWAP无统计意义。
    # 激进/摆动模式：完全跳过（摆动持仓与日内VWAP无关）
    # 日内标准模式：仅使用最后一个交易日的数据计算当日VWAP
    if aggressive_mode or not is_intraday:
        gates["vwap"] = {"pass": "skip",
                         "note": "摆动/激进模式不使用VWAP（日内重置指标，跨天无效）"}
    else:
        vwap_gate = {"pass": "skip", "note": "日内数据不足，跳过VWAP"}
        if not hist_5d.empty and len(hist_5d) >= 10:
            # 只取今天的数据（最后一个交易日）
            try:
                today_str = hist_5d.index[-1].date()
                today_data = hist_5d[hist_5d.index.date == today_str]
                if len(today_data) >= 5:
                    tp   = (today_data["High"] + today_data["Low"] + today_data["Close"]) / 3
                    vwap = (tp * today_data["Volume"]).cumsum() / today_data["Volume"].cumsum()
                    vwap_now = float(vwap.iloc[-1])
                    price_5m = float(today_data["Close"].iloc[-1])
                    dev = (price_5m - vwap_now) / vwap_now * 100 if vwap_now > 0 else 0
                    if direction == "LONG":
                        vwap_ok = dev > -3 and dev < 5
                        vwap_gate = {
                            "pass": vwap_ok,
                            "note": (f"当日VWAP {'上方' if dev > 0 else '下方'}{abs(dev):.1f}%"
                                     + ("，位置合理" if vwap_ok else
                                        "，偏离过大（>5%追高 或 <-3%趋势已破）")),
                        }
                    else:
                        vwap_ok = dev < 3 and dev > -5
                        vwap_gate = {"pass": vwap_ok, "note": f"当日VWAP偏离{dev:.1f}%"}
            except Exception:
                pass
        gates["vwap"] = vwap_gate

    # ── Gate G：ATR 止损空间 ─────────────────────────────
    atr_val  = _calc_atr(hist_1y)
    stop_d   = atr_val * 1.5
    stop_pct = stop_d / price * 100
    # 激进模式：高动量小市值股日均ATR可达5-7%，放宽止损上限至12%
    max_stop = 12.0 if aggressive_mode else 8.0

    if stop_pct > max_stop:
        gates["stop_distance"] = {"pass": False,
                                   "note": f"ATR止损{stop_pct:.1f}%过大（>{max_stop:.0f}%），风险不可控"}
    elif stop_pct < 0.3:
        gates["stop_distance"] = {"pass": False,
                                   "note": f"ATR止损{stop_pct:.1f}%过小（<0.3%），信号可能无效"}
    else:
        gates["stop_distance"] = {"pass": True,
                                   "note": f"1.5ATR止损{stop_pct:.1f}%（${stop_d:.2f}），合理"}

    # ── Gate H：价格位置 ──────────────────────────────────
    # 【算法修正】原逻辑将"接近新高"标为警告，与动量突破理论相反。
    # Minervini/O'Neil：在VCP/杯柄突破点买入 = 历史新高附近，这正是最强买点。
    # 正确逻辑：
    #   激进/动量模式：近新高 = 突破候选，有利信号
    #   保守模式：在高位且未放量突破时保持警惕
    high_20d   = float(close.tail(20).max())
    high_52w   = float(close.tail(252).max()) if len(close) >= 252 else high_20d
    pct_20h    = (price - high_20d) / high_20d * 100
    pct_52w    = (price - high_52w) / high_52w * 100

    if direction == "LONG":
        if aggressive_mode:
            # 激进模式：接近20日高点 = 潜在突破位，是有利信号
            if pct_20h >= -2:
                gates["near_high"] = {"pass": True,
                                       "note": f"价格距20日高点{abs(pct_20h):.1f}%，接近突破位——等放量确认"}
            elif pct_20h >= -10:
                gates["near_high"] = {"pass": True,
                                       "note": f"价格在整理区间（距高点{abs(pct_20h):.1f}%），等待突破信号"}
            else:
                gates["near_high"] = {"pass": "warn",
                                       "note": f"距20日高点{abs(pct_20h):.1f}%，整理幅度偏大，确认趋势未破"}
        else:
            # 标准模式：接近高点时提示注意阻力，但不直接否决
            if pct_20h > -1:
                gates["near_high"] = {"pass": "warn",
                                       "note": f"距20日高点仅{abs(pct_20h):.1f}%，注意短期阻力，需放量突破确认"}
            else:
                gates["near_high"] = {"pass": True,
                                       "note": f"距20日高点{abs(pct_20h):.1f}%，位置安全"}
    else:
        gates["near_high"] = {"pass": True, "note": f"做空：距高点{abs(pct_20h):.1f}%"}

    # ── Gate I：财报前禁入 ─────────────────────────────────
    try:
        cal = tk.calendar
        next_earnings = None
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            next_earnings = (list(ed)[0] if hasattr(ed, "__iter__")
                             and not isinstance(ed, str) else ed) if ed else None
        elif cal is not None and hasattr(cal, "columns") and "Earnings Date" in cal.columns:
            next_earnings = cal["Earnings Date"].iloc[0]

        if next_earnings:
            import pandas as _pd
            days_to_earn = (_pd.Timestamp(next_earnings).tz_localize(None)
                            - _pd.Timestamp(now.replace(tzinfo=None))).days
            if days_to_earn < 0:
                # yfinance 财报日期未更新（返回上季度日期），视为安全但注明
                gates["earnings_blackout"] = {
                    "pass": True,
                    "note": f"财报已过（{abs(days_to_earn)}天前），等待下季度日历更新",
                }
            elif days_to_earn <= 3:
                gates["earnings_blackout"] = {
                    "pass": False,
                    "note": f"财报在 {days_to_earn} 天后！禁止建仓（不赌二元事件）",
                }
            elif days_to_earn <= 7:
                gates["earnings_blackout"] = {
                    "pass": "warn",
                    "note": f"财报在 {days_to_earn} 天后，建议等财报结果再行动",
                }
            else:
                gates["earnings_blackout"] = {
                    "pass": True,
                    "note": f"距下次财报 {days_to_earn} 天，安全期",
                }
        else:
            gates["earnings_blackout"] = {"pass": True, "note": "无近期财报记录"}
    except Exception:
        gates["earnings_blackout"] = {"pass": True, "note": "财报日期未知，默认通过"}

    # ── 板块轮动背景门（非阻断，但影响评分） ──────────────
    # 复用已下载的 info 字段，避免在 check_sector_gate 里重复发起 HTTP 请求
    try:
        from .sector_rotation import check_sector_gate, _INDUSTRY_MAP, _SECTOR_MAP
        precomputed_etf = (
            _INDUSTRY_MAP.get(info.get("industry", ""))
            or _SECTOR_MAP.get(info.get("sector", ""))
        )
        sector_g = check_sector_gate(ticker, sector_etf=precomputed_etf)
        gates["sector_rotation"] = {"pass": sector_g["pass"], "note": sector_g["note"]}
        gates["_sector_meta"]    = {k: sector_g.get(k) for k in ("rank", "etf", "accel", "heat")}
    except Exception:
        gates["sector_rotation"] = {"pass": True, "note": "板块门跳过（模块加载失败）"}
        gates["_sector_meta"]    = {}

    # ── PDT 检查（小账户核心保护） ───────────────────────
    pdt_status = check_pdt_risk(portfolio, account_type, day_trades_used)
    pdt_trigger = will_trigger_pdt(day_trades_used, portfolio, is_intraday, account_type)

    if pdt_trigger.get("trigger"):
        gates["pdt_rule"] = {
            "pass": False,
            "note": pdt_trigger["note"],
            "action": pdt_trigger.get("action", ""),
        }
    elif pdt_trigger.get("safe") == "warn":
        gates["pdt_rule"] = {
            "pass": "warn",
            "note": pdt_trigger["note"],
        }
    else:
        gates["pdt_rule"] = {
            "pass": True,
            "note": pdt_trigger.get("note", "PDT检查通过"),
        }

    # PDT gate 写入后计算 hard_fail，确保 PDT 触发能走 ABORT 路径
    hard_fail = [k for k, v in gates.items()
                 if k != "_sector_meta" and v.get("pass") is False]

    # 所有 gate 写完后再算分，score 能完整看到全部门关状态（含 pdt_rule/earnings_quality）
    score = _calc_score(gates, vix_val, aggressive_mode)

    # ── 激进模式额外加分项 ────────────────────────────────
    bonus = 0
    bonus_notes = []
    if aggressive_mode:
        # RS > 85 加分（系统已在九关里检测，这里用简化指标）
        try:
            ret_3m = float((close.iloc[-1] - close.iloc[-63]) / close.iloc[-63]) if len(close) >= 63 else 0
            spy_hist = yf.Ticker("SPY").history(period="3mo")
            spy_3m   = float((spy_hist["Close"].iloc[-1] - spy_hist["Close"].iloc[-63])
                              / spy_hist["Close"].iloc[-63]) if len(spy_hist) >= 63 else 0
            outperform = ret_3m - spy_3m
            if outperform > 0.15:
                bonus += 10
                bonus_notes.append(f"跑赢SPY {outperform*100:.0f}%（+10分）")
            elif outperform > 0.05:
                bonus += 5
                bonus_notes.append(f"跑赢SPY {outperform*100:.0f}%（+5分）")
        except Exception:
            pass
        # 财报加速
        rev_growth = info.get("revenueGrowth")
        if rev_growth and float(rev_growth) > 0.25:
            bonus += 8
            bonus_notes.append(f"营收加速 {float(rev_growth)*100:.0f}%（+8分）")

    # ── 财报质量门（EPS加速度 + CANSLIM + PEAD）─────────────
    try:
        from .earnings_analyzer import full_earnings_analysis, check_position_correlation
        ea = full_earnings_analysis(ticker)
        if ea.get("ok"):
            cs_score = ea.get("canslim_score", 0)
            grade    = ea.get("overall_grade", "")
            # CANSLIM A级财报 → 大加分
            if cs_score >= 80:
                bonus += 15
                bonus_notes.append(f"财报A级（CANSLIM{cs_score}分，+15）")
            elif cs_score >= 60:
                bonus += 8
                bonus_notes.append(f"财报B级（CANSLIM{cs_score}分，+8）")
            # 财报D级 + 方向做多 → 扣分
            elif cs_score < 40 and direction == "LONG":
                bonus -= 12
                bonus_notes.append(f"财报D级（CANSLIM{cs_score}分，-12）")

            # PEAD 信号（财报后漂移）
            pead = ea.get("pead", {})
            if pead.get("ok") and pead.get("latest_surprise_pct", 0) > 5:
                cons = pead.get("pead_consistency", 0)
                if cons > 60:
                    bonus += 8
                    bonus_notes.append(f"PEAD做多信号（历史{cons:.0f}%延续率，+8）")

            # 质量因子（ROE/毛利率）
            qual = ea.get("quality_factors", {})
            if qual.get("ok") and qual.get("quality_grade") == "A":
                bonus += 5
                bonus_notes.append("质量因子A级（高ROE+高毛利率，+5）")

            gates["earnings_quality"] = {
                "pass": cs_score >= 40 or direction == "SHORT",
                "note": f"CANSLIM评分{cs_score}/100（{grade}）— {ea.get('action', '')}",
            }
    except Exception:
        pass

    # ── 宏观过滤（FOMC/CPI/传导链）─────────────────────────
    try:
        from .macro_filter import macro_gate_check
        macro = macro_gate_check(ticker)
        if macro.get("block"):
            return {
                "verdict": "ABORT",
                "reason":  f"宏观否决：{macro['reason']}",
                "score": 0, "gates": gates, "entry_plan": None,
            }
        if macro.get("penalty", 0) > 0:
            bonus -= macro["penalty"]
            bonus_notes.append(f"宏观扣分：{macro['reason']}（-{macro['penalty']}）")
        if macro.get("bonus", 0) > 0:
            bonus += macro["bonus"]
            bonus_notes.append(f"宏观加分：{macro['reason']}（+{macro['bonus']}）")
    except Exception:
        pass

    # ── 机构追踪加分（智能资金共振）────────────────────────
    # 无论标准/激进，有机构信号都加分（上限+20，不影响一票否决逻辑）
    try:
        from .smart_money import detect_unusual_options, smart_money_flow, detect_short_squeeze
        # 期权流向（v2）：bid/ask位置推断主动买方方向，比纯Vol/OI更可信但仍是弱信号
        # 加分权重从±10/8 收窄至±5/3（置信度折扣）
        uoa = detect_unusual_options(ticker)
        if uoa.get("ok"):
            if uoa.get("bias") == "bullish" and abs(uoa.get("net_call_flow", 0)) > 300:
                bonus += 5
                bonus_notes.append(
                    f"期权Call主动买入主导（净流向{uoa.get('net_call_flow',0):+.0f}手，+5，弱信号）"
                )
            elif uoa.get("bias") == "bearish" and direction == "LONG":
                bonus -= 3
                bonus_notes.append(
                    f"期权Put主动买入（净流向{uoa.get('net_put_flow',0):+.0f}手，-3，可能对冲）"
                )

        # 智能资金流向：收盘段净流入 +10
        smf = smart_money_flow(ticker)
        if smf.get("ok") and smf.get("smf_bias") == "bullish":
            bonus += 10
            bonus_notes.append(f"智能资金收盘净流入（+10）")
        elif smf.get("ok") and smf.get("smf_bias") == "bearish" and direction == "LONG":
            bonus -= 8
            bonus_notes.append(f"机构尾盘出货警告（-8）")

        # 空头挤压：高挤压潜力做多 +5
        if direction == "LONG":
            sqz = detect_short_squeeze(ticker)
            if sqz.get("ok") and sqz.get("squeeze_score", 0) >= 60:
                bonus += 5
                bonus_notes.append(f"逼空潜力高（空仓{sqz['short_float_pct']}%，+5）")
    except Exception:
        pass  # 智能资金模块失败不影响主流程

    # ── SEC Form 4 内部人买卖（真实可验证信号，替代失效的旧UOA）───────
    try:
        from .insider_tracker import insider_summary
        ins = insider_summary(ticker)
        if ins.get("ok") and ins.get("score_delta", 0) != 0:
            delta = ins["score_delta"]
            bonus += delta
            bonus_notes.append(
                f"内部人{'+' if delta > 0 else ''}{delta}（{ins.get('note', '')}，SEC Form 4）"
            )
    except Exception:
        pass  # 不影响主流程

    go_threshold   = 65 if aggressive_mode else 75
    wait_threshold = 45 if aggressive_mode else 55

    adjusted_score = min(100, score + bonus)

    if hard_fail:
        verdict = "ABORT"
        verdict_reason = f"以下检查未通过（一票否决）：{', '.join(hard_fail)}"
    elif adjusted_score >= go_threshold:
        verdict = "GO"
        verdict_reason = (f"{'激进模式' if aggressive_mode else '标准模式'}评分{adjusted_score}"
                          f"（门槛{go_threshold}）{'，激进加分：' + '、'.join(bonus_notes) if bonus_notes else ''}，可以入场")
    elif adjusted_score >= wait_threshold:
        verdict = "WAIT"
        verdict_reason = f"评分{adjusted_score}（需≥{go_threshold}），建议等待更好信号"
    else:
        verdict = "ABORT"
        verdict_reason = f"评分{adjusted_score}过低，保持空仓"

    # PDT触发时强制降级
    if pdt_trigger.get("trigger") and verdict == "GO":
        verdict = "WAIT"
        verdict_reason = (f"信号通过（分数{adjusted_score}），但PDT限制：{pdt_trigger['note']}。"
                          f"建议：{pdt_trigger.get('action','次日建仓转为摆动交易')}")

    # ── 操作计划（激进模式仓位计算） ─────────────────────
    entry_plan = None
    if verdict == "GO":
        # 激进模式：3%风险，单仓上限50%
        # 标准模式：1%风险，单仓上限25%
        if aggressive_mode:
            risk_pct    = 0.03
            max_pos_pct = 0.50
            rule_note   = "激进3%风险规则：$2k-$5k账户，单笔最大亏损不超过账户3%"
        elif portfolio < 10_000:
            risk_pct    = 0.015
            max_pos_pct = 0.25
            rule_note   = "小账户1.5%风险规则：单笔最大亏损不超过账户1.5%"
        else:
            risk_pct    = 0.01
            max_pos_pct = 0.25
            rule_note   = "1%风险规则：任何情况下单笔最大亏损不超过总资金1%"

        max_risk       = portfolio * risk_pct
        shares_by_risk = max(1, int(max_risk / stop_d)) if stop_d > 0 else 1
        shares_by_cap  = max(1, int(portfolio * max_pos_pct / price)) if price > 0 else 1
        shares         = min(shares_by_risk, shares_by_cap)
        pos_size       = shares * price
        actual_risk    = shares * stop_d

        # 期望值计算（使用配置常量，避免硬编码）
        win_rate  = EV_WIN_RATE
        rr        = EV_RR_RATIO
        ev        = win_rate * (rr * actual_risk) - (1 - win_rate) * actual_risk
        ev_pct    = ev / portfolio * 100

        stop_price   = round(price - stop_d if direction == "LONG" else price + stop_d, 2)
        target1      = round(price + atr_val * 2   if direction == "LONG" else price - atr_val * 2,   2)
        target2      = round(price + atr_val * 3.5 if direction == "LONG" else price - atr_val * 3.5, 2)

        entry_plan = {
            "direction":          direction,
            "mode":               "激进动量" if aggressive_mode else "标准",
            "entry_price":        round(price, 2),
            "stop_loss":          stop_price,
            "target_1":           target1,
            "target_2":           target2,
            "target_1_gain_pct":  round(abs(target1 - price) / price * 100, 1),
            "target_2_gain_pct":  round(abs(target2 - price) / price * 100, 1),
            "shares":             shares,
            "position_usd":       round(pos_size, 2),
            "position_pct":       round(pos_size / portfolio * 100, 1),
            "max_risk_usd":       round(actual_risk, 2),
            "max_risk_pct":       round(actual_risk / portfolio * 100, 2),
            "trade_type":         "摆动交易（PDT安全）" if not is_intraday else "日内交易",
            "expected_value_usd": round(ev, 2),
            "expected_value_pct": round(ev_pct, 2),
            "rule":               rule_note,
            "exit_rules": [
                f"止损：价格{'跌破' if direction=='LONG' else '涨破'} ${stop_price}，立即市价清仓，不等反弹",
                f"目标1（${target1}，+{round(abs(target1-price)/price*100,1)}%）到达：减仓50%，止损上移至成本价",
                f"目标2（${target2}，+{round(abs(target2-price)/price*100,1)}%）到达：清仓剩余50%",
                "时间止损：摆动持有超过7天仍未盈利，减半仓离场" if not is_intraday
                else "收盘前15分钟无论盈亏清仓（日内规则）",
                "绝不补仓亏损仓位，绝不移动止损扩大亏损",
            ],
        }

    return {
        "ticker":           ticker,
        "direction":        direction,
        "price":            round(price, 2),
        "verdict":          verdict,
        "score":            adjusted_score,
        "score_base":       score,
        "bonus":            bonus,
        "bonus_notes":      bonus_notes,
        "go_threshold":     go_threshold,
        "reason":           verdict_reason,
        "gates":            gates,
        "entry_plan":       entry_plan,
        "pdt_status":       pdt_status,
        "aggressive_mode":  aggressive_mode,
        "account_info": {
            "value":           portfolio,
            "type":            account_type,
            "day_trades_used": day_trades_used,
            "day_trades_left": pdt_status.get("day_trades_left"),
            "strategy_mode":   pdt_status.get("strategy_mode"),
        },
        "cold_rules": (
            [
                "激进模式核心规则（$2k-$5k账户）",
                "只做动量最强的股票（RS>85，VCP/杯柄突破）",
                "每笔最多亏3%账户：$2000账户 = 最多亏$60",
                "3:1盈亏比：每笔目标最少 $180，期望值 +$84",
                "摆动持有3-10天（绕开PDT），不做日内",
                "止损触发立即执行，不等、不挂、不拖",
            ] if aggressive_mode else [
                "永远不追价：挂单等价格来，价格不来就放弃",
                "止损是纪律：触发即执行，不等反弹，不平均",
                "1%风险上限：不管多确定，单笔风险不超1%",
                "开盘15分钟禁止入场：等待方向确认",
                "不持有隔夜（日内模式）：收盘前清仓" if is_intraday
                else "摆动模式：次日不可同天卖出（PDT），持有至目标或止损",
            ]
        ),
        "timestamp":  now.strftime("%Y-%m-%d %H:%M ET"),
    }


# ── 股票扫描器（批量筛选） ────────────────────────────────────

def scan_tickers(tickers: list, portfolio: float = 100_000,
                 direction: str = "LONG") -> dict:
    """
    批量扫描，按冷静模型分数排序
    用于「按策略选股」功能
    """
    results = []
    errors  = []

    for ticker in tickers[:20]:  # 单次最多20只，防止超时
        try:
            r = cold_decision(ticker, portfolio, direction)
            results.append({
                "ticker":  ticker,
                "verdict": r["verdict"],
                "score":   r["score"],
                "price":   r["price"],
                "gates_failed": [k for k, v in r.get("gates", {}).items()
                                 if v.get("pass") is False],
                "reason":  r["reason"],
            })
        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})

    go_list   = sorted([r for r in results if r["verdict"] == "GO"],
                       key=lambda x: x["score"], reverse=True)
    wait_list = sorted([r for r in results if r["verdict"] == "WAIT"],
                       key=lambda x: x["score"], reverse=True)
    abort_list = [r for r in results if r["verdict"] == "ABORT"]

    return {
        "direction":   direction,
        "total":       len(tickers),
        "go_count":    len(go_list),
        "wait_count":  len(wait_list),
        "abort_count": len(abort_list),
        "go":          go_list,
        "wait":        wait_list[:5],
        "abort":       abort_list[:5],
        "errors":      errors,
        "note":        "分数≥75=GO（标准）/≥65=GO（激进，$5k以下自动启用），<门槛或有红灯=ABORT",
    }


# ── 辅助函数 ─────────────────────────────────────────────────

def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs    = gain / loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def _calc_atr(hist: pd.DataFrame, period: int = 14) -> float:
    tr = pd.concat([
        hist["High"] - hist["Low"],
        (hist["High"] - hist["Close"].shift()).abs(),
        (hist["Low"]  - hist["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    val = float(tr.ewm(com=period - 1, adjust=False).mean().iloc[-1])
    return val if val == val else float(hist["Close"].iloc[-1]) * 0.015


def _calc_score(gates: dict, vix: float, aggressive_mode: bool = False) -> int:
    base = 100

    if aggressive_mode:
        # 激进模式权重：
        # - 趋势、RSI、成交量是核心（权重高）
        # - 时间窗口权重低（摆动交易时间不敏感）
        # - near_high 在激进模式已反向，不扣分
        # - VWAP 被跳过，不扣分
        deduct = {
            "time_window":   10,   # 摆动模式时间限制少（仅开盘/收盘5分钟屏蔽）
            "trend":         25,   # 核心：趋势方向是激进策略最重要的
            "rsi":           15,   # 核心：RSI区间已扩展，仍是重要信号
            "volume":        15,   # 核心：放量是突破有效性的关键
            "stop_distance": 20,   # 核心：止损太大/太小都危险
            "vwap":           0,   # 激进/摆动跳过，不扣分
            "near_high":      0,   # 激进模式近高是利好，不扣分
        }
    else:
        # 标准模式权重
        deduct = {
            "time_window":   20,   # 日内交易时间很重要
            "trend":         20,
            "rsi":           15,
            "volume":        10,
            "stop_distance": 15,
            "vwap":           8,   # 日内VWAP有参考价值（已修正为当日VWAP）
            "near_high":      5,   # 保守模式接近高点略有压力
        }

    for key, cost in deduct.items():
        g = gates.get(key, {})
        p = g.get("pass", True)
        if p is False:
            base -= cost
        elif p == "warn":
            base -= cost // 2
        # "skip" 不扣分

    # VIX 扣分（激进模式稍宽）
    if vix > 40:
        base -= 15
    elif vix > 28:
        base -= (5 if aggressive_mode else 10)

    # 板块轮动背景（非阻断，加减分）
    # rank/accel/heat 存储在 _sector_meta 避免污染 sector_rotation gate 结构
    sg      = gates.get("sector_rotation", {})
    sg_meta = gates.get("_sector_meta", {})
    sg_pass = sg.get("pass", True)
    rank    = sg_meta.get("rank")
    if sg_pass == "warn":
        base -= 8    # 逆风板块扣分
    elif sg_pass is True and isinstance(rank, int) and rank <= 3:
        base += SECTOR_TOP_BONUS    # 顺风板块加分
        if sg_meta.get("accel"):
            base += SECTOR_ACCEL_BONUS  # 加速流入额外加分

    return max(0, min(100, base))
