"""
模拟交易引擎（Paper Trading Engine）

功能：
  1. 执行模拟买入/卖出，持久化到 JSON 交易日志
  2. 真实成本模型：滑点 + 买卖价差
  3. Kelly Criterion 动态仓位建议（基于实际统计胜率）
  4. 绩效指标：Sharpe比率、Sortino比率、最大回撤、胜率
  5. 模拟盘 vs 真实盘 对比报告
  6. 连续亏损熔断（Wall Street 风控铁律）

Wall Street 核心纪律：
  - 没有经过模拟盘验证的策略，不允许用真钱
  - 最大回撤超过10%，停止所有新仓位
  - 连续亏损5笔，强制暂停交易1周
"""

import json
import os
import threading
import uuid
from datetime import datetime, date
import numpy as np
import yfinance as yf
import pytz

ET      = pytz.timezone("America/New_York")
_DATA   = os.path.join(os.path.dirname(__file__), "..", "data")
_LOG    = os.path.join(_DATA, "paper_trades.json")
_REAL   = os.path.join(_DATA, "real_trades.json")
_PT_LOCK = threading.Lock()  # 防止并发读写持仓文件

os.makedirs(_DATA, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 全局配置常量（修改参数在此处）
# ══════════════════════════════════════════════════════════════

# ── 滑点配置 ─────────────────────────────────────────────────
DEFAULT_SLIPPAGE_PCT        = 0.05     # 默认滑点（市价单，大盘股）
MAX_SLIPPAGE_PCT            = 1.0      # 最大允许滑点（拦截异常输入）
MAX_POSITION_PCT            = 0.50     # 单仓上限（账户净值50%）

# ── 持仓约束 ─────────────────────────────────────────────────
MAX_CONCURRENT_POSITIONS    = 2        # 最多同时持有2仓（$2k账户，3仓=150%仓位风险）
MAX_TOTAL_EXPOSURE_PCT      = 0.80     # 总仓位上限80%账户净值（留20%应急缓冲）

# ── 熔断器 ───────────────────────────────────────────────────
CB_LOSS_TRIGGER             = 5        # 连续亏损N笔触发熔断
CB_DRAWDOWN_TRIGGER         = -6.0     # 单日回撤超过6%触发熔断（原10%与最坏路径不自洽：5笔×止损可达-20%）

# ── 追踪止损 ─────────────────────────────────────────────────
TRAIL_STOP_DEFAULT          = 8.0      # 默认追踪止损%
TRAIL_STOP_MIN              = 0.1      # 最小追踪止损%（防误操作）
TRAIL_STOP_MAX              = 50.0     # 最大追踪止损%（防误操作）

# ── 统计门限 ─────────────────────────────────────────────────
SHARPE_MIN_TRADES           = 5        # Sharpe 最少需要N笔数据
KELLY_WARN_TRADES           = 30       # Kelly 建议≥30笔样本
RR_RATIO_FALLBACK           = 99.9     # 无亏损记录时的盈亏比占位值

# ── 精度 ─────────────────────────────────────────────────────
PRICE_DECIMALS              = 4        # 价格保留小数位
PNL_DECIMALS                = 2        # P&L 保留小数位

# ── 无风险利率（Sharpe 分子需减去 Rf）────────────────────────
RF_ANNUAL_PCT               = 4.5      # 美国短期国债年化率（%），每季度手动更新

# ══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# 交易日志 I/O
# ─────────────────────────────────────────────────────────────

def _load(path: str) -> dict:
    if not os.path.exists(path):
        return {"trades": [], "account": {}, "positions": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"trades": [], "account": {}, "positions": {}}


def _save(data: dict, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)  # 原子替换，防止崩溃后 JSON 截断归零


# ─────────────────────────────────────────────────────────────
# 初始化账户
# ─────────────────────────────────────────────────────────────

def init_account(account_value: float, mode: str = "paper",
                  label: str = "默认账户") -> dict:
    """
    初始化模拟盘或真实盘账户。

    参数：
      mode — "paper"（模拟盘）或 "real"（真实盘记录）
    """
    path = _LOG if mode == "paper" else _REAL
    data = _load(path)
    data["account"] = {
        "initial_value": account_value,
        "current_value": account_value,
        "cash":          account_value,
        "peak_value":    account_value,
        "mode":          mode,
        "label":         label,
        "created_at":    str(datetime.now(ET)),
        "circuit_breaker": {
            "active":           False,
            "consecutive_losses": 0,
            "max_drawdown_pct": 0.0,
        },
    }
    _save(data, path)
    return {"ok": True, "account": data["account"]}


# ─────────────────────────────────────────────────────────────
# 开仓
# ─────────────────────────────────────────────────────────────

def open_position(ticker: str, shares: int, entry_price: float,
                   stop_loss: float, target: float,
                   strategy: str = "", mode: str = "paper",
                   slippage_pct: float = 0.05, **kwargs) -> dict:
    """
    开仓记录。

    真实成本模拟：
      - 滑点（slippage）：市价单平均滑点 0.05%（大盘股）至 0.3%（小盘股）
      - 买卖价差：已含在 slippage_pct 中
    """
    if entry_price <= 0 or not np.isfinite(entry_price):
        return {"ok": False, "error": f"入场价无效：{entry_price}"}
    if not isinstance(shares, (int, float)) or shares <= 0:
        return {"ok": False, "error": f"手数无效：{shares}"}
    slippage_pct = max(0.0, min(MAX_SLIPPAGE_PCT, slippage_pct))

    path = _LOG if mode == "paper" else _REAL
    cold_result   = kwargs.get("cold_result")
    debate_result = kwargs.get("debate_result")

    with _PT_LOCK:
        data = _load(path)
        acct = data.get("account", {})

        if acct.get("circuit_breaker", {}).get("active"):
            return {"ok": False, "error": "熔断器激活！当前禁止开新仓位",
                    "reason": f"连续{CB_LOSS_TRIGGER}笔亏损或回撤超过{abs(CB_DRAWDOWN_TRIGGER)}%，需冷静期1周"}

        exec_price   = entry_price * (1 + slippage_pct / 100)
        total_cost   = exec_price * shares
        available    = acct.get("cash", 0)

        acct_value    = acct.get("current_value") or available
        max_single    = acct_value * MAX_POSITION_PCT
        if total_cost > max_single:
            return {"ok": False,
                    "error": f"超出单仓上限（账户{MAX_POSITION_PCT*100:.0f}%=${max_single:.2f}），请减少股数"}

        open_positions = {k: v for k, v in data.get("positions", {}).items()
                          if v.get("status") == "open"}

        # 防摊平：同标的已有持仓时禁止加仓（对应用户"越补越深"反模式）
        existing_tickers = {v["ticker"] for v in open_positions.values()}
        if ticker in existing_tickers:
            return {"ok": False, "error": f"{ticker} 已有持仓，禁止同向加仓（防摊平保护）"}

        if len(open_positions) >= MAX_CONCURRENT_POSITIONS:
            return {"ok": False,
                    "error": (f"已有 {len(open_positions)} 个持仓，达到上限 "
                              f"{MAX_CONCURRENT_POSITIONS}（风控约束），等待现有仓位平仓后再入场")}

        current_exposure = sum(p.get("entry_price", 0) * p.get("shares", 0)
                               for p in open_positions.values())
        if acct_value > 0 and (current_exposure + total_cost) / acct_value > MAX_TOTAL_EXPOSURE_PCT:
            used_pct = current_exposure / acct_value * 100
            return {"ok": False,
                    "error": (f"总仓位将达 {(current_exposure+total_cost)/acct_value*100:.0f}%"
                              f"（已用 {used_pct:.0f}%，上限 {MAX_TOTAL_EXPOSURE_PCT*100:.0f}%），拒绝新仓")}

        if total_cost > available:
            return {"ok": False, "error": f"资金不足：需${total_cost:.2f}，可用${available:.2f}"}

        trade_id = str(uuid.uuid4())[:8]
        now      = str(datetime.now(ET))

        position = {
            "id":           trade_id,
            "ticker":       ticker.upper(),
            "shares":       shares,
            "entry_price":  round(exec_price, 4),
            "entry_ideal":  round(entry_price, 4),
            "slippage_cost": round((exec_price - entry_price) * shares, 4),
            "stop_loss":    round(stop_loss, 4),
            "target":       round(target, 4),
            "strategy":     strategy,
            "status":       "open",
            "mode":         mode,
            "opened_at":    now,
            "closed_at":    None,
            "exit_price":   None,
            "pnl":          None,
            "pnl_pct":      None,
            "exit_reason":  None,
        }

        data.setdefault("positions", {})[trade_id] = position
        acct["cash"] = round(available - total_cost, 4)
        data["account"] = acct
        data.setdefault("trades", []).append({
            "event": "open", "id": trade_id, "ticker": ticker,
            "shares": shares, "price": exec_price, "at": now,
        })
        _save(data, path)
        ret = {
            "ok":         True,
            "trade_id":   trade_id,
            "exec_price": round(exec_price, 4),
            "slippage":   round((exec_price - entry_price) * shares, 2),
            "total_cost": round(total_cost, 2),
            "cash_left":  round(acct["cash"], 2),
            "note":       f"{'模拟' if mode=='paper' else '真实'}开仓成功",
        }

    # 锁外记录入场信号快照，避免持锁期间做文件IO
    if cold_result:
        try:
            from .feedback import record_entry_signals
            record_entry_signals(trade_id, ticker, cold_result, debate_result)
        except Exception:
            pass

    return ret


# ─────────────────────────────────────────────────────────────
# 平仓
# ─────────────────────────────────────────────────────────────

def close_position(trade_id: str, exit_price: float,
                    exit_reason: str = "手动平仓",
                    mode: str = "paper",
                    slippage_pct: float = 0.05) -> dict:
    """平仓并记录 P&L，更新熔断器状态。"""
    path = _LOG if mode == "paper" else _REAL

    # 先在锁外读一次持仓，确认存在后再抓网络价格（避免持锁期间网络阻塞）
    with _PT_LOCK:
        data_pre = _load(path)
    pos_pre = data_pre.get("positions", {}).get(trade_id)
    if not pos_pre:
        return {"ok": False, "error": f"找不到仓位 {trade_id}"}
    if pos_pre.get("status") != "open":
        return {"ok": False, "error": "仓位已关闭"}

    # 锁外抓剩余持仓实时价格（网络IO不持锁）
    remaining_tickers = list(set(
        p["ticker"] for p in data_pre.get("positions", {}).values()
        if p.get("status") == "open" and p.get("id") != trade_id
    ))
    live_prices = {}
    try:
        for tk in remaining_tickers:
            h = yf.Ticker(tk).history(period="1d")
            if not h.empty:
                live_prices[tk] = float(h["Close"].iloc[-1])
    except Exception:
        pass

    with _PT_LOCK:
        data = _load(path)
        pos = data.get("positions", {}).get(trade_id)
        if not pos or pos.get("status") != "open":
            return {"ok": False, "error": "仓位状态已改变，平仓终止"}

        exec_price = exit_price * (1 - slippage_pct / 100)
        shares     = pos["shares"]
        entry      = pos["entry_price"]
        pnl        = (exec_price - entry) * shares
        pnl_pct    = (exec_price - entry) / entry * 100
        proceeds   = exec_price * shares

        pos.update({
            "status":      "closed",
            "closed_at":   str(datetime.now(ET)),
            "exit_price":  round(exec_price, 4),
            "exit_ideal":  round(exit_price, 4),
            "exit_slippage": round((exit_price - exec_price) * shares, 4),
            "pnl":         round(pnl, 4),
            "pnl_pct":     round(pnl_pct, 2),
            "exit_reason": exit_reason,
        })

        acct = data.get("account", {})
        acct["cash"] = round(acct.get("cash", 0) + proceeds, 4)

        cb = acct.setdefault("circuit_breaker", {"active": False, "consecutive_losses": 0})
        if pnl < 0:
            cb["consecutive_losses"] = cb.get("consecutive_losses", 0) + 1
        else:
            cb["consecutive_losses"] = 0
        if cb["consecutive_losses"] >= CB_LOSS_TRIGGER:
            cb["active"] = True
            cb["reason"] = f"连续亏损{CB_LOSS_TRIGGER}笔，强制暂停1周"

        remaining_open = {k: p for k, p in data["positions"].items()
                          if p.get("status") == "open" and k != trade_id}
        total_open = sum(
            live_prices.get(p["ticker"], p["entry_price"]) * p["shares"]
            for p in remaining_open.values()
        )
        acct["current_value"] = round(acct["cash"] + total_open, 4)
        acct["peak_value"]    = max(acct.get("peak_value", acct["current_value"]),
                                    acct["current_value"])
        peak = acct["peak_value"]
        dd_pct = (acct["current_value"] - peak) / peak * 100 if peak > 0 else 0.0
        cb["max_drawdown_pct"] = round(min(cb.get("max_drawdown_pct", 0), dd_pct), 2)
        if dd_pct < CB_DRAWDOWN_TRIGGER:
            cb["active"] = True
            cb["reason"] = f"最大回撤达{abs(dd_pct):.1f}%，超过{abs(CB_DRAWDOWN_TRIGGER):.0f}%熔断线"

        data["positions"][trade_id] = pos
        data["account"] = acct
        data.setdefault("trades", []).append({
            "event": "close", "id": trade_id, "ticker": pos["ticker"],
            "pnl": pnl, "pnl_pct": pnl_pct, "reason": exit_reason,
            "at": str(datetime.now(ET)),
        })
        _save(data, path)
        ret = {
            "ok":           True,
            "trade_id":     trade_id,
            "pnl":          round(pnl, 2),
            "pnl_pct":      round(pnl_pct, 2),
            "result":       "盈利" if pnl > 0 else "亏损",
            "exec_price":   round(exec_price, 4),
            "proceeds":     round(proceeds, 2),
            "cash_after":   round(acct["cash"], 2),
            "account_value": acct["current_value"],
            "circuit_breaker": cb,
        }

    # 锁外记录反馈学习结果
    try:
        from .feedback import record_exit_result
        opened_at = pos.get("opened_at", "")
        hold_days = 0
        if opened_at:
            open_dt  = datetime.fromisoformat(str(opened_at)[:19])
            hold_days= max(0, (datetime.now() - open_dt).days)
        record_exit_result(trade_id, pnl_pct, hold_days, exit_reason)
    except Exception:
        pass

    return ret


# ─────────────────────────────────────────────────────────────
# 追踪止损（Trailing Stop）
# ─────────────────────────────────────────────────────────────

def update_trailing_stop(trade_id: str, current_price: float,
                          trail_pct: float = 8.0, mode: str = "paper") -> dict:
    """
    追踪止损：股价每创新高，止损跟随上移（锁定利润）。

    trail_pct=8% → 止损始终维持在最高价的 92%。
    止损只升不降——一旦上移就不会因为回调而下移。
    """
    trail_pct = max(TRAIL_STOP_MIN, min(TRAIL_STOP_MAX, trail_pct))
    path = _LOG if mode == "paper" else _REAL
    data = _load(path)
    pos  = data.get("positions", {}).get(trade_id)
    if not pos or pos.get("status") != "open":
        return {"ok": False, "error": "仓位不存在或已关闭"}

    highest_seen = pos.get("highest_price", pos["entry_price"])
    if current_price > highest_seen:
        highest_seen = current_price
        pos["highest_price"] = round(highest_seen, 4)

    new_stop = round(highest_seen * (1 - trail_pct / 100), 4)
    old_stop = pos["stop_loss"]

    if new_stop > old_stop:
        pos["stop_loss"]     = new_stop
        pos["trailing_stop"] = True
        data["positions"][trade_id] = pos
        _save(data, path)
        return {
            "ok":       True,
            "updated":  True,
            "old_stop": old_stop,
            "new_stop": new_stop,
            "highest":  highest_seen,
            "note": f"追踪止损上移至${new_stop:.2f}（最高价${highest_seen:.2f}×{100-trail_pct:.0f}%）",
        }
    return {"ok": True, "updated": False, "current_stop": old_stop,
            "note": "止损无需上移"}


# ─────────────────────────────────────────────────────────────
# 盯市（Mark to Market）
# ─────────────────────────────────────────────────────────────

def mark_to_market(mode: str = "paper") -> dict:
    """获取所有持仓的当前市值，更新账户总值。"""
    path = _LOG if mode == "paper" else _REAL
    data = _load(path)
    positions = data.get("positions", {})
    open_pos  = {k: v for k, v in positions.items() if v.get("status") == "open"}

    if not open_pos:
        acct = data.get("account", {})
        return {"ok": True, "total_value": acct.get("current_value", 0),
                "open_positions": [], "cash": acct.get("cash", 0)}

    tickers = list(set(p["ticker"] for p in open_pos.values()))
    prices  = {}
    try:
        for tk in tickers:
            hist = yf.Ticker(tk).history(period="1d")
            if not hist.empty:
                prices[tk] = float(hist["Close"].iloc[-1])
    except Exception:
        pass

    open_summary = []
    total_pos_val = 0.0
    for tid, pos in open_pos.items():
        cur_price = prices.get(pos["ticker"], pos["entry_price"])
        mkt_val   = cur_price * pos["shares"]
        unreal_pnl = (cur_price - pos["entry_price"]) * pos["shares"]
        unreal_pct = (cur_price - pos["entry_price"]) / pos["entry_price"] * 100
        total_pos_val += mkt_val

        # 检查是否触及止损或目标
        alert = None
        alert_type = None
        if cur_price <= pos["stop_loss"]:
            alert = f"⚠️ 触及止损价 ${pos['stop_loss']:.2f}！应立即执行止损"
            alert_type = "stop_loss"
        elif cur_price >= pos["target"]:
            alert = f"✅ 达到目标价 ${pos['target']:.2f}！可考虑减仓"
            alert_type = "target"

        open_summary.append({
            "id":           tid,
            "ticker":       pos["ticker"],
            "shares":       pos["shares"],
            "entry_price":  pos["entry_price"],
            "current_price": round(cur_price, 2),
            "market_value": round(mkt_val, 2),
            "unrealized_pnl": round(unreal_pnl, 2),
            "unrealized_pct": round(unreal_pct, 2),
            "stop_loss":    pos["stop_loss"],
            "target":       pos["target"],
            "alert":        alert,
            "alert_type":   alert_type,
        })

    acct = data.get("account", {})
    cash = acct.get("cash", 0)
    total_val = round(cash + total_pos_val, 2)

    # 更新历史峰值并持久化，确保两次平仓之间的浮盈高点不丢失
    if total_val > acct.get("peak_value", 0):
        acct["peak_value"] = total_val
        data["account"] = acct
        _save(data, path)

    peak  = acct.get("peak_value", total_val)
    dd_pct = (total_val - peak) / peak * 100 if peak > 0 else 0.0

    return {
        "ok":             True,
        "total_value":    total_val,
        "cash":           round(cash, 2),
        "open_pos_value": round(total_pos_val, 2),
        "drawdown_pct":   round(dd_pct, 2),
        "open_positions": open_summary,
        "alerts":         [p["alert"] for p in open_summary if p.get("alert")],
    }


# ─────────────────────────────────────────────────────────────
# 绩效报告（Sharpe / Sortino / MaxDD / Kelly更新）
# ─────────────────────────────────────────────────────────────

def performance_report(mode: str = "paper") -> dict:
    """
    生成完整绩效报告。

    指标说明：
      Sharpe  > 1.0  可接受；> 2.0 优秀；> 3.0 顶尖
      Sortino 比Sharpe更严格（只惩罚下行波动）
      Max DD  < 10%  良好；< 5%  顶尖
      Kelly   用实际胜率计算最优仓位比例
    """
    path  = _LOG if mode == "paper" else _REAL
    data  = _load(path)
    acct  = data.get("account", {})
    trades = [t for t in data.get("trades", []) if t.get("event") == "close"]

    if not trades:
        return {"ok": True, "note": "暂无已关闭交易，无法计算绩效"}

    pnls     = [t["pnl"] for t in trades]
    pnl_pcts = [t["pnl_pct"] for t in trades if "pnl_pct" in t]

    total    = len(pnls)
    win_rate = sum(1 for p in pnls if p > 0) / total if total > 0 else 0

    # Kelly 盈亏比用 pnl_pct（百分比），避免大仓位主导均值
    if pnl_pcts:
        win_pcts  = [p for p in pnl_pcts if p > 0]
        loss_pcts = [p for p in pnl_pcts if p < 0]
        avg_win   = float(np.mean(win_pcts))  if win_pcts  else 0
        avg_loss  = float(np.mean(loss_pcts)) if loss_pcts else 0
    else:
        wins     = [p for p in pnls if p > 0]
        losses   = [p for p in pnls if p < 0]
        avg_win  = float(np.mean(wins))  if wins   else 0
        avg_loss = float(np.mean(losses)) if losses else 0

    if avg_loss != 0:
        rr_ratio = abs(avg_win / avg_loss)
    elif avg_win > 0:
        rr_ratio = RR_RATIO_FALLBACK
    else:
        rr_ratio = 0.0

    # Kelly Criterion（用真实统计数据）
    kelly_f = 0.0
    if rr_ratio > 0:
        kelly_f = win_rate - (1 - win_rate) / rr_ratio
    half_kelly    = max(0, kelly_f / 2)
    kelly_usd     = acct.get("current_value", 2000) * half_kelly
    negative_edge = kelly_f < 0  # 期望值为负，不应交易

    # Sharpe / Sortino（按实际年化交易频率，非错误的√252日化）
    if len(pnl_pcts) >= 5:
        pnl_arr   = np.array(pnl_pcts)
        mean_r    = np.mean(pnl_arr)
        std_r     = np.std(pnl_arr, ddof=1)
        # 摆动交易年化系数：基于实际第一笔到现在的时间跨度（不是从1月1日起算）
        trade_times = [t.get("at", "") for t in data.get("trades", []) if t.get("event") == "close" and t.get("at")]
        try:
            first_dt   = datetime.fromisoformat(str(sorted(trade_times)[0])[:19])
            span_days  = max((datetime.now() - first_dt).days, 30)
            trades_per_year = total / (span_days / 365)
        except Exception:
            trades_per_year = max(total, 1) * 4  # 无法确定时按年化4×估算
        ann_factor = np.sqrt(trades_per_year)
        # Sharpe：超额收益（减去无风险利率）/ 波动率
        rf_per_trade = RF_ANNUAL_PCT / max(trades_per_year, 1)  # 无风险收益率（每笔，%）
        sharpe  = float((mean_r - rf_per_trade) / std_r * ann_factor) if std_r > 0 else 0
        # Sortino：分母 = 下行半偏差（全部N笔的负偏差均方根，非仅亏损笔的std）
        downside_sq = float(np.mean(np.minimum(pnl_arr, 0) ** 2))
        down_std = float(np.sqrt(downside_sq)) if downside_sq > 0 else std_r
        sortino  = float((mean_r - rf_per_trade) / down_std * ann_factor) if down_std > 0 else 0
    else:
        sharpe = sortino = 0.0

    # Max Drawdown（账户级别）
    init_v    = acct.get("initial_value", 1)
    cur_v     = acct.get("current_value", init_v)
    peak_v    = acct.get("peak_value", init_v)
    total_pnl = round(cur_v - init_v, 2)
    total_pct = round(total_pnl / init_v * 100, 2) if init_v else 0
    max_dd    = acct.get("circuit_breaker", {}).get("max_drawdown_pct", 0)
    consec_l  = acct.get("circuit_breaker", {}).get("consecutive_losses", 0)

    # Expected Value
    ev_per_trade = win_rate * avg_win + (1 - win_rate) * avg_loss

    return {
        "ok":        True,
        "mode":      mode,
        "summary": {
            "total_trades":    total,
            "win_rate":        round(win_rate * 100, 1),
            "avg_win_usd":     round(avg_win, 2),
            "avg_loss_usd":    round(avg_loss, 2),
            "risk_reward":     round(rr_ratio, 2),
            "ev_per_trade":    round(ev_per_trade, 2),
            "total_pnl":       total_pnl,
            "total_pnl_pct":   total_pct,
            "current_value":   cur_v,
        },
        "risk_metrics": {
            "sharpe_ratio":    round(sharpe, 2),
            "sortino_ratio":   round(sortino, 2),
            "max_drawdown_pct": max_dd,
            "consecutive_losses": consec_l,
            "circuit_breaker": acct.get("circuit_breaker", {}).get("active", False),
        },
        "kelly": {
            "actual_win_rate":  round(win_rate * 100, 1),
            "actual_rr_ratio":  round(rr_ratio, 2),
            "full_kelly_pct":   round(kelly_f * 100, 1),
            "half_kelly_pct":   round(half_kelly * 100, 1),
            "half_kelly_usd":   round(kelly_usd, 2),
            "negative_edge": negative_edge,
            "note": (
                "🚨 策略期望值为负（Kelly<0），当前参数下不应交易，需重新审查入场条件"
                if negative_edge else
                f"基于{total}笔数据的Kelly建议：${kelly_usd:.0f}（账户{half_kelly*100:.0f}%）"
                if total >= 30 else
                f"样本量{total}笔（建议≥30笔），当前Kelly仅供参考，误差较大"
            ),
        },
        "grades": {
            "sharpe":  ("A" if sharpe > 2 else "B" if sharpe > 1 else "C" if sharpe > 0.5 else "D"),
            "max_dd":  ("A" if max_dd > -5 else "B" if max_dd > -10 else "C" if max_dd > -20 else "D"),
            "win_rate": ("A" if win_rate > 0.65 else "B" if win_rate > 0.55 else "C" if win_rate > 0.45 else "D"),
        },
    }


# ─────────────────────────────────────────────────────────────
# 模拟盘 vs 真实盘对比
# ─────────────────────────────────────────────────────────────

def compare_paper_vs_real() -> dict:
    """对比模拟盘和真实盘绩效，量化'执行摩擦'。"""
    paper_perf = performance_report("paper")
    real_perf  = performance_report("real")

    if (not paper_perf.get("ok") or not real_perf.get("ok")
            or "summary" not in paper_perf or "summary" not in real_perf):
        return {"ok": True, "note": "需要同时有已关闭交易才能对比模拟盘与真实盘"}

    paper_pnl = paper_perf.get("summary", {}).get("total_pnl_pct", 0)
    real_pnl  = real_perf.get("summary",  {}).get("total_pnl_pct", 0)
    gap       = paper_pnl - real_pnl

    friction_analysis = []
    if gap > 5:
        friction_analysis.append("🔴 执行摩擦严重：真实盘比模拟盘少赚5%+，检查是否有追价、滑点、情绪化操作")
    elif gap > 2:
        friction_analysis.append("🟡 轻微执行差距：可能有滑点或情绪化修改止损")
    elif abs(gap) <= 2:
        friction_analysis.append("🟢 执行纪律优秀：模拟盘与真实盘表现接近，策略稳健")
    else:
        friction_analysis.append("真实盘表现优于模拟盘，检查模拟盘参数是否过于保守")

    return {
        "ok": True,
        "paper": {"pnl_pct": paper_pnl, "trades": paper_perf.get("summary", {}).get("total_trades", 0)},
        "real":  {"pnl_pct": real_pnl,  "trades": real_perf.get("summary",  {}).get("total_trades", 0)},
        "gap_pct": round(gap, 2),
        "friction_analysis": friction_analysis,
        "paper_kelly": paper_perf.get("kelly", {}),
        "real_kelly":  real_perf.get("kelly", {}),
    }


# ─────────────────────────────────────────────────────────────
# 重置熔断器（审查后手动解除）
# ─────────────────────────────────────────────────────────────

def reset_circuit_breaker(mode: str = "paper", confirm: bool = False) -> dict:
    if not confirm:
        return {"ok": False, "error": "需要 confirm=True 才能重置熔断器。请先复盘亏损原因再解除。"}
    path = _LOG if mode == "paper" else _REAL
    data = _load(path)
    cb   = data.get("account", {}).get("circuit_breaker", {})
    cb["active"]             = False
    cb["consecutive_losses"] = 0
    data["account"]["circuit_breaker"] = cb
    _save(data, path)
    return {"ok": True, "note": "熔断器已重置。请确保已完成交易复盘再开始新仓位。"}


# ─────────────────────────────────────────────────────────────
# 列出所有持仓和历史
# ─────────────────────────────────────────────────────────────

def list_positions(mode: str = "paper") -> dict:
    path = _LOG if mode == "paper" else _REAL
    data = _load(path)
    pos  = data.get("positions", {})
    return {
        "ok":      True,
        "mode":    mode,
        "open":    [v for v in pos.values() if v.get("status") == "open"],
        "closed":  [v for v in pos.values() if v.get("status") == "closed"],
        "account": data.get("account", {}),
    }


# ─────────────────────────────────────────────────────────────
# 执行偏差日志（P0-C：关闭信号→实际执行的黑洞监控）
# ─────────────────────────────────────────────────────────────

_EXEC_LOG = os.path.join(_DATA, "execution_log.json")


def log_execution(ticker: str, signal_price: float, actual_price: float,
                  signal_time: str, action: str = "entered",
                  note: str = "") -> dict:
    """
    记录信号价格 vs 实际执行价格，量化执行摩擦。

    action: "entered"（已执行）| "skipped"（信号超出限价，跳过）| "manual_override"（手动改单）
    """
    try:
        with open(_EXEC_LOG, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"logs": []}

    deviation_pct = ((actual_price - signal_price) / signal_price * 100
                     if signal_price > 0 else 0)
    data["logs"].append({
        "at":            str(datetime.now(ET)),
        "ticker":        ticker.upper(),
        "signal_price":  round(signal_price, 4),
        "actual_price":  round(actual_price, 4),
        "deviation_pct": round(deviation_pct, 3),
        "action":        action,
        "signal_time":   signal_time,
        "note":          note,
    })
    # 原子写入：防止进程崩溃时 execution_log 截断
    _tmp = _EXEC_LOG + ".tmp"
    with open(_tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(_tmp, _EXEC_LOG)
    return {"ok": True, "deviation_pct": round(deviation_pct, 3)}


def execution_deviation_report() -> dict:
    """汇总执行偏差统计：量化信号→实际下单的系统性摩擦。"""
    if not os.path.exists(_EXEC_LOG):
        return {"ok": True, "note": "暂无执行记录。请在每次下单后调用 log_execution()。"}
    with open(_EXEC_LOG, "r", encoding="utf-8") as f:
        data = json.load(f)
    logs = data.get("logs", [])
    if not logs:
        return {"ok": True, "note": "执行日志为空"}

    entered  = [l for l in logs if l["action"] == "entered"]
    skipped  = [l for l in logs if l["action"] == "skipped"]
    override = [l for l in logs if l["action"] == "manual_override"]
    deviations = [l["deviation_pct"] for l in entered]
    avg_dev  = float(np.mean(deviations)) if deviations else 0
    return {
        "ok":            True,
        "total_signals": len(logs),
        "entered":       len(entered),
        "skipped":       len(skipped),
        "manual_override": len(override),
        "avg_deviation_pct": round(avg_dev, 3),
        "max_deviation_pct": round(max(deviations), 3) if deviations else 0,
        "execution_rate": f"{len(entered)/len(logs)*100:.0f}%" if logs else "0%",
        "note": (f"平均入场价偏差 {avg_dev:+.2f}%（正数=追高，负数=抄底）。"
                 f"跳过 {len(skipped)} 次信号（限价纪律）。"
                 f"手动改单 {len(override)} 次（待分析原因）。"),
    }
