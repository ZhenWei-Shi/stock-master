"""
PDT（Pattern Day Trader）规则保护模块

美国法规：账户净值 < $25,000 的保证金账户，
  每 5 个滚动交易日内最多执行 3 笔"日内交易"
  （同一交易日内买入并卖出同一证券 = 1 笔日内交易）

违规后果：
  第一次警告 → 账户标记为 PDT
  持续违规  → 账户被限制为纯现金交易，90天内无法使用保证金

现金账户例外：纯现金账户（非保证金）不受 PDT 约束，
  但受"T+2结算"限制（卖出后2天内资金才到账）

本模块功能：
  1. 检查本周已用 PDT 次数（用户自行输入或从文件读取）
  2. 判断某笔计划交易是否会触发 PDT 警告
  3. 根据账户规模推荐正确的交易策略（日内 vs 摆动）
  4. 提供摆动交易时间框架建议（绕过 PDT 的合法方式）
"""

import json
import os
import threading
from datetime import datetime, date, timedelta
import pytz

ET   = pytz.timezone("America/New_York")
_DB  = os.path.join(os.path.dirname(__file__), "..", "data", "pdt_log.json")
_PDT_LOCK = threading.Lock()

PDT_THRESHOLD   = 25_000   # 账户净值门槛（美元）
PDT_WEEK_LIMIT  = 3        # 5个交易日内最多日内交易次数


# ─────────────────────────────────────────────────────────────
# PDT 日志持久化（5日滚动窗口）
# ─────────────────────────────────────────────────────────────

def _load_pdt_log() -> dict:
    if not os.path.exists(_DB):
        return {}
    try:
        with open(_DB, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_pdt_log(data: dict):
    os.makedirs(os.path.dirname(_DB), exist_ok=True)
    try:
        tmp = _DB + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, _DB)
    except OSError:
        pass


def record_day_trade(ticker: str):
    """平仓时如果当日开平（日内交易），调用此函数持久化记录"""
    with _PDT_LOCK:
        data = _load_pdt_log()
        today = str(date.today())
        data.setdefault(today, []).append({
            "ticker": ticker.upper(),
            "at": datetime.now(ET).isoformat(),
        })
        _save_pdt_log(data)


def get_rolling_day_trades() -> int:
    """统计过去5个交易日（近似跳过周末）内的日内交易次数（真实PDT滚动窗口）"""
    data = _load_pdt_log()
    today = date.today()
    count = 0
    trading_days = 0
    for offset in range(14):  # 最多回溯14日历日，取5个交易日
        d = today - timedelta(days=offset)
        if d.weekday() < 5:  # 跳过周末
            count += len(data.get(str(d), []))
            trading_days += 1
            if trading_days >= 5:
                break
    return count


# ─────────────────────────────────────────────────────────────
# 主入口：PDT 风险评估
# ─────────────────────────────────────────────────────────────

def check_pdt_risk(account_value: float, account_type: str = "margin",
                   day_trades_used: int = 0) -> dict:
    """
    评估当前账户的 PDT 风险状态。

    参数：
      account_value   — 账户当前净值（美元）
      account_type    — "margin"（保证金）或 "cash"（现金账户）
      day_trades_used — 本周已用日内交易次数（用户自行记录）

    返回：
      status          — SAFE / WARNING / DANGER / EXEMPT
      day_trades_left — 本周剩余日内交易次数
      strategy_mode   — 推荐策略模式
      recommendations — 针对性建议
    """
    # 现金账户不受 PDT 约束
    if account_type == "cash":
        return {
            "status":           "EXEMPT",
            "reason":           "现金账户不受 PDT 约束（但受 T+2 资金结算限制）",
            "day_trades_left":  999,
            "strategy_mode":    "swing_or_intraday",
            "recommendations":  [
                "现金账户可以无限次日内交易，但资金需等待 T+2 结算",
                "卖出后当天资金不可再用，需等2个交易日",
                "警惕'善意违规'（Good Faith Violation）：用未结算资金买入后卖出",
                "适合摆动交易（3-10天持有），避免资金周转问题",
            ],
            "t2_warning": True,
        }

    # 账户净值超过门槛，完全豁免
    if account_value >= PDT_THRESHOLD:
        return {
            "status":           "EXEMPT",
            "reason":           f"账户净值 ${account_value:,.0f} ≥ $25,000，不受 PDT 限制",
            "day_trades_left":  999,
            "account_value":    account_value,
            "strategy_mode":    "all_strategies_available",
            "recommendations":  [
                "账户净值超过 PDT 门槛，所有策略均可使用",
                "建议继续保持净值 > $25,000，避免跌破后被标记",
                "跌破 $25,000 当日即受 PDT 约束",
            ],
        }

    # 受 PDT 约束的账户
    remaining = max(0, PDT_WEEK_LIMIT - day_trades_used)
    pct_used  = day_trades_used / PDT_WEEK_LIMIT * 100

    if day_trades_used >= PDT_WEEK_LIMIT:
        status = "DANGER"
        reason = f"本周日内交易已用完 {day_trades_used}/{PDT_WEEK_LIMIT}，再交易将触发 PDT 违规"
    elif day_trades_used == PDT_WEEK_LIMIT - 1:
        status = "WARNING"
        reason = f"本周仅剩 {remaining} 次日内交易机会，谨慎使用"
    else:
        status = "SAFE"
        reason = f"本周已用 {day_trades_used}/{PDT_WEEK_LIMIT} 次，剩余 {remaining} 次"

    # 根据账户大小推荐策略
    if account_value < 5_000:
        strategy_mode = "swing_only"
        mode_desc     = "纯摆动模式（3-15天持有），避免任何日内操作"
    elif account_value < 10_000:
        strategy_mode = "swing_preferred"
        mode_desc     = "以摆动为主，偶尔使用保留的日内交易次数"
    else:
        strategy_mode = "swing_with_selective_intraday"
        mode_desc     = "摆动为主，每周精选 1-2 次日内交易机会"

    return {
        "status":              status,
        "reason":              reason,
        "account_value":       account_value,
        "pdt_threshold":       PDT_THRESHOLD,
        "day_trades_used":     day_trades_used,
        "day_trades_limit":    PDT_WEEK_LIMIT,
        "day_trades_left":     remaining,
        "pct_used":            round(pct_used, 0),
        "strategy_mode":       strategy_mode,
        "strategy_mode_desc":  mode_desc,
        "account_type":        account_type,
        "recommendations":     _get_recommendations(account_value, remaining, status),
        "swing_timeframes":    _swing_timeframes(account_value),
        "pdt_explained":       (
            "PDT Rule：账户净值 < $25,000 的保证金账户，"
            "5个交易日内同一证券当天买卖（含期权）不超过3次。"
            "违规后账户将被标记，持续违规则限制为现金账户90天。"
        ),
    }


def will_trigger_pdt(day_trades_used: int, account_value: float,
                      is_intraday: bool, account_type: str = "margin") -> dict:
    """
    判断某笔计划交易是否会触发 PDT 警告。
    在 cold_model 或 quant_strategy 调用前检查。
    """
    if account_type == "cash" or account_value >= PDT_THRESHOLD:
        return {"trigger": False, "safe": True, "note": "不受 PDT 约束"}

    if not is_intraday:
        return {"trigger": False, "safe": True, "note": "摆动交易不计入 PDT"}

    if day_trades_used >= PDT_WEEK_LIMIT:
        return {
            "trigger": True,
            "safe":    False,
            "note":    f"⚠️ 已用完本周 {PDT_WEEK_LIMIT} 次！此交易将触发 PDT 违规",
            "action":  "改为明天建仓并持有过夜（转换为摆动交易）",
        }
    elif day_trades_used == PDT_WEEK_LIMIT - 1:
        return {
            "trigger": False,
            "safe":    "warn",
            "note":    f"本周最后 1 次日内交易机会，请确认值得使用",
            "action":  "建议留给更高质量的信号，或改用摆动策略",
        }
    return {
        "trigger": False,
        "safe":    True,
        "note":    f"日内交易机会充足（本周已用{day_trades_used}/{PDT_WEEK_LIMIT}）",
    }


# ─────────────────────────────────────────────────────────────
# 摆动交易专用框架（PDT 限制下的正确策略）
# ─────────────────────────────────────────────────────────────

def swing_trade_plan(ticker: str, entry: float, stop: float, target: float,
                     account_value: float, hold_days: int = 5) -> dict:
    """
    摆动交易计划生成器

    摆动交易 = 持有 2-15 天，不在同一天买卖，绕过 PDT 规则
    持有期间可以利用过夜动量、财报后效应、周期性规律
    """
    if entry <= 0 or stop <= 0 or target <= 0:
        return {"error": "价格参数无效"}

    stop_pct   = abs(entry - stop) / entry * 100
    target_pct = abs(target - entry) / entry * 100
    rr         = target_pct / stop_pct if stop_pct > 0 else 0

    # 仓位计算（1% 风险规则 + 小账户上限保护）
    max_risk     = account_value * 0.01
    stop_dist    = abs(entry - stop)
    raw_shares   = int(max_risk / stop_dist) if stop_dist > 0 else 0
    max_pos_pct  = 0.20 if account_value < 10_000 else 0.25  # 小账户单仓上限20%
    max_by_pct   = int(account_value * max_pos_pct / entry)
    shares       = min(raw_shares, max_by_pct)
    pos_size     = shares * entry
    pos_pct      = pos_size / account_value * 100
    actual_risk  = shares * stop_dist

    # 持有天数建议
    if hold_days < 2:
        hold_days = 2
    elif hold_days > 15:
        hold_days = 15

    # 税务提示
    tax_note = None
    if hold_days < 30:
        tax_note = "持有不足30天：短期资本利得，按普通收入税率（10-37%）征税"
    elif hold_days >= 365:
        tax_note = "持有超1年：长期资本利得，税率仅15%（中等收入）或20%（高收入），省税显著"

    return {
        "type":            "摆动交易（Swing Trade）",
        "pdt_safe":        True,
        "ticker":          ticker,
        "account_value":   account_value,
        "entry":           round(entry, 2),
        "stop_loss":       round(stop, 2),
        "target":          round(target, 2),
        "stop_pct":        round(stop_pct, 1),
        "target_pct":      round(target_pct, 1),
        "risk_reward":     round(rr, 2),
        "rr_adequate":     rr >= 2.0,
        "shares":          shares,
        "position_size":   round(pos_size, 2),
        "position_pct":    round(pos_pct, 1),
        "max_risk":        round(actual_risk, 2),
        "max_risk_pct":    round(actual_risk / account_value * 100, 2),
        "hold_days_plan":  hold_days,
        "tax_note":        tax_note,
        "exit_rules": [
            f"止损：价格跌破 ${stop:.2f}，次日开盘市价卖出（不要等盘中）",
            f"目标：价格涨至 ${target:.2f}，可分批卖出 50%+50%",
            f"时间止损：持有超 {hold_days} 天无进展，无论盈亏减半仓",
            "盈利后将止损移至成本价（保本止损）",
        ],
        "warning": "不可在建仓当天卖出！否则计入 PDT 日内交易次数",
    }


# ─────────────────────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────────────────────

def _get_recommendations(account_value: float, remaining: int, status: str) -> list:
    recs = []
    if status == "DANGER":
        recs += [
            "🚫 本周禁止日内交易！将所有新仓计划改为次日入场（持有过夜）",
            "利用剩余时间做功课：研究下周要操作的标的",
            "专注于找 VCP / 杯柄形态，等待周一重置",
        ]
    elif status == "WARNING":
        recs += [
            f"⚠️ 仅剩 {remaining} 次日内交易机会，只用于最高置信度信号（≥80分）",
            "此时最优策略：找摆动交易机会，保存日内次数备用",
        ]

    if account_value < 5_000:
        recs += [
            f"账户 ${account_value:,.0f}：首要目标是增长到 $25,000 解锁 PDT 豁免",
            "专注于 2-5 只股票，不要分散精力",
            "每月目标 3-5% 收益（复利后 12 个月 ≈ 43-80% 增长）",
            "绝对禁止：加杠杆 3x ETF、期权、低价股（< $5）",
        ]
    elif account_value < 15_000:
        recs += [
            f"账户 ${account_value:,.0f}：以摆动交易为主，偶尔精选日内机会",
            "单笔最大仓位 ≤ 20% 总资产",
            "可以开始学习简单期权（买 Call/Put），但每笔不超过账户 5%",
        ]
    else:
        recs += [
            f"账户 ${account_value:,.0f}：接近 PDT 门槛，谨慎管理净值",
            "避免大幅回撤跌破 $25,000",
        ]

    return recs


def _swing_timeframes(account_value: float) -> list:
    """根据账户规模推荐摆动交易时间框架"""
    if account_value < 5_000:
        return [
            {"frame": "1-3周（5-15交易日）", "reason": "规避 PDT，利用中期趋势，降低交易频率"},
            {"frame": "主要看日线图", "reason": "日线趋势更稳定，噪音少"},
            {"frame": "每周最多2笔新仓", "reason": "小账户集中持仓，监控压力小"},
        ]
    elif account_value < 15_000:
        return [
            {"frame": "3-10交易日", "reason": "最平衡的风险收益窗口"},
            {"frame": "日线 + 4小时辅助", "reason": "4小时确认日线信号"},
            {"frame": "每周3-4笔交易", "reason": "避免过度交易"},
        ]
    else:
        return [
            {"frame": "2-7交易日（摆动）或当天（日内）", "reason": "灵活切换"},
            {"frame": "日线主框架，15分钟执行", "reason": "精准入场"},
        ]
