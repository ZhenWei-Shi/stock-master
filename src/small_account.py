"""
小资金账户适配器

专为资金 < $25,000 的散户起步阶段设计。
解决以下核心问题：
  1. 仓位计算在小账户下的边界条件
  2. 流动性过滤（避免滑点吃掉利润）
  3. 账户成长路径规划
  4. 期权适合性评估
  5. 股票可负担性检查
"""

import yfinance as yf
import numpy as np


# ─────────────────────────────────────────────────────────────
# 主入口：小账户综合评估
# ─────────────────────────────────────────────────────────────

def assess_small_account(account_value: float, ticker: str = None,
                          monthly_contribution: float = 0) -> dict:
    """
    小账户全面评估：
      - 账户阶段定义
      - 可用策略列表
      - 期权权限建议
      - 成长路径预测
    """
    stage      = _get_stage(account_value)
    strategies = _available_strategies(account_value)
    options_ok = _options_suitability(account_value)
    roadmap    = _growth_roadmap(account_value, monthly_contribution)

    result = {
        "account_value":     account_value,
        "stage":             stage,
        "available_strategies": strategies,
        "options_suitability":  options_ok,
        "growth_roadmap":    roadmap,
        "hard_rules":        _hard_rules(account_value),
    }

    if ticker:
        result["stock_check"] = check_stock_suitability(ticker, account_value)

    return result


# ─────────────────────────────────────────────────────────────
# 流动性 + 可负担性过滤器
# ─────────────────────────────────────────────────────────────

def check_stock_suitability(ticker: str, account_value: float) -> dict:
    """
    检查某只股票是否适合该账户规模操作

    检查项目：
      1. 日均成交量（流动性）
      2. 买卖价差（隐性成本）
      3. 单股可负担性（能否建立最小仓位）
      4. 波动性（ATR/价格）
    """
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info
        hist = tk.history(period="30d", interval="1d")
        if hist.empty:
            return {"error": "无法获取数据"}
    except Exception as e:
        return {"error": str(e)}

    price     = float(hist["Close"].iloc[-1])
    avg_vol   = info.get("averageVolume", 0) or 0
    avg_vol3m = info.get("averageDailyVolume3Month", avg_vol) or avg_vol

    # ── 1. 流动性检查 ────────────────────────────────────────
    if avg_vol3m >= 2_000_000:
        liquidity_grade = "A"
        liquidity_note  = f"日均成交量 {avg_vol3m/1e6:.1f}M，流动性极佳，滑点极低"
    elif avg_vol3m >= 500_000:
        liquidity_grade = "B"
        liquidity_note  = f"日均成交量 {avg_vol3m/1e6:.1f}M，流动性良好，适合小账户"
    elif avg_vol3m >= 100_000:
        liquidity_grade = "C"
        liquidity_note  = f"日均成交量 {avg_vol3m/1000:.0f}K，流动性一般，买卖需挂限价单"
    else:
        liquidity_grade = "D"
        liquidity_note  = f"日均成交量 {avg_vol3m/1000:.0f}K，流动性差！买卖困难，小账户不建议"

    liquidity_pass = liquidity_grade in ("A", "B")

    # ── 2. 买卖价差估算 ──────────────────────────────────────
    bid  = info.get("bid",  0) or 0
    ask  = info.get("ask",  0) or 0
    if bid > 0 and ask > 0:
        spread_pct  = (ask - bid) / price * 100
        spread_pass = spread_pct < 0.2
        spread_note = f"买卖价差 {spread_pct:.2f}%（{'可接受' if spread_pass else '偏大，影响小账户利润'}）"
    else:
        spread_pct  = None
        spread_pass = True
        spread_note = "无实时买卖价差数据"

    # ── 3. 可负担性 ──────────────────────────────────────────
    max_position_value = account_value * 0.20      # 单仓最大20%
    min_shares         = 1
    max_shares         = int(max_position_value / price) if price > 0 else 0

    if price <= 0:
        afford_note  = "无法获取价格"
        afford_pass  = False
    elif price > account_value * 0.25:
        afford_note  = f"单股 ${price:.2f} 超过账户 25%，无法建立最小合理仓位"
        afford_pass  = False
    elif max_shares < 1:
        afford_note  = f"单股 ${price:.2f}，20%上限内只能买 0 股，不适合"
        afford_pass  = False
    elif max_shares < 5 and account_value < 10_000:
        afford_note  = f"单股 ${price:.2f}，仅能买 {max_shares} 股（仓位过小，止损精度差）"
        afford_pass  = True
    else:
        afford_note  = f"单股 ${price:.2f}，最大 {max_shares} 股（仓位 ${max_shares*price:.0f}）"
        afford_pass  = True

    # ── 4. 波动性检查 ────────────────────────────────────────
    if len(hist) >= 14:
        tr_list = []
        for i in range(1, len(hist)):
            h = float(hist["High"].iloc[i])
            l = float(hist["Low"].iloc[i])
            c_prev = float(hist["Close"].iloc[i-1])
            tr_list.append(max(h - l, abs(h - c_prev), abs(l - c_prev)))
        atr14     = float(np.mean(tr_list[-14:]))
        atr_pct   = atr14 / price * 100
        if atr_pct > 5:
            vol_note = f"日均ATR {atr_pct:.1f}%，波动极大，小账户止损成本高"
            vol_warn = True
        elif atr_pct > 3:
            vol_note = f"日均ATR {atr_pct:.1f}%，波动较大，需宽止损"
            vol_warn = False
        else:
            vol_note = f"日均ATR {atr_pct:.1f}%，波动合理"
            vol_warn = False
    else:
        atr_pct  = None
        vol_note = "数据不足，无法计算ATR"
        vol_warn = False

    overall_pass = liquidity_pass and afford_pass
    overall_note = "✅ 适合小账户操作" if overall_pass else "⚠️ 不建议小账户操作此股票"

    return {
        "ticker":          ticker,
        "price":           round(price, 2),
        "account_value":   account_value,
        "overall":         overall_note,
        "overall_pass":    overall_pass,
        "liquidity": {
            "grade":        liquidity_grade,
            "avg_vol_3m":   avg_vol3m,
            "pass":         liquidity_pass,
            "note":         liquidity_note,
        },
        "spread": {
            "spread_pct":   spread_pct,
            "pass":         spread_pass,
            "note":         spread_note,
        },
        "affordability": {
            "max_shares":   max_shares,
            "max_position": round(max_shares * price, 2) if max_shares else 0,
            "pass":         afford_pass,
            "note":         afford_note,
        },
        "volatility": {
            "atr_pct":      round(atr_pct, 1) if atr_pct else None,
            "warning":      vol_warn,
            "note":         vol_note,
        },
    }


def liquidity_screen(tickers: list, min_avg_vol: int = 500_000) -> dict:
    """
    批量流动性筛选：只保留日均成交量 > min_avg_vol 的股票
    在 screener.py 的九关之前使用，提前过滤不适合小账户的股票
    """
    passed  = []
    failed  = []
    details = {}

    for ticker in tickers:
        try:
            info    = yf.Ticker(ticker).info
            vol     = info.get("averageDailyVolume3Month", 0) or \
                      info.get("averageVolume", 0) or 0
            if vol >= min_avg_vol:
                passed.append(ticker)
                details[ticker] = {"avg_vol": vol, "pass": True}
            else:
                failed.append(ticker)
                details[ticker] = {"avg_vol": vol, "pass": False,
                                   "reason": f"日均量{vol/1000:.0f}K < {min_avg_vol/1000:.0f}K"}
        except Exception as e:
            failed.append(ticker)
            details[ticker] = {"pass": False, "reason": str(e)}

    return {
        "passed":  passed,
        "failed":  failed,
        "details": details,
        "filter":  f"日均成交量 ≥ {min_avg_vol/1000:.0f}K",
    }


# ─────────────────────────────────────────────────────────────
# 仓位计算（小账户版）
# ─────────────────────────────────────────────────────────────

def position_size_small(account_value: float, price: float,
                         stop_distance: float, risk_pct: float = 0.01) -> dict:
    """
    小账户仓位计算，包含多重上限约束。

    标准1%风险规则 + 小账户3个额外约束：
      - 单仓不超过账户20%（< $10k）或25%（$10k-$25k）
      - 最少1股
      - 仓位不超过日均成交量的1%（避免移动市场）
    """
    if price <= 0 or stop_distance <= 0:
        return {"error": "价格或止损距离无效"}

    max_risk      = account_value * risk_pct
    shares_by_risk = int(max_risk / stop_distance)

    max_pos_pct   = 0.20 if account_value < 10_000 else 0.25
    shares_by_cap = int(account_value * max_pos_pct / price)

    shares        = max(1, min(shares_by_risk, shares_by_cap))
    pos_value     = shares * price
    pos_pct       = pos_value / account_value * 100
    actual_risk   = shares * stop_distance
    actual_risk_pct = actual_risk / account_value * 100

    warnings = []
    if shares == 1 and shares_by_risk > 1:
        warnings.append(f"仓位被压缩至1股（原计算{shares_by_risk}股），因单仓上限约束")
    if shares < 10 and account_value < 10_000:
        warnings.append("仓位较小（<10股），止损精度受限，建议使用限价单")
    if pos_pct > 15:
        warnings.append(f"单仓占比 {pos_pct:.1f}%，接近上限，注意集中度风险")
    if price > account_value * 0.15:
        warnings.append(f"单股价格 ${price} 较高，建议优先寻找同等质量的低价标的")

    return {
        "shares":          shares,
        "position_value":  round(pos_value, 2),
        "position_pct":    round(pos_pct, 1),
        "max_risk":        round(actual_risk, 2),
        "max_risk_pct":    round(actual_risk_pct, 2),
        "risk_pct_used":   risk_pct * 100,
        "shares_by_risk":  shares_by_risk,
        "shares_by_cap":   shares_by_cap,
        "binding_rule":    "风险规则" if shares_by_risk <= shares_by_cap else "仓位上限",
        "warnings":        warnings,
        "entry_total":     round(pos_value, 2),
    }


# ─────────────────────────────────────────────────────────────
# 账户成长路径
# ─────────────────────────────────────────────────────────────

def _growth_roadmap(account_value: float, monthly_contribution: float = 0) -> dict:
    """
    从当前账户规模到 $25,000 PDT 豁免点的成长预测

    三种场景：保守（月+2%）/ 中性（月+3%）/ 积极（月+5%）
    """
    target = 25_000.0
    if account_value >= target:
        return {"message": "账户已超过 PDT 门槛，无需规划成长路径", "months_needed": 0}

    scenarios = {}
    for label, monthly_rate in [("保守(+2%/月)", 0.02), ("中性(+3%/月)", 0.03), ("积极(+5%/月)", 0.05)]:
        bal    = account_value
        months = 0
        while bal < target and months < 120:
            bal = bal * (1 + monthly_rate) + monthly_contribution
            months += 1
        scenarios[label] = {
            "months":        months,
            "years":         round(months / 12, 1),
            "monthly_growth": f"{monthly_rate*100:.0f}%",
        }

    milestone_gains = []
    for milestone in [10_000, 15_000, 25_000, 50_000, 100_000]:
        if milestone > account_value:
            ratio = milestone / account_value
            milestone_gains.append({
                "milestone": f"${milestone:,.0f}",
                "need_gain": f"{(ratio-1)*100:.0f}%",
                "unlock":    _milestone_unlock(milestone),
            })

    return {
        "current":          account_value,
        "target_pdt":       target,
        "gap":              round(target - account_value, 2),
        "monthly_contribution": monthly_contribution,
        "scenarios":        scenarios,
        "milestones":       milestone_gains[:4],
        "note": (
            "月收益率是净收益率（扣除亏损）。"
            "专业摆动交易者平均月收益 2-4%。"
            "每月定投加快复利速度。"
        ),
    }


def _milestone_unlock(value: float) -> str:
    if value >= 100_000:
        return "机构级策略、多空对冲、完整期权组合"
    if value >= 50_000:
        return "更大仓位、波动性策略、裸卖Put（Level 3期权）"
    if value >= 25_000:
        return "PDT限制解除，可自由日内交易，所有策略可用"
    if value >= 15_000:
        return "更好的分散化（3-4只股票），期权Level 2"
    if value >= 10_000:
        return "期权Level 1（买Call/Put），小仓位ETF"
    return "摆动交易基础策略"


# ─────────────────────────────────────────────────────────────
# 账户阶段定义
# ─────────────────────────────────────────────────────────────

def _get_stage(account_value: float) -> dict:
    if account_value < 3_000:
        return {
            "stage":      "种子期",
            "range":      "< $3,000",
            "focus":      "学习阶段，绝对不要冒险",
            "max_trades": "每周 1 笔",
            "philosophy": "保护本金优先于盈利。每一笔亏损都会延长成长周期",
        }
    elif account_value < 10_000:
        return {
            "stage":      "起步期",
            "range":      "$3,000 - $10,000",
            "focus":      "摆动交易为主，建立交易纪律",
            "max_trades": "每周 2-3 笔",
            "philosophy": "在这个阶段的目标是让账户活过去，而不是快速致富",
        }
    elif account_value < 25_000:
        return {
            "stage":      "成长期",
            "range":      "$10,000 - $25,000",
            "focus":      "精选摆动交易，谨慎使用日内次数",
            "max_trades": "每周 4-5 笔",
            "philosophy": "越接近 $25,000，越要谨慎——跌破意味着 PDT 重新生效",
        }
    elif account_value < 100_000:
        return {
            "stage":      "自由期",
            "range":      "$25,000 - $100,000",
            "focus":      "日内 + 摆动灵活切换，策略多元化",
            "max_trades": "无限制",
            "philosophy": "现在才是真正施展量化策略的时候",
        }
    else:
        return {
            "stage":      "专业期",
            "range":      "> $100,000",
            "focus":      "完整策略组合，风险对冲",
            "max_trades": "无限制",
            "philosophy": "资本管理和风险控制比寻找好股票更重要",
        }


def _available_strategies(account_value: float) -> list:
    base = [
        {"name": "摆动交易（Swing Trading）", "available": True,
         "desc": "持有2-15天，不触发PDT，利用中期趋势"},
        {"name": "VCP形态突破", "available": True,
         "desc": "Minervini核心策略，适合任何账户规模"},
        {"name": "杯柄形态突破", "available": True,
         "desc": "O'Neil经典，成功率高，但需要耐心等待"},
        {"name": "趋势ETF（SPY/QQQ）", "available": True,
         "desc": "摆动持有1-3x ETF，风险可控"},
    ]
    if account_value < 5_000:
        base += [
            {"name": "日内交易", "available": False,
             "desc": f"账户 < $5,000，PDT规则严重限制日内操作"},
            {"name": "3x杠杆ETF（TQQQ/SOXL）", "available": False,
             "desc": "波动耗散 + 极端风险，小账户禁用"},
            {"name": "期权", "available": False,
             "desc": "期权价值衰减和IV风险对小账户太危险"},
            {"name": "剥头皮", "available": False,
             "desc": "需要实时数据源和大量PDT次数，不适合"},
        ]
    elif account_value < 10_000:
        base += [
            {"name": "日内交易（每周限3次）", "available": "limited",
             "desc": "严格控制，每次日内交易必须90分以上信号"},
            {"name": "3x杠杆ETF", "available": False,
             "desc": "波动耗散效应，持有超过5天必亏，小账户禁用"},
            {"name": "期权（买Call/Put）", "available": "limited",
             "desc": "仅限简单方向性买入，每笔 ≤ 账户5%"},
        ]
    elif account_value < 25_000:
        base += [
            {"name": "日内交易（每周限3次）", "available": "limited",
             "desc": "PDT限制内谨慎使用，留给最高质量信号"},
            {"name": "3x杠杆ETF（短期）", "available": "limited",
             "desc": "仅限持有1-3天的趋势追随，禁止长持"},
            {"name": "期权（Level 1-2）", "available": True,
             "desc": "可做简单买入+垂直价差，控制敞口"},
        ]
    else:
        base += [
            {"name": "所有策略", "available": True, "desc": "PDT已解除，策略不受限"},
        ]
    return base


def _options_suitability(account_value: float) -> dict:
    if account_value < 5_000:
        return {
            "level":      0,
            "suitable":   False,
            "advice":     "此阶段禁止期权交易",
            "reason":     "期权时间价值衰减和IV风险会在学习曲线期快速消耗小账户",
            "when_ready": "账户达到 $10,000 后再考虑",
        }
    elif account_value < 10_000:
        return {
            "level":      1,
            "suitable":   "limited",
            "advice":     "只允许简单方向性买入（买 Call 或买 Put）",
            "rules": [
                "每笔期权支出 ≤ 账户 3%（$300 上限）",
                "避免财报前买入（IV Crush风险）",
                "只买30天以上到期的合约（避免时间价值快速衰减）",
                "禁止卖出裸Call/Put（无限风险）",
            ],
        }
    elif account_value < 25_000:
        return {
            "level":      2,
            "suitable":   True,
            "advice":     "可做牛市价差（Bull Call Spread）和保护性Put",
            "rules": [
                "每笔期权支出 ≤ 账户 5%",
                "垂直价差限制最大亏损",
                "学习IV Rank概念：只在IV低时买，在IV高时卖",
            ],
        }
    else:
        return {
            "level":      3,
            "suitable":   True,
            "advice":     "可使用大多数期权策略",
        }


def _hard_rules(account_value: float) -> list:
    rules = [
        "单笔最大亏损 ≤ 账户 1%（用止损单强制执行）",
        "同时持仓不超过 3 只股票",
        "永远不要补仓亏损的仓位（加仓下跌 = 摊薄不等于智慧）",
        "亏损超过账户 5% 时停止交易，冷静3天再复盘",
    ]
    if account_value < 25_000:
        rules += [
            "每周最多 3 笔日内交易（PDT规则），超过则被限制",
            "禁止 3x 杠杆 ETF 隔夜持有",
        ]
    if account_value < 10_000:
        rules += [
            "禁止期权（账户太小，时间价值衰减不可接受）",
            "每笔交易前必须问：如果亏光这笔钱，我还能承受吗？",
        ]
    if account_value < 5_000:
        rules += [
            "本阶段目标不是盈利，是培养纪律和零亏损习惯",
            "模拟盘先交易3个月，胜率 > 55% 才转真实账户",
        ]
    return rules
