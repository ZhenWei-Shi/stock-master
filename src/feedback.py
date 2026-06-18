"""
交易反馈学习系统（Feedback Learning）

核心逻辑：
  1. 开仓时快照当时的所有信号状态
  2. 平仓后记录结果，建立"条件→结果"数据库
  3. 定期分析：哪些信号组合在亏损前反复出现
  4. 生成动态"危险条件"列表，反馈给 cold_decision
  5. 发送每周学习报告到 Telegram

目标：
  不是预测价格，而是识别"高概率亏损的市场条件"并回避
  这是专业量化基金的标准做法：anti-pattern recognition

重要说明：
  系统不会自动修改策略参数（防止过拟合近期市场）
  只生成警告，最终由人类决策是否调整
  需要至少 30 笔历史交易才能产生可信的模式（MIN_TRADES_FOR_LEARNING = 30）
"""

import os
import json
import numpy as np
from datetime import datetime
import pytz

ET      = pytz.timezone("America/New_York")
_DATA   = os.path.join(os.path.dirname(__file__), "..", "data")
_FB_LOG = os.path.join(_DATA, "feedback_log.json")
_PATTERN= os.path.join(_DATA, "learned_patterns.json")

os.makedirs(_DATA, exist_ok=True)

MIN_TRADES_FOR_LEARNING = 30   # 初步参考；≥50笔才具统计显著性


# ─────────────────────────────────────────────────────────────
# 开仓时记录信号快照
# ─────────────────────────────────────────────────────────────

def record_entry_signals(trade_id: str, ticker: str,
                          cold_result: dict, debate_result: dict = None):
    """
    开仓时调用，记录当时所有信号状态。
    cold_result 来自 cold_decision() 的返回值。
    """
    signals = {
        "trade_id":       trade_id,
        "ticker":         ticker,
        "entry_time":     str(datetime.now(ET)),
        "score":          cold_result.get("score", 0),
        "verdict":        cold_result.get("verdict", ""),
        "aggressive_mode": cold_result.get("aggressive_mode", False),
        "gates": {},
        "market_state":   cold_result.get("market_state", {}).get("state", ""),
        "vix":            cold_result.get("market_state", {}).get("vix", 0),
        "bonus":          cold_result.get("bonus", 0),
        "result":         None,  # 平仓后填入
        "pnl_pct":        None,
        "max_loss_pct":   None,
        "hold_days":      None,
    }

    # 提取各关信号状态（gates 是 dict: {name: {"pass": True/False/"warn", "note": ...}}）
    for gate_name, gate_data in cold_result.get("gates", {}).items():
        p = gate_data.get("pass")
        signals["gates"][gate_name] = "pass" if p is True else ("warn" if p == "warn" else "fail")

    # 辩论结论
    if debate_result:
        signals["debate_conclusion"] = debate_result.get("verdict", {}).get("conclusion", "")
        signals["bull_conviction"]   = debate_result.get("bull_analyst", {}).get("conviction", "")
        signals["bear_severity"]     = debate_result.get("bear_analyst", {}).get("severity", "")
        signals["risk_kelly"]        = debate_result.get("risk_officer", {}).get(
            "kelly_criterion", {}).get("half_kelly_pct", 0)

    _append_feedback(signals)
    return signals


def record_exit_result(trade_id: str, pnl_pct: float,
                        hold_days: int, exit_reason: str = ""):
    """平仓后调用，填写结果。"""
    data = _load_feedback()
    for entry in data:
        if entry.get("trade_id") == trade_id:
            entry["result"]       = "win" if pnl_pct > 0 else "loss"
            entry["pnl_pct"]      = round(pnl_pct, 2)
            entry["hold_days"]    = hold_days
            entry["exit_reason"]  = exit_reason
            entry["exit_time"]    = str(datetime.now(ET))
            break
    _save_feedback(data)


# ─────────────────────────────────────────────────────────────
# 核心分析：找出亏损交易的共同条件
# ─────────────────────────────────────────────────────────────

def analyze_patterns() -> dict:
    """
    分析历史交易，找出"亏损时反复出现的条件"。

    输出：
      danger_conditions — 高频亏损信号组合（出现时降低信心）
      safe_conditions   — 高频盈利信号组合（参考正向条件）
      stats             — 整体统计
    """
    data  = [d for d in _load_feedback() if d.get("result") in ("win", "loss")]
    total = len(data)

    if total < MIN_TRADES_FOR_LEARNING:
        return {
            "ready":   False,
            "total":   total,
            "needed":  MIN_TRADES_FOR_LEARNING,
            "message": f"数据不足：已有{total}笔，需{MIN_TRADES_FOR_LEARNING}笔才能开始学习",
        }

    wins   = [d for d in data if d["result"] == "win"]
    losses = [d for d in data if d["result"] == "loss"]

    win_rate  = len(wins) / total
    avg_win   = float(np.mean([d["pnl_pct"] for d in wins]))   if wins   else 0
    avg_loss  = float(np.mean([d["pnl_pct"] for d in losses])) if losses else 0
    worst_loss= float(min([d["pnl_pct"] for d in losses]))      if losses else 0

    # ── 分析各关状态在盈/亏交易中的频率 ──────────────────────
    gate_ids = set()
    for d in data:
        gate_ids.update(d.get("gates", {}).keys())

    gate_analysis = {}
    for gid in gate_ids:
        win_pass  = sum(1 for d in wins   if d.get("gates",{}).get(gid) == "pass")
        loss_pass = sum(1 for d in losses if d.get("gates",{}).get(gid) == "pass")
        win_fail  = sum(1 for d in wins   if d.get("gates",{}).get(gid) == "fail")
        loss_fail = sum(1 for d in losses if d.get("gates",{}).get(gid) == "fail")

        # 这个关"亮红灯但仍然入场"的情况，在亏损中占比多少
        if loss_fail + win_fail > 3:
            loss_fail_rate = loss_fail / (loss_fail + win_fail)
            gate_analysis[gid] = {
                "loss_when_fail_pct": round(loss_fail_rate * 100, 1),
                "danger": loss_fail_rate > 0.65,  # 超过65%的"带伤入场"最终亏损
            }

    # ── 市场状态分析 ──────────────────────────────────────────
    state_stats = {}
    for d in data:
        state = d.get("market_state", "unknown")
        if state not in state_stats:
            state_stats[state] = {"win": 0, "loss": 0}
        state_stats[state][d["result"]] += 1

    danger_market_states = [
        s for s, v in state_stats.items()
        if v.get("loss", 0) / max(v.get("win", 0) + v.get("loss", 0), 1) > 0.6
        and v.get("win", 0) + v.get("loss", 0) >= 3
    ]

    # ── VIX 分析 ──────────────────────────────────────────────
    high_vix_losses = [d for d in losses if (d.get("vix") or 0) > 25]
    vix_danger = len(high_vix_losses) / len(losses) > 0.5 if losses else False

    # ── 保存学习到的模式 ──────────────────────────────────────
    patterns = {
        "updated_at":          str(datetime.now(ET)),
        "total_trades":        total,
        "win_rate":            round(win_rate * 100, 1),
        "avg_win_pct":         round(avg_win, 2),
        "avg_loss_pct":        round(avg_loss, 2),
        "worst_single_loss":   round(worst_loss, 2),
        "danger_gates":        {k: v for k, v in gate_analysis.items() if v.get("danger")},
        "danger_market_states": danger_market_states,
        "vix_danger_above_25": vix_danger,
        "gate_analysis":       gate_analysis,
        "state_stats":         state_stats,
        "ready":               True,
    }
    with open(_PATTERN, "w", encoding="utf-8") as f:
        json.dump(patterns, f, ensure_ascii=False, indent=2)

    return patterns


# ─────────────────────────────────────────────────────────────
# 供 cold_decision 查询：当前是否触发反向模式
# ─────────────────────────────────────────────────────────────

def check_anti_patterns(cold_result: dict) -> dict:
    """
    在 cold_decision 做最终判断前调用。
    检查当前信号是否命中历史亏损模式。

    返回：
      warnings  — 触发了哪些危险模式（列表）
      penalty   — 建议扣减的分数
      block     — 是否建议强制否决
    """
    if not os.path.exists(_PATTERN):
        return {"warnings": [], "penalty": 0, "block": False}

    with open(_PATTERN, "r", encoding="utf-8") as f:
        patterns = json.load(f)

    if not patterns.get("ready"):
        return {"warnings": [], "penalty": 0, "block": False}

    warnings = []
    penalty  = 0

    # 检查危险大盘状态
    market_state = cold_result.get("market_state", {}).get("state", "")
    if market_state in patterns.get("danger_market_states", []):
        warnings.append(f"⚠️ 历史数据：{market_state} 环境下亏损率>60%")
        penalty += 10

    # 检查 VIX
    vix = cold_result.get("market_state", {}).get("vix", 0)
    if patterns.get("vix_danger_above_25") and vix > 25:
        warnings.append(f"⚠️ VIX={vix:.0f}>25，历史数据：高恐慌期亏损集中")
        penalty += 8

    # 检查危险关卡
    current_gates = {}
    for gate_name, gate_data in cold_result.get("gates", {}).items():
        p = gate_data.get("pass")
        current_gates[gate_name] = "pass" if p is True else ("warn" if p == "warn" else "fail")

    for gid, info in patterns.get("danger_gates", {}).items():
        if current_gates.get(gid) == "fail":
            rate = info.get("loss_when_fail_pct", 0)
            warnings.append(f"⚠️ 关卡{gid}亮红：历史{rate:.0f}%情况下带伤入场最终亏损")
            penalty += 12

    # 超过3个危险信号时建议否决
    block = len(warnings) >= 3 or penalty >= 30

    return {
        "warnings": warnings,
        "penalty":  penalty,
        "block":    block,
        "data_basis": f"基于{patterns.get('total_trades', 0)}笔历史交易",
    }


# ─────────────────────────────────────────────────────────────
# 每周学习报告
# ─────────────────────────────────────────────────────────────

def weekly_learning_report() -> str:
    """生成每周策略学习报告，发送到 Telegram。"""
    patterns = analyze_patterns()

    if not patterns.get("ready"):
        return (
            f"📚 <b>学习系统</b>\n\n"
            f"数据积累中：{patterns['total']} / {patterns['needed']} 笔\n"
            f"继续运行模拟盘，数据足够后自动开始学习"
        )

    danger_gates  = patterns.get("danger_gates", {})
    danger_states = patterns.get("danger_market_states", [])

    report = (
        f"📚 <b>每周策略学习报告</b>\n\n"
        f"样本：{patterns['total_trades']} 笔交易\n"
        f"胜率：{patterns['win_rate']}%\n"
        f"平均盈利：+{patterns['avg_win_pct']}%\n"
        f"平均亏损：{patterns['avg_loss_pct']}%\n"
        f"最大单笔亏损：{patterns['worst_single_loss']}%\n\n"
    )

    if danger_gates:
        report += "🔴 <b>高风险信号（建议加强重视）</b>\n"
        for gid, info in danger_gates.items():
            report += f"  关卡{gid}：带伤入场 {info['loss_when_fail_pct']}% 亏损\n"
        report += "\n"

    if danger_states:
        report += f"⚠️ <b>危险市场环境</b>：{', '.join(danger_states)}\n\n"

    if patterns.get("vix_danger_above_25"):
        report += "📊 VIX>25 时亏损集中，高恐慌期建议减仓或不开仓\n\n"

    report += "💡 系统已将以上模式加入实时过滤，入场前自动预警"
    return report


# ─────────────────────────────────────────────────────────────
# 文件 I/O
# ─────────────────────────────────────────────────────────────

def _load_feedback() -> list:
    if not os.path.exists(_FB_LOG):
        return []
    with open(_FB_LOG, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_feedback(data: list):
    with open(_FB_LOG, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _append_feedback(entry: dict):
    data = _load_feedback()
    data.append(entry)
    _save_feedback(data)
