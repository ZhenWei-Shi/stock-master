"""
美联储加息/降息隐含概率（Fed Funds Futures Implied Rate Probability）

背景：2026-09-16追踪ASTS时讨论"模型对FOMC的处理有没有加息可能性参考"，
发现 macro_filter.py 现有机制全是事后反应型——FOMC日历只回答"今天是不是
决议日"用于禁入窗口，TLT涨跌幅/新闻关键词都是决议已发生后才触发的代理信号，
完全没有"决议前市场怎么定价"的前瞻性数据。

CME FedWatch Tool 官方就是用 30天联邦基金利率期货（CBOT代码 ZQ）价格反推
市场对下次FOMC决议结果的隐含概率，方法论公开透明，非黑箱：

  实测确认：yfinance 用 "ZQ=F" 能免费拿到该期货的front-month（最近到期月）
  连续合约价格，不需要CME官方FedWatch API（编程访问收费，EOD数据$25/月起）。
  但 yfinance 不支持按月份代码（如 ZQV26.CBT）指定具体到期月，只能拿到
  front-month，这是本模块相对CME官方多合约完整曲线的已知简化。

算法（标准CME FedWatch方法论的单档简化版）：
  30天期货价格隐含"当月平均EFFR" = 100 - futures_price
  若FOMC决议日落在front-month合约对应的自然月内（第d天，当月共N天）：
    (d-1)/N * R_current + (N-d+1)/N * R_new = implied_avg_rate
    解出 R_new，delta = R_new - R_current
    只考虑标准±25bp一档：prob_hike = clamp(delta/0.25, 0, 1)（delta>0时）
                          prob_cut  = clamp(-delta/0.25, 0, 1)（delta<0时）
                          prob_hold = 1 - prob_hike - prob_cut
  若决议不在front-month合约的自然月内（决议月份还未成为front month，通常
  发生在决议前1个月以上）：无法用当前front-month合约反映该次决议定价，
  降级返回"距决议较远，暂不支持"提示，不强行给出误导性数字。

已知局限（先记录，非本次范围）：
  1. 只处理标准±25bp一档，若市场实际price in 50bp+的跳档概率（历史罕见，
     2022年那轮加息周期出现过），delta会明显超出±0.25，本模块仍按线性
     公式外推给方向性参考，但不代表精确的分档概率分布（CME官方用更完整
     的多合约曲线做多档拆分，本模块是简化近似）。
  2. CURRENT_FED_RATE_MID 需要在每次FOMC决议后手动更新（跟 macro_filter.py
     的 _FOMC_DATES_20XX 年度日历同一类需要人工维护的硬编码，非本模块能
     自动感知，过期时用 _check_rate_const_staleness() 打印告警）。
  3. yfinance 只给 front-month 连续合约，无法像CME官方那样用多个到期月
     合约拼出完整的未来利率路径曲线，只能预测"下一次"决议，不能预测
     再往后的会议。

集成方式（与 short_volume_monitor.py 同一先例）：
  - scheduler.py 09:00晨报并行任务刷新 data/rate_expectations_snapshot.json
  - cold_model.py 新增 fed_rate_expectation gate，只读快照，pass 恒为 True——
    全新信号，先展示不参与打分/否决，观察一段时间验证稳定性后再决定是否
    正式接入 macro 否决权重
  - Telegram /fedwatch 按需查询
"""
from __future__ import annotations

import os
import json
import calendar as _calendar_mod
from datetime import datetime, date

import yfinance as yf
import pytz

from .macro_filter import _ALL_FOMC

ET_TZ = pytz.timezone("America/New_York")

_DATA          = os.path.join(os.path.dirname(__file__), "..", "data")
_SNAPSHOT_FILE = os.path.join(_DATA, "rate_expectations_snapshot.json")
_SNAPSHOT_MAX_AGE_HOURS = 12   # 期货价格盘中变化不大，比价格类gate的2小时阈值更宽松

# ══════════════════════════════════════════════════════════════
# 需要人工维护的常量（每次FOMC决议后更新，同macro_filter.py年度日历一样）
# ══════════════════════════════════════════════════════════════

CURRENT_FED_RATE_MID  = 3.875   # 当前目标区间中值（3.75%-4.00%）
CURRENT_FED_RATE_ASOF = "2026-09-16"   # 该常量对应的决议生效日期

RATE_STEP = 0.25   # 标准加/降息一档（25个基点）

_STALENESS_WARNED = False


def _check_rate_const_staleness():
    """
    若已经过了一次比 CURRENT_FED_RATE_ASOF 更晚的FOMC决议日，说明常量
    可能已过期未更新，打印一次性告警（每次进程启动最多一次）。
    """
    global _STALENESS_WARNED
    if _STALENESS_WARNED:
        return
    try:
        asof = date.fromisoformat(CURRENT_FED_RATE_ASOF)
        today = datetime.now(ET_TZ).date()
        passed_newer_fomc = any(
            asof < date.fromisoformat(fd) <= today for fd in _ALL_FOMC
        )
        if passed_newer_fomc:
            import warnings
            warnings.warn(
                "[rate_expectations] CURRENT_FED_RATE_MID 记录的决议日"
                f"（{CURRENT_FED_RATE_ASOF}）早于已发生的更新FOMC决议，"
                "请在 rate_expectations.py 中手动更新为最新目标利率区间中值。",
                RuntimeWarning, stacklevel=3,
            )
    except Exception:
        pass
    _STALENESS_WARNED = True


def _next_fomc_date(today: date) -> date | None:
    """从 macro_filter._ALL_FOMC 中找今天（含今天）之后最近的一次决议日。"""
    upcoming = sorted(
        date.fromisoformat(fd) for fd in _ALL_FOMC if date.fromisoformat(fd) >= today
    )
    return upcoming[0] if upcoming else None


def _fetch_front_month_price() -> float | None:
    """
    拉取 ZQ=F（30天联邦基金利率期货，CBOT）最新收盘价。
    yfinance 只提供 front-month 连续合约，非本次决议可用时由调用方降级处理。
    """
    try:
        hist = yf.Ticker("ZQ=F").history(period="5d")
        if hist.empty:
            return None
        return round(float(hist["Close"].iloc[-1]), 4)
    except Exception:
        return None


def _calc_from_next_fomc(as_of: date, next_fomc: date, futures_price: float | None,
                          current_rate: float) -> dict:
    """
    核心数学计算：给定"已经确定"的下次决议日期，算隐含加息/降息/维持概率。
    独立抽出（不查FOMC日历），便于pytest任意构造月初/月中/跨月边界，
    不依赖 macro_filter._ALL_FOMC 的真实日期分布。
    """
    days_away = (next_fomc - as_of).days

    if futures_price is None:
        return {
            "ok": False,
            "next_fomc_date": str(next_fomc),
            "days_away": days_away,
            "note": "期货价格获取失败（ZQ=F暂无数据），无法计算隐含概率",
        }

    # front-month 合约到期月份必须覆盖决议日所在自然月，否则该合约还没
    # 反映这次决议的定价（通常发生在决议前1个月以上查询时）。
    # yfinance 只返回 front-month，这里用"查询当天所在月份 == 决议月份"
    # 作为近似判断（front-month 通常就是当前自然月的合约）。
    if (as_of.year, as_of.month) != (next_fomc.year, next_fomc.month):
        return {
            "ok": False,
            "next_fomc_date": str(next_fomc),
            "days_away": days_away,
            "note": f"距下次FOMC决议还有{days_away}天，本月期货合约暂未反映该次决议定价，"
                    "需等决议月份内查询（通常决议前2-4周该合约才会成为front-month）",
        }

    n_days_in_month = _calendar_mod.monthrange(next_fomc.year, next_fomc.month)[1]
    d = next_fomc.day   # 决议日是当月第几天

    implied_avg_rate = round(100 - futures_price, 4)

    if d <= 1:
        # 决议在月初，当月几乎全程都是决议后的新利率，直接近似相等
        implied_new_rate = implied_avg_rate
    else:
        pre_days  = d - 1
        post_days = n_days_in_month - d + 1
        implied_new_rate = (implied_avg_rate * n_days_in_month
                             - pre_days * current_rate) / post_days

    delta = round(implied_new_rate - current_rate, 4)

    if delta > 0:
        prob_hike = max(0.0, min(1.0, delta / RATE_STEP))
        prob_cut  = 0.0
    elif delta < 0:
        prob_cut  = max(0.0, min(1.0, -delta / RATE_STEP))
        prob_hike = 0.0
    else:
        prob_hike = prob_cut = 0.0
    prob_hold = round(1.0 - prob_hike - prob_cut, 4)
    prob_hike = round(prob_hike, 4)
    prob_cut  = round(prob_cut, 4)

    extreme = abs(delta) > RATE_STEP * 1.5   # 隐含跳档超过1.5档，简化模型可能失真

    if prob_hike >= 0.6:
        lean = f"市场倾向加息25bp（隐含概率约{prob_hike:.0%}）"
    elif prob_cut >= 0.6:
        lean = f"市场倾向降息25bp（隐含概率约{prob_cut:.0%}）"
    elif prob_hold >= 0.6:
        lean = f"市场倾向维持不变（隐含概率约{prob_hold:.0%}）"
    else:
        lean = "市场定价分歧较大，无明显一致预期"

    note = (f"下次FOMC决议{next_fomc}（{days_away}天后）：{lean}。"
            f"隐含新利率约{implied_new_rate:.2f}%（当前{current_rate:.2f}%）")
    if extreme:
        note += "；⚠️隐含变动超过一档25bp，简化模型（只处理标准单档）可能失真，仅供方向参考"

    return {
        "ok": True,
        "next_fomc_date": str(next_fomc),
        "days_away": days_away,
        "current_rate": current_rate,
        "implied_avg_rate": implied_avg_rate,
        "implied_new_rate": round(implied_new_rate, 4),
        "delta_pct": delta,
        "prob_hike": prob_hike,
        "prob_cut": prob_cut,
        "prob_hold": prob_hold,
        "extreme": extreme,
        "note": note,
    }


def calc_rate_probability(as_of: date, futures_price: float | None,
                           current_rate: float = CURRENT_FED_RATE_MID) -> dict:
    """
    对外接口：从 macro_filter._ALL_FOMC 查出下次决议日期，再委托
    _calc_from_next_fomc 做核心计算。
    """
    next_fomc = _next_fomc_date(as_of)
    if next_fomc is None:
        return {"ok": False, "note": "FOMC日历未覆盖到更远的会议日期（需要年度更新）"}
    return _calc_from_next_fomc(as_of, next_fomc, futures_price, current_rate)


def save_rate_expectation_snapshot() -> dict:
    """生成隐含概率快照并写入文件，供 cold_model 全天读取（不逐股重复请求）。"""
    _check_rate_const_staleness()
    today = datetime.now(ET_TZ).date()
    price = _fetch_front_month_price()
    result = calc_rate_probability(today, price)

    snapshot = {
        "generated_at": str(datetime.now(ET_TZ)),
        **result,
    }
    os.makedirs(_DATA, exist_ok=True)
    tmp = _SNAPSHOT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, _SNAPSHOT_FILE)
    return snapshot


def rate_expectation_gate_check() -> dict:
    """
    供 cold_model.py 调用，只读快照，无网络请求。pass 恒为 True——全新信号，
    先展示观察，不参与 GO/ABORT 打分，跟 short_volume gate 同一先例。
    """
    try:
        if not os.path.exists(_SNAPSHOT_FILE):
            return {"pass": True, "note": "加息概率快照不存在，跳过（等下次scheduler刷新）"}

        with open(_SNAPSHOT_FILE, encoding="utf-8") as f:
            snap = json.load(f)

        gen_at = snap.get("generated_at", "")
        if gen_at:
            try:
                updated = datetime.fromisoformat(str(gen_at))
                age_h = (datetime.now(ET_TZ) - updated).total_seconds() / 3600
                if age_h > _SNAPSHOT_MAX_AGE_HOURS:
                    return {"pass": True, "note": f"加息概率快照已{age_h:.1f}小时未刷新，仅供参考"}
            except Exception:
                pass

        if not snap.get("ok"):
            return {"pass": True, "note": snap.get("note", "加息概率暂不可用")}

        return {
            "pass": True,
            "note": snap.get("note", ""),
            "prob_hike": snap.get("prob_hike"),
            "prob_cut": snap.get("prob_cut"),
            "prob_hold": snap.get("prob_hold"),
            "next_fomc_date": snap.get("next_fomc_date"),
        }
    except Exception as e:
        return {"pass": True, "note": f"加息概率检查跳过（{e}）"}


def format_rate_expectation_telegram() -> str:
    """/fedwatch 按需查询的Telegram格式化输出。"""
    result = rate_expectation_gate_check()
    lines = ["🏛 <b>美联储利率隐含概率</b>（Fed Funds期货推算，仅供参考不参与打分）"]
    # 用 is None 判断而非真值判断：100%确定维持不变时 prob_hike/prob_cut
    # 均为 0.0，真值判断会把这个合法结果误判成"无数据"。
    if result.get("prob_hold") is None:
        lines.append(result.get("note", "暂无数据"))
        return "\n".join(lines)

    lines.append(result["note"])
    lines.append(
        f"加息25bp {result['prob_hike']:.0%} / 维持不变 {result['prob_hold']:.0%} "
        f"/ 降息25bp {result['prob_cut']:.0%}"
    )
    lines.append(
        "<i>（大白话：这是用30天联邦基金利率期货价格倒推出来的市场共识，"
        "不是官方预测，只覆盖\"下一次\"决议，且只按标准25bp一档估算，"
        "跳档概率更大的极端情况会失真）</i>"
    )
    return "\n".join(lines)
