"""
动态宏观数据（完全免费，通过 yfinance 获取）

已实现：
  - 收益率曲线（10Y-3M / 10Y-5Y 利差，判断是否倒挂）
  - 跨资产风险偏好（SPY/黄金/美元/铜/VIX/原油/长债，综合Risk-On/Off评分）
  - 板块轮动（11大板块20日涨幅排名）——注意：cold_model.py 的
    sector_rotation gate 已用 src/sector_rotation.py 实现同类功能，
    本模块这部分仅用于 full_breadth_report() 的展示，不重复接入 cold_model。

尚未实现（docstring曾提及但代码里没有，2026-07-21复核时更正）：
  - 市场广度（% 股票在200MA上方）——需要扫描大量个股，暂未做，非本次范围。

2026-07-21接入 cold_model.py：新增 save_breadth_snapshot() / breadth_gate_check()，
采用跟 macro_filter.py 完全一致的"快照文件"模式——由 scheduler 定时刷新写入
data/breadth_snapshot.json，cold_decision() 只读快照不发起实时请求，避免每
扫描一只股票就重新拉一遍收益率曲线+7个跨资产ticker（11次yfinance调用/股）。
"""

import json
import os
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import pytz

ET = pytz.timezone("America/New_York")
_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
_SNAPSHOT_PATH = os.path.join(_DATA, "breadth_snapshot.json")
_SNAPSHOT_MAX_AGE_HOURS = 2   # 跟 macro_filter.py 的陈旧判定口径保持一致


# ─────────────────────────────────────────────────────────────
# 收益率曲线
# ─────────────────────────────────────────────────────────────

def get_yield_curve() -> dict:
    """
    拉取美债收益率，计算关键利差。
    yfinance Ticker 代码：
      ^IRX = 13周（3个月）
      ^FVX = 5年
      ^TNX = 10年
      ^TYX = 30年
    2Y 用 ETF SHY 代理（直接 2Y 无 yfinance Ticker）
    """
    tickers = {"3M": "^IRX", "5Y": "^FVX", "10Y": "^TNX", "30Y": "^TYX"}
    yields  = {}
    for label, sym in tickers.items():
        try:
            tk = yf.Ticker(sym)
            hist = tk.history(period="5d")
            if not hist.empty:
                yields[label] = round(float(hist["Close"].iloc[-1]), 3)
        except Exception:
            pass

    spread_10_3m = round(yields.get("10Y", 0) - yields.get("3M", 0), 3)
    spread_10_5y = round(yields.get("10Y", 0) - yields.get("5Y", 0), 3)

    if spread_10_3m < -0.5:
        curve_signal = "DEEPLY_INVERTED"
        curve_note   = f"曲线深度倒挂（10Y-3M={spread_10_3m:.2f}%），历史上预示12-18月内衰退"
    elif spread_10_3m < 0:
        curve_signal = "INVERTED"
        curve_note   = f"曲线倒挂（10Y-3M={spread_10_3m:.2f}%），需警惕"
    elif spread_10_3m < 0.5:
        curve_signal = "FLAT"
        curve_note   = f"曲线平坦（10Y-3M={spread_10_3m:.2f}%），方向不明"
    else:
        curve_signal = "NORMAL"
        curve_note   = f"曲线正常（10Y-3M={spread_10_3m:.2f}%），经济扩张期"

    return {
        "yields":        yields,
        "spread_10_3m":  spread_10_3m,
        "spread_10_5y":  spread_10_5y,
        "signal":        curve_signal,
        "note":          curve_note,
        "inverted":      spread_10_3m < 0,
    }


# ─────────────────────────────────────────────────────────────
# 跨资产风险偏好（Risk-On / Risk-Off）
# ─────────────────────────────────────────────────────────────

def get_risk_sentiment() -> dict:
    """
    通过跨资产价格判断当前风险偏好。

    Risk-On：  股市涨 + 美元跌 + 黄金平 + 铜涨 + VIX低
    Risk-Off：  股市跌 + 美元涨 + 黄金涨 + 铜跌 + VIX高
    """
    symbols = {
        "SPY":  "大盘（SPY）",
        "GLD":  "黄金",
        "DX-Y.NYB": "美元指数",
        "HG=F": "铜（工业需求）",
        "^VIX": "VIX",
        "CL=F": "WTI原油",
        "TLT":  "长债（20Y+）",
    }
    data = {}
    for sym, label in symbols.items():
        try:
            hist = yf.Ticker(sym).history(period="10d")
            if len(hist) >= 5:
                ret_5d = float((hist["Close"].iloc[-1] - hist["Close"].iloc[-5])
                               / hist["Close"].iloc[-5] * 100)
                data[sym] = {
                    "label":   label,
                    "price":   round(float(hist["Close"].iloc[-1]), 2),
                    "ret_5d":  round(ret_5d, 2),
                }
        except Exception:
            pass

    vix = data.get("^VIX", {}).get("price", 20)
    spy_ret = data.get("SPY", {}).get("ret_5d", 0)
    gld_ret = data.get("GLD", {}).get("ret_5d", 0)
    dxy_ret = data.get("DX-Y.NYB", {}).get("ret_5d", 0)
    cu_ret  = data.get("HG=F", {}).get("ret_5d", 0)

    # 综合评分
    score = 0
    score += 1 if spy_ret > 0 else -1
    score += 1 if gld_ret < 0 else -1   # 黄金跌 = 不需要避险
    score += 1 if dxy_ret < 0 else -1   # 美元跌 = 流动性宽松
    score += 1 if cu_ret  > 0 else -1   # 铜涨 = 工业需求强
    score -= 1 if vix > 25 else 0

    if score >= 3:
        mode = "RISK_ON"
        note = "风险偏好强烈，大盘/铜/流动性三箭齐发，成长股有利"
    elif score >= 1:
        mode = "MILD_RISK_ON"
        note = "风险偏好偏多，可正常持股"
    elif score >= -1:
        mode = "NEUTRAL"
        note = "跨资产信号混乱，谨慎操作"
    elif score >= -3:
        mode = "MILD_RISK_OFF"
        note = "避险情绪升温，控制仓位"
    else:
        mode = "RISK_OFF"
        note = "全面避险，美元/黄金/债券齐涨，大盘承压"

    return {
        "mode":     mode,
        "score":    score,
        "note":     note,
        "vix":      vix,
        "assets":   data,
        "spy_5d":   spy_ret,
        "gold_5d":  gld_ret,
        "dxy_5d":   dxy_ret,
        "copper_5d": cu_ret,
    }


# ─────────────────────────────────────────────────────────────
# 板块轮动
# ─────────────────────────────────────────────────────────────

def get_sector_rotation() -> dict:
    """
    11个标准板块 ETF 近20日表现，识别领先/落后板块。
    """
    sectors = {
        "XLK": "科技",    "XLF": "金融",   "XLV": "医疗",
        "XLE": "能源",    "XLI": "工业",   "XLY": "非必选消费",
        "XLP": "必选消费", "XLB": "材料",  "XLU": "公用事业",
        "XLC": "通信",    "XLRE": "房地产",
    }
    results = []
    for sym, name in sectors.items():
        try:
            hist = yf.Ticker(sym).history(period="30d")
            if len(hist) >= 20:
                ret = float((hist["Close"].iloc[-1] - hist["Close"].iloc[-20])
                            / hist["Close"].iloc[-20] * 100)
                results.append({"etf": sym, "sector": name, "ret_20d": round(ret, 1)})
        except Exception:
            pass

    results.sort(key=lambda x: x["ret_20d"], reverse=True)
    leaders  = results[:3]
    laggards = results[-3:]

    leader_names  = "/".join(r["sector"] for r in leaders)
    laggard_names = "/".join(r["sector"] for r in laggards)
    return {
        "leaders":  leaders,
        "laggards": laggards,
        "all":      results,
        "note": f"领涨：{leader_names}  落后：{laggard_names}",
    }


# ─────────────────────────────────────────────────────────────
# 综合报告
# ─────────────────────────────────────────────────────────────

def full_breadth_report() -> dict:
    now = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    yc  = get_yield_curve()
    rs  = get_risk_sentiment()
    sr  = get_sector_rotation()

    # 综合操作建议
    if rs["mode"] == "RISK_OFF" or yc["signal"] == "DEEPLY_INVERTED":
        action = "🔴 全面防御：收益率曲线深度倒挂或全球避险，大幅缩仓"
    elif rs["mode"] == "MILD_RISK_OFF" or yc["inverted"]:
        action = "🟡 保守操作：曲线倒挂或避险情绪，控制新仓规模"
    elif rs["mode"] in ("RISK_ON", "MILD_RISK_ON") and not yc["inverted"]:
        action = "🟢 正常操作：风险偏好健康，曲线正常，可积极布局"
    else:
        action = "⚪ 观望：信号混合，等待方向明朗"

    return {
        "time":           now,
        "yield_curve":    yc,
        "risk_sentiment": rs,
        "sector_rotation": sr,
        "action":         action,
    }


def format_breadth_telegram(report: dict) -> str:
    yc = report["yield_curve"]
    rs = report["risk_sentiment"]
    sr = report["sector_rotation"]

    leaders_str  = ", ".join(
        "{0}({1:+.1f}%)".format(r["sector"], r["ret_20d"]) for r in sr["leaders"]
    )
    laggards_str = ", ".join(
        "{0}({1:+.1f}%)".format(r["sector"], r["ret_20d"]) for r in sr["laggards"]
    )
    y10  = yc["yields"].get("10Y", "?")
    y3m  = yc["yields"].get("3M",  "?")
    vix  = rs["vix"]
    lines = [
        f"🌐 <b>市场宏观快照</b>  {report['time']}",
        "",
        f"📈 风险偏好：{rs['mode']}  VIX={vix:.1f}",
        f"   {rs['note']}",
        "",
        f"📉 收益率曲线：{yc['signal']}",
        f"   {yc['note']}",
        f"   10Y={y10}%  3M={y3m}%",
        "",
        "🔄 板块轮动：",
        f"   领涨：{leaders_str}",
        f"   落后：{laggards_str}",
        "",
        f"💡 {report['action']}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# cold_model 集成接口（快照模式，跟 macro_filter.py 同一套设计）
# ─────────────────────────────────────────────────────────────

def save_breadth_snapshot() -> dict:
    """生成宏观广度报告并写入快照文件，供 cold_model 全天读取（不再逐股实时拉取）。"""
    report = full_breadth_report()
    snapshot = {
        "generated_at":   str(datetime.now(ET)),
        "yc_signal":      report["yield_curve"]["signal"],
        "yc_inverted":    report["yield_curve"]["inverted"],
        "yc_note":        report["yield_curve"]["note"],
        "rs_mode":        report["risk_sentiment"]["mode"],
        "rs_score":       report["risk_sentiment"]["score"],
        "rs_note":        report["risk_sentiment"]["note"],
        "action":         report["action"],
    }
    os.makedirs(_DATA, exist_ok=True)
    tmp = _SNAPSHOT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, _SNAPSHOT_PATH)
    return snapshot


def breadth_gate_check() -> dict:
    """
    cold_model 调用此函数得到宏观广度层面的软性提示（只读快照，无网络请求）。
    只做 warn，不做硬否决——收益率曲线倒挂是持续数月的慢变量，不适合像VIX
    单日恐慌那样一票否决所有新仓，而是提醒"大环境偏弱，正常权衡即可"。
    """
    try:
        if not os.path.exists(_SNAPSHOT_PATH):
            return {"pass": True, "note": "宏观广度快照不存在，跳过（等下次scheduler刷新）"}

        with open(_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            snap = json.load(f)

        gen_at = snap.get("generated_at", "")
        if gen_at:
            try:
                updated = datetime.fromisoformat(str(gen_at))
                age_h = (datetime.now(ET) - updated).total_seconds() / 3600
                if age_h > _SNAPSHOT_MAX_AGE_HOURS:
                    return {"pass": True, "note": f"宏观广度快照已{age_h:.1f}小时，建议刷新，暂按中性处理"}
            except Exception:
                pass

        yc_inverted = snap.get("yc_inverted", False)
        rs_mode     = snap.get("rs_mode", "NEUTRAL")

        if rs_mode == "RISK_OFF" or snap.get("yc_signal") == "DEEPLY_INVERTED":
            return {"pass": "warn",
                    "note": f"宏观环境偏弱：{snap.get('rs_note','')}；{snap.get('yc_note','')}"}
        if rs_mode == "MILD_RISK_OFF" or yc_inverted:
            return {"pass": "warn",
                    "note": f"宏观环境略偏谨慎：{snap.get('rs_note','')}"}
        return {"pass": True, "note": f"宏观环境：{rs_mode}，{snap.get('rs_note','')}"}

    except Exception as e:
        return {"pass": True, "note": f"宏观广度检查跳过（{e}）"}
