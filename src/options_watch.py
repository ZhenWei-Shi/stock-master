"""
期权纸上交易自动盯盘（2026-09-30新增）

wiki stock-master/options-log 里的纸上期权价差原来靠人盯，平仓条件容易错过。
这里把每笔交易的规则写进 data/options_paper.json，每个交易日16:10（收盘后，用当日
收盘价和CBOE最后报价）检查一次，触发就按对手价记平仓并推送Telegram。

每笔交易的规则（开仓时写死）：
  invalidate_close_above + invalidate_vol_ratio：标的收盘高于该价且量比≥该值 → 失效平仓
  time_exit + min_profit_pct：到该日仍未达到"按对手价平仓价值 ≥ 成本×(1+min_profit_pct)" → 时间止损
  到期日：按标的收盘价算内在价值结算
平仓价值按对手价：买入腿按bid卖出、卖出腿按ask买回（与options-log规则4一致）。

只记账、不下单。用法：python -m src.options_watch [--status]
"""
import json
import math
import os
import sys
from datetime import date, datetime

import pytz

ET = pytz.timezone("America/New_York")
_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
_FILE = os.path.join(_DATA, "options_paper.json")

# 2026-09-29 options-log 第1笔。"10/14仍未明显盈利就平仓"里的"明显盈利"定为+30%。
SEED_TRADES = [{
    "id": 1, "underlying": "ASTS", "expiry": "2026-10-16", "cp": "P",
    "long_strike": 62.0, "short_strike": 60.0, "qty": 1, "debit": 1.00,
    "opened": "2026-09-29",
    "invalidate_close_above": 63.5, "invalidate_vol_ratio": 1.5,
    "time_exit": "2026-10-14", "min_profit_pct": 0.30,
    "status": "open",
}]


def _load() -> list:
    try:
        with open(_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return [dict(t) for t in SEED_TRADES]


def _save(trades: list):
    os.makedirs(_DATA, exist_ok=True)
    tmp = _FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(trades, f, ensure_ascii=False, indent=1)
    os.replace(tmp, _FILE)


def intrinsic(tr: dict, spot: float) -> float:
    """到期内在价值（每股）。熊市看跌价差：max(0,K高-S) - max(0,K低-S)。"""
    k1, k2 = tr["long_strike"], tr["short_strike"]
    if tr["cp"] == "P":
        return max(0.0, k1 - spot) - max(0.0, k2 - spot)
    return max(0.0, spot - k1) - max(0.0, spot - k2)


def evaluate(tr: dict, today: date, close: float, vol_ratio: float | None, quotes: dict | None) -> dict:
    """
    按事先写死的规则判断今天要不要平仓（纯函数）。
    quotes: {"long_bid","long_ask","short_bid","short_ask"}，取不到时为None。
    返回 {"action": "hold"/"close", "reason", "exit_value", "mark_natural", "mark_mid"}。
    """
    out = {"action": "hold", "reason": "", "exit_value": None, "mark_natural": None, "mark_mid": None,
           "quote_issue": None, "raw_mark_natural": None, "raw_mark_mid": None}
    issue = "缺少两腿报价"
    if quotes:
        try:
            lb, la, sb, sa = (float(quotes[k]) for k in ["long_bid", "long_ask", "short_bid", "short_ask"])
            width = abs(tr["long_strike"] - tr["short_strike"])
            natural, mid = lb - sa, (lb + la - sb - sa) / 2
            issue = None
            if not all(math.isfinite(x) and x >= 0 for x in [lb, la, sb, sa]):
                issue = "腿报价含非法数值"
            elif lb > la or sb > sa:
                issue = "腿报价买卖倒挂"
            else:
                out["raw_mark_natural"], out["raw_mark_mid"] = round(natural, 2), round(mid, 2)
                if natural < 0 or natural > width + 0.01 or mid < 0 or mid > width + 0.01:
                    issue = "价差标记超出零至宽度范围"
            if quotes.get("source") == "cboe":
                stamp = quotes.get("quote_timestamp")
                try:
                    if not stamp or datetime.fromisoformat(stamp).date() != today:
                        issue = "CBOE报价日期未知或不是检查当日"
                except (TypeError, ValueError):
                    issue = "CBOE报价时间无法解析"
            if issue is None:
                out["mark_natural"], out["mark_mid"] = round(natural, 2), round(mid, 2)
        except (KeyError, TypeError, ValueError):
            issue = "腿报价字段缺失或格式错误"
    out["quote_issue"] = issue
    expiry = date.fromisoformat(tr["expiry"])
    if not math.isfinite(close) or close < 0:
        return {**out, "action": "review", "reason": "标的收盘价非法，不能确认结算"}
    if today >= expiry:
        v = round(intrinsic(tr, close), 2)
        return {**out, "action": "close", "reason": f"到期结算（{tr['underlying']}收盘${close:.2f}）", "exit_value": v}
    if (close > tr["invalidate_close_above"] and vol_ratio is not None
            and vol_ratio >= tr["invalidate_vol_ratio"]):
        if issue:
            return {**out, "action": "review", "reason": f"失效条件已触发，但{issue}，待核实平仓价值"}
        return {**out, "action": "close", "exit_value": out["mark_natural"],
                "reason": f"失效条件：收盘${close:.2f}>{tr['invalidate_close_above']}且量比{vol_ratio:.1f}x"}
    if today >= date.fromisoformat(tr["time_exit"]):
        target = tr["debit"] * (1 + tr["min_profit_pct"])
        mark = out["mark_natural"]
        if issue:
            return {**out, "action": "review", "reason": f"已到时间检查日，但{issue}，不能确认时间退出或盈亏"}
        if mark < target:
            mark_text = "未知" if mark is None else f"${mark:.2f}"
            return {**out, "action": "close", "exit_value": mark,
                    "reason": f"时间止损：{tr['time_exit']}起对手价平仓价值{mark_text}<目标${target:.2f}"}
    return out


def _market_data(tr: dict) -> tuple:
    """标的当日收盘价、量比（当日量/前20日均量），以及两条腿的CBOE报价。"""
    import yfinance as yf
    from .alpaca_options import occ_symbol
    from .gex_cboe import fetch_cboe_chain
    h = yf.Ticker(tr["underlying"]).history(period="2mo")
    if h.empty or h.index[-1].date() != datetime.now(ET).date():
        raise ValueError("标的当日日线缺失，不能用旧收盘价结算")
    close = float(h["Close"].iloc[-1])
    avg20 = float(h["Volume"].iloc[-21:-1].mean())
    vol_ratio = float(h["Volume"].iloc[-1]) / avg20 if avg20 > 0 else None
    quotes = None
    try:
        exp = date.fromisoformat(tr["expiry"])
        chain = fetch_cboe_chain(tr["underlying"])
        by_sym = {o["option"]: o for o in chain.get("options", [])}
        lg = by_sym.get(occ_symbol(tr["underlying"], exp, tr["cp"], tr["long_strike"]))
        sh = by_sym.get(occ_symbol(tr["underlying"], exp, tr["cp"], tr["short_strike"]))
        if lg and sh:
            quotes = {"long_bid": lg.get("bid"), "long_ask": lg.get("ask"),
                      "short_bid": sh.get("bid"), "short_ask": sh.get("ask"),
                      "source": "cboe", "quote_timestamp": chain.get("quote_timestamp")}
    except Exception:
        pass
    return close, vol_ratio, quotes


def run_options_watch(today: date | None = None, market=None) -> list:
    """每天16:10：检查所有未平仓的纸上期权，触发就记平仓。返回要推送的消息列表。"""
    today = today or datetime.now(ET).date()
    market = market or _market_data
    trades = _load()
    msgs = []
    for tr in trades:
        if tr.get("status") != "open":
            continue
        close, vol_ratio, quotes = market(tr)
        r = evaluate(tr, today, close, vol_ratio, quotes)
        tr["last_check"] = {"date": today.isoformat(), "close": round(close, 2),
                            "vol_ratio": round(vol_ratio, 2) if vol_ratio else None,
                            "mark_natural": r["mark_natural"], "mark_mid": r["mark_mid"],
                            "quote_issue": r["quote_issue"], "raw_mark_natural": r["raw_mark_natural"],
                            "raw_mark_mid": r["raw_mark_mid"], "quotes": quotes,
                            "action": r["action"]}
        if r["action"] == "review":
            msgs.append(f"⚠️ 期权纸上交易#{tr['id']}报价待核实：{r['reason']}；未确认平仓、不记盈亏")
        if r["action"] == "close":
            tr.update(status="closed", closed=today.isoformat(), exit_reason=r["reason"], exit_value=r["exit_value"])
            if r["exit_value"] is not None:
                tr["pnl"] = round((r["exit_value"] - tr["debit"]) * 100 * tr["qty"], 2)
            msgs.append(f"🧾 <b>期权纸上交易#{tr['id']}平仓</b>（{tr['underlying']} {tr['expiry'][5:]} "
                        f"{tr['long_strike']:g}/{tr['short_strike']:g}{tr['cp']}）\n{r['reason']}\n"
                        + (f"平仓价值${r['exit_value']:.2f} vs 成本${tr['debit']:.2f} → 盈亏${tr['pnl']:+.0f}"
                           if r["exit_value"] is not None else "取不到报价，平仓价值待人工补记")
                        + "\n（记得更新wiki options-log）")
    _save(trades)
    return msgs


def status_line() -> str:
    lines = []
    for tr in _load():
        head = f"期权#{tr['id']} {tr['underlying']} {tr['long_strike']:g}/{tr['short_strike']:g}{tr['cp']} {tr['expiry'][5:]}"
        if tr.get("status") == "open":
            lc = tr.get("last_check") or {}
            mark = lc.get("mark_natural")
            lines.append(f"{head}：持仓中，成本${tr['debit']:.2f}"
                         + (f"，{lc['date'][5:]}对手价${mark:.2f}、{tr['underlying']}收${lc['close']}" if mark is not None else "")
                         + f"；失效线${tr['invalidate_close_above']}（需放量{tr['invalidate_vol_ratio']}x），"
                           f"{tr['time_exit'][5:]}起未到${tr['debit'] * (1 + tr['min_profit_pct']):.2f}平仓")
        else:
            lines.append(f"{head}：已平仓（{tr.get('closed')}，{tr.get('exit_reason')}，盈亏${tr.get('pnl', 0):+.0f}）")
    return "\n".join(lines)


if __name__ == "__main__":
    if "--status" in sys.argv:
        print(status_line())
    else:
        for m in run_options_watch():
            print(m)
        print(status_line())
