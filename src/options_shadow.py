"""
卖put价差影子样本（2026-10-10用户决定，10/12起；只记账、不下单）

背景：options_auto的真实仓位每个仓位同时只能1笔、平均持有2-5周，满20笔已平仓要约1.5年。
影子样本：每个交易日10:30按**完全相同**的选价差规则（pick_spread、rank_scan）假想开一笔，
对照SPY、扫描各一笔，不受"同时最多1笔"和期权购买力限制；之后在真实仓位的每次检查时点
（9:45、10:30、11:05-15:05约每30分钟、15:30）按同样的平仓规则（exit_decision：止盈50%、
止损3倍且连续两次检查超线、剩≤21天）平掉。真实仓位照跑，用来核对影子成交价是否现实。

记账口径：
  - 开仓收入 = 对手价（卖腿bid − 买腿ask），与真实首次限价相同；另记中间价
  - 止盈 = 收入×TAKE_PROFIT（真实是挂在这个价的限价单）；止损/时间平仓 = 当次检查的对手价平仓成本
  - 盈亏按每股记（×100为每张），不管张数；VIX>VIX_MAX的日子不开（同真实）
  - 张数按账户价值算、但不看已占用的购买力（只要账户10%放得下1张）

判定标准（2026-10-10登记，不事后改）。对照、扫描两个仓位分别判断，"成立"要求：
  1. 已平仓 ≥ MIN_CLOSED（120笔，约6个月）
  2. 每笔平均盈亏 > 0，且开仓价再扣$HAIRCUT（一次改价步长，模拟真实成交比对手价差）后仍 > 0
  3. Newey-West t > 2（每天开一笔、持有期重叠，样本不独立；滞后阶数 = 平均持有交易日数）
  另：扫描每笔平均是否高于对照只作描述；影子 vs 真实同日开仓的收入差每次汇总时列出，
  若真实成交系统性低于影子超过$HAIRCUT，影子结论按该差额重新扣减后再看。

用法：python -m src.options_shadow [--summary]
"""
import json
import math
import os
import sys
from datetime import date, datetime

from .options_auto import ET, MIN_IV_RV, TAKE_PROFIT, VIX_MAX, arm_of, close_debit, exit_decision, \
    pick_spread, rank_scan, stop_breached

_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "options_shadow.json")
START = "2026-10-12"
MIN_CLOSED = 120
HAIRCUT = 0.02


def _load(path: str | None = None) -> list:
    try:
        with open(path or _FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def _save(rows: list, path: str | None = None):
    path = path or _FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def has_open(path: str | None = None) -> bool:
    return any(r["status"] == "open" for r in _load(path))


# ── 纯函数 ────────────────────────────────────────────────────

def shadow_plans(control: dict | None, scan: list, account_value: float, today: date) -> list:
    """
    今天该开的影子仓位：[(arm, 候选, plan)]。候选里要有rows/spot（_candidate返回值）。
    plan用账户价值、不看购买力重新选（真实开仓会因已有持仓占用购买力而选不出）。
    """
    def replan(c):
        return {**c, "plan": pick_spread(c["rows"], c["spot"], today, account_value, None)} if c.get("rows") else c
    out = []
    if control:
        c = replan(control)
        if c["plan"]["ok"]:
            out.append(("control", c))
    ranked = rank_scan([replan(c) for c in scan if c.get("iv_rv") is not None and c["iv_rv"] >= MIN_IV_RV])
    if ranked:
        out.append(("scan", ranked[0]))
    return out


def new_record(arm: str, cand: dict, vix, today: date, now: str, next_id: int) -> dict:
    p = cand["plan"]
    return {"id": next_id, "arm": arm, "opened": today.isoformat(), "at": now, "underlying": cand["underlying"],
            "expiry": p["expiry"].isoformat(), "dte": p["dte"], "short_sym": p["short_sym"], "long_sym": p["long_sym"],
            "short_strike": p["short_strike"], "long_strike": p["long_strike"], "width": p["width"],
            "credit": p["credit"], "mid": round(p["mid"], 2) if p.get("mid") is not None else None,
            "short_delta": p["short_delta"], "iv_rv": cand.get("iv_rv"), "vix": vix,
            "above_ma200": cand.get("above_ma200"), "status": "open", "stop_breach": False}


def check_one(rec: dict, debit: float | None, today: date) -> dict | None:
    """按真实规则判断一笔影子仓位；该平就返回平仓字段，否则None。原地更新stop_breach（纯逻辑）。"""
    d = exit_decision(rec, debit, today, prior_breach=rec.get("stop_breach", False))
    if debit is not None:
        rec["stop_breach"] = stop_breached(rec, debit)
    rec["last_debit"] = debit
    if d["action"] != "close" or debit is None:
        return None
    exit_debit = round(rec["credit"] * TAKE_PROFIT, 2) if d["reason"].startswith("止盈") else debit
    return {"status": "closed", "closed": today.isoformat(), "exit_debit": exit_debit, "reason": d["reason"],
            "pnl": round(rec["credit"] - exit_debit, 2)}


def nw_t(xs: list, lag: int) -> float | None:
    """Newey-West t（Bartlett核）。xs按开仓顺序。"""
    n = len(xs)
    if n < 3:
        return None
    m = sum(xs) / n
    d = [x - m for x in xs]
    var = sum(v * v for v in d) / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1)) * sum(d[i] * d[i - k] for i in range(k, n)) / n
    return m / math.sqrt(var / n) if var > 0 else None


def _trading_days(a: str, b: str) -> int:
    import numpy as np
    return int(np.busday_count(a, b))


def arm_stats(rows: list, arm: str) -> dict:
    closed = sorted((r for r in rows if r["arm"] == arm and r["status"] == "closed"), key=lambda r: (r["opened"], r["id"]))
    n = len(closed)
    if not n:
        return {"n": 0, "open": sum(1 for r in rows if r["arm"] == arm and r["status"] == "open")}
    pnl = [r["pnl"] for r in closed]
    hold = sum(_trading_days(r["opened"], r["closed"]) for r in closed) / n
    lag = max(1, round(hold))
    mean = sum(pnl) / n
    return {"n": n, "open": sum(1 for r in rows if r["arm"] == arm and r["status"] == "open"),
            "mean": mean, "mean_hc": mean - HAIRCUT, "t": nw_t(pnl, lag), "hold": hold,
            "win": sum(1 for x in pnl if x > 0) / n,
            "mean_risk": sum(r["pnl"] / (r["width"] - r["credit"]) for r in closed) / n}


def verdict(s: dict) -> str:
    if s["n"] < MIN_CLOSED:
        return f"样本未满{MIN_CLOSED}，不下结论"
    ok = s["mean"] > 0 and s["mean_hc"] > 0 and (s["t"] or 0) > 2
    return "✅ 按事先标准成立" if ok else "❌ 按事先标准不成立"


# ── 与options_auto的接口 ──────────────────────────────────────

def open_daily(control: dict | None, scan: list, account_value: float, vix, today: date,
               path: str | None = None) -> list:
    """10:30调用：按今天的候选开影子仓位。返回说明。"""
    if today.isoformat() < START:
        return []
    if vix is not None and vix > VIX_MAX:
        return [f"影子：VIX {vix}>{VIX_MAX:g}，今天不开"]
    rows = _load(path)
    if any(r["opened"] == today.isoformat() for r in rows):
        return ["影子：今天已开过"]
    now = datetime.now(ET).isoformat(timespec="minutes")
    notes = []
    for arm, c in shadow_plans(control, scan, account_value, today):
        rows.append(new_record(arm, c, vix, today, now, max([r["id"] for r in rows] or [0]) + 1))
        notes.append(f"影子开{arm} {c['underlying']} {c['plan']['short_strike']:g}/{c['plan']['long_strike']:g}P "
                     f"收入${c['plan']['credit']:.2f}")
    _save(rows, path)
    return notes or ["影子：今天没有合格价差"]


def check_all(market, today: date, path: str | None = None) -> list:
    """每次检查时点调用：一次批量取报价，按规则平掉该平的影子仓位。返回说明。"""
    rows = _load(path)
    opened = [r for r in rows if r["status"] == "open"]
    if not opened:
        return []
    syms = sorted({s for r in opened for s in (r["short_sym"], r["long_sym"])})
    q = {}
    for i in range(0, len(syms), 100):
        q.update(market.quotes(syms[i:i + 100]))
    closed = []
    for r in opened:
        upd = check_one(r, close_debit(q.get(r["short_sym"]), q.get(r["long_sym"])), today)
        if upd:
            r.update(upd)
            closed.append(f"{r['underlying']} {r['opened'][5:]} {upd['reason'][:2]}{upd['pnl']:+.2f}")
    _save(rows, path)
    return [f"影子持仓{len(opened) - len(closed)}笔" + (f"，平仓：{'，'.join(closed)}" if closed else "")]


def calibration(rows: list, real: list) -> list:
    """影子 vs 真实同日同仓位开仓：真实成交收入 − 影子收入。"""
    out = []
    for t in real:
        if t.get("credit") is None or not t.get("submitted"):
            continue
        sh = [r for r in rows if r["opened"] == t["submitted"] and r["arm"] == arm_of(t)
              and r["underlying"] == t["underlying"]]
        if sh:
            out.append(round(t["credit"] - sh[0]["credit"], 2))
    return out


def summarize(path: str | None = None) -> str:
    from .options_auto import _load as load_real
    rows = _load(path)
    lines = [f"卖put价差影子样本（{START}起，每天每仓位1笔，只记账）"]
    for arm, name in (("control", "对照SPY"), ("scan", "扫描")):
        s = arm_stats(rows, arm)
        if not s["n"]:
            lines.append(f"  {name}：已平0笔，持仓{s['open']}笔")
            continue
        t = "—" if s["t"] is None else f"{s['t']:+.2f}"
        lines.append(f"  {name}：已平{s['n']}笔（持仓{s['open']}），每笔均${s['mean']:+.3f}/股"
                     f"（扣${HAIRCUT}后${s['mean_hc']:+.3f}），占最大亏损{s['mean_risk']:+.1%}，胜率{s['win']:.0%}，"
                     f"平均持有{s['hold']:.1f}个交易日，NW t={t}  {verdict(s)}")
    cal = calibration(rows, [t for t in load_real() if t.get("status") in ("open", "pending_close", "closed")])
    if cal:
        lines.append(f"  校准：真实成交−影子收入 {len(cal)}笔，平均${sum(cal) / len(cal):+.3f}"
                     + ("（真实系统性更差，超过扣减额，影子结论要再扣）" if sum(cal) / len(cal) < -HAIRCUT else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize())
