"""
ETF卖put价差自动开仓（2026-09-30新增，Alpaca模拟账户，前向实验）

背景：用户希望由程序判断期权开仓时机、不再手工纸上交易。本项目的方向性信号
（九关/cold_model）回测没有优势，拿来买期权等于付费押随机方向；期权里有外部实证的
机械优势是"卖指数put"（波动率风险溢价），见wiki stock-master/options-log。
学术上期权偏贵主要在指数层面（相关性风险溢价），个股期权平均并不明显偏贵、还有
财报跳空，所以只扫ETF、不碰个股。只下Alpaca模拟账户（alpaca_client写死paper=True）。

两个仓位（对照实验，2026-09-30用户确定）：
  control：固定SPY（原规则，作为基准）
  scan：   在SCAN_UNIVERSE里按"隐含波动率/过去20日实际波动率"(IV/RV)挑最高的一只
           （要求≥MIN_IV_RV：期权比实际波动还便宜时不卖）
  一年后比较两个仓位，看"挑"是否比"固定SPY"好。每天把所有ETF的IV/RV记进
  options_scan_log.json，以后要换排序方法（如IV Rank）时有历史数据。

规则（开仓前登记，不事后改；改规则要在wiki记录日期和原因）：
  开仓：每个仓位同时最多1笔、每只ETF最多1笔；VIX>VIX_MAX全部不开（跳空风险保护）
        账户里有不是本模块开的期权持仓时不开（避免和人工单混在一起）
        到期DTE_MIN-DTE_MAX天，取最接近DTE_TARGET的到期日
        卖出腿：|delta|在DELTA_MIN-DELTA_MAX之间、最接近DELTA_TARGET
        宽度按WIDTHS依次尝试，要求 收入≥MIN_CREDIT（每股），
        且对手价收入≥中间价收入×MIN_FILL_RATIO（买卖价差太宽的不做）
        （2026-09-30首次运行前修正：原规则"收入≥宽度20%"与"卖delta 0.20"数学上几乎
        不可能同时满足——窄价差的 收入/宽度 ≈ 卖出腿|delta|，按对手价还更低，
        当天10:30 SPY和全部ETF都被它挡掉。优势来自IV>RV，不来自收入占宽度比例）
        张数 = floor(账户价值×8% ÷ 每张最大亏损)，再受期权购买力限制；
        账户涨了自动多开、跌了自动少开，不足1张就不开
        限价 = 对手价收入（卖出腿bid − 买入腿ask），与options-log规则4口径一致
  平仓（每次检查按对手价 = 卖出腿ask − 买入腿bid 估算平仓成本）：
        止盈：平仓成本 ≤ 收入×TAKE_PROFIT
        止损：平仓成本 ≥ 收入×STOP_MULT（亏损约为收入的2倍）
        时间：剩余天数 ≤ EXIT_DTE
        开仓当天不平仓（账户<$25k，同日开平算一次日内交易，PDT限制）

Alpaca多腿（mleg）限价单符号：正数=借方（付钱），负数=贷方（收钱）。Alpaca文档页
没写，只在Python SDK参考里有；传成正数会把收钱的单当成付钱的单。

用法：python -m src.options_auto [--dry-run] [--no-open] [--status]
"""
import argparse
import json
import math
import os
import sys
from datetime import date, datetime, timedelta

import pytz

from . import alpaca_client
from .alpaca_options import MAX_LOSS_PCT, open_option_positions

ET = pytz.timezone("America/New_York")
_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
_FILE = os.path.join(_DATA, "options_auto.json")
_SCAN_LOG = os.path.join(_DATA, "options_scan_log.json")

CONTROL = "SPY"
SCAN_UNIVERSE = ("QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "GLD", "TLT")
DTE_MIN, DTE_MAX, DTE_TARGET = 30, 45, 35
DELTA_MIN, DELTA_MAX, DELTA_TARGET = 0.15, 0.25, 0.20
WIDTHS = (2.0, 1.0)
MIN_CREDIT = 0.10          # 每股最低收入，太少覆盖不了费用
MIN_FILL_RATIO = 0.70
MIN_IV_RV = 1.0
TAKE_PROFIT = 0.50
STOP_MULT = 3.0
EXIT_DTE = 21
VIX_MAX = 35.0
RISK_FREE = 0.04          # 只用于greeks缺失时自己算delta/IV
ACTIVE = ("pending_open", "open", "pending_close")


# ── 纯函数 ─────────────────────────────────────────────────────

def parse_occ(sym: str) -> tuple:
    """OCC代码 → (标的, 到期日, C/P, 行权价)。例：SPY261106P00640000"""
    root, ymd, cp, k = sym[:-15], sym[-15:-9], sym[-9], sym[-8:]
    return root, datetime.strptime(ymd, "%y%m%d").date(), cp, int(k) / 1000


def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_put(spot: float, k: float, t: float, iv: float, r: float = RISK_FREE) -> tuple:
    """Black-Scholes看跌期权 → (价格, delta)。t单位为年。"""
    d1 = (math.log(spot / k) + (r + iv * iv / 2) * t) / (iv * math.sqrt(t))
    d2 = d1 - iv * math.sqrt(t)
    price = k * math.exp(-r * t) * _ncdf(-d2) - spot * _ncdf(-d1)
    return price, _ncdf(d1) - 1


def implied_vol(price: float, spot: float, k: float, t: float) -> float | None:
    """由看跌期权价格二分反推隐含波动率；价格不合理时返回None。"""
    if price <= 0 or t <= 0:
        return None
    lo, hi = 0.01, 3.0
    if not bs_put(spot, k, t, lo)[0] <= price <= bs_put(spot, k, t, hi)[0]:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_put(spot, k, t, mid)[0] < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def implied_put_delta(price: float, spot: float, k: float, t: float) -> float | None:
    iv = implied_vol(price, spot, k, t)
    return None if iv is None else bs_put(spot, k, t, iv)[1]


def pick_expiry(rows: list, today: date):
    exps = sorted({r["expiry"] for r in rows if DTE_MIN <= (r["expiry"] - today).days <= DTE_MAX})
    return min(exps, key=lambda e: (abs((e - today).days - DTE_TARGET), e)) if exps else None


def atm_iv(rows: list, spot: float, expiry: date, today: date) -> float | None:
    """该到期日最接近平值的put的隐含波动率（快照没给就由中间价反推）。"""
    same = [r for r in rows if r["expiry"] == expiry and r["bid"] and r["ask"]]
    if not same:
        return None
    r = min(same, key=lambda r: abs(r["strike"] - spot))
    if r.get("iv"):
        return float(r["iv"])
    return implied_vol((r["bid"] + r["ask"]) / 2, spot, r["strike"], (expiry - today).days / 365)


def size_qty(max_loss_per: float, account_value: float, buying_power: float | None) -> int:
    """张数 = floor(账户8% ÷ 每张最大亏损)，再受期权购买力限制（纯函数）。"""
    if max_loss_per <= 0:
        return 0
    q = math.floor(account_value * MAX_LOSS_PCT / 100 / max_loss_per + 1e-9)
    if buying_power is not None:
        q = min(q, math.floor(buying_power / max_loss_per + 1e-9))
    return max(q, 0)


def pick_spread(rows: list, spot: float, today: date, account_value: float,
                buying_power: float | None = None) -> dict:
    """
    从某只ETF的put期权快照里按规则选价差（纯函数）。
    rows: [{"symbol","expiry","strike","bid","ask","delta"(可为None),"iv"(可选)}]
    返回 {"ok": True, ...计划} 或 {"ok": False, "note": 原因}。
    """
    expiry = pick_expiry(rows, today)
    if expiry is None:
        return {"ok": False, "note": f"没有{DTE_MIN}-{DTE_MAX}天到期的合约"}
    t = (expiry - today).days / 365
    by_strike = {r["strike"]: r for r in rows if r["expiry"] == expiry}

    cands = []
    for r in by_strike.values():
        if not r["bid"] or not r["ask"] or r["ask"] < r["bid"]:
            continue
        d = r.get("delta")
        if d is None:
            d = implied_put_delta((r["bid"] + r["ask"]) / 2, spot, r["strike"], t)
        if d is not None and DELTA_MIN <= abs(d) <= DELTA_MAX:
            cands.append((abs(abs(d) - DELTA_TARGET), r, d))
    if not cands:
        return {"ok": False, "note": f"{expiry}没有|delta|在{DELTA_MIN}-{DELTA_MAX}的put"}
    _, short, delta = min(cands, key=lambda c: c[0])

    tried = []
    for w in WIDTHS:
        long = by_strike.get(round(short["strike"] - w, 2))
        if not long or not long["ask"]:
            tried.append(f"宽{w:g}：没有{short['strike'] - w:g}行权价")
            continue
        credit = round(short["bid"] - long["ask"], 2)
        mid = (short["bid"] + short["ask"]) / 2 - (long["bid"] + long["ask"]) / 2
        max_loss = round((w - credit) * 100, 2)
        qty = size_qty(max_loss, account_value, buying_power)
        if credit < MIN_CREDIT:
            tried.append(f"宽{w:g}：收入${credit:.2f}<${MIN_CREDIT:.2f}")
        elif mid > 0 and credit < round(mid * MIN_FILL_RATIO, 2):
            tried.append(f"宽{w:g}：对手价${credit:.2f}<中间价${mid:.2f}的{MIN_FILL_RATIO:.0%}（买卖价差太宽）")
        elif qty < 1:
            tried.append(f"宽{w:g}：每张最大亏损${max_loss:.0f}，账户{MAX_LOSS_PCT:.0f}%或购买力不够1张")
        else:
            return {"ok": True, "expiry": expiry, "dte": (expiry - today).days,
                    "short_sym": short["symbol"], "long_sym": long["symbol"],
                    "short_strike": short["strike"], "long_strike": long["strike"], "width": w,
                    "credit": credit, "qty": qty, "max_loss": round(max_loss * qty, 2),
                    "short_delta": round(delta, 3)}
    return {"ok": False, "note": f"卖{short['strike']:g}P（delta {delta:.2f}）没有合格宽度：" + "；".join(tried)}


def rank_scan(cands: list) -> list:
    """扫描候选按IV/RV从高到低；只留规则可做且IV/RV≥MIN_IV_RV的（纯函数）。"""
    ok = [c for c in cands if c["plan"]["ok"] and c.get("iv_rv") is not None and c["iv_rv"] >= MIN_IV_RV]
    return sorted(ok, key=lambda c: -c["iv_rv"])


def close_debit(q_short: dict | None, q_long: dict | None) -> float | None:
    """按对手价估算平仓成本（每股）：买回卖出腿按ask，卖掉买入腿按bid。"""
    if not q_short or not q_long or not q_short.get("ask"):
        return None
    return round(max(0.0, q_short["ask"] - (q_long.get("bid") or 0)), 2)


def exit_decision(tr: dict, debit: float | None, today: date) -> dict:
    """按登记的平仓规则判断（纯函数）。返回 {"action": "hold"/"close", "reason"}。"""
    if tr.get("opened") == today.isoformat():
        return {"action": "hold", "reason": "开仓当天不平（PDT）"}
    dte = (date.fromisoformat(tr["expiry"]) - today).days
    credit = tr["credit"]
    if debit is not None and debit <= round(credit * TAKE_PROFIT, 2):
        return {"action": "close", "reason": f"止盈：平仓成本${debit:.2f}≤收入${credit:.2f}的{TAKE_PROFIT:.0%}"}
    if debit is not None and debit >= round(credit * STOP_MULT, 2):
        return {"action": "close", "reason": f"止损：平仓成本${debit:.2f}≥收入${credit:.2f}的{STOP_MULT:g}倍"}
    if dte <= EXIT_DTE:
        return {"action": "close", "reason": f"时间：剩{dte}天≤{EXIT_DTE}天"}
    if debit is None:
        return {"action": "hold", "reason": "取不到报价"}
    return {"action": "hold", "reason": f"平仓成本${debit:.2f}，剩{dte}天"}


def arm_of(tr: dict) -> str:
    return tr.get("arm", "control")   # 2026-09-30扫描上线前的记录都是SPY对照


# ── 存取与行情 ─────────────────────────────────────────────────

def _load(path: str | None = None) -> list:
    try:
        with open(path or _FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def _save(trades: list, path: str | None = None):
    path = path or _FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(trades, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, path)


def _snap_row(sym: str, snap) -> dict | None:
    q = getattr(snap, "latest_quote", None)
    if q is None:
        return None
    g = getattr(snap, "greeks", None)
    _, exp, _, k = parse_occ(sym)
    return {"symbol": sym, "expiry": exp, "strike": k,
            "bid": float(q.bid_price or 0), "ask": float(q.ask_price or 0),
            "delta": getattr(g, "delta", None) if g else None,
            "iv": getattr(snap, "implied_volatility", None)}


class AlpacaMarket:
    """真实行情：期权快照走Alpaca，标的价/实际波动率/VIX走yfinance。测试里替换成假对象。"""

    def __init__(self):
        self.data = alpaca_client.option_data_client()

    def put_chain(self, underlying: str, spot: float, today: date) -> list:
        from alpaca.data.requests import OptionChainRequest
        from alpaca.trading.enums import ContractType
        req = OptionChainRequest(
            underlying_symbol=underlying, type=ContractType.PUT,
            expiration_date_gte=today + timedelta(days=DTE_MIN),
            expiration_date_lte=today + timedelta(days=DTE_MAX),
            strike_price_gte=round(spot * 0.85, 2), strike_price_lte=round(spot * 1.02, 2))
        rows = [_snap_row(s, v) for s, v in self.data.get_option_chain(req).items()]
        return [r for r in rows if r]

    def quotes(self, symbols: list) -> dict:
        from alpaca.data.requests import OptionSnapshotRequest
        snaps = self.data.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=symbols))
        return {s: _snap_row(s, v) for s, v in snaps.items()}

    def vix(self) -> float | None:
        import yfinance as yf
        v = yf.Ticker("^VIX").history(period="5d")["Close"]
        return round(float(v.iloc[-1]), 2) if len(v) else None

    def underlying(self, sym: str) -> dict:
        """标的价、20日实际波动率（年化）、是否在200日线上方。"""
        import numpy as np
        import yfinance as yf
        h = yf.Ticker(sym).history(period="1y")["Close"]
        spot = float(h.iloc[-1])
        rets = np.log(h / h.shift(1)).dropna().iloc[-20:]
        return {"spot": round(spot, 2), "rv20": round(float(rets.std() * math.sqrt(252)), 4),
                "above_ma200": bool(spot > float(h.iloc[-200:].mean()))}


def _order_status(o) -> str:
    s = getattr(o, "status", "")
    return str(getattr(s, "value", s)).lower()


def _submit_mleg(client, legs: list, limit_price: float, qty: int = 1):
    """legs: [(symbol, "buy"/"sell", intent)]。limit_price：正=借方，负=贷方。"""
    from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
    return client.submit_order(LimitOrderRequest(
        qty=qty, limit_price=round(limit_price, 2), order_class=OrderClass.MLEG,
        time_in_force=TimeInForce.DAY,
        legs=[OptionLegRequest(symbol=s, ratio_qty=1,
                               side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                               position_intent=getattr(PositionIntent, intent))
              for s, side, intent in legs]))


def _label(tr: dict) -> str:
    return (f"{tr['underlying']} {tr['expiry'][5:]} {tr['short_strike']:g}/{tr['long_strike']:g}P"
            + (f"×{tr['qty']}" if tr.get("qty", 1) > 1 else ""))


ARM_NAME = {"control": "对照SPY", "scan": "扫描"}


# ── 主流程 ─────────────────────────────────────────────────────

def _sync_orders(client, trades: list, today: date) -> list:
    """把挂单的成交/失效结果写回记录。"""
    msgs = []
    for tr in trades:
        oid = tr.get("open_order_id") if tr["status"] == "pending_open" else \
            tr.get("close_order_id") if tr["status"] == "pending_close" else None
        if not oid:
            continue
        o = client.get_order_by_id(oid)
        st = _order_status(o)
        px = abs(float(o.filled_avg_price)) if getattr(o, "filled_avg_price", None) else None
        tag = f"[{ARM_NAME[arm_of(tr)]}]"
        if st == "filled" and tr["status"] == "pending_open":
            filled = getattr(o, "filled_at", None)
            tr.update(status="open", credit=px if px is not None else tr["planned_credit"],
                      opened=(filled.astimezone(ET).date() if filled else today).isoformat())
            msgs.append(f"✅ <b>卖put价差成交</b>{tag} {_label(tr)}\n收入${tr['credit']:.2f}/股"
                        f"（最大亏损${(tr['width'] - tr['credit']) * 100 * tr['qty']:.0f}）")
        elif st == "filled" and tr["status"] == "pending_close":
            tr.update(status="closed", closed=today.isoformat(),
                      exit_debit=px if px is not None else tr.get("planned_exit_debit"))
            tr["pnl"] = round((tr["credit"] - tr["exit_debit"]) * 100 * tr["qty"], 2)
            msgs.append(f"🧾 <b>卖put价差平仓</b>{tag} {_label(tr)}\n{tr['close_reason']}\n"
                        f"收入${tr['credit']:.2f} − 平仓${tr['exit_debit']:.2f} → 盈亏${tr['pnl']:+.0f}")
        elif st in ("canceled", "expired", "rejected", "done_for_day"):
            if tr["status"] == "pending_open":
                tr["status"] = "not_filled"
                msgs.append(f"⚪ 卖put价差开仓单未成交（{st}）{tag}：{_label(tr)}，下个交易日重新选")
            else:
                tr["status"] = "open"   # 平仓单没成交，下次检查重新挂
                tr.pop("close_order_id", None)
    return msgs


def _candidate(market, sym: str, today: date, acct_value: float, bp: float | None) -> dict:
    u = market.underlying(sym)
    rows = market.put_chain(sym, u["spot"], today)
    exp = pick_expiry(rows, today)
    iv = atm_iv(rows, u["spot"], exp, today) if exp else None
    iv_rv = round(iv / u["rv20"], 2) if iv and u["rv20"] else None
    return {"underlying": sym, **u, "iv": round(iv, 4) if iv else None, "iv_rv": iv_rv,
            "plan": pick_spread(rows, u["spot"], today, acct_value, bp)}


def _open(client, trades: list, cand: dict, arm: str, vix, acct_value: float, today: date) -> str:
    plan = cand["plan"]
    o = _submit_mleg(client, [(plan["short_sym"], "sell", "SELL_TO_OPEN"),
                              (plan["long_sym"], "buy", "BUY_TO_OPEN")], -plan["credit"], plan["qty"])
    tr = {"id": max([t["id"] for t in trades] or [0]) + 1, "arm": arm, "underlying": cand["underlying"],
          "expiry": plan["expiry"].isoformat(), "short_strike": plan["short_strike"],
          "long_strike": plan["long_strike"], "width": plan["width"], "qty": plan["qty"],
          "short_sym": plan["short_sym"], "long_sym": plan["long_sym"],
          "planned_credit": plan["credit"], "short_delta": plan["short_delta"],
          "context": {"spot": cand["spot"], "rv20": cand["rv20"], "iv": cand["iv"], "iv_rv": cand["iv_rv"],
                      "above_ma200": cand["above_ma200"], "vix": vix, "account": acct_value, "dte": plan["dte"]},
          "submitted": today.isoformat(), "open_order_id": str(o.id), "status": "pending_open"}
    trades.append(tr)
    return (f"📥 <b>卖put价差挂开仓单</b>[{ARM_NAME[arm]}] {_label(tr)}（{plan['dte']}天）\n"
            f"限价贷方${plan['credit']:.2f}×{plan['qty']}张，卖出腿delta {plan['short_delta']}，"
            f"最大亏损${plan['max_loss']:.0f}；IV/RV {cand['iv_rv']}，VIX {vix}")


def run(dry_run: bool = False, allow_open: bool = True, today: date | None = None,
        client=None, market=None) -> dict:
    """一次检查：同步挂单 → 按规则平仓 → 空出的仓位按规则开仓。返回 {"msgs", "status"}。"""
    today = today or datetime.now(ET).date()
    client = client or alpaca_client.paper_trading_client()
    if client is None:
        return {"msgs": [], "status": "未配置ALPACA密钥，跳过"}
    market = market or AlpacaMarket()
    trades = _load()
    msgs = [] if dry_run else _sync_orders(client, trades, today)
    notes = []

    for tr in (t for t in trades if t["status"] == "open"):
        q = market.quotes([tr["short_sym"], tr["long_sym"]])
        debit = close_debit(q.get(tr["short_sym"]), q.get(tr["long_sym"]))
        d = exit_decision(tr, debit, today)
        tr["last_check"] = {"date": today.isoformat(), "close_debit": debit}
        notes.append(f"{_label(tr)}：{d['reason']}")
        if d["action"] != "close" or debit is None:
            continue
        if dry_run:
            msgs.append(f"[空跑] 会平仓{_label(tr)}：{d['reason']}")
            continue
        o = _submit_mleg(client, [(tr["short_sym"], "buy", "BUY_TO_CLOSE"),
                                  (tr["long_sym"], "sell", "SELL_TO_CLOSE")], debit, tr["qty"])
        tr.update(status="pending_close", close_order_id=str(o.id), close_reason=d["reason"],
                  planned_exit_debit=debit)
        msgs.append(f"📤 卖put价差挂平仓单[{ARM_NAME[arm_of(tr)]}] {_label(tr)}：{d['reason']}，限价借方${debit:.2f}")

    active = [t for t in trades if t["status"] in ACTIVE]
    free_arms = [a for a in ("control", "scan") if not any(arm_of(t) == a for t in active)]
    if allow_open:
        ours = {s for t in active for s in (t["short_sym"], t["long_sym"])}
        foreign = [p.symbol for p in open_option_positions(client) if p.symbol not in ours]
        vix = market.vix()
        if foreign:
            notes.append(f"账户里有不是本模块开的期权持仓{foreign[:2]}，不开新仓")
        elif vix is not None and vix > VIX_MAX:
            notes.append(f"VIX {vix}>{VIX_MAX:g}，不开仓")
        else:
            acct = client.get_account()
            value = float(acct.portfolio_value)
            bp = getattr(acct, "options_buying_power", None) or getattr(acct, "buying_power", None)
            bp = float(bp) if bp is not None else None
            held = {t["underlying"] for t in active}
            opened_msgs = []

            if "control" in free_arms and CONTROL not in held:
                c = _candidate(market, CONTROL, today, value, bp)
                if not c["plan"]["ok"]:
                    notes.append(f"对照SPY不开：{c['plan']['note']}")
                elif dry_run:
                    opened_msgs.append(f"[空跑] 对照会卖{_label_plan(c)}")
                else:
                    opened_msgs.append(_open(client, trades, c, "control", vix, value, today))
                    bp = None if bp is None else bp - c["plan"]["max_loss"]

            # 扫描每天都跑（10:30那次），记录全部ETF的IV/RV；有空仓位才下单
            scan = []
            for sym in SCAN_UNIVERSE:
                try:
                    scan.append(_candidate(market, sym, today, value, bp))
                except Exception as e:
                    scan.append({"underlying": sym, "error": str(e)[:100], "plan": {"ok": False}})
            if not dry_run:
                log = _load(_SCAN_LOG)
                log.append({"date": today.isoformat(), "vix": vix, "rows": [
                    {k: c.get(k) for k in ("underlying", "spot", "rv20", "iv", "iv_rv", "error")}
                    | {"plan_ok": c["plan"]["ok"]} for c in scan]})
                _save(log[-750:], _SCAN_LOG)
            ranked = [c for c in rank_scan(scan) if c["underlying"] not in held]
            top = "，".join(f"{c['underlying']} {c['iv_rv']}" for c in
                           sorted((c for c in scan if c.get("iv_rv")), key=lambda c: -c["iv_rv"])[:3])
            notes.append(f"IV/RV前3：{top or '无'}")
            if "scan" in free_arms:
                if not ranked:
                    notes.append(f"扫描不开：没有IV/RV≥{MIN_IV_RV:g}且规则可做的ETF")
                elif dry_run:
                    opened_msgs.append(f"[空跑] 扫描会卖{_label_plan(ranked[0])}")
                else:
                    opened_msgs.append(_open(client, trades, ranked[0], "scan", vix, value, today))
            msgs += opened_msgs
    if not dry_run:
        _save(trades)
    return {"msgs": msgs, "status": "；".join(notes) or status_line(trades)}


def _label_plan(c: dict) -> str:
    p = c["plan"]
    return (f"{c['underlying']} {p['expiry']} {p['short_strike']:g}/{p['long_strike']:g}P×{p['qty']}，"
            f"收入${p['credit']:.2f}，delta {p['short_delta']}，最大亏损${p['max_loss']:.0f}，IV/RV {c.get('iv_rv')}")


def status_line(trades: list | None = None) -> str:
    trades = _load() if trades is None else trades
    out = []
    for arm in ("control", "scan"):
        mine = [t for t in trades if arm_of(t) == arm]
        closed = [t for t in mine if t["status"] == "closed"]
        active = [t for t in mine if t["status"] in ACTIVE]
        parts = [f"{_label(t)} {t['status']}" for t in active] or ["无持仓"]
        if closed:
            pnl = sum(t.get("pnl", 0) for t in closed)
            wins = sum(1 for t in closed if t.get("pnl", 0) > 0)
            parts.append(f"已平{len(closed)}笔（胜{wins}），累计${pnl:+.0f}")
        out.append(f"{ARM_NAME[arm]}：" + "，".join(parts))
    return "卖put价差 " + "；".join(out)


def run_in_subprocess(allow_open: bool = True, timeout: int = 600) -> dict:
    """供scheduler调用：子进程运行，alpaca-py/yfinance不常驻scheduler内存。"""
    import subprocess
    root = os.path.join(os.path.dirname(__file__), "..")
    args = [sys.executable, "-m", "src.options_auto", "--json"] + ([] if allow_open else ["--no-open"])
    r = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()
        raise RuntimeError(tail[-1] if tail else f"exit {r.returncode}")
    return json.loads(r.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ETF卖put价差自动开仓（Alpaca模拟账户）")
    ap.add_argument("--dry-run", action="store_true", help="只算不下单、不写记录")
    ap.add_argument("--no-open", action="store_true", help="只检查平仓，不开新仓")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--json", action="store_true", help="最后一行输出JSON（供scheduler解析）")
    a = ap.parse_args()
    if a.status:
        print(status_line())
        sys.exit(0)
    res = run(dry_run=a.dry_run, allow_open=not a.no_open)
    # 顺带给动量账本的Alpaca镜像对账（本地-20%止损平仓后Alpaca跟着卖）：放在这个
    # 每天两次的子进程里，scheduler不用改也不用常驻alpaca-py
    try:
        from .momentum_book import sync_alpaca
        res["msgs"] += [f"📈 动量账本{x}" for x in sync_alpaca(dry_run=a.dry_run)]
    except Exception as e:
        res["msgs"].append(f"❌ 动量账本Alpaca对账失败：{str(e)[:120]}")
    if a.json:
        print(json.dumps(res, ensure_ascii=False))
    else:
        for m in res["msgs"]:
            print(m)
        print(res["status"])
