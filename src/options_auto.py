"""
SPY卖put价差自动开仓（2026-09-30新增，Alpaca模拟账户，前向实验）

背景：用户希望由程序判断期权开仓时机、不再手工纸上交易。本项目的方向性信号
（九关/cold_model）回测没有优势，拿来买期权等于付费押随机方向；期权里有外部实证的
机械优势是"卖指数put"（波动率风险溢价），见wiki stock-master/options-log
"自动卖put价差"。所以这里只做一件事：按事先写死的规则，在SPY上机械地卖出虚值
put价差，限定最大亏损。只下Alpaca模拟账户（alpaca_client写死paper=True）。

规则（开仓前登记，不事后改；改规则要在wiki记录日期和原因）：
  开仓：同时最多1笔（沿用options-log规则3）；VIX>VIX_MAX不开（跳空风险保护）
        到期DTE_MIN-DTE_MAX天，取最接近DTE_TARGET的到期日
        卖出腿：|delta|在DELTA_MIN-DELTA_MAX之间、最接近DELTA_TARGET
        宽度按WIDTHS依次尝试，要求 收入≥宽度×MIN_CREDIT_RATIO 且 最大亏损≤账户8%
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

UNDERLYING = "SPY"
DTE_MIN, DTE_MAX, DTE_TARGET = 30, 45, 35
DELTA_MIN, DELTA_MAX, DELTA_TARGET = 0.15, 0.25, 0.20
WIDTHS = (2.0, 1.0)
MIN_CREDIT_RATIO = 0.20
TAKE_PROFIT = 0.50
STOP_MULT = 3.0
EXIT_DTE = 21
VIX_MAX = 35.0
RISK_FREE = 0.04          # 只用于greeks缺失时自己算delta
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


def implied_put_delta(price: float, spot: float, k: float, t: float) -> float | None:
    """由期权中间价二分反推隐含波动率，再算delta；价格不合理时返回None。"""
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
    return bs_put(spot, k, t, (lo + hi) / 2)[1]


def pick_spread(rows: list, spot: float, today: date, account_value: float) -> dict:
    """
    从put期权快照里按规则选价差（纯函数）。
    rows: [{"symbol","expiry","strike","bid","ask","delta"(可为None)}]
    返回 {"ok": True, ...计划} 或 {"ok": False, "note": 原因}。
    """
    exps = sorted({r["expiry"] for r in rows if DTE_MIN <= (r["expiry"] - today).days <= DTE_MAX})
    if not exps:
        return {"ok": False, "note": f"没有{DTE_MIN}-{DTE_MAX}天到期的合约"}
    expiry = min(exps, key=lambda e: (abs((e - today).days - DTE_TARGET), e))
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

    max_loss_cap = account_value * MAX_LOSS_PCT / 100
    tried = []
    for w in WIDTHS:
        long = by_strike.get(short["strike"] - w)
        if not long or not long["ask"]:
            tried.append(f"宽{w:g}：没有{short['strike'] - w:g}行权价")
            continue
        credit = round(short["bid"] - long["ask"], 2)
        max_loss = round((w - credit) * 100, 2)
        if credit < w * MIN_CREDIT_RATIO:
            tried.append(f"宽{w:g}：收入${credit:.2f}<宽度{MIN_CREDIT_RATIO:.0%}")
        elif max_loss > max_loss_cap:
            tried.append(f"宽{w:g}：最大亏损${max_loss:.0f}>账户{MAX_LOSS_PCT:.0f}%（${max_loss_cap:.0f}）")
        else:
            return {"ok": True, "expiry": expiry, "dte": (expiry - today).days,
                    "short_sym": short["symbol"], "long_sym": long["symbol"],
                    "short_strike": short["strike"], "long_strike": long["strike"], "width": w,
                    "credit": credit, "max_loss": max_loss, "short_delta": round(delta, 3)}
    return {"ok": False, "note": f"卖{short['strike']:g}P（delta {delta:.2f}）没有合格宽度：" + "；".join(tried)}


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


# ── 存取与行情 ─────────────────────────────────────────────────

def _load() -> list:
    try:
        with open(_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def _save(trades: list):
    os.makedirs(_DATA, exist_ok=True)
    tmp = _FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(trades, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, _FILE)


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
    """真实行情：期权快照走Alpaca，标的价/VIX走yfinance。测试里替换成假对象。"""

    def __init__(self):
        self.data = alpaca_client.option_data_client()

    def put_chain(self, spot: float, today: date) -> list:
        from alpaca.data.requests import OptionChainRequest
        from alpaca.trading.enums import ContractType
        req = OptionChainRequest(
            underlying_symbol=UNDERLYING, type=ContractType.PUT,
            expiration_date_gte=today + timedelta(days=DTE_MIN),
            expiration_date_lte=today + timedelta(days=DTE_MAX),
            strike_price_gte=round(spot * 0.85), strike_price_lte=round(spot))
        rows = [_snap_row(s, v) for s, v in self.data.get_option_chain(req).items()]
        return [r for r in rows if r]

    def quotes(self, symbols: list) -> dict:
        from alpaca.data.requests import OptionSnapshotRequest
        snaps = self.data.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=symbols))
        return {s: _snap_row(s, v) for s, v in snaps.items()}

    def context(self) -> dict:
        """标的价、是否在200日线上方、VIX（只作记录和VIX开仓保护）。"""
        import yfinance as yf
        h = yf.Ticker(UNDERLYING).history(period="1y")["Close"]
        vix = yf.Ticker("^VIX").history(period="5d")["Close"]
        spot = float(h.iloc[-1])
        return {"spot": round(spot, 2), "above_ma200": bool(spot > float(h.iloc[-200:].mean())),
                "vix": round(float(vix.iloc[-1]), 2) if len(vix) else None}


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
    return f"{tr['underlying']} {tr['expiry'][5:]} {tr['short_strike']:g}/{tr['long_strike']:g}P"


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
        if st == "filled" and tr["status"] == "pending_open":
            filled = getattr(o, "filled_at", None)
            tr.update(status="open", credit=px if px is not None else tr["planned_credit"],
                      opened=(filled.astimezone(ET).date() if filled else today).isoformat())
            msgs.append(f"✅ <b>SPY卖put价差成交</b> {_label(tr)}\n收入${tr['credit']:.2f}/股"
                        f"（最大亏损${(tr['width'] - tr['credit']) * 100:.0f}）")
        elif st == "filled" and tr["status"] == "pending_close":
            tr.update(status="closed", closed=today.isoformat(),
                      exit_debit=px if px is not None else tr.get("planned_exit_debit"))
            tr["pnl"] = round((tr["credit"] - tr["exit_debit"]) * 100 * tr["qty"], 2)
            msgs.append(f"🧾 <b>SPY卖put价差平仓</b> {_label(tr)}\n{tr['close_reason']}\n"
                        f"收入${tr['credit']:.2f} − 平仓${tr['exit_debit']:.2f} → 盈亏${tr['pnl']:+.0f}")
        elif st in ("canceled", "expired", "rejected", "done_for_day"):
            if tr["status"] == "pending_open":
                tr["status"] = "not_filled"
                msgs.append(f"⚪ SPY卖put价差开仓单未成交（{st}）：{_label(tr)}，下个交易日重新选")
            else:
                tr["status"] = "open"   # 平仓单没成交，下次检查重新挂
                tr.pop("close_order_id", None)
    return msgs


def run(dry_run: bool = False, allow_open: bool = True, today: date | None = None,
        client=None, market=None) -> dict:
    """一次检查：同步挂单 → 按规则平仓 → 没有持仓时按规则开仓。返回 {"msgs", "status"}。"""
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
        msgs.append(f"📤 SPY卖put价差挂平仓单 {_label(tr)}：{d['reason']}，限价借方${debit:.2f}")

    active = [t for t in trades if t["status"] in ACTIVE]
    if allow_open and not active:
        if open_option_positions(client):
            notes.append("Alpaca账户里已有其他期权持仓，按同时最多1笔不开仓")
        else:
            ctx = market.context()
            if ctx["vix"] is not None and ctx["vix"] > VIX_MAX:
                notes.append(f"VIX {ctx['vix']}>{VIX_MAX:g}，不开仓")
            else:
                acct = float(client.get_account().portfolio_value)
                plan = pick_spread(market.put_chain(ctx["spot"], today), ctx["spot"], today, acct)
                if not plan["ok"]:
                    notes.append(f"不开仓：{plan['note']}")
                elif dry_run:
                    msgs.append(f"[空跑] 会卖{UNDERLYING} {plan['expiry']} {plan['short_strike']:g}/"
                                f"{plan['long_strike']:g}P，收入${plan['credit']:.2f}，delta {plan['short_delta']}，"
                                f"最大亏损${plan['max_loss']:.0f}（账户${acct:,.0f}，VIX {ctx['vix']}）")
                else:
                    o = _submit_mleg(client, [(plan["short_sym"], "sell", "SELL_TO_OPEN"),
                                              (plan["long_sym"], "buy", "BUY_TO_OPEN")], -plan["credit"])
                    tr = {"id": max([t["id"] for t in trades] or [0]) + 1, "underlying": UNDERLYING,
                          "expiry": plan["expiry"].isoformat(), "short_strike": plan["short_strike"],
                          "long_strike": plan["long_strike"], "width": plan["width"], "qty": 1,
                          "short_sym": plan["short_sym"], "long_sym": plan["long_sym"],
                          "planned_credit": plan["credit"], "short_delta": plan["short_delta"],
                          "context": {**ctx, "account": acct, "dte": plan["dte"]},
                          "submitted": today.isoformat(), "open_order_id": str(o.id),
                          "status": "pending_open"}
                    trades.append(tr)
                    msgs.append(f"📥 <b>SPY卖put价差挂开仓单</b> {_label(tr)}（{plan['dte']}天）\n"
                                f"限价贷方${plan['credit']:.2f}，卖出腿delta {plan['short_delta']}，"
                                f"最大亏损${plan['max_loss']:.0f}；VIX {ctx['vix']}"
                                f"{'，SPY在200日线上方' if ctx['above_ma200'] else '，SPY在200日线下方'}")
    if not dry_run:
        _save(trades)
    return {"msgs": msgs, "status": "；".join(notes) or status_line(trades)}


def status_line(trades: list | None = None) -> str:
    trades = _load() if trades is None else trades
    closed = [t for t in trades if t["status"] == "closed"]
    active = [t for t in trades if t["status"] in ACTIVE]
    parts = [f"{_label(t)} {t['status']}" for t in active] or ["无持仓"]
    if closed:
        pnl = sum(t.get("pnl", 0) for t in closed)
        wins = sum(1 for t in closed if t.get("pnl", 0) > 0)
        parts.append(f"已平{len(closed)}笔（胜{wins}），累计${pnl:+.0f}")
    return "SPY卖put：" + "，".join(parts)


def run_in_subprocess(allow_open: bool = True, timeout: int = 300) -> dict:
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
    ap = argparse.ArgumentParser(description="SPY卖put价差自动开仓（Alpaca模拟账户）")
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
