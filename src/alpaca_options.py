"""
Alpaca期权模拟交易（2026-09-29新增，配合wiki stock-master/options-log）

用途：把options-log的纸上交易从"手工记对手价"换成在Alpaca模拟账户里真实挂单，
成交价、持仓、盈亏由Alpaca记录。只做模拟账户（paper=True写死）。本模块不会自动下单——
只能由人通过命令行显式执行open（自动下单的只有src/options_auto.py的SPY卖put价差），且下单前强制检查options-log的规则：
  规则1：单笔最大亏损 ≤ 账户8%
  规则2：价差 ≥ 14天到期
  规则3：同时最多1笔期权持仓

用法（需先配置ALPACA_API_KEY/ALPACA_SECRET_KEY，见alpaca_client）：
  python -m src.alpaca_options quote ASTS 2026-10-16 62 60 P
  python -m src.alpaca_options open  ASTS 2026-10-16 62 60 P --limit 1.00 --confirm
  python -m src.alpaca_options positions

⚠️ 多腿（MLEG）限价单的limit_price符号约定（借方为正）按Alpaca文档实现，
   尚未在真实模拟账户上验证过，第一次下单时要核对订单详情。
"""
import argparse
from datetime import date, datetime

import pytz

from . import alpaca_client

ET = pytz.timezone("America/New_York")

MAX_LOSS_PCT      = 8.0    # options-log规则1
MIN_SPREAD_DTE    = 14     # options-log规则2
MAX_OPEN_SPREADS  = 1      # options-log规则3


def occ_symbol(root: str, expiry: date, cp: str, strike: float) -> str:
    """OCC期权代码，例：('ASTS', 2026-10-16, 'P', 62) → ASTS261016P00062000"""
    return f"{root.upper()}{expiry:%y%m%d}{cp.upper()}{int(round(strike * 1000)):08d}"


def vertical_quote(chain: dict, long_sym: str, short_sym: str) -> dict:
    """
    由期权快照算价差报价（纯函数）。chain: {symbol: {"bid":..,"ask":..}}。
    对手价（natural）= 买腿ask − 卖腿bid，是options-log规则4的记账口径。
    """
    lg, sh = chain.get(long_sym), chain.get(short_sym)
    if not lg or not sh:
        missing = [s for s, q in ((long_sym, lg), (short_sym, sh)) if not q]
        return {"ok": False, "note": f"期权链里没有{', '.join(missing)}"}
    natural = round(lg["ask"] - sh["bid"], 2)
    mid = round((lg["bid"] + lg["ask"]) / 2 - (sh["bid"] + sh["ask"]) / 2, 2)
    return {"ok": True, "natural": natural, "mid": mid,
            "friction_pct": round((natural - mid) / natural * 100, 1) if natural > 0 else None,
            "long": lg, "short": sh}


def check_rules(debit: float, width: float, expiry: date, account_value: float,
                open_spreads: int, qty: int = 1, today: date | None = None) -> list:
    """返回违反的规则列表（空=可以下单，纯函数）。debit为每股借方，1张=100股。"""
    today = today or datetime.now(ET).date()
    problems = []
    max_loss = debit * 100 * qty
    if debit <= 0 or debit >= width:
        problems.append(f"借方${debit:.2f}不合理（应在0和价差宽度${width:.2f}之间）")
    if account_value and max_loss > account_value * MAX_LOSS_PCT / 100:
        problems.append(f"最大亏损${max_loss:.0f}超过账户{MAX_LOSS_PCT:.0f}%（${account_value * MAX_LOSS_PCT / 100:.0f}）")
    dte = (expiry - today).days
    if dte < MIN_SPREAD_DTE:
        problems.append(f"距到期{dte}天，少于价差最短{MIN_SPREAD_DTE}天")
    if open_spreads >= MAX_OPEN_SPREADS:
        problems.append(f"已有{open_spreads}笔期权持仓，上限{MAX_OPEN_SPREADS}笔")
    return problems


def fetch_chain(underlying: str, expiry: date, cp: str) -> dict:
    """Alpaca期权快照 → {symbol: {"bid","ask","iv","delta"}}；未配置返回{}。"""
    client = alpaca_client.option_data_client()
    if client is None:
        return {}
    from alpaca.data.requests import OptionChainRequest
    from alpaca.trading.enums import ContractType
    req = OptionChainRequest(underlying_symbol=underlying.upper(), expiration_date=expiry,
                             type=ContractType.PUT if cp.upper() == "P" else ContractType.CALL)
    out = {}
    for sym, snap in client.get_option_chain(req).items():
        q = getattr(snap, "latest_quote", None)
        if q is None:
            continue
        g = getattr(snap, "greeks", None)
        out[sym] = {"bid": float(q.bid_price or 0), "ask": float(q.ask_price or 0),
                    "iv": getattr(snap, "implied_volatility", None),
                    "delta": getattr(g, "delta", None) if g else None}
    return out


def open_option_positions(client) -> list:
    def _cls(p):
        c = getattr(p, "asset_class", None)
        return getattr(c, "value", c)
    return [p for p in client.get_all_positions() if _cls(p) == "us_option"]


def submit_vertical(underlying: str, expiry: date, long_strike: float, short_strike: float,
                    cp: str, limit_price: float, qty: int = 1, confirm: bool = False) -> dict:
    """在模拟账户挂价差限价单。confirm=False时只做检查、不下单。"""
    client = alpaca_client.paper_trading_client()
    if client is None:
        return {"ok": False, "note": "未配置ALPACA_API_KEY/ALPACA_SECRET_KEY"}
    long_sym = occ_symbol(underlying, expiry, cp, long_strike)
    short_sym = occ_symbol(underlying, expiry, cp, short_strike)
    acct = client.get_account()
    # 每个价差占两条腿的持仓记录
    open_spreads = (len(open_option_positions(client)) + 1) // 2
    problems = check_rules(limit_price, abs(long_strike - short_strike), expiry,
                           float(acct.portfolio_value), open_spreads, qty)
    if problems:
        return {"ok": False, "note": "违反options-log规则：" + "；".join(problems)}
    if not confirm:
        return {"ok": True, "dry_run": True, "legs": [long_sym, short_sym], "limit": limit_price,
                "note": "检查通过，加--confirm才会真正在模拟账户下单"}

    from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
    order = client.submit_order(LimitOrderRequest(
        qty=qty, limit_price=round(limit_price, 2), order_class=OrderClass.MLEG,
        time_in_force=TimeInForce.DAY,
        legs=[OptionLegRequest(symbol=long_sym, ratio_qty=1, side=OrderSide.BUY,
                               position_intent=PositionIntent.BUY_TO_OPEN),
              OptionLegRequest(symbol=short_sym, ratio_qty=1, side=OrderSide.SELL,
                               position_intent=PositionIntent.SELL_TO_OPEN)],
    ))
    return {"ok": True, "order_id": str(order.id), "status": str(order.status), "legs": [long_sym, short_sym]}


def _main():
    ap = argparse.ArgumentParser(description="Alpaca期权模拟交易（价差）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("quote", "open"):
        p = sub.add_parser(name)
        p.add_argument("underlying")
        p.add_argument("expiry", help="YYYY-MM-DD")
        p.add_argument("long_strike", type=float)
        p.add_argument("short_strike", type=float)
        p.add_argument("cp", choices=["P", "C", "p", "c"])
        if name == "open":
            p.add_argument("--limit", type=float, required=True, help="每股借方限价")
            p.add_argument("--qty", type=int, default=1)
            p.add_argument("--confirm", action="store_true")
    sub.add_parser("positions")
    a = ap.parse_args()

    if not alpaca_client.is_configured():
        print("未配置ALPACA_API_KEY/ALPACA_SECRET_KEY")
        return
    if a.cmd == "positions":
        for p in open_option_positions(alpaca_client.paper_trading_client()):
            print(p.symbol, p.qty, p.avg_entry_price, p.unrealized_pl)
        return
    expiry = date.fromisoformat(a.expiry)
    if a.cmd == "quote":
        chain = fetch_chain(a.underlying, expiry, a.cp)
        q = vertical_quote(chain, occ_symbol(a.underlying, expiry, a.cp, a.long_strike),
                           occ_symbol(a.underlying, expiry, a.cp, a.short_strike))
        print(q if not q["ok"] else
              f"对手价${q['natural']:.2f}  中间价${q['mid']:.2f}  摩擦{q['friction_pct']}%  "
              f"最大亏损${q['natural'] * 100:.0f}  最大盈利${(abs(a.long_strike - a.short_strike) - q['natural']) * 100:.0f}")
    else:
        print(submit_vertical(a.underlying, expiry, a.long_strike, a.short_strike, a.cp,
                              a.limit, a.qty, a.confirm))


if __name__ == "__main__":
    _main()
