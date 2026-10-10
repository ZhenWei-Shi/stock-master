"""
行业ETF动量账本（2026-10-10用户决定，2026-10-30月末首次调仓，本地模拟记账、不下单）

用途：月度动量账本（momentum_book，前2只个股）的对照组。10/5-10/9动量账本回撤约10%，主要
来自AXTI+MU同在半导体；这里用同样的12-1个月动量，在37只行业/细分行业ETF里持有前5只，
回答"集中押2只个股"和"分散到5个行业ETF"哪个更好。

依据（wiki stock-master/overview"替代策略回测"，PR#22）：行业ETF 37只、12-1个月前5只，
相对全池等权+5.6%/年，t=1.35——方向对、统计上不显著；没有选股后见之明偏差。

规则（与backtest_alt.rank_portfolio一致，事先定好）：
  - 股票池：backtest_alt.ETF_UNIVERSE（37只）
  - 每月最后一个交易日15:40（跟在动量账本后面）：按close[t-21]/close[t-252]-1排名，持有前N_HOLD只，
    全部调回等权、满仓（零股）；新进入/退出的品种按单边COST_BPS扣成本（同回测：2×换手×成本）
  - 不设止损、不过风控层（ETF，回测里也没有）；不同步Alpaca（Alpaca账户已被动量账本+期权占满）
  - 初始INIT_VALUE（与动量账本相同），记账文件data/etf_momentum_book.json
  - 每次调仓同时记下动量账本净值和SPY收盘价，方便逐月对比

判定（登记）：满12个月后比较两本账的逐月收益：年化、最大回撤、相对同期SPY的超额。12个样本不够下
  统计结论，只看方向和回撤；结论写进wiki，不据此改动量账本规则（要改需另行登记）。

用法：python -m src.etf_momentum_book [--dry-run] [--status]
"""
import json
import os
import sys
from datetime import datetime

import pandas as pd

from .backtest_alt import ETF_UNIVERSE
from .momentum_book import ET, fractional_shares, is_last_trading_day, rank_momentum

_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "etf_momentum_book.json")
INIT_VALUE = 2000.0
N_HOLD = 5
COST_BPS = 10


def _load(path: str | None = None) -> dict:
    try:
        with open(path or _FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save(book: dict, path: str | None = None):
    path = path or _FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(book, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def book_value(book: dict, prices: dict) -> float:
    """现金 + 持仓市值；缺价格的持仓按上次调仓价（纯函数）。"""
    last = book.get("last_prices", {})
    return book.get("cash", 0.0) + sum(q * prices.get(t, last.get(t, 0.0)) for t, q in book.get("holdings", {}).items())


def plan(ranked: pd.Series, prices: dict, held: list, value: float) -> dict:
    """
    纯函数。按排名取前N_HOLD只（跳过没价格的），全部调回等权、满仓。
    成本 = 价值 × 2 × 换手 × COST_BPS（换手 = 新进入只数/N_HOLD，首次建仓为1）。
    """
    targets = [t for t in ranked.index if prices.get(t)][:N_HOLD]
    if not targets:
        return {"targets": [], "buy": [], "sell": held, "turnover": 0.0, "cost": 0.0, "holdings": {}}
    new = [t for t in targets if t not in held]
    turnover = 1.0 if not held else len(new) / N_HOLD
    cost = round(value * 2 * turnover * COST_BPS / 1e4, 2)
    each = (value - cost) / len(targets)
    return {"targets": targets, "buy": new, "sell": [t for t in held if t not in targets],
            "turnover": turnover, "cost": cost,
            "holdings": {t: fractional_shares(each, prices[t]) for t in targets}}


def _momentum_book_value() -> float | None:
    try:
        from .paper_trading import list_positions
        return (list_positions("momentum").get("account") or {}).get("current_value")
    except Exception:
        return None


def rebalance(dry_run: bool = False, force: bool = False, path: str | None = None, closes=None) -> dict:
    """每月最后一个交易日执行；其他日子直接返回（force=True可手动触发）。"""
    today = datetime.now(ET).date()
    if not force and not is_last_trading_day(today):
        return {"ok": True, "skipped": "非月末最后一个交易日"}
    if closes is None:
        import yfinance as yf
        tickers = list(ETF_UNIVERSE) + ["SPY"]
        closes = yf.download(tickers, period="15mo", auto_adjust=True, progress=False, threads=True)["Close"]
        closes = closes.dropna(how="all")
    spy = float(closes["SPY"].dropna().iloc[-1]) if "SPY" in closes else None
    etf = closes[[c for c in closes.columns if c in ETF_UNIVERSE]]
    ranked = rank_momentum(etf)
    prices = {t: float(etf[t].dropna().iloc[-1]) for t in etf.columns if etf[t].notna().any()}

    book = _load(path) or {"created": today.isoformat(), "initial_value": INIT_VALUE, "cash": INIT_VALUE,
                           "holdings": {}, "last_prices": {}, "history": []}
    value = round(book_value(book, prices), 2)
    p = plan(ranked, prices, list(book["holdings"]), value)
    entry = {"date": today.isoformat(), "value_before": value, "targets": p["targets"], "buy": p["buy"],
             "sell": p["sell"], "turnover": p["turnover"], "cost": p["cost"],
             "top10": [(t, round(float(v) * 100, 1)) for t, v in ranked.head(10).items()],
             "momentum_book_value": _momentum_book_value(), "spy": spy}
    if not dry_run and p["targets"]:
        book["holdings"] = p["holdings"]
        book["cash"] = round(value - p["cost"] - sum(q * prices[t] for t, q in p["holdings"].items()), 2)
        book["last_prices"] = {t: prices[t] for t in p["holdings"]}
        book["history"].append(entry)
        _save(book, path)
    return {"ok": True, "dry_run": dry_run, **entry}


def format_rebalance(r: dict) -> str:
    if r.get("skipped"):
        return ""
    head = "📊 <b>行业ETF动量账本换仓</b>（对照组，本地模拟" + ("，空跑" if r.get("dry_run") else "") + "）"
    lines = [head, f"调仓前净值：${r['value_before']:,.2f}",
             "排名前5（12-1个月）：" + "，".join(f"{t} {v:+.0f}%" for t, v in r["top10"][:5]),
             f"目标持仓（等权）：{', '.join(r['targets']) or '无'}"]
    if r["buy"] or r["sell"]:
        lines.append(f"买入{', '.join(r['buy']) or '无'}；卖出{', '.join(r['sell']) or '无'}；成本${r['cost']:.2f}")
    else:
        lines.append("持仓不变，只调回等权")
    if r.get("momentum_book_value") is not None:
        lines.append(f"同日个股动量账本净值：${r['momentum_book_value']:,.2f}")
    return "\n".join(lines)


def status_line(path: str | None = None) -> str:
    book = _load(path)
    if not book:
        return "行业ETF动量账本：尚未建仓（每月最后一个交易日15:40调仓，首次2026-10-30）"
    h = book["history"][-1] if book.get("history") else {}
    return (f"行业ETF动量账本：持有{', '.join(book['holdings'])}；上次调仓{h.get('date')}，"
            f"调仓前净值${h.get('value_before', 0):,.2f}，共调仓{len(book.get('history', []))}次")


if __name__ == "__main__":
    if "--status" in sys.argv:
        print(status_line())
    else:
        res = rebalance(dry_run="--dry-run" in sys.argv, force=True)
        print(format_rebalance(res) or res)
