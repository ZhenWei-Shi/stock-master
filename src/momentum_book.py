"""
月度动量账本（2026-09-29新增：个股波段从"3-10天技术面择时"改为月度横截面动量）

依据（wiki stock-master/overview"替代策略回测"）：横截面动量是唯一方向对、量级
符合文献的机制——行业ETF（无选股偏差）+5.6%/年，t=1.35，统计上还不显著；个股池
的结果被后见之明偏差放大（+33%/年不可信）。用户2026-09-29决定个股波段改用它，
以独立的模拟账本前向运行，与原模拟盘（九关时代）分开记账。

规则（事先定好）：
  - 股票池：watchlist + 板块代表股（与回测一致），去掉ETF
  - 零股（2026-09-30起）：按金额买，股数保留4位小数（向下取整）。原先用整数股、
    只选"股价≤单仓预算"的股票，$2,000账本买不起MU、LITE这类高价股，实际持仓
    不一定是真正的前两名；Alpaca支持零股，账本按零股记才和将来能下的单一致
  - 每月最后一个交易日15:40：按过去12个月涨幅（跳过最近1个月）排名，持有前N_HOLD只，
    每只目标权重WEIGHT（2只×39%，留出滑点余量，不碰paper_trading的80%总仓位上限）
  - 换仓：卖出跌出前N的，买入新进入的；仍在前N的不动（减少换手）
  - 买入前过风控层（risk_layer），忽略财报风控——持有一个月约1/3概率跨财报，
    若按财报禁入会系统性排除近期有财报的股票，让选股带偏；被其他风控否决的顺延下一名
  - 保护性止损：入场价-20%（防单只暴跌），不设止盈，不受10天时间止损约束
  - 评估：每月记录"持仓组合 vs 同池等权"的超额收益，12个月后再看；
    12个样本不足以下统计结论，主要看方向和回撤是否可接受

用法：python -m src.momentum_book [--dry-run]
"""
import json
import math
import os
import sys
from datetime import date, datetime, timedelta

import pandas as pd
import pytz

ET = pytz.timezone("America/New_York")
_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
_LOGFILE = os.path.join(_DATA, "momentum_log.json")

MODE = "momentum"
STRATEGY = "Momentum/Monthly"
INIT_VALUE = 2000.0
N_HOLD = 2
# 39%而不是40%：开仓会加0.05%滑点，2×40%贴边时可能算出80.04%，被paper_trading
# 的80%总仓位上限拒绝第二笔（2026-09-29重启前清查发现）
WEIGHT = 0.39
LOOKBACK, SKIP = 252, 21
STOP_PCT = 0.20
RISK_IGNORE = ("earnings_blackout",)
SHARE_DECIMALS = 4


def fractional_shares(budget: float, price: float) -> float:
    """按金额折算零股，向下取整到SHARE_DECIMALS位，保证花费不超过预算。"""
    q = 10 ** SHARE_DECIMALS
    return math.floor(budget / price * q) / q


def universe(watchlist: list | None = None) -> list:
    from .backtest import default_universe
    from .sector_rotation import SECTOR_TICKERS
    etfs = set(SECTOR_TICKERS) | {"QQQ", "SPY"}
    return sorted(set(default_universe()) | {t.upper() for t in (watchlist or [])} - etfs)


def is_last_trading_day(d: date) -> bool:
    from .event_lab import sessions_between
    month_end = (pd.Timestamp(d) + pd.offsets.MonthEnd(0)).date()
    s = sessions_between(d, month_end)
    return bool(s) and s[-1] == d


def rank_momentum(closes: pd.DataFrame) -> pd.Series:
    """最后一行为"今天"：close[t-SKIP]/close[t-LOOKBACK]-1，降序（纯函数）。缺历史的剔除。"""
    if len(closes) <= LOOKBACK:
        return pd.Series(dtype=float)
    past = closes.iloc[-1 - SKIP] / closes.iloc[-1 - LOOKBACK] - 1
    ok = closes.iloc[-1 - LOOKBACK].notna() & closes.iloc[-1].notna()
    return past[ok & past.notna()].sort_values(ascending=False)


def plan_rebalance(ranked: pd.Series, prices: dict, held: list, book_value: float,
                   risk_ok=lambda t: True) -> dict:
    """
    决定卖什么、买什么（纯函数）。按排名往下找，跳过没有价格的和风控否决的，
    凑够N_HOLD只目标；已持有且仍在目标里的不动。零股，所以不再有"买不起"。
    """
    budget = book_value * WEIGHT
    targets, skipped = [], []
    for t in ranked.index:
        if len(targets) >= N_HOLD:
            break
        px = prices.get(t)
        if not px:
            skipped.append((t, "无价格"))
            continue
        if t not in held and not risk_ok(t):
            skipped.append((t, "风控否决"))
            continue
        targets.append(t)
    return {"targets": targets, "sell": [t for t in held if t not in targets],
            "buy": [t for t in targets if t not in held], "skipped": skipped,
            "shares": {t: fractional_shares(budget, prices[t]) for t in targets if t not in held}}


def _load_log() -> list:
    try:
        with open(_LOGFILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def _append_log(entry: dict):
    log = _load_log()
    log.append(entry)
    os.makedirs(_DATA, exist_ok=True)
    tmp = _LOGFILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, _LOGFILE)


def ensure_book():
    from .paper_trading import list_positions, init_account
    acct = list_positions(MODE).get("account") or {}
    if not acct.get("initial_value"):
        init_account(INIT_VALUE, mode=MODE, label="月度动量账本")


def rebalance(watchlist: list | None = None, dry_run: bool = False, force: bool = False) -> dict:
    """每月最后一个交易日执行；其他日子直接返回（force=True可手动触发）。"""
    import yfinance as yf
    from .paper_trading import list_positions, open_position, close_position
    from .risk_layer import risk_check

    today = datetime.now(ET).date()
    if not force and not is_last_trading_day(today):
        return {"ok": True, "skipped": "非月末最后一个交易日"}

    ensure_book()
    tickers = universe(watchlist)
    raw = yf.download(tickers, period="15mo", auto_adjust=True, progress=False, threads=True)["Close"]
    closes = raw.dropna(how="all")
    ranked = rank_momentum(closes)
    prices = {t: float(closes[t].dropna().iloc[-1]) for t in ranked.index if closes[t].notna().any()}

    book = list_positions(MODE)
    open_pos = book.get("open", [])
    held = [p["ticker"] for p in open_pos]
    value = (book.get("account") or {}).get("current_value") or INIT_VALUE
    plan = plan_rebalance(ranked, prices, held, value,
                          risk_ok=lambda t: risk_check(t, portfolio=value, ignore=RISK_IGNORE)["ok"])

    actions = []
    if not dry_run:
        for p in open_pos:
            if p["ticker"] in plan["sell"]:
                r = close_position(p["id"], prices.get(p["ticker"], p["entry_price"]),
                                   exit_reason="月度动量换仓", mode=MODE)
                actions.append(f"卖出{p['ticker']}：{'成功' if r.get('ok') else r.get('error')}")
        for t in plan["buy"]:
            px, n = prices[t], plan["shares"][t]
            r = open_position(t, n, px, stop_loss=round(px * (1 - STOP_PCT), 2), target=round(px * 100, 2),
                              strategy=STRATEGY, mode=MODE)
            actions.append(f"买入{t} {n:g}股@{px:.2f}（${n * px:,.0f}）：{'成功' if r.get('ok') else r.get('error')}")

    top = [(t, round(float(v) * 100, 1)) for t, v in ranked.head(10).items()]
    _append_log({"date": today.isoformat(), "dry_run": dry_run, "book_value": value, "top10": top,
                 "held_before": held, **{k: plan[k] for k in ("targets", "sell", "buy", "skipped")},
                 "actions": actions})
    return {"ok": True, "top10": top, "plan": plan, "actions": actions}


def format_rebalance(r: dict) -> str:
    if r.get("skipped") and "plan" not in r:
        return ""
    p = r["plan"]
    lines = ["📈 <b>月度动量换仓</b>（模拟账本）",
             "排名前5（过去12个月涨幅，跳过最近1个月）：" + "，".join(f"{t} {v:+.0f}%" for t, v in r["top10"][:5]),
             f"目标持仓：{', '.join(p['targets']) or '无'}"]
    if p["skipped"]:
        lines.append("跳过：" + "，".join(f"{t}({why})" for t, why in p["skipped"][:5]))
    lines += r["actions"] or ["无需换仓"]
    return "\n".join(lines)


if __name__ == "__main__":
    res = rebalance(dry_run="--dry-run" in sys.argv, force=True)
    print(format_rebalance(res) or res)
