"""
模拟盘绩效报告（2026-09-29新增，基于quantstats）

paper_trading.performance_report只有逐笔统计（胜率/盈亏比/Kelly），没有资金曲线
层面的指标（最大回撤、Sharpe、相对SPY）。这里从data/paper_trades.json的持仓记录
还原每日资金曲线（现金 + 持仓按当日收盘价估值），交给quantstats算标准指标。

内存：quantstats连带加载matplotlib/seaborn，import一次约160MB，服务器只有1GB，
所以不在scheduler进程里import——scheduler用子进程跑本模块（run_in_subprocess），
跑完即退出释放内存。本模块顶层不import quantstats，scheduler可以安全引用。

用法：
  python -m src.performance_report            # 打印Telegram格式文本
  python -m src.performance_report --html out.html   # 生成quantstats完整网页报告（本地）
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

from .backtest import wilson_ci

ET = pytz.timezone("America/New_York")
_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
_LOG = os.path.join(_DATA, "paper_trades.json")

# 2026-09-16起有时间止损，此前9笔全是止损出场，两种出场制度的样本不能混在一起看
TIME_STOP_SINCE = "2026-09-16"


def _day(ts) -> pd.Timestamp:
    return pd.Timestamp(str(ts)[:10])


def load_ledger(path: str = _LOG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_equity_curve(positions: list, init_value: float, closes: dict,
                       calendar: pd.DatetimeIndex) -> pd.Series:
    """
    每日资金 = 初始资金 - 已开仓成本 + 已平仓回款 + 未平仓按收盘价估值（纯函数）。
    closes: {ticker: 日收盘价Series}；calendar: 交易日序列（通常用SPY的日期）。
    """
    cal = pd.DatetimeIndex(calendar).tz_localize(None).normalize()
    equity = pd.Series(float(init_value), index=cal)
    for p in positions:
        shares = float(p.get("shares") or 0)
        if not shares or not p.get("opened_at"):
            continue
        o = _day(p["opened_at"])
        c = _day(p["closed_at"]) if p.get("closed_at") else None
        cost = float(p["entry_price"]) * shares
        equity[cal >= o] -= cost
        if c is not None and p.get("exit_price") is not None:
            equity[cal >= c] += float(p["exit_price"]) * shares
        held = (cal >= o) & ((cal < c) if c is not None else True)
        if not held.any():
            continue
        px = closes.get(p["ticker"])
        if px is None or len(px) == 0:
            # 取不到价格时按开仓价估值，不能让已扣掉的成本凭空消失
            equity[held] += float(p["entry_price"]) * shares
            continue
        px = pd.Series(px.values, index=pd.DatetimeIndex(px.index).tz_localize(None).normalize())
        px = px[~px.index.duplicated()].reindex(cal).ffill().fillna(float(p["entry_price"]))
        equity[held] += px[held] * shares
    return equity


def trade_summary(positions: list) -> dict:
    closed = [p for p in positions if p.get("status") == "closed" and p.get("pnl_pct") is not None]

    def _stats(rows):
        n = len(rows)
        if not n:
            return {"n": 0}
        wins = sum(1 for p in rows if p["pnl"] > 0)
        return {"n": n, "wins": wins, "win_ci": wilson_ci(wins, n),
                "avg_pct": round(float(np.mean([p["pnl_pct"] for p in rows])), 2),
                "pnl": round(float(sum(p["pnl"] for p in rows)), 2)}

    # 按平仓日切分：9/16前开仓、之后被时间止损平掉的单子属于新出场制度
    new = [p for p in closed if str(p.get("closed_at", "")) >= TIME_STOP_SINCE]
    by_reason = {}
    for p in closed:
        by_reason.setdefault(p.get("exit_reason") or "未知", []).append(p)
    return {"all": _stats(closed), "time_stop_regime": _stats(new),
            "by_reason": {k: _stats(v) for k, v in by_reason.items()},
            "open": [p["ticker"] for p in positions if p.get("status") == "open"]}


def curve_metrics(equity: pd.Series, bench: pd.Series) -> dict:
    """用quantstats算资金曲线指标（此处才import quantstats）。"""
    import quantstats as qs

    rets = equity.pct_change().dropna()
    b = pd.Series(bench.values, index=pd.DatetimeIndex(bench.index).tz_localize(None).normalize())
    b = b[~b.index.duplicated()].reindex(equity.index).ffill().pct_change().dropna()
    greeks = qs.stats.greeks(rets, b) if len(rets) > 5 else {}
    return {
        "total_ret": round(float(equity.iloc[-1] / equity.iloc[0] - 1) * 100, 2),
        "bench_ret": round(float((1 + b).prod() - 1) * 100, 2),
        "max_dd": round(float(qs.stats.max_drawdown(rets)) * 100, 2),
        "sharpe": round(float(qs.stats.sharpe(rets)), 2),
        "sortino": round(float(qs.stats.sortino(rets)), 2),
        "volatility": round(float(qs.stats.volatility(rets)) * 100, 1),
        "beta": round(float(greeks.get("beta", np.nan)), 2) if len(greeks) else None,
        "worst_day": round(float(rets.min()) * 100, 2),
        "days": len(equity),
    }


def build_report(ledger: dict | None = None, fetch=None) -> dict:
    ledger = ledger or load_ledger()
    positions = list(ledger["positions"].values()) if isinstance(ledger["positions"], dict) else ledger["positions"]
    acct = ledger["account"]
    start = min(_day(p["opened_at"]) for p in positions) if positions else _day(acct["created_at"])

    if fetch is None:
        import yfinance as yf

        def fetch(t):
            return yf.Ticker(t).history(start=start - pd.Timedelta(days=5))["Close"]

    spy = fetch("SPY")
    cal = pd.DatetimeIndex(spy.index).tz_localize(None).normalize()
    cal = cal[cal >= start]
    closes = {t: fetch(t) for t in {p["ticker"] for p in positions}}
    equity = build_equity_curve(positions, acct["initial_value"], closes, cal)
    return {"equity": equity, "metrics": curve_metrics(equity, spy), "trades": trade_summary(positions),
            "account": acct, "spy": spy}


def format_telegram(rep: dict) -> str:
    m, t, a = rep["metrics"], rep["trades"], rep["account"]

    def line(name, s):
        if not s.get("n"):
            return f"  {name}：0笔"
        return (f"  {name}：{s['n']}笔 {s['wins']}胜（胜率95%区间{s['win_ci'][0]}-{s['win_ci'][1]}%）"
                f"，平均{s['avg_pct']:+.2f}%，合计${s['pnl']:+.0f}")

    lines = [
        f"📊 <b>模拟盘周报</b>  {datetime.now(ET):%Y-%m-%d}",
        f"账户 ${rep['equity'].iloc[-1]:,.0f}（起始${a['initial_value']:,}，{m['days']}个交易日）",
        f"总收益 {m['total_ret']:+.1f}%  vs 同期SPY {m['bench_ret']:+.1f}%",
        f"最大回撤 {m['max_dd']:.1f}%  Sharpe {m['sharpe']}  Sortino {m['sortino']}  年化波动 {m['volatility']}%"
        + (f"  beta {m['beta']}" if m.get("beta") is not None else ""),
        "",
        "<b>逐笔</b>",
        line("全部", t["all"]),
        line(f"时间止损制度({TIME_STOP_SINCE}起)", t["time_stop_regime"]),
    ]
    for reason, s in t["by_reason"].items():
        lines.append(line(f"出场:{reason}", s))
    if t["open"]:
        lines.append(f"  持仓中：{', '.join(t['open'])}")
    lines.append("\n（资金曲线按每日收盘价还原；样本很少时各项比率参考意义有限）")
    return "\n".join(lines)


def run_in_subprocess(timeout: int = 300) -> str:
    """供scheduler/telegram调用：子进程生成报告，避免quantstats常驻scheduler内存。"""
    root = os.path.join(os.path.dirname(__file__), "..")
    r = subprocess.run([sys.executable, "-m", "src.performance_report"], cwd=root,
                       capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if r.returncode != 0:
        raise RuntimeError((r.stderr or "").strip().splitlines()[-1] if r.stderr else f"exit {r.returncode}")
    return r.stdout.strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="模拟盘绩效报告（quantstats）")
    ap.add_argument("--html", default=None, help="生成quantstats完整网页报告到该路径")
    a = ap.parse_args()
    rep = build_report()
    if a.html:
        import quantstats as qs
        spy = rep["spy"]
        b = pd.Series(spy.values, index=pd.DatetimeIndex(spy.index).tz_localize(None).normalize())
        qs.reports.html(rep["equity"].pct_change().dropna(), benchmark=b.pct_change().dropna(),
                        output=a.html, title="stock-master 模拟盘")
        print(f"已生成 {a.html}")
    else:
        print(format_telegram(rep))
