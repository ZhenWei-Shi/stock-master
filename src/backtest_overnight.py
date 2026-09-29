"""
隔夜放量策略回测（2026-09-29，用户想法，本地研究用）

用户想法：前一天按"成交量/资金大"选股收盘买入；第二天开盘后一段时间内跌到止损就卖，
盈利就动态止盈。T-1买、T卖不算日内交易，不受PDT限制。

相关文献（GitHub上无成熟实现，只有个位数星的"收盘买开盘卖"小实验）：
  - 隔夜收益效应：美股收益大部分发生在收盘→次日开盘（Cliff/Cooper/Gulen 2008；
    Lou/Polk/Skouras 2019）
  - 高成交量溢价：成交量异常放大的股票之后倾向上涨（Gervais/Kaniel/Mingelgrin 2001）

入场（T-1收盘价）：
  V1 放量上涨：量比≥RVOL_MIN（对前20日均量）且收涨
  V2 放量下跌：量比≥RVOL_MIN且收跌（对照）
  V3 成交额前5：当天成交额（收盘价×成交量）最大的5只
  R  随机：同股票、同笔数的随机日子（对照组）
出场（T日）：
  E0 开盘卖（纯隔夜）         E1 收盘卖
  E2 用户规则（需要小时线，yfinance最多约2年）：开盘已低于止损→按开盘价卖；
     第一根小时K线内触及止损→按止损价卖；之后动态止盈——从当天最高点回撤TRAIL就卖
     （用上一根K线为止的最高点判断，避免同一根K线里先高后低的顺序偷看）；都没触发→收盘卖
成本：单边COST_BPS（含滑点），每笔往返扣2倍。

用法：python -m src.backtest_overnight
"""
import math
import os
import pickle

import numpy as np
import pandas as pd
import yfinance as yf

from .backtest import _cache_path, default_universe, load_prices

RVOL_MIN = 2.0
TOP_DOLLAR_N = 5
COST_BPS = 5
STOP_PCT = 0.02
TRAIL_PCT = 0.015


def entry_signals(hist: pd.DataFrame) -> pd.DataFrame:
    """逐日：量比、涨跌、成交额（纯函数）。量比用前20日均量，不含当天。"""
    c, v = hist["Close"], hist["Volume"]
    out = pd.DataFrame(index=hist.index)
    out["rvol"] = v / v.shift(1).rolling(20).mean()
    out["up"] = c > c.shift(1)
    out["dollar_vol"] = c * v
    return out


def daily_exits(hist: pd.DataFrame) -> pd.DataFrame:
    """T-1收盘入场：E0=T开盘/T-1收盘-1，E1=T收盘/T-1收盘-1（对齐到入场日T-1）。"""
    c, o = hist["Close"], hist["Open"]
    return pd.DataFrame({"e0": o.shift(-1) / c - 1, "e1": c.shift(-1) / c - 1}, index=hist.index)


def managed_exit(entry: float, bars: pd.DataFrame, stop_pct: float = STOP_PCT,
                 trail_pct: float = TRAIL_PCT) -> tuple:
    """
    用户规则在T日小时线上的出场（纯函数）。bars为T日按时间排序的小时K线。
    返回 (收益率, 原因)。
    """
    if bars.empty:
        return (float("nan"), "no_data")
    stop = entry * (1 - stop_pct)
    o = float(bars["Open"].iloc[0])
    if o <= stop:
        return (o / entry - 1, "gap_stop")
    first = bars.iloc[0]
    if float(first["Low"]) <= stop:
        return (stop / entry - 1, "first_hour_stop")
    peak = max(o, float(first["High"]))
    for _, b in bars.iloc[1:].iterrows():
        trigger = peak * (1 - trail_pct)
        if trigger > entry and float(b["Low"]) <= trigger:
            px = min(trigger, float(b["Open"]))
            return (px / entry - 1, "trail")
        if float(b["Low"]) <= stop:
            return (min(stop, float(b["Open"])) / entry - 1, "stop")
        peak = max(peak, float(b["High"]))
    return (float(bars["Close"].iloc[-1]) / entry - 1, "close")


def load_hourly(tickers: list) -> dict:
    path = _cache_path(f"hourly_{len(tickers)}")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    raw = yf.download(tickers, period="730d", interval="60m", auto_adjust=True,
                      group_by="ticker", progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            df = raw[t].dropna(how="all")
        except KeyError:
            continue
        if len(df):
            df.index = df.index.tz_convert("America/New_York")
            out[t] = df
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def _stats(x) -> str:
    x = pd.Series(x, dtype=float).dropna() * 100
    if len(x) < 3:
        return f"n={len(x)}"
    t = x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
    return f"n={len(x):5d} 平均{x.mean():+.3f}%(t={t:+.1f}) 胜率{(x > 0).mean() * 100:.0f}%"


def main():
    tickers = [t for t in default_universe() if t != "QQQ"]
    prices = load_prices(tickers, 5)
    tickers = [t for t in tickers if t in prices]
    cost = 2 * COST_BPS / 1e4
    rng = np.random.default_rng(0)

    rows = []
    for t in tickers:
        h = prices[t]
        sig = entry_signals(h).join(daily_exits(h))
        sig["ticker"] = t
        rows.append(sig.iloc[21:-1])
    d = pd.concat(rows).reset_index(names="date")
    d["dv_rank"] = d.groupby("date")["dollar_vol"].rank(ascending=False)

    groups = {
        "V1 放量上涨": (d["rvol"] >= RVOL_MIN) & d["up"],
        "V2 放量下跌": (d["rvol"] >= RVOL_MIN) & ~d["up"],
        f"V3 成交额前{TOP_DOLLAR_N}": d["dv_rank"] <= TOP_DOLLAR_N,
        "全部股票日（基准）": pd.Series(True, index=d.index),
    }
    lines = [f"== 隔夜放量策略：{len(tickers)}只股票，5年日线，单边成本{COST_BPS}bp（下面收益已扣往返成本）=="]
    for name, m in groups.items():
        sub = d[m]
        lines.append(f"{name:14s} E0开盘卖 {_stats(sub['e0'] - cost)}  |  E1收盘卖 {_stats(sub['e1'] - cost)}")
    lines.append("  不扣成本的纯隔夜收益（E0）：" + "  ".join(
        f"{n.split()[0]} {d[m]['e0'].mean() * 100:+.3f}%" for n, m in groups.items()))

    # E2：用户规则，小时线（约2年）
    hourly = load_hourly(tickers)
    res = {k: [] for k in ("V1", "V2", "V3", "R")}
    reasons = {k: {} for k in res}
    d2 = d[d["date"] >= min(df.index.min().tz_localize(None).normalize() for df in hourly.values())]
    by_day = {t: {k: g for k, g in df.groupby(df.index.date)} for t, df in hourly.items()}
    picks = {"V1": d2[(d2["rvol"] >= RVOL_MIN) & d2["up"]],
             "V2": d2[(d2["rvol"] >= RVOL_MIN) & ~d2["up"]],
             "V3": d2[d2["dv_rank"] <= TOP_DOLLAR_N]}
    picks["R"] = d2.sample(n=len(picks["V1"]), random_state=0)
    for k, sub in picks.items():
        for _, r in sub.iterrows():
            days = by_day.get(r["ticker"], {})
            d0 = r["date"].date()
            nxt = sorted(x for x in days if x > d0)
            if not nxt or d0 not in days:
                continue
            # 入场价也取小时线（T-1最后一根K线收盘）：日线auto_adjust会按分红回调历史价格，
            # 与小时线口径不一致，2026-09-29首次运行时因此让随机对照也虚增到每笔+1.8%
            entry = days[d0].sort_index()["Close"].iloc[-1]
            ret, why = managed_exit(float(entry), days[nxt[0]].sort_index())
            res[k].append(ret - cost)
            reasons[k][why] = reasons[k].get(why, 0) + 1
    lines.append(f"== E2 用户规则（止损{STOP_PCT:.0%}、第一小时止损、回撤{TRAIL_PCT:.1%}动态止盈、否则收盘卖），"
                 f"小时线{d2['date'].min():%Y-%m}起，已扣成本 ==")
    names = {"V1": "V1 放量上涨", "V2": "V2 放量下跌", "V3": f"V3 成交额前{TOP_DOLLAR_N}", "R": "R  随机对照"}
    for k in ("V1", "V2", "V3", "R"):
        lines.append(f"{names[k]:14s} {_stats(res[k])}  出场分布{reasons[k]}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
