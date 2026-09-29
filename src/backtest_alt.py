"""
替代策略回测（2026-09-29，本地研究用）

背景：技术面近似回测证明"3-10天技术面择时"没有优势（见wiki stock-master/overview
"核心策略诊断"）。这里验证文献里证据更扎实的三种机制，判断核心策略该往哪换：

  1. 横截面动量：每月末按过去12个月（跳过最近1个月）涨幅排序，持有前N只一个月
  2. 短期反转：每周末买过去一周跌得最多的一组，持有一周（检验"短周期是反转不是动量"）
  3. 财报后漂移（PEAD）：财报超预期后，按事件后N个交易日相对SPY的超额收益分组

对照组都是"同一股票池等权持有"——动量/反转组合比它好才算有优势；PEAD看超额收益
是否随超预期幅度单调变化。

局限：股票池是当前存在的股票（幸存者偏差，10年窗口比5年更严重）；未计融资/借券；
组合层面按收盘价成交、按单边成本cost_bps扣换手成本。

用法：python -m src.backtest_alt [momentum|reversal|pead|all]
"""
import math
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from .backtest import _cache_path, default_universe, load_prices

YEARS = 10
COST_BPS = 10          # 单边交易成本（含滑点）


def research_universe() -> list:
    from .backtest_research import LOSERS_2021
    return sorted(set(default_universe()) | set(LOSERS_2021) - {"QQQ"})


def closes_frame(prices: dict, tickers: list) -> pd.DataFrame:
    df = pd.DataFrame({t: prices[t]["Close"] for t in tickers if t in prices})
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    return df.sort_index()


# ─────────────────────────────────────────────────────────────
# 统计
# ─────────────────────────────────────────────────────────────

def perf_stats(rets: pd.Series, periods_per_year: int) -> dict:
    r = rets.dropna()
    if len(r) < 3:
        return {"n": len(r)}
    eq = (1 + r).cumprod()
    return {
        "n": len(r),
        "ann_ret": round(float((eq.iloc[-1]) ** (periods_per_year / len(r)) - 1) * 100, 1),
        "ann_vol": round(float(r.std(ddof=1) * math.sqrt(periods_per_year)) * 100, 1),
        "sharpe": round(float(r.mean() / r.std(ddof=1) * math.sqrt(periods_per_year)), 2) if r.std() > 0 else None,
        "max_dd": round(float((eq / eq.cummax() - 1).min()) * 100, 1),
    }


def excess_stats(strat: pd.Series, bench: pd.Series, periods_per_year: int) -> dict:
    """策略相对基准的逐期超额收益：年化均值与t值。"""
    d = (strat - bench).dropna()
    if len(d) < 3:
        return {"n": len(d)}
    se = d.std(ddof=1) / math.sqrt(len(d))
    return {"n": len(d), "ann_excess": round(float(d.mean() * periods_per_year) * 100, 2),
            "t": round(float(d.mean() / se), 2) if se > 0 else None,
            "hit": round(float((d > 0).mean()) * 100, 1)}


# ─────────────────────────────────────────────────────────────
# 1. 横截面动量 / 2. 短期反转（同一个排序组合框架）
# ─────────────────────────────────────────────────────────────

def rank_portfolio(closes: pd.DataFrame, freq: str, lookback: int, skip: int,
                   n_hold: int, pick: str = "top", cost_bps: float = COST_BPS,
                   min_history: int = 252) -> tuple:
    """
    每个调仓日（freq="ME"月末/"W-FRI"周五）按 close[t-skip]/close[t-lookback]-1 排序，
    等权持有前（pick="top"）或后（pick="bottom"）n_hold只到下个调仓日（纯函数）。
    只在调仓日已有min_history根历史的股票里排序，避免用到未上市前的空数据。
    返回 (策略逐期收益, 同期全池等权收益, 平均换手率)。
    """
    rebal = closes.resample(freq).last().index
    rebal = [closes.index[closes.index <= d][-1] for d in rebal if (closes.index <= d).any()]
    rebal = sorted(set(rebal))
    strat, bench, turnover = [], [], []
    prev = set()
    for a, b in zip(rebal[:-1], rebal[1:]):
        ia = closes.index.get_loc(a)
        if ia < max(lookback, min_history):
            continue
        hist_ok = closes.iloc[ia - min_history].notna() & closes.iloc[ia].notna() & closes.loc[b].notna()
        past = closes.iloc[ia - skip] / closes.iloc[ia - lookback] - 1
        past = past[hist_ok & past.notna()]
        if len(past) < n_hold * 2:
            continue
        chosen = past.nlargest(n_hold).index if pick == "top" else past.nsmallest(n_hold).index
        period_ret = closes.loc[b] / closes.loc[a] - 1
        cur = set(chosen)
        to = 1.0 if not prev else len(cur - prev) / n_hold
        turnover.append(to)
        strat.append((b, float(period_ret[chosen].mean()) - 2 * to * cost_bps / 1e4))
        bench.append((b, float(period_ret[past.index].mean())))
        prev = cur
    s = pd.Series(dict(strat)).sort_index()
    bm = pd.Series(dict(bench)).sort_index()
    return s, bm, float(np.mean(turnover)) if turnover else float("nan")


def run_momentum(closes: pd.DataFrame, spy: pd.Series) -> list:
    lines = [f"== 1. 横截面动量（月末调仓，等权，单边成本{COST_BPS}bp）股票池{closes.shape[1]}只 =="]
    spy_m = spy.resample("ME").last().pct_change()
    for lb, sk, name in ((252, 21, "12-1个月"), (126, 21, "6-1个月"), (63, 0, "3个月")):
        for n in (5, 10, 20):
            s, bm, to = rank_portfolio(closes, "ME", lb, sk, n)
            p, e = perf_stats(s, 12), excess_stats(s, bm, 12)
            lines.append(f"  {name} 前{n:2d}只：年化{p['ann_ret']:+.1f}% 波动{p['ann_vol']}% Sharpe {p['sharpe']} "
                         f"回撤{p['max_dd']}%  | 超额(vs全池等权) {e['ann_excess']:+.2f}%/年 t={e['t']} 胜月{e['hit']}%  换手{to:.0%}")
        s_bot, bm, _ = rank_portfolio(closes, "ME", lb, sk, 10, pick="bottom")
        e = excess_stats(s_bot, bm, 12)
        lines.append(f"  {name} 后10只（输家）：超额 {e['ann_excess']:+.2f}%/年 t={e['t']}")
    _, bm, _ = rank_portfolio(closes, "ME", 252, 21, 10)
    pb = perf_stats(bm, 12)
    ps = perf_stats(spy_m.reindex(bm.index), 12)
    lines.append(f"  基准：全池等权 年化{pb['ann_ret']:+.1f}% Sharpe {pb['sharpe']} 回撤{pb['max_dd']}%；"
                 f"SPY 年化{ps['ann_ret']:+.1f}% Sharpe {ps['sharpe']} 回撤{ps['max_dd']}%  （{bm.index[0]:%Y-%m}起）")
    return lines


# 行业/细分行业ETF：ETF不存在"按今天的结果挑股票"的后见之明偏差，是检验动量最干净的股票池
ETF_UNIVERSE = [
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    "SMH", "SOXX", "IGV", "FDN", "IBB", "XBI", "IHI", "XPH", "KRE", "KBE", "IAI", "KIE",
    "ITA", "IYT", "XHB", "ITB", "XRT", "XOP", "OIH", "GDX", "XME", "TAN", "ICLN", "URA", "LIT", "VNQ",
]


def run_bias_checks(prices: dict, closes: pd.DataFrame) -> list:
    """动量结果的后见之明偏差检验：分股票池、分时段、ETF股票池。"""
    from .backtest_research import LOSERS_2021
    lines = ["== 1b. 动量偏差检验（12-1个月、前10只/ETF前5只）=="]

    def one(cl, n, label):
        s, bm, _ = rank_portfolio(cl, "ME", 252, 21, n)
        e, p = excess_stats(s, bm, 12), perf_stats(s, 12)
        pb = perf_stats(bm, 12)
        return (f"  {label:34s} 策略年化{p['ann_ret']:+.1f}% vs 等权{pb['ann_ret']:+.1f}%  "
                f"超额{e['ann_excess']:+.2f}%/年 t={e['t']}  ({s.index[0]:%Y-%m}起，{e['n']}个月)")

    losers = [t for t in LOSERS_2021 if t in closes.columns]
    winners = [t for t in closes.columns if t not in set(LOSERS_2021)]
    lines.append(one(closes[winners], 10, f"只用默认池（今天的赢家）{len(winners)}只"))
    lines.append(one(closes[losers], 10, f"只用2021热门后大跌池 {len(losers)}只"))

    s, bm, _ = rank_portfolio(closes, "ME", 252, 21, 10)
    mid = s.index[len(s) // 2]
    for lab, m in (("前半段", s.index <= mid), ("后半段", s.index > mid)):
        e = excess_stats(s[m], bm[m], 12)
        lines.append(f"  全池 {lab}（{s.index[m][0]:%Y-%m}~{s.index[m][-1]:%Y-%m}）：超额{e['ann_excess']:+.2f}%/年 t={e['t']}")
    etf_prices = load_prices(ETF_UNIVERSE, YEARS)
    etf_cl = closes_frame(etf_prices, [t for t in ETF_UNIVERSE if t in etf_prices])
    lines.append(one(etf_cl, 5, f"行业ETF {etf_cl.shape[1]}只（无选股偏差）"))
    lines.append(one(etf_cl, 3, f"行业ETF 前3只"))
    s6, bm6, _ = rank_portfolio(etf_cl, "ME", 126, 21, 5)
    e6 = excess_stats(s6, bm6, 12)
    lines.append(f"  行业ETF 6-1个月 前5只：超额{e6['ann_excess']:+.2f}%/年 t={e6['t']}")
    return lines


def run_reversal(closes: pd.DataFrame) -> list:
    lines = ["== 2. 短期反转（周五调仓，按过去一周涨跌排序，持有一周）=="]
    for n in (5, 10):
        for pick, name in (("bottom", "买过去一周跌最多"), ("top", "买过去一周涨最多")):
            s, bm, to = rank_portfolio(closes, "W-FRI", 5, 0, n, pick=pick)
            e = excess_stats(s, bm, 52)
            e0 = excess_stats(s + 2 * to * COST_BPS / 1e4, bm, 52)
            lines.append(f"  {name} {n}只：超额(扣成本) {e['ann_excess']:+.2f}%/年 t={e['t']}  "
                         f"| 不扣成本 {e0['ann_excess']:+.2f}%/年 t={e0['t']}  换手{to:.0%}/周")
    return lines


# ─────────────────────────────────────────────────────────────
# 3. PEAD
# ─────────────────────────────────────────────────────────────

def load_earnings_table(tickers: list) -> pd.DataFrame:
    """[ticker, ts(带时区的公告时间), surprise_pct]；取不到的标的跳过。"""
    path = _cache_path(f"earnings_table_{len(tickers)}")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    rows = []
    for t in tickers:
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=48)
        except Exception:
            continue
        if ed is None or ed.empty:
            continue
        for ts, r in ed.iterrows():
            if pd.notna(r.get("Surprise(%)")) and pd.notna(r.get("Reported EPS")):
                rows.append((t, ts, float(r["Surprise(%)"])))
    out = pd.DataFrame(rows, columns=["ticker", "ts", "surprise_pct"])
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def reaction_day(ts: pd.Timestamp, index: pd.DatetimeIndex):
    """公告后第一个能交易到消息的交易日：盘后(≥16:00)公告→下一交易日，否则当天（纯函数）。"""
    t = ts.tz_convert("America/New_York") if ts.tzinfo else ts
    day = pd.Timestamp(t.date())
    after_close = t.hour >= 16
    pos = index.searchsorted(day, side="right" if after_close else "left")
    return index[pos] if pos < len(index) else None


def event_returns(closes: pd.DataFrame, spy: pd.Series, events: pd.DataFrame,
                  horizons=(5, 20, 60)) -> pd.DataFrame:
    """
    每个财报事件：反应日收盘入场（此时已知超预期幅度和当天涨跌），计算之后h个交易日
    相对SPY的超额收益；另记反应日当天的跳空反应（car0，相对前一日收盘）。
    """
    spy = spy.copy()
    spy.index = pd.DatetimeIndex(spy.index).tz_localize(None).normalize()
    spy = spy.reindex(closes.index).ffill()
    rows = []
    idx = closes.index
    for _, ev in events.iterrows():
        if ev["ticker"] not in closes.columns:
            continue
        d = reaction_day(ev["ts"], idx)
        if d is None:
            continue
        i = idx.get_loc(d)
        c = closes[ev["ticker"]]
        if i < 1 or pd.isna(c.iloc[i]) or pd.isna(c.iloc[i - 1]):
            continue
        row = {"ticker": ev["ticker"], "date": d, "surprise_pct": ev["surprise_pct"],
               "car0": (c.iloc[i] / c.iloc[i - 1] - 1) - (spy.iloc[i] / spy.iloc[i - 1] - 1)}
        for h in horizons:
            if i + h < len(idx) and pd.notna(c.iloc[i + h]):
                row[f"x{h}"] = (c.iloc[i + h] / c.iloc[i] - 1) - (spy.iloc[i + h] / spy.iloc[i] - 1)
        rows.append(row)
    return pd.DataFrame(rows)


def adjust_for_stock_drift(ev: pd.DataFrame, closes: pd.DataFrame, spy: pd.Series,
                           horizons=(5, 20, 60)) -> pd.DataFrame:
    """
    x{h}（相对SPY）再减去该股票全样本所有交易日的平均h日超额收益，得到a{h}。
    股票池是事后挑的（整体大幅跑赢SPY），不扣这一项，任何事件分组都会显得"有超额"——
    2026-09-29实测：不扣时"全部财报"60日+2.37%(t=5.9)，扣掉后-0.30%。
    """
    spy = spy.copy()
    spy.index = pd.DatetimeIndex(spy.index).tz_localize(None).normalize()
    spy = spy.reindex(closes.index).ffill()
    out = ev.copy()
    for h in horizons:
        base = (closes.shift(-h) / closes - 1).sub(spy.shift(-h) / spy - 1, axis=0).mean()
        out[f"a{h}"] = out[f"x{h}"] - out["ticker"].map(base)
    return out


def run_pead(prices: dict, tickers: list, spy: pd.Series) -> list:
    closes = closes_frame(prices, tickers)
    events = load_earnings_table(tickers)
    ev = adjust_for_stock_drift(event_returns(closes, spy, events), closes, spy)
    lines = [f"== 3. 财报后漂移（PEAD）：{ev['ticker'].nunique()}只股票 {len(ev)}次财报，"
             f"{ev['date'].min():%Y-%m}~{ev['date'].max():%Y-%m}；反应日收盘入场；"
             f"异常收益=相对SPY再扣除该股自身平均超额（去掉股票池后见之明）=="]

    def grp(name, m):
        sub = ev[m]
        parts = [f"  {name:22s} n={len(sub):4d}"]
        for h in (5, 20, 60):
            x = sub[f"a{h}"].dropna() * 100
            if len(x) > 2:
                parts.append(f"{h}日 {x.mean():+.2f}%(t={x.mean() / (x.std(ddof=1) / math.sqrt(len(x))):+.1f})")
        return "  ".join(parts)

    s = ev["surprise_pct"]
    lines.append(grp("全部", s.notna()))
    lines.append(grp("超预期>10%", s > 10))
    lines.append(grp("超预期0~10%", (s > 0) & (s <= 10)))
    lines.append(grp("低于预期", s < 0))
    q = ev["car0"]
    lines.append("  -- 按反应日当天的跳空反应分组（市场对财报的实际反应）--")
    lines.append(grp("当天跑赢SPY>5%", q > 0.05))
    lines.append(grp("当天跑输SPY>5%", q < -0.05))
    lines.append(grp("超预期>10% 且 当天>+5%", (s > 10) & (q > 0.05)))
    lines.append(grp("低于预期 且 当天<-5%", (s < 0) & (q < -0.05)))
    lines.append(grp("超预期>10% 但 当天<-5%", (s > 10) & (q < -0.05)))
    lines.append("  注：同一时期多只股票的事件相关，t值偏乐观；超额未扣交易成本（单次往返约0.2%）")
    return lines


def main(which: str = "all"):
    tickers = research_universe()
    prices = load_prices(tickers, YEARS)
    tickers = [t for t in tickers if t in prices]
    closes = closes_frame(prices, tickers)
    spy = prices["SPY"]["Close"]
    spy_n = spy.copy()
    spy_n.index = pd.DatetimeIndex(spy_n.index).tz_localize(None).normalize()
    out = []
    if which in ("momentum", "all"):
        out += run_momentum(closes, spy_n) + [""]
        out += run_bias_checks(prices, closes) + [""]
    if which in ("reversal", "all"):
        out += run_reversal(closes) + [""]
    if which in ("pead", "all"):
        out += run_pead(prices, tickers, spy)
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
