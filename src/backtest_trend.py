"""
ETF趋势跟踪（时间序列动量）回测（2026-10-02，本地研究用）

背景：个股3-10天技术面择时已证明和随机入场无差别（wiki stock-master/overview）。
文献里证据最扎实的"趋势"是资产层面的时间序列动量（Moskowitz-Ooi-Pedersen 2012；
Faber 2007的10个月均线）：每月末看每个资产自己过去的走势，涨势中就持有、否则换成短债。
文献里它主要降低大跌时的回撤，不一定提高收益。

规则（月末收盘出信号、持有到下个月末，等权分配给有足够历史的资产，单边成本COST_BPS）：
  tsmom：过去N个月总收益 > 同期短债(SHY)收益 → 持有该资产，否则这一份换成SHY
  sma  ：月末收盘价 > 过去N个月月末收盘均值 → 持有，否则换成SHY
对照：同一股票池等权一直持有（每月再平衡，同样的资产可选范围）、SPY一直持有。

判定标准（2026-10-02跑之前登记，不事后改）：
  资产类别池里，tsmom-12 和 sma-10 两条都满足以下三条，才算"成立"：
    1. 最大回撤比等权一直持有小至少1/3
    2. Sharpe不低于等权一直持有
    3. 前后两半时段分别看，Sharpe都不低于等权一直持有超过0.1
  收益不要求更高。行业ETF池和SPY单独择时只作描述，不参与判定。
  参数稳健性（6/9/12个月）只作描述。

局限：资产池是今天挑的（偏差比个股小）；auto_adjust收盘价含分红；没有计税；
Sharpe未扣无风险利率（两边同口径，比较有效）。

用法：python -m src.backtest_trend
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
import yfinance as yf

from .backtest import _cache_path
from .backtest_alt import ETF_UNIVERSE, excess_stats, perf_stats

COST_BPS = 10
CASH = "SHY"
ASSET_CLASSES = ["SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "VNQ", "DBC"]
CRISES = {"2008金融危机": ("2007-11", "2009-03"), "2020疫情": ("2020-02", "2020-03"),
          "2022加息": ("2022-01", "2022-10")}


def load_monthly(tickers: list, start: str) -> pd.DataFrame:
    """月末收盘价（auto_adjust，含分红），去掉当前未走完的月份。"""
    path = _cache_path(f"trend_{start}_{len(tickers)}")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    raw = yf.download(sorted(set(tickers)), start=start, auto_adjust=True, progress=False, threads=True)["Close"]
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    m = raw.resample("ME").last()
    m = m[m.index < pd.Timestamp.today().normalize().replace(day=1)]
    with open(path, "wb") as f:
        pickle.dump(m, f)
    return m


def trend_portfolio(monthly: pd.DataFrame, cash: pd.Series, rule: str, lookback: int,
                    cost_bps: float = COST_BPS) -> tuple:
    """
    纯函数。monthly：各资产月末价；cash：短债月末价。
    第i个月末出信号，赚第i+1个月的收益；只有已有lookback+1个月历史的资产参与（等权1/N）。
    返回 (策略逐月收益, 等权一直持有逐月收益, 平均持有风险资产比例, 平均月换手)。
    """
    rets = monthly.pct_change()
    cash_ret = cash.pct_change()
    strat, bench, expo, turns = {}, {}, [], []
    prev = None
    for i in range(lookback, len(monthly) - 1):
        now, nxt = monthly.index[i], monthly.index[i + 1]
        hist = monthly.iloc[i - lookback: i + 1]
        ok = hist.notna().all() & monthly.iloc[i + 1].notna()
        names = list(ok[ok].index)
        if not names or pd.isna(cash.iloc[i - lookback]) or pd.isna(cash_ret.iloc[i + 1]):
            continue
        if rule == "tsmom":
            past = monthly.iloc[i][names] / monthly.iloc[i - lookback][names] - 1
            on = past > cash.iloc[i] / cash.iloc[i - lookback] - 1
        elif rule == "sma":
            on = monthly.iloc[i][names] > monthly.iloc[i - lookback + 1: i + 1][names].mean()
        else:
            raise ValueError(rule)
        w = pd.Series(0.0, index=list(monthly.columns) + [CASH])
        w[[n for n in names if on[n]]] = 1 / len(names)
        w[CASH] = 1 - w[names].sum()
        traded = float(w.sum()) if prev is None else float((w - prev).abs().sum())
        nr = rets.loc[nxt].reindex(w.index).fillna(0.0)
        nr[CASH] = cash_ret.loc[nxt]
        strat[nxt] = float((w * nr).sum()) - traded * cost_bps / 1e4
        bench[nxt] = float(rets.loc[nxt][names].mean())
        expo.append(1 - w[CASH])
        turns.append(traded)
        prev = w
    return (pd.Series(strat).sort_index(), pd.Series(bench).sort_index(),
            float(np.mean(expo)) if expo else float("nan"), float(np.mean(turns)) if turns else float("nan"))


def window_return(rets: pd.Series, start: str, end: str) -> float:
    r = rets[(rets.index >= pd.Timestamp(start)) & (rets.index <= pd.Timestamp(end) + pd.offsets.MonthEnd(0))]
    return round(float((1 + r).prod() - 1) * 100, 1) if len(r) else float("nan")


def judge(strat: pd.Series, bench: pd.Series, dd_cut: float = 1 / 3) -> dict:
    """按登记标准判断一条规则（纯函数）。dd_cut：回撤至少要比基准小的比例。"""
    ps, pb = perf_stats(strat, 12), perf_stats(bench, 12)
    half = len(strat) // 2
    halves = [(perf_stats(strat.iloc[a:b], 12)["sharpe"], perf_stats(bench.iloc[a:b], 12)["sharpe"])
              for a, b in ((0, half), (half, len(strat)))]
    c1 = ps["max_dd"] >= pb["max_dd"] * (1 - dd_cut)     # 回撤是负数：-30 >= -45*2/3
    c2 = ps["sharpe"] >= pb["sharpe"]
    c3 = all(s >= b - 0.1 for s, b in halves)
    return {"dd_ok": c1, "sharpe_ok": c2, "halves_ok": c3, "pass": c1 and c2 and c3, "halves": halves}


def _fmt(p: dict) -> str:
    calmar = round(p["ann_ret"] / -p["max_dd"], 2) if p.get("max_dd") else None
    return f"年化{p['ann_ret']:+.1f}% 波动{p['ann_vol']}% Sharpe {p['sharpe']} 回撤{p['max_dd']}% Calmar {calmar}"


def run_pool(name: str, monthly: pd.DataFrame, cash: pd.Series, spy: pd.Series, judged: bool) -> list:
    lines = [f"== {name}（{monthly.shape[1]}只，月末调仓，单边成本{COST_BPS}bp，空仓部分持有{CASH}）=="]
    verdicts = []
    _, bench, _, _ = trend_portfolio(monthly, cash, "tsmom", 12)
    for rule, lbs in (("tsmom", (6, 9, 12)), ("sma", (6, 10, 12))):
        for lb in lbs:
            s, bm, ex, to = trend_portfolio(monthly, cash, rule, lb)
            s, bm = s.reindex(bench.index).dropna(), bm.reindex(bench.index).dropna()   # 统一起点
            e = excess_stats(s, bm, 12)
            crisis = "  ".join(f"{k}{window_return(s, *v):+.1f}%/{window_return(bm, *v):+.1f}%" for k, v in CRISES.items())
            lines.append(f"  {rule}-{lb:2d}：{_fmt(perf_stats(s, 12))}  持仓{ex:.0%} 换手{to:.0%}/月 "
                         f"| 超额 {e['ann_excess']:+.2f}%/年 t={e['t']}")
            lines.append(f"      危机期（策略/一直持有）：{crisis}")
            if judged and (rule, lb) in (("tsmom", 12), ("sma", 10)):
                j = judge(s, bm)
                verdicts.append(j["pass"])
                lines.append(f"      判定：回撤小1/3 {'✓' if j['dd_ok'] else '✗'}  Sharpe不低 {'✓' if j['sharpe_ok'] else '✗'}  "
                             f"两半时段 {'✓' if j['halves_ok'] else '✗'} {j['halves']}")
    spy_r = spy.pct_change().reindex(bench.index)
    crisis = "  ".join(f"{k}{window_return(bench, *v):+.1f}%" for k, v in CRISES.items())
    lines.append(f"  对照 等权一直持有：{_fmt(perf_stats(bench, 12))}  危机期：{crisis}")
    lines.append(f"  对照 SPY一直持有：{_fmt(perf_stats(spy_r, 12))}  （{bench.index[0]:%Y-%m}至{bench.index[-1]:%Y-%m}）")
    if judged:
        lines.append(f"  ★ 登记标准判定：{'成立' if verdicts and all(verdicts) else '不成立'}")
    return lines


def main():
    ac = load_monthly(ASSET_CLASSES + [CASH], "2005-01-01")
    out = run_pool("资产类别", ac[ASSET_CLASSES], ac[CASH], ac["SPY"], judged=True) + [""]
    out += run_pool("SPY单独择时", ac[["SPY"]], ac[CASH], ac["SPY"], judged=False) + [""]
    sec = load_monthly(ETF_UNIVERSE + [CASH, "SPY"], "2015-01-01")
    out += run_pool("行业ETF", sec[[t for t in ETF_UNIVERSE if t in sec]], sec[CASH], sec["SPY"], judged=False)
    print("\n".join(out))


if __name__ == "__main__":
    main()
