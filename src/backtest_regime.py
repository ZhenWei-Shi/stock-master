"""
市场状态开关回测（2026-10-02，本地研究用）

背景：ETF趋势跟踪回测（backtest_trend，PR#47）显示时间序列动量的价值是"该不该在场"，
不是选股方向。这里测它当开关用：SPY处于下跌趋势时，现有两块策略暂停、资金放短期国债，
看能不能少亏而不多付代价。

开关（月末出信号，管下一个月）：
  sma-10  ：SPY月末收盘 > 过去10个月月末收盘均值 → 开
  tsmom-12：SPY过去12个月总收益 > 同期短期国债收益 → 开
被开关管的策略（逐月收益）：
  卖put   ：CBOE PutWrite指数^PUT（每月卖平值SPX put、国债担保），1997起。实际仓位是
            0.20 delta的ETF卖put价差，^PUT只是方向上的近似
  ETF动量 ：行业ETF 37只，12-1个月动量前5只等权、月末调仓（backtest_alt.rank_portfolio），
            2006起；最接近月度动量账本、又没有选股后见之明偏差
  个股动量：月度动量账本自己的股票池（momentum_book.universe），前2只，近10年；
            有幸存者偏差，只作描述
关掉的月份赚短期国债收益（^IRX折算），每次开关切换扣一次单边成本COST_BPS。

判定标准（2026-10-02跑之前登记，不事后改）：
  卖put、ETF动量两块分别判断；一块"成立"要求sma-10和tsmom-12两个开关都满足：
    1. 最大回撤比不加开关小至少1/4
    2. Sharpe不低于不加开关
    3. 前后两半时段Sharpe都不低于不加开关超过0.1
  个股动量只作描述。

用法：python -m src.backtest_regime
"""
import os
import pickle

import numpy as np
import pandas as pd
import yfinance as yf

from .backtest import _cache_path, load_prices
from .backtest_alt import ETF_UNIVERSE, excess_stats, perf_stats, rank_portfolio
from .backtest_trend import COST_BPS, _fmt, judge, window_return

CRISES = {"2001-02熊市": ("2001-01", "2002-09"), "2008金融危机": ("2007-11", "2009-03"),
          "2020疫情": ("2020-02", "2020-03"), "2022加息": ("2022-01", "2022-10")}


def load_daily(tickers: list, start: str) -> pd.DataFrame:
    path = _cache_path(f"regime_{start}_{len(tickers)}")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    raw = yf.download(sorted(set(tickers)), start=start, auto_adjust=True, progress=False, threads=True)["Close"]
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
    with open(path, "wb") as f:
        pickle.dump(raw, f)
    return raw


def to_monthly(daily: pd.Series) -> pd.Series:
    """日线 → 月末值，索引为月份Period；去掉当前没走完的月份。"""
    m = daily.dropna().resample("ME").last()
    m.index = m.index.to_period("M")
    return m[m.index < pd.Timestamp.today().to_period("M")]


def cash_from_irx(irx_daily: pd.Series) -> pd.Series:
    """13周国债年化收益率(%) → 逐月收益：用上月末的收益率赚这个月。"""
    y = to_monthly(irx_daily)
    return (y.shift(1) / 100 / 12).dropna()


def regime_signal(spy_m: pd.Series, cash_ret: pd.Series, rule: str, lookback: int) -> pd.Series:
    """纯函数。返回月份Period→bool：该月末的信号（管下一个月）。"""
    if rule == "sma":
        ref = spy_m.rolling(lookback).mean()
        sig = spy_m > ref
    elif rule == "tsmom":
        cash_idx = (1 + cash_ret.reindex(spy_m.index).fillna(0)).cumprod()
        ref = spy_m.shift(lookback)
        sig = (spy_m / ref) > (cash_idx / cash_idx.shift(lookback))
    else:
        raise ValueError(rule)
    return sig[ref.notna()].astype(bool)


def apply_regime(rets: pd.Series, signal: pd.Series, cash_ret: pd.Series,
                 cost_bps: float = COST_BPS) -> tuple:
    """
    纯函数。rets/cash_ret：月份Period→当月收益；signal：月份Period→月末信号。
    第p个月是否在场由p-1月末的信号决定；状态切换的那个月扣一次cost_bps。
    返回 (加开关后的逐月收益, 同期不加开关的逐月收益, 在场比例)。
    """
    out, base, on_list = {}, {}, []
    prev = None
    for p, r in rets.dropna().items():
        on = signal.get(p - 1)
        c = cash_ret.get(p)
        if on is None or c is None or pd.isna(c):
            continue
        val = r if on else c
        if prev is not None and on != prev:
            val -= cost_bps / 1e4
        out[p], base[p] = val, r
        on_list.append(on)
        prev = on
    return pd.Series(out), pd.Series(base), float(np.mean(on_list)) if on_list else float("nan")


def _as_ts(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = s.index.to_timestamp("M")
    return s


def run_sleeve(name: str, rets: pd.Series, spy_m: pd.Series, cash_ret: pd.Series, judged: bool) -> list:
    lines = [f"== {name}（月末开关，切换扣{COST_BPS}bp，关掉时赚短期国债）=="]
    verdicts = []
    for rule, lb in (("sma", 10), ("tsmom", 12), ("sma", 6), ("sma", 12)):
        sig = regime_signal(spy_m, cash_ret, rule, lb)
        f, b, on = apply_regime(rets, sig, cash_ret)
        f, b = _as_ts(f), _as_ts(b)
        e = excess_stats(f, b, 12)
        crisis = "  ".join(f"{k}{window_return(f, *v):+.1f}%/{window_return(b, *v):+.1f}%"
                           for k, v in CRISES.items() if f.index[0] <= pd.Timestamp(v[0] + "-01"))
        switches = int((pd.Series(sig.reindex(rets.index)).astype(float).diff().abs() > 0).sum())
        lines.append(f"  开关{rule}-{lb:2d}：{_fmt(perf_stats(f, 12))}  在场{on:.0%} 切换{switches}次 "
                     f"| 相对不加开关 {e['ann_excess']:+.2f}%/年 t={e['t']}")
        lines.append(f"      危机期（加开关/不加）：{crisis}")
        if judged and (rule, lb) in (("sma", 10), ("tsmom", 12)):
            j = judge(f, b, dd_cut=1 / 4)
            verdicts.append(j["pass"])
            lines.append(f"      判定：回撤小1/4 {'✓' if j['dd_ok'] else '✗'}  Sharpe不低 {'✓' if j['sharpe_ok'] else '✗'}  "
                         f"两半时段 {'✓' if j['halves_ok'] else '✗'} {j['halves']}")
    lines.append(f"  不加开关：{_fmt(perf_stats(b, 12))}  （{b.index[0]:%Y-%m}至{b.index[-1]:%Y-%m}）")
    if judged:
        lines.append(f"  ★ 登记标准判定：{'成立' if verdicts and all(verdicts) else '不成立'}")
    return lines


def main():
    base = load_daily(["SPY", "^PUT", "^IRX"], "1995-01-01")
    spy_m = to_monthly(base["SPY"])
    cash_ret = cash_from_irx(base["^IRX"])
    out = run_sleeve("卖put（^PUT）", to_monthly(base["^PUT"]).pct_change().dropna(), spy_m, cash_ret, True) + [""]

    etf = load_daily(ETF_UNIVERSE, "2005-01-01")
    s, _, _ = rank_portfolio(etf, "ME", 252, 21, 5)
    s.index = pd.DatetimeIndex(s.index).to_period("M")
    out += run_sleeve(f"ETF动量（行业ETF {etf.shape[1]}只，12-1个月前5只）", s, spy_m, cash_ret, True) + [""]

    from .backtest_alt import closes_frame
    from .momentum_book import universe
    tickers = universe()
    prices = load_prices(tickers, 10)
    cl = closes_frame(prices, [t for t in tickers if t in prices])
    s2, _, _ = rank_portfolio(cl, "ME", 252, 21, 2)
    s2.index = pd.DatetimeIndex(s2.index).to_period("M")
    out += run_sleeve(f"个股动量（动量账本股票池{cl.shape[1]}只，前2只，有幸存者偏差）", s2, spy_m, cash_ret, False)
    print("\n".join(out))


if __name__ == "__main__":
    main()
