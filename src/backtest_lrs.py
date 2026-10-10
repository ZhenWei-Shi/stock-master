"""
杠杆趋势轮动回测（Leverage Rotation，2026-10-10，用户要求"更冒险、回报更高"，本地研究用）

想法来源：Gayed & Bilello《Leverage for the Long Run》（SSRN 2016）——指数在200日均线上方时持有
杠杆ETF，跌破换成短期国债。GitHub调研（2026-10-10）：cozec/LRS（无许可证，只借思路）复现3倍QQQ
回撤-81%→-53%，但2010后牛市跑输一直持有；它的"Hybrid Sniper"调参版只多1.5%/年，是过拟合的反面
教材。nateGeorge/simulate_leveraged_ETFs（Apache）思路：用指数模拟杠杆ETF并对照真实ETF验证；
这里没有用它的代码（它按真实ETF拟合倍数、不含融资成本），改用标准公式。

杠杆ETF模拟（逐日）：
  r_lev = L × r_指数全收益 − (L−1) × (r_f + 融资利差) − 管理费/252
  r_f = 13周国债(^IRX)；管理费0.95%/年；融资利差用真实UPRO(2009-06起)/TQQQ(2010-02起)校准，
  并报告模拟与真实的年化差和逐日相关性。
指数：标普500全收益^SP500TR（1988起）；纳指100用^NDX价格（1985起，不含约0.7%/年股息，偏保守），
  1999-03起换成QQQ复权价。

规则（逐日）：第t天收盘看信号，管第t+1天的收益（不偷看当天）。
  进场：指数收盘 > N日均线×(1+b)；出场：指数收盘 < N日均线×(1−b)；中间维持原状态。
  空仓时赚短期国债；每次切换扣COST_BPS。另测一个月末版：月末收盘>10个月均线。
参数网格（共TRIALS组，DSR按这个数打折）：
  指数{标普500, 纳指100} × 杠杆{2, 3} × 均线{100,150,200,250}日 × 缓冲{0,1,2,3}% = 64组
  + 月末sma-10 × 指数 × 杠杆 = 4组

样本切分：样本内 = 起点 ~ 2021-09；样本外 = 2021-10 ~ 2026-09（调参时不看）。
选参（只用样本内）：每个(指数,杠杆)组内，按"该点和相邻参数（均线±50日、缓冲±1%）的样本内Sharpe
  中位数"选最高的一组——选平稳区，不选单点峰值。

判定标准（2026-10-10跑之前登记，不事后改）。对选出的每组（4组：两个指数×两个杠杆）分别判断，
  "成立"要求全部满足：
  1. 样本外年化 > SPY一直持有的样本外年化（目标是回报更高）
  2. 全期最大回撤 ≥ −60%（3倍一直持有在2000/2008接近归零）
  3. 样本内、样本外Sharpe都 ≥ SPY一直持有（不能只是放大风险）
  4. 样本内前后两半Sharpe都不低于SPY一直持有超过0.1
  5. Deflated Sharpe（Bailey & López de Prado 2014，按TRIALS组和各组Sharpe的离散度打折，
     基准为SPY一直持有的Sharpe）≥ 0.90
  只要有一组成立，就可以开前向模拟账本；成立组里选样本内回撤较小的。

用法：python -m src.backtest_lrs
"""
import math
import os
import pickle

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from .backtest import _cache_path
from .backtest_alt import perf_stats

COST_BPS = 10
EXPENSE = 0.0095
IS_END = "2021-09-30"
OOS_END = "2026-09-30"
MAS = (100, 150, 200, 250)
BUFFERS = (0.0, 0.01, 0.02, 0.03)
LEVERAGES = (2, 3)
INDEXES = {"标普500": "SPX", "纳指100": "NDX"}
REAL_3X = {"SPX": "UPRO", "NDX": "TQQQ"}
TRIALS = len(INDEXES) * len(LEVERAGES) * (len(MAS) * len(BUFFERS) + 1)
CRISES = {"2000-02互联网泡沫": ("2000-03-24", "2002-10-09"), "2008金融危机": ("2007-10-09", "2009-03-09"),
          "2020疫情": ("2020-02-19", "2020-03-23"), "2022加息": ("2022-01-03", "2022-10-12")}


# ── 数据 ──────────────────────────────────────────────────────

def load(tickers: list, start: str = "1985-01-01") -> pd.DataFrame:
    path = _cache_path(f"lrs_{start}_{'_'.join(sorted(tickers))}")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    raw = yf.download(sorted(set(tickers)), start=start, auto_adjust=True, progress=False, threads=True)["Close"]
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
    with open(path, "wb") as f:
        pickle.dump(raw, f)
    return raw


def index_series(px: pd.DataFrame) -> dict:
    """返回 {"SPX": (信号用价格, 全收益逐日收益), "NDX": ...}。"""
    spx_tr = px["^SP500TR"].dropna()
    ndx = px["^NDX"].dropna()
    qqq = px["QQQ"].dropna()
    ndx_ret = ndx.pct_change()
    q_ret = qqq.pct_change()
    ndx_ret.loc[q_ret.index[1]:] = q_ret.loc[q_ret.index[1]:]   # 1999-03起用QQQ复权（含股息）
    return {"SPX": (px["^GSPC"].dropna(), spx_tr.pct_change().dropna()),
            "NDX": (ndx, ndx_ret.dropna())}


def daily_rf(irx: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """13周国债年化(%) → 逐日收益；用前一天的收益率。"""
    y = irx.reindex(idx).ffill().shift(1)
    return (y / 100 / 252).fillna(0)


# ── 纯函数 ────────────────────────────────────────────────────

def simulate_leveraged(r: pd.Series, rf: pd.Series, lev: float, spread: float = 0.0,
                       expense: float = EXPENSE) -> pd.Series:
    """逐日杠杆ETF收益：L×r − (L−1)×(rf+利差) − 管理费。"""
    rf = rf.reindex(r.index).fillna(0)
    return lev * r - (lev - 1) * (rf + spread / 252) - expense / 252


def ma_signal(price: pd.Series, ma: int, buffer: float) -> pd.Series:
    """逐日在场状态（带缓冲的迟滞）：>MA×(1+b)进，<MA×(1−b)出，中间维持。"""
    m = price.rolling(ma).mean()
    up, dn = price > m * (1 + buffer), price < m * (1 - buffer)
    state, out = False, []
    for u, d, ok in zip(up.values, dn.values, m.notna().values):
        if not ok:
            out.append(np.nan)
            continue
        if u:
            state = True
        elif d:
            state = False
        out.append(state)
    return pd.Series(out, index=price.index)


def monthly_sma_signal(price: pd.Series, months: int = 10) -> pd.Series:
    """月末收盘>过去N个月月末均值 → 下个月整月在场；展开成逐日。"""
    me = price.resample("ME").last()
    sig = (me > me.rolling(months).mean()).where(me.rolling(months).mean().notna())
    daily = sig.reindex(price.index, method="ffill")
    return daily.where(daily.notna())


def apply_signal(lev_ret: pd.Series, cash: pd.Series, sig: pd.Series, cost_bps: float = COST_BPS) -> pd.Series:
    """第t天收盘的信号管第t+1天；切换那天扣一次成本。返回逐日策略收益（从信号可用起）。"""
    s = sig.shift(1).reindex(lev_ret.index)
    s = s[s.notna()].astype(bool)
    r = np.where(s, lev_ret.reindex(s.index), cash.reindex(s.index).fillna(0))
    sw = s.astype(int).diff().abs().fillna(0) > 0
    return pd.Series(r - sw.values * cost_bps / 1e4, index=s.index)


def sharpe_daily(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else float("nan")


def deflated_sharpe(r: pd.Series, sr_bench: float, trial_srs: list) -> float:
    """
    Deflated Sharpe Ratio（Bailey & López de Prado 2014），逐日口径。
    门槛 = 基准Sharpe + 试了N组时纯噪声下"最好那组"的期望Sharpe；返回真实Sharpe超过门槛的概率。
    """
    r = r.dropna()
    n, sr = len(r), sharpe_daily(r)
    k = len(trial_srs)
    v = float(np.var(trial_srs, ddof=1)) if k > 1 else 0.0
    g = 0.5772156649
    e_max = math.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / k) + g * norm.ppf(1 - 1 / (k * math.e))) if k > 1 else 0.0
    sr0 = sr_bench + e_max
    skew, kurt = float(r.skew()), float(r.kurt()) + 3
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr0) * math.sqrt(n - 1) / denom))


def stats(r: pd.Series) -> dict:
    return perf_stats(r, 252)


def window_ret(r: pd.Series, a: str, b: str) -> float:
    w = r[(r.index >= a) & (r.index <= b)]
    return round(float((1 + w).prod() - 1) * 100, 1) if len(w) else float("nan")


def neighbors(ma: int, b: float) -> list:
    return [(m, x) for m in MAS for x in BUFFERS
            if abs(m - ma) <= 50 and abs(round((x - b) * 100)) <= 1]


# ── 主流程 ────────────────────────────────────────────────────

def calibrate(idx: dict, rf: pd.Series, px: pd.DataFrame) -> tuple:
    """用真实3倍ETF校准融资利差：使模拟与真实的年化收益差最小。返回 ({指数: 利差}, 报告行)。"""
    lines, spreads = [], {}
    for key, real in REAL_3X.items():
        rr = px[real].dropna().pct_change().dropna()
        base = idx[key][1].reindex(rr.index).dropna()
        rr = rr.reindex(base.index)
        best = None
        for sp in np.arange(0.0, 0.0301, 0.0025):
            sim = simulate_leveraged(base, rf, 3, sp)
            gap = stats(sim)["ann_ret"] - stats(rr)["ann_ret"]
            if best is None or abs(gap) < abs(best[1]):
                best = (sp, gap, sim)
        sp, gap, sim = best
        spreads[key] = float(sp)
        corr = float(np.corrcoef(sim, rr)[0, 1])
        lines.append(f"  {real}（{rr.index[0]:%Y-%m}起）：真实年化{stats(rr)['ann_ret']:+.1f}% 回撤{stats(rr)['max_dd']}%"
                     f"｜模拟（利差{sp:.2%}）年化{stats(sim)['ann_ret']:+.1f}% 回撤{stats(sim)['max_dd']}%"
                     f"｜年化差{gap:+.1f}% 逐日相关{corr:.4f}")
    return spreads, lines


def run():
    px = load(["^GSPC", "^SP500TR", "^NDX", "QQQ", "SPY", "^IRX", "UPRO", "TQQQ"])
    idx = index_series(px)
    rf = daily_rf(px["^IRX"], px.index)
    spreads, cal_lines = calibrate(idx, rf, px)
    out = ["== 模拟杠杆ETF对照真实ETF（校准融资利差）=="] + cal_lines + [""]

    spy = px["SPY"].dropna().pct_change().dropna()
    spy_is, spy_oos = spy[:IS_END], spy[IS_END:OOS_END].iloc[1:]
    sr_spy_is, sr_spy_oos = sharpe_daily(spy_is), sharpe_daily(spy_oos)

    results = {}
    for name, key in INDEXES.items():
        price, r = idx[key]
        for lev in LEVERAGES:
            lr = simulate_leveraged(r, rf, lev, spreads[key])
            for ma in MAS:
                for b in BUFFERS:
                    results[(key, lev, ma, b)] = apply_signal(lr, rf, ma_signal(price, ma, b))
            results[(key, lev, "M10", None)] = apply_signal(lr, rf, monthly_sma_signal(price))
            results[(key, lev, "BH", None)] = lr.copy()
    trial_srs = [sharpe_daily(v[:IS_END]) for k, v in results.items() if k[2] != "BH"]
    assert len(trial_srs) == TRIALS

    out.append(f"== 基准：SPY一直持有 ==")
    out.append(f"  样本内（{spy_is.index[0]:%Y-%m}~2021-09）：{_f(stats(spy_is))}")
    out.append(f"  样本外（2021-10~2026-09）：{_f(stats(spy_oos))}")
    out.append("")

    verdicts = []
    for name, key in INDEXES.items():
        for lev in LEVERAGES:
            grid = {(ma, b): sharpe_daily(results[(key, lev, ma, b)][:IS_END]) for ma in MAS for b in BUFFERS}
            plateau = {p: float(np.median([grid[q] for q in neighbors(*p)])) for p in grid}
            ma, b = max(plateau, key=plateau.get)
            out.append(f"== {name} {lev}倍（全期{results[(key, lev, ma, b)].index[0]:%Y-%m}起）==")
            out.append("  样本内年化Sharpe网格（行=均线，列=缓冲0/1/2/3%）：")
            for m in MAS:
                out.append(f"    {m:3d}日  " + "  ".join(f"{grid[(m, x)] * math.sqrt(252):.2f}" for x in BUFFERS))
            out.append(f"    月末sma-10：{sharpe_daily(results[(key, lev, 'M10', None)][:IS_END]) * math.sqrt(252):.2f}")
            out.append(f"  选中（平稳区中位数最高）：{ma}日均线、缓冲{b:.0%}")
            for label, k in ((f"轮动{ma}日/{b:.0%}", (key, lev, ma, b)), ("月末sma-10", (key, lev, "M10", None)),
                             (f"{lev}倍一直持有", (key, lev, "BH", None))):
                s = results[k]
                s_is, s_oos = s[:IS_END], s[IS_END:OOS_END].iloc[1:]
                out.append(f"  {label}：全期 {_f(stats(s[:OOS_END]))}")
                out.append(f"      样本内 {_f(stats(s_is))}")
                out.append(f"      样本外 {_f(stats(s_oos))}")
                out.append("      危机期：" + "  ".join(f"{c}{window_ret(s, *w):+.1f}%" for c, w in CRISES.items()
                                                     if s.index[0] <= pd.Timestamp(w[0])))
            s = results[(key, lev, ma, b)]
            s_is, s_oos = s[:IS_END], s[IS_END:OOS_END].iloc[1:]
            half = len(s_is) // 2
            spy_al = spy.reindex(s_is.index).dropna()
            h = [(sharpe_daily(s_is.iloc[a:z]), sharpe_daily(spy_al.reindex(s_is.index[a:z]).dropna()))
                 for a, z in ((0, half), (half, len(s_is)))]
            dsr = deflated_sharpe(s_is, sr_spy_is, trial_srs)
            c = [stats(s_oos)["ann_ret"] > stats(spy_oos)["ann_ret"],
                 stats(s[:OOS_END])["max_dd"] >= -60,
                 sharpe_daily(s_is) >= sr_spy_is and sharpe_daily(s_oos) >= sr_spy_oos,
                 all(a * math.sqrt(252) >= z * math.sqrt(252) - 0.1 for a, z in h),
                 dsr >= 0.90]
            verdicts.append(((name, lev, ma, b), all(c), stats(s_is)["max_dd"]))
            out.append(f"  判定：①样本外跑赢SPY {_ok(c[0])} ②回撤≥-60% {_ok(c[1])} ③Sharpe不低于SPY {_ok(c[2])} "
                       f"④两半时段 {_ok(c[3])} ({h[0][0] * 15.87:.2f}/{h[0][1] * 15.87:.2f}, {h[1][0] * 15.87:.2f}/{h[1][1] * 15.87:.2f}) "
                       f"⑤DSR={dsr:.3f} {_ok(c[4])} → {'成立' if all(c) else '不成立'}")
            out.append("")
    ok = [v for v in verdicts if v[1]]
    out.append(f"★ 登记标准判定（试了{TRIALS}组）：" + (
        "成立 " + "、".join(f"{n}{l}倍{m}日/{b:.0%}" for (n, l, m, b), _, _ in ok)
        + f"；前向账本建议用 {max(ok, key=lambda v: v[2])[0]}" if ok else "全部不成立"))
    return "\n".join(out)


def _f(p: dict) -> str:
    calmar = round(p["ann_ret"] / -p["max_dd"], 2) if p.get("max_dd") else None
    return f"年化{p['ann_ret']:+.1f}% 波动{p['ann_vol']}% Sharpe {p['sharpe']} 回撤{p['max_dd']}% Calmar {calmar}"


def _ok(x: bool) -> str:
    return "✓" if x else "✗"


if __name__ == "__main__":
    print(run())
