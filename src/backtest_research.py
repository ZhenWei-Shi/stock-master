"""
回测研究脚本（2026-09-29，本地运行）：在backtest.py基础上回答四个问题

  1. ablation   —— 逐个关掉硬门，哪些门真正有用？
                   另外对每个门做"边际效果"：其余门都通过的日子里，该门通过 vs 不通过
  2. pullback   —— 路径B（回调企稳）5年只触发4次，是设计还是bug？拆开各子条件看
  3. short      —— 做空方向的技术面信号有没有优势（决定put信号值不值得继续做）
  4. universe   —— 换一个"2021年热门、之后大跌"的股票池，看结论是否依赖幸存者偏差

每个结果都和"同股票、同笔数、同出场规则的随机入场"对照组比较——绝对收益会
被股票池本身的涨跌带偏，只有相对随机入场的差值才说明信号有没有用。

用法：python -m src.backtest_research [ablation|pullback|short|universe|all]
"""
import math
import sys

import numpy as np
import pandas as pd

from .backtest import (
    HARD_GATES, PULLBACK_ZONE_MAX_PCT, PULLBACK_ZONE_MIN_PCT, compute_signals, ablate,
    simulate_trades, random_control_trades, trade_stats, default_universe, load_prices,
    load_earnings_dates, _rsi_series, _check_pullback_setup,
)

# 2021年热门/高估值、之后大幅下跌的股票（有意用"事后看跌得多"的反向偏差，
# 与默认池"事后看涨得多"对照；两者结论一致才说明结论不依赖选股池）
LOSERS_2021 = [
    "PTON", "ZM", "TDOC", "ROKU", "DOCU", "PYPL", "SNAP", "PINS", "ETSY", "CHWY",
    "LCID", "RIVN", "NIO", "XPEV", "PLUG", "FSLR", "ENPH", "SEDG", "RUN", "CHPT",
    "BYND", "FUBO", "SPCE", "UPST", "AFRM", "SOFI", "HOOD", "COIN", "MRNA", "BNTX",
    "NVAX", "BIIB", "INTC", "WBA", "DIS", "BABA", "JD", "PDD", "SE", "SHOP",
    "U", "RBLX", "DKNG", "PATH", "TWLO", "OKTA", "ZS", "CRWD", "NET", "DDOG",
]


def _mean_se(x) -> tuple:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan"), float("nan"), len(x))
    return (float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))), len(x))


def compare_to_random(prices: dict, trades: pd.DataFrame, direction: str = "LONG", seeds: int = 3) -> dict:
    """真实交易 vs 随机入场（多个种子合并）：平均收益差及其标准误。"""
    if trades.empty or trades["ret_pct"].notna().sum() == 0:
        return {"n": 0}
    real = trades["ret_pct"].dropna()
    ctrl = pd.concat([random_control_trades(prices, trades, seed=s, direction=direction) for s in range(seeds)])
    ctrl = ctrl["ret_pct"].dropna()
    m1, se1, n1 = _mean_se(real)
    m2, se2, n2 = _mean_se(ctrl)
    diff_se = math.sqrt(se1 ** 2 + (se2 ** 2)) if not (math.isnan(se1) or math.isnan(se2)) else float("nan")
    return {"n": n1, "avg": m1, "se": se1, "win": float((real > 0).mean() * 100),
            "rand_avg": m2, "rand_win": float((ctrl > 0).mean() * 100),
            "diff": m1 - m2, "diff_se": diff_se, "t": (m1 - m2) / diff_se if diff_se else float("nan")}


def _fmt(name: str, r: dict) -> str:
    if not r.get("n"):
        return f"{name:30s} n=0"
    return (f"{name:30s} n={r['n']:5d}  平均{r['avg']:+.2f}%(±{1.96 * r['se']:.2f})  胜率{r['win']:.1f}%  "
            f"| 随机{r['rand_avg']:+.2f}% 胜率{r['rand_win']:.1f}%  | 差值{r['diff']:+.2f}% (t={r['t']:+.1f})")


def _all_signals(prices, earnings, tickers, direction="LONG", max_fails=1) -> dict:
    spy, vix = prices["SPY"]["Close"], prices["^VIX"]["Close"]
    return {t: compute_signals(prices[t], spy, vix, earnings.get(t), direction=direction, max_fails=max_fails)
            for t in tickers if t in prices}


def _trades(prices, sigs, direction="LONG", transform=None) -> pd.DataFrame:
    out = []
    for t, sig in sigs.items():
        s = transform(sig) if transform else sig
        tr = simulate_trades(prices[t], s, t, direction=direction)
        for x in tr:
            row = sig.loc[x["entry_date"]]
            for g in HARD_GATES:
                x[f"g_{g}"] = bool(row[g])
        out += tr
    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────
# 1. ablation
# ─────────────────────────────────────────────────────────────

def run_ablation(prices, earnings, tickers) -> list:
    sigs = _all_signals(prices, earnings, tickers, max_fails=1)
    lines = ["== 1. 逐个关掉硬门（LONG，激进模式）=="]
    base = _trades(prices, sigs)
    lines.append(_fmt("全部硬门（基准）", compare_to_random(prices, base)))
    for g in HARD_GATES:
        tr = _trades(prices, sigs, transform=lambda s, g=g: ablate(s, (g,)))
        lines.append(_fmt(f"关掉 {g}", compare_to_random(prices, tr)))
        # 边际效果：关掉g之后的交易里，g本来通过 vs 不通过
        passed, failed = tr[tr[f"g_{g}"]], tr[~tr[f"g_{g}"]]
        mp, sp, npass = _mean_se(passed["ret_pct"])
        mf, sf, nfail = _mean_se(failed["ret_pct"])
        if nfail >= 2:
            d_se = math.sqrt(sp ** 2 + sf ** 2)
            lines.append(f"    └ {g}的边际效果：通过{npass}笔 平均{mp:+.2f}%  vs  不通过{nfail}笔 平均{mf:+.2f}%  "
                         f"差{mp - mf:+.2f}% (t={(mp - mf) / d_se:+.1f})")
    lines.append("-- 用加分项当过滤器（在全部硬门通过的基础上）--")
    for name, cond in (("只做动能确认强(conviction≥3)", lambda s: s["conviction"].fillna(-99) >= 3),
                       ("只做动能确认弱(conviction<0)", lambda s: s["conviction"].fillna(99) < 0),
                       ("只做路径A(突破区)", lambda s: s["path"] == "A"),
                       ("只做非路径A", lambda s: s["path"] != "A")):
        tr = _trades(prices, sigs, transform=lambda s, c=cond: s.assign(go=s["go"] & c(s)))
        lines.append(_fmt(name, compare_to_random(prices, tr)))
    go_all = pd.concat([s[s["go"]] for s in sigs.values()])
    lines.append(f"注：激进模式下score = 100 − VIX扣分(≤15) + bonus(−15..+25)，全部硬门通过时score最低约70，"
                 f"始终≥65——回测里GO日共{len(go_all)}天，score<65的{int((go_all['score'] < 65).sum())}天。"
                 f"即score/bonus在技术面层面从不改变GO结论，只有硬门起作用。")
    return lines


# ─────────────────────────────────────────────────────────────
# 2. 路径B诊断
# ─────────────────────────────────────────────────────────────

def run_pullback(prices, earnings, tickers) -> list:
    counts = {k: 0 for k in ("zone", "support", "vol", "rsi_low", "rsi_rebound", "confirmed",
                             "confirmed_trend_strict", "confirmed_trend_relaxed", "trend_strict_in_zone")}
    relaxed_sigs = {}
    spy, vix = prices["SPY"]["Close"], prices["^VIX"]["Close"]
    for t in tickers:
        if t not in prices:
            continue
        h = prices[t]
        c, v = h["Close"], h["Volume"]
        ma20, ma50, ma200 = (c.rolling(n).mean() for n in (20, 50, 200))
        rsi = _rsi_series(c)
        p20 = (c / c.rolling(20).max() - 1) * 100
        strict = (c > ma20) & (ma20 > ma50) & (ma50 > ma200)
        relaxed = (c > ma20 * 0.98) & (ma20 > ma50) & (ma50 > ma200)   # 与路径B自己的2%缓冲一致
        sig = compute_signals(h, spy, vix, earnings.get(t), max_fails=1)
        sig = sig.assign(go=False)
        for i in range(252, len(c)):
            if not (PULLBACK_ZONE_MIN_PCT <= p20.iloc[i] < PULLBACK_ZONE_MAX_PCT):
                continue
            counts["zone"] += 1
            counts["trend_strict_in_zone"] += bool(strict.iloc[i])
            pb = _check_pullback_setup(c.iloc[:i + 1], v.iloc[:i + 1], rsi.iloc[:i + 1], ma20.iloc[i], c.iloc[i])
            counts["support"] += pb["support_held"]
            counts["vol"] += pb["vol_dried_up"]
            counts["rsi_low"] += bool(rsi.iloc[max(0, i - 9):i + 1].min() < 40)
            counts["rsi_rebound"] += pb["rsi_rebounding"]
            if pb["confirmed"]:
                counts["confirmed"] += 1
                counts["confirmed_trend_strict"] += bool(strict.iloc[i])
                counts["confirmed_trend_relaxed"] += bool(relaxed.iloc[i])
                d = c.index[i]
                others_ok = all(bool(sig.at[d, g]) for g in HARD_GATES if g != "trend") if d in sig.index else False
                if relaxed.iloc[i] and others_ok:
                    sig.at[d, "go"] = True
                    sig.at[d, "path"] = "B"
        relaxed_sigs[t] = sig
    z = max(counts["zone"], 1)
    lines = ["== 2. 路径B（回调企稳）为什么几乎不触发 ==",
             f"回调区（距20日高点-5%~-15%）股票日：{counts['zone']}",
             f"  其中严格trend门通过（价格>MA20>MA50>MA200）：{counts['trend_strict_in_zone']}（{counts['trend_strict_in_zone'] / z * 100:.1f}%）",
             f"  子条件单独成立：MA20支撑未破(>98%MA20) {counts['support'] / z * 100:.1f}%  "
             f"回调缩量 {counts['vol'] / z * 100:.1f}%  近10日RSI曾<40 {counts['rsi_low'] / z * 100:.1f}%  "
             f"RSI企稳回升 {counts['rsi_rebound'] / z * 100:.1f}%",
             f"  三个子条件同时成立（路径B确认）：{counts['confirmed']}（{counts['confirmed'] / z * 100:.2f}%）",
             f"  确认 且 严格trend通过：{counts['confirmed_trend_strict']}   确认 且 放宽trend（价格>MA20×98%）：{counts['confirmed_trend_relaxed']}"]
    tr = _trades(prices, relaxed_sigs)
    lines.append(_fmt("路径B+放宽trend（其余硬门通过）", compare_to_random(prices, tr)))
    return lines


# ─────────────────────────────────────────────────────────────
# 3. 做空
# ─────────────────────────────────────────────────────────────

def run_short(prices, earnings, tickers) -> list:
    sigs = _all_signals(prices, earnings, tickers, direction="SHORT", max_fails=0)
    lines = ["== 3. 做空方向（SHORT技术面gate：价格<MA20<MA50、RSI 20-55、量比≥0.8、止损≤12%、财报前7天禁入）=="]
    tr = _trades(prices, sigs, direction="SHORT")
    lines.append(_fmt("SHORT全部信号", compare_to_random(prices, tr, direction="SHORT")))
    below200 = {t: s.assign(go=s["go"] & (prices[t]["Close"] < prices[t]["Close"].rolling(200).mean())
                            .reindex(s.index).fillna(False)) for t, s in sigs.items()}
    tr2 = _trades(prices, below200, direction="SHORT")
    lines.append(_fmt("SHORT且价格<MA200", compare_to_random(prices, tr2, direction="SHORT")))
    if not tr.empty:
        st = trade_stats(tr)
        lines.append(f"  出场分布：{tr['reason'].value_counts().to_dict()}  平均持有{st.get('avg_bars')}根K线（未计融券/期权成本）")
    return lines


# ─────────────────────────────────────────────────────────────
# 4. 股票池
# ─────────────────────────────────────────────────────────────

def run_universe(years) -> list:
    lines = ["== 4. 换股票池：2021年热门、之后大跌的50只（反向幸存者偏差）=="]
    prices = load_prices(LOSERS_2021, years)
    tickers = [t for t in LOSERS_2021 if t in prices]
    earnings = load_earnings_dates(tickers)
    sigs = _all_signals(prices, earnings, tickers, max_fails=0)
    tr = _trades(prices, sigs)
    lines.append(f"有数据的股票 {len(tickers)} 只")
    lines.append(_fmt("LONG全部信号", compare_to_random(prices, tr)))
    ssigs = _all_signals(prices, earnings, tickers, direction="SHORT", max_fails=0)
    lines.append(_fmt("SHORT全部信号", compare_to_random(prices, _trades(prices, ssigs, "SHORT"), "SHORT")))
    return lines


def main(which: str = "all", years: int = 5):
    tickers = default_universe()
    prices = load_prices(tickers, years)
    tickers = [t for t in tickers if t in prices]
    earnings = load_earnings_dates(tickers)
    out = []
    if which in ("ablation", "all"):
        out += run_ablation(prices, earnings, tickers) + [""]
    if which in ("pullback", "all"):
        out += run_pullback(prices, earnings, tickers) + [""]
    if which in ("short", "all"):
        out += run_short(prices, earnings, tickers) + [""]
    if which in ("universe", "all"):
        out += run_universe(years)
    text = "\n".join(out)
    print(text)
    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
