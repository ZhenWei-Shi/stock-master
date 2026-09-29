"""
九关模型"技术面近似回测"（2026-09-29重建，研究用，本地运行，不部署到服务器）

为什么要有：模拟盘约每周1笔，时间止损制度下到2026-09-28只有3笔干净样本，
按这个速度攒到30笔要到2027年春。用历史日线把cold_model的技术面gate逐日重放，
一次就能在几十只股票、几年数据上拿到几百上千个信号样本。

口径（尽量复用cold_model的函数，避免回测和实盘两套逻辑）：
  - 逐日计算的gate：trend、rsi、volume、stop_distance（激进模式12%）、near_high
    四段式（路径A突破区/路径B回调企稳）、earnings_blackout（财报前7天禁入，用yfinance
    历史财报日）、动能确认综合指数（VCP/MACD/OBV/相对SPY强度）、VIX扣分
  - 算分：cold_model._calc_score + 同样的bonus规则；GO = 无硬否决且分数≥65
  - 出场：与模拟盘实际行为一致——只有1.5ATR止损（跳空低开按开盘价成交）和
    持仓超过10个日历日的时间止损自动平仓；到达目标价只提醒不平仓（见
    trading_agent.run_monitor）。另提供take_profit_atr参数做对照
  - 入场价：信号日收盘价（对应15:30扫描），与entry_plan一致

没法回测的部分（结果因此偏乐观，应视为"技术面信号的上限"）：
  - earnings_quality（CANSLIM，D级一票否决）、营收加速/PEAD等基本面——没有时点数据
  - 宏观否决（FOMC/CPI日）、debt_event、news_event、板块轮动加分、期权/资金流向
  - 辩论层CONDITIONAL减半仓、PDT、盘中VWAP
  - 股票池是当前的watchlist+板块代表股，有幸存者偏差

用法：
  python -m src.backtest                 # 默认5年，打印报告
  python -m src.backtest --years 3 --tp 2.0
"""
import argparse
import math
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from .cold_model import (
    ATR_STOP_MULT, GO_THRESHOLD_AGG, MAX_STOP_PCT_AGG, MIN_STOP_PCT,
    RSI_AGG_MIN, RSI_AGG_MAX, VOL_RATIO_MIN, SELLOFF_VOL_RATIO, PRICE_DROP_WARN_PCT,
    NEAR_HIGH_BREAKOUT_PCT, PULLBACK_ZONE_MAX_PCT, PULLBACK_ZONE_MIN_PCT, PULLBACK_BONUS,
    CONV_STRONG_THRESHOLD, CONV_MAX_BONUS, CONV_MAX_PENALTY,
    MACD_MOMENTUM_LOOKBACK, VP_DIVERGENCE_LOOKBACK, VCP_LOOKBACK_DAYS,
    _rsi_series, _calc_macd_hist, _check_vcp_contraction, _check_pullback_setup,
    _momentum_conviction, _calc_score,
)

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
_CACHE_DIR = os.path.join(_DATA, "backtest_cache")

EARNINGS_BLACKOUT_DAYS = 7      # 与cold_model earnings_blackout一致：7天内禁入
MAX_HOLD_CALENDAR_DAYS = 10     # 与paper_trading一致
RS_LOOKBACK            = 63     # 相对SPY强度：3个月


def default_universe() -> list:
    """当前watchlist + 板块代表股（与动态watchlist的候选来源一致）。"""
    from .sector_rotation import SECTOR_TICKERS
    from .scheduler import DEFAULT_WATCHLIST
    tickers = set(DEFAULT_WATCHLIST) | {"ASTS", "MU", "QQQ"}
    for lst in SECTOR_TICKERS.values():
        tickers |= set(lst)
    return sorted(tickers)


# ─────────────────────────────────────────────────────────────
# 取数（带本地缓存，同一天重复跑不重复下载）
# ─────────────────────────────────────────────────────────────

def _cache_path(name: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"{name}_{datetime.now():%Y%m%d}.pkl")


def load_prices(tickers: list, years: int) -> dict:
    """返回 {ticker: OHLCV DataFrame}，含SPY和^VIX。"""
    path = _cache_path(f"prices_{years}y_{len(tickers)}")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    allt = sorted(set(tickers) | {"SPY", "^VIX"})
    raw = yf.download(allt, period=f"{years}y", auto_adjust=True, group_by="ticker",
                      progress=False, threads=True)
    out = {}
    for t in allt:
        try:
            df = raw[t].dropna(how="all")
            if len(df) > 260:
                out[t] = df[["Open", "High", "Low", "Close", "Volume"]]
        except KeyError:
            pass
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def load_earnings_dates(tickers: list) -> dict:
    """{ticker: 升序的财报日期列表（date）}；取不到的标的返回空列表（不做财报过滤）。"""
    path = _cache_path(f"earnings_{len(tickers)}")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    out = {}
    for t in tickers:
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=40)
            out[t] = sorted({d.date() for d in ed.index}) if ed is not None else []
        except Exception:
            out[t] = []
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


# ─────────────────────────────────────────────────────────────
# 逐日重放技术面gate
# ─────────────────────────────────────────────────────────────

def days_to_next_earnings(index: pd.DatetimeIndex, earnings: list) -> np.ndarray:
    """每个交易日距下一次财报的日历天数；之后没有已知财报日时为inf。"""
    if not earnings:
        return np.full(len(index), np.inf)
    ed = np.array([np.datetime64(d) for d in earnings])
    days = index.tz_localize(None).values.astype("datetime64[D]") if index.tz else index.values.astype("datetime64[D]")
    pos = np.searchsorted(ed, days, side="left")
    out = np.full(len(days), np.inf)
    ok = pos < len(ed)
    out[ok] = (ed[pos[ok]] - days[ok]).astype(int)
    return out


def compute_signals(hist: pd.DataFrame, spy_close: pd.Series, vix: pd.Series,
                    earnings: list | None = None, start: int = 252) -> pd.DataFrame:
    """
    对一只股票逐日计算激进模式LONG的技术面verdict。返回按日期索引的DataFrame：
    go / score / path(A=突破区,B=回调企稳,-) / conviction / stop_frac / fails。
    """
    close, high, low, vol = hist["Close"], hist["High"], hist["Low"], hist["Volume"]
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    rsi = _rsi_series(close)
    vol_m20 = vol.rolling(20).mean()
    vol_ratio = vol / vol_m20
    price_chg = close.pct_change() * 100
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=13, adjust=False).mean()
    stop_pct = atr * ATR_STOP_MULT / close * 100
    pct_20h = (close / close.rolling(20).max() - 1) * 100
    macd_h = _calc_macd_hist(close)
    sign = np.sign(close.diff().fillna(0))
    obv = (sign * vol).cumsum()
    spy = spy_close.reindex(close.index).ffill()
    rs = close.pct_change(RS_LOOKBACK) - spy.pct_change(RS_LOOKBACK)
    vix_a = vix.reindex(close.index).ffill().fillna(20.0)
    d2e = days_to_next_earnings(close.index, earnings or [])

    trend_ok = (close > ma20) & (ma20 > ma50) & (ma50 > ma200) & ma200.notna()
    rsi_ok = rsi.between(RSI_AGG_MIN, RSI_AGG_MAX)
    heavy_sell = (price_chg < PRICE_DROP_WARN_PCT) & (vol_ratio > SELLOFF_VOL_RATIO)
    vol_ok = (vol_ratio >= VOL_RATIO_MIN) & ~heavy_sell
    stop_ok = stop_pct.between(MIN_STOP_PCT, MAX_STOP_PCT_AGG)
    earn_ok = d2e > EARNINGS_BLACKOUT_DAYS

    rows = []
    idx = close.index
    for i in range(start, len(close)):
        fails = [name for name, ok in (("trend", trend_ok.iloc[i]), ("rsi", rsi_ok.iloc[i]),
                                       ("volume", vol_ok.iloc[i]), ("stop_distance", stop_ok.iloc[i]),
                                       ("earnings_blackout", earn_ok[i])) if not ok]
        if fails:
            rows.append((idx[i], False, None, "-", None, None, ",".join(fails)))
            continue

        p20 = pct_20h.iloc[i]
        path, near, pullback = "-", True, None
        if p20 >= NEAR_HIGH_BREAKOUT_PCT:
            path = "A"
        elif p20 >= PULLBACK_ZONE_MAX_PCT:
            path = "-"
        elif p20 >= PULLBACK_ZONE_MIN_PCT:
            pullback = _check_pullback_setup(close.iloc[:i + 1], vol.iloc[:i + 1], rsi.iloc[:i + 1],
                                             ma20.iloc[i], close.iloc[i])
            near = True if pullback["confirmed"] else "warn"
            path = "B" if pullback["confirmed"] else "-"
        else:
            near = "warn"

        vcp = _check_vcp_contraction(hist.iloc[i + 1 - VCP_LOOKBACK_DAYS:i + 1])
        lb = MACD_MOMENTUM_LOOKBACK
        macd_pass = True if (macd_h.iloc[i] > 0 or macd_h.iloc[i] > macd_h.iloc[i - lb]) else "warn"
        vlb = VP_DIVERGENCE_LOOKBACK
        diverge = close.iloc[i] - close.iloc[i - vlb] > 0 and obv.iloc[i] - obv.iloc[i - vlb] <= 0
        vp_pass = "warn" if diverge else True
        rs_i = rs.iloc[i]
        conv, _ = _momentum_conviction("LONG", vcp["contracted"], macd_pass, vp_pass,
                                       None if pd.isna(rs_i) else float(rs_i), {}, {})

        gates = {"trend": {"pass": True}, "rsi": {"pass": True}, "volume": {"pass": True},
                 "stop_distance": {"pass": True}, "near_high": {"pass": near},
                 "vwap": {"pass": "skip"}, "time_window": {"pass": True}}
        score = _calc_score(gates, float(vix_a.iloc[i]), aggressive_mode=True)
        bonus = 0
        if conv >= CONV_STRONG_THRESHOLD:
            bonus += min(CONV_MAX_BONUS, conv)
        elif conv < 0:
            bonus -= min(CONV_MAX_PENALTY, abs(conv) * 2)
        if pullback is not None and pullback["confirmed"]:
            bonus += PULLBACK_BONUS
        adjusted = min(100, score + bonus)
        rows.append((idx[i], adjusted >= GO_THRESHOLD_AGG, adjusted, path, conv,
                     float(stop_pct.iloc[i]) / 100, ""))

    return pd.DataFrame(rows, columns=["date", "go", "score", "path", "conviction", "stop_frac", "fails"]).set_index("date")


# ─────────────────────────────────────────────────────────────
# 逐笔出场模拟（与模拟盘规则一致）
# ─────────────────────────────────────────────────────────────

def simulate_trades(hist: pd.DataFrame, signals: pd.DataFrame, ticker: str = "",
                    take_profit_atr: float | None = None,
                    max_hold_days: int = MAX_HOLD_CALENDAR_DAYS) -> list:
    """
    信号日收盘入场；之后每天先看止损（开盘已低于止损→按开盘价成交，否则最低价
    触及→按止损价成交），再看止盈（可选），最后看时间止损（持有超过max_hold_days
    个日历日→按当天收盘价平仓）。同一标的持仓期间不再开新仓（防摊平）。
    数据结束时仍持仓的交易标记为open，不计入统计。
    """
    o, h, l, c = (hist[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    dates = hist.index
    pos_of = {d: k for k, d in enumerate(dates)}
    trades = []
    busy_until = -1
    for d, s in signals[signals["go"]].iterrows():
        i = pos_of.get(d)
        if i is None or i <= busy_until:
            continue
        entry = c[i]
        stop = entry * (1 - s["stop_frac"])
        tp = entry * (1 + s["stop_frac"] / ATR_STOP_MULT * take_profit_atr) if take_profit_atr else None
        exit_px, reason, j = None, "open", i
        for j in range(i + 1, len(c)):
            if o[j] <= stop:
                exit_px, reason = o[j], "stop_gap"
            elif l[j] <= stop:
                exit_px, reason = stop, "stop"
            elif tp is not None and o[j] >= tp:
                exit_px, reason = o[j], "target_gap"
            elif tp is not None and h[j] >= tp:
                exit_px, reason = tp, "target"
            elif (dates[j] - dates[i]).days > max_hold_days:
                exit_px, reason = c[j], "time"
            if exit_px is not None:
                break
        busy_until = j
        trades.append({
            "ticker": ticker, "entry_date": d, "entry": round(entry, 4),
            "exit_date": dates[j] if exit_px is not None else None,
            "exit": round(exit_px, 4) if exit_px is not None else None,
            "reason": reason, "bars": j - i,
            "ret_pct": round((exit_px / entry - 1) * 100, 3) if exit_px is not None else None,
            "path": s["path"], "score": s["score"], "conviction": s["conviction"],
            "stop_frac": s["stop_frac"],
        })
    return trades


# ─────────────────────────────────────────────────────────────
# 统计
# ─────────────────────────────────────────────────────────────

def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round((centre - half) * 100, 1), round((centre + half) * 100, 1))


def trade_stats(df: pd.DataFrame) -> dict:
    done = df[df["ret_pct"].notna()]
    n = len(done)
    if n == 0:
        return {"n": 0}
    wins = int((done["ret_pct"] > 0).sum())
    r = done["ret_pct"]
    avg_win = float(r[r > 0].mean()) if wins else 0.0
    avg_loss = float(r[r <= 0].mean()) if wins < n else 0.0
    return {
        "n": n,
        "win_rate": round(wins / n * 100, 1),
        "win_ci": wilson_ci(wins, n),
        "avg_ret": round(float(r.mean()), 2),
        "avg_ret_se": round(float(r.std(ddof=1) / math.sqrt(n)), 2) if n > 1 else None,
        "median_ret": round(float(r.median()), 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "payoff": round(avg_win / abs(avg_loss), 2) if avg_loss else None,
        "avg_bars": round(float(done["bars"].mean()), 1),
    }


def baseline_forward_returns(prices: dict, tickers: list, bars: int, start: int = 252) -> dict:
    """对照组：同一股票池里"任意一天收盘买入、持有bars根K线"的平均收益。"""
    rets = []
    for t in tickers:
        c = prices[t]["Close"].to_numpy(dtype=float)
        if len(c) > start + bars:
            rets.append(c[start + bars:] / c[start:-bars] - 1)
    r = np.concatenate(rets) * 100 if rets else np.array([])
    return {"n": len(r), "avg_ret": round(float(r.mean()), 2) if len(r) else None,
            "win_rate": round(float((r > 0).mean() * 100), 1) if len(r) else None}


def random_control_trades(prices: dict, trades: pd.DataFrame, seed: int = 0,
                          start: int = 252) -> pd.DataFrame:
    """
    对照组：每只股票随机挑与真实信号同样多的入场日，止损宽度用当天的1.5ATR，
    出场规则完全相同。用来回答"GO信号比随便哪天买入好在哪里"——单看组合
    总收益会把股票池本身的涨幅（幸存者偏差、高beta）误当成信号的功劳。
    """
    rng = np.random.default_rng(seed)
    out = []
    for t, k in trades["ticker"].value_counts().items():
        hist = prices[t]
        c, h, l = hist["Close"], hist["High"], hist["Low"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        stop_frac = (tr.ewm(com=13, adjust=False).mean() * ATR_STOP_MULT / c).clip(upper=MAX_STOP_PCT_AGG / 100)
        days = rng.choice(np.arange(start, len(c) - 1), size=min(k * 3, len(c) - 1 - start), replace=False)
        sig = pd.DataFrame({"go": False, "path": "R", "score": None, "conviction": None,
                            "stop_frac": stop_frac}, index=c.index)
        sig.iloc[np.sort(days), sig.columns.get_loc("go")] = True
        # 抽3倍候选日：同一标的持仓期间的信号会被跳过，保证最后能凑够k笔
        out += simulate_trades(hist, sig, t)[:k]
    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────
# 组合层（vectorbt）
# ─────────────────────────────────────────────────────────────

def run_portfolio(prices: dict, trades: pd.DataFrame, init_cash: float = 2000.0):
    """
    用vectorbt按账户规则模拟资金曲线：共享现金、单仓按"3%风险÷止损宽度"定仓位
    且不超过50%，现金不够时新信号自动放弃（近似MAX_CONCURRENT_POSITIONS=2和
    80%总仓位上限）。出场价/出场日直接用simulate_trades的结果。
    """
    import vectorbt as vbt

    done = trades[trades["exit_date"].notna()]
    tickers = sorted(done["ticker"].unique())
    close = pd.DataFrame({t: prices[t]["Close"] for t in tickers}).ffill()
    price = close.copy()
    size = pd.DataFrame(np.nan, index=close.index, columns=tickers)
    for _, tr in done.iterrows():
        # 入场：目标仓位=账户总值×比例；出场：目标仓位0，按simulate_trades的出场价成交
        size.at[tr["entry_date"], tr["ticker"]] = min(0.03 / tr["stop_frac"], 0.5)
        size.at[tr["exit_date"], tr["ticker"]] = 0.0
        price.at[tr["exit_date"], tr["ticker"]] = tr["exit"]

    # allow_partial=False：现金不够买满目标仓位时整笔放弃（与模拟盘"超上限拒绝开仓"一致）
    return vbt.Portfolio.from_orders(
        close, size, size_type="targetpercent", price=price, init_cash=init_cash,
        cash_sharing=True, group_by=True, call_seq="auto", allow_partial=False, freq="1D",
    )


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────

def run_backtest(tickers: list | None = None, years: int = 5,
                 take_profit_atr: float | None = None, with_earnings: bool = True) -> dict:
    tickers = tickers or default_universe()
    prices = load_prices(tickers, years)
    spy, vix = prices["SPY"]["Close"], prices["^VIX"]["Close"]
    earnings = load_earnings_dates(tickers) if with_earnings else {}

    all_trades, go_days, cand_days = [], 0, 0
    for t in tickers:
        if t not in prices:
            continue
        sig = compute_signals(prices[t], spy, vix, earnings.get(t))
        go_days += int(sig["go"].sum())
        cand_days += len(sig)
        all_trades += simulate_trades(prices[t], sig, t, take_profit_atr)

    trades = pd.DataFrame(all_trades)
    return {"prices": prices, "trades": trades, "go_days": go_days, "days": cand_days,
            "tickers": [t for t in tickers if t in prices], "years": years,
            "take_profit_atr": take_profit_atr,
            "earnings_covered": sum(1 for t in tickers if earnings.get(t))}


def report(res: dict, with_portfolio: bool = True, n_control: int = 20) -> str:
    tr = res["trades"]
    tp = res["take_profit_atr"]
    exit_rule = f"止损+时间止损+目标{tp}ATR" if tp else "止损+时间止损"
    lines = [f"== 九关模型技术面近似回测：{len(res['tickers'])}只股票，{res['years']}年，出场={exit_rule} ==",
             f"GO信号日 {res['go_days']} / {res['days']} 个股票日（{res['go_days'] / max(res['days'], 1) * 100:.1f}%），"
             f"财报日覆盖 {res['earnings_covered']} 只"]
    if tr.empty:
        return "\n".join(lines + ["无交易"])

    def fmt(name, s):
        if not s.get("n"):
            return f"{name:14s} n=0"
        return (f"{name:14s} n={s['n']:4d}  胜率{s['win_rate']:5.1f}% (95%CI {s['win_ci'][0]}-{s['win_ci'][1]})  "
                f"平均{s['avg_ret']:+.2f}%(±{(s['avg_ret_se'] or 0) * 1.96:.2f})  中位{s['median_ret']:+.2f}%  "
                f"盈亏比{s['payoff']}  平均持有{s['avg_bars']}根")

    lines.append(fmt("全部交易", trade_stats(tr)))
    for p, name in (("A", "路径A突破区"), ("B", "路径B回调企稳"), ("-", "其他位置")):
        lines.append(fmt(name, trade_stats(tr[tr["path"] == p])))
    for r in sorted(tr["reason"].unique()):
        lines.append(fmt(f"出场:{r}", trade_stats(tr[tr["reason"] == r])))
    tr_y = tr.assign(year=pd.to_datetime(tr["entry_date"]).dt.year)
    for y in sorted(tr_y["year"].unique()):
        lines.append(fmt(f"{y}年", trade_stats(tr_y[tr_y["year"] == y])))

    ctrl = random_control_trades(res["prices"], tr)
    lines.append(fmt("对照:随机入场", trade_stats(ctrl)))
    avg_bars = int(round(trade_stats(tr)["avg_bars"]))
    base = baseline_forward_returns(res["prices"], res["tickers"], max(avg_bars, 1))
    lines.append(f"对照组：任意一天买入持有{avg_bars}根K线  n={base['n']}  "
                 f"胜率{base['win_rate']}%  平均{base['avg_ret']:+.2f}%")

    if with_portfolio:
        pf = run_portfolio(res["prices"], tr)
        st = pf.stats()
        spy = res["prices"]["SPY"]["Close"]
        spy_ret = (spy.iloc[-1] / spy.loc[pf.wrapper.index[0]:].iloc[0] - 1) * 100
        lines.append(f"组合($2000起，共享现金，单仓≤50%)：总收益{st['Total Return [%]']:.1f}%  "
                     f"最大回撤{st['Max Drawdown [%]']:.1f}%  Sharpe {st['Sharpe Ratio']:.2f}  "
                     f"成交{st['Total Trades']}笔  同期SPY {spy_ret:.1f}%")
        # 组合收益路径依赖很强（3000多个信号里只成交约300笔），单个对照不够，
        # 跑n_control组随机入场看真实结果排在分布的什么位置
        ctrl_ret, ctrl_sharpe = [], []
        for seed in range(n_control):
            cst = run_portfolio(res["prices"], random_control_trades(res["prices"], tr, seed)).stats()
            ctrl_ret.append(cst["Total Return [%]"])
            ctrl_sharpe.append(cst["Sharpe Ratio"])
        ctrl_ret, ctrl_sharpe = np.array(ctrl_ret), np.array(ctrl_sharpe)
        p10, p50, p90 = np.percentile(ctrl_ret, [10, 50, 90])
        lines.append(f"  随机入场对照组合×{n_control}：总收益 p10 {p10:.0f}% / 中位 {p50:.0f}% / p90 {p90:.0f}%；"
                     f"真实信号收益高于{(ctrl_ret < st['Total Return [%]']).mean() * 100:.0f}%的对照、"
                     f"Sharpe高于{(ctrl_sharpe < st['Sharpe Ratio']).mean() * 100:.0f}%的对照"
                     f"（≥95%才算显著）")
        res["portfolio"] = pf
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="九关模型技术面近似回测")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--tp", type=float, default=None, help="可选：目标价ATR倍数（模拟盘默认不自动止盈）")
    ap.add_argument("--tickers", default=None, help="逗号分隔，默认watchlist+板块代表股")
    ap.add_argument("--no-earnings", action="store_true")
    ap.add_argument("--save", action="store_true", help="交易明细存到data/backtest_trades.csv")
    a = ap.parse_args()
    res = run_backtest(a.tickers.split(",") if a.tickers else None, a.years, a.tp, not a.no_earnings)
    print(report(res))
    if a.save:
        res["trades"].to_csv(os.path.join(_DATA, "backtest_trades.csv"), index=False)
