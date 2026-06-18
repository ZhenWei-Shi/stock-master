"""
策略回测引擎

引用项目：
  backtesting.py ⭐5,000+ (kernc/backtesting.py)  — 核心回测框架
  quantstats     ⭐6,500  (ranaroussi/quantstats)  — 组合绩效分析

策略：VWAP动量（MA20 代理 VWAP）+ RSI 过滤 + ATR 止盈止损
quantstats 新增：Sortino、Calmar、Recovery Factor、Tail Ratio
"""
import yfinance as yf
import pandas as pd
import numpy as np

try:
    from backtesting import Backtest, Strategy
    BACKTEST_OK = True
except ImportError:
    BACKTEST_OK = False

try:
    import quantstats as qs
    QS_OK = True
except ImportError:
    QS_OK = False


def _rsi(close_arr, period=14):
    s = pd.Series(close_arr.astype(float))
    d = s.diff()
    g = d.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    l = (-d.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    r = g / l.replace(0, 1e-9)
    out = (100 - 100 / (1 + r)).fillna(50)
    return out.values


def _atr(high_arr, low_arr, close_arr, period=14):
    hi = pd.Series(high_arr.astype(float))
    lo = pd.Series(low_arr.astype(float))
    cl = pd.Series(close_arr.astype(float))
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean().fillna(method="bfill").values


def _ma(close_arr, n=20):
    s = pd.Series(close_arr.astype(float))
    return s.rolling(n, min_periods=1).mean().values


if BACKTEST_OK:
    class VWAPMomentumStrategy(Strategy):
        """
        VWAP动量策略
        入场：价格 > MA20（代理VWAP） + RSI > 52
        出场：价格跌破 MA20 或 RSI < 44 或 ATR止损/止盈触发
        """
        n_ma      = 20
        rsi_entry = 52
        rsi_exit  = 44
        atr_stop  = 1.5
        atr_tp    = 2.5

        def init(self):
            self.ma  = self.I(_ma,  self.data.Close, self.n_ma)
            self.rsi = self.I(_rsi, self.data.Close, 14)
            self.atr_v = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)

        def next(self):
            price = self.data.Close[-1]
            above = price > self.ma[-1]
            rsi   = self.rsi[-1]
            atr   = self.atr_v[-1]

            if above and rsi > self.rsi_entry and not self.position:
                sl = price - atr * self.atr_stop
                tp = price + atr * self.atr_tp
                self.buy(sl=max(sl, 0.01), tp=tp)

            elif self.position.is_long:
                if not above or rsi < self.rsi_exit:
                    self.position.close()


def run_backtest(ticker: str, period: str = "1y") -> dict:
    if not BACKTEST_OK:
        return {"error": "backtesting 库未安装"}

    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty or len(hist) < 60:
            return {"error": f"历史数据不足（仅 {len(hist)} 天）"}

        data = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
        data.index = pd.DatetimeIndex(data.index).tz_localize(None)
        data = data[data["Volume"] > 0].dropna()

        bt    = Backtest(data, VWAPMomentumStrategy, cash=100_000, commission=0.001)
        stats = bt.run()

        def safe(key, default=0):
            v = stats.get(key, default)
            return 0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v

        base = {
            "ticker":              ticker,
            "period":              period,
            "bars":                len(data),
            "total_return_pct":    round(float(safe("Return [%]")), 2),
            "buy_hold_return_pct": round(float(safe("Buy & Hold Return [%]")), 2),
            "win_rate_pct":        round(float(safe("Win Rate [%]")), 1),
            "num_trades":          int(safe("# Trades")),
            "max_drawdown_pct":    round(float(safe("Max. Drawdown [%]")), 2),
            "sharpe":              round(float(safe("Sharpe Ratio")), 2),
            "profit_factor":       round(float(safe("Profit Factor")), 2),
            "avg_trade_pct":       round(float(safe("Avg. Trade [%]")), 2),
            "expectancy_pct":      round(float(safe("Expectancy [%]")), 2),
        }

        # quantstats ⭐6,500 增强分析
        qs_metrics = {}
        if QS_OK:
            try:
                eq  = stats._equity_curve["Equity"]
                ret = eq.pct_change().dropna()
                qs_metrics = {
                    "sortino":          round(float(qs.stats.sortino(ret)),         2),
                    "calmar":           round(float(qs.stats.calmar(ret)),          2),
                    "max_drawdown_pct": round(float(qs.stats.max_drawdown(ret))*100,2),
                    "volatility_ann":   round(float(qs.stats.volatility(ret))*100,  2),
                    "avg_win_pct":      round(float(qs.stats.avg_win(ret))*100,      2),
                    "avg_loss_pct":     round(float(qs.stats.avg_loss(ret))*100,     2),
                    "best_day_pct":     round(float(qs.stats.best(ret))*100,         2),
                    "worst_day_pct":    round(float(qs.stats.worst(ret))*100,        2),
                    "recovery_factor":  round(float(qs.stats.recovery_factor(ret)), 2),
                    "payoff_ratio":     round(float(qs.stats.payoff_ratio(ret)),     2),
                    "tail_ratio":       round(float(qs.stats.tail_ratio(ret)),       2),
                    "library": "quantstats ⭐6,500 (ranaroussi/quantstats)",
                }
            except Exception as qe:
                qs_metrics = {"error": str(qe)}

        return {**base, "quantstats": qs_metrics}

    except Exception as e:
        return {"error": str(e)}
