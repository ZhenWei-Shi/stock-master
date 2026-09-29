"""
Alpaca接入（2026-09-29新增）：备用行情源 + 期权模拟账户的客户端工厂

没配置密钥时本模块什么都不做（is_configured()为False，各函数返回None），
不影响现有流程。配置方法：在启动scheduler的shell里export
  ALPACA_API_KEY=...  ALPACA_SECRET_KEY=...
（与Telegram token同样的做法，见wiki stock-master/overview"部署环境"）。
只使用Alpaca的paper（模拟）账户，代码里写死paper=True。

行情口径：免费档可以拿延迟15分钟以上的全市场SIP历史数据；实时只有IEX一家
交易所（成交量约占全市场2-3%），拿IEX补日线会让量比严重偏低、误杀volume gate，
所以这里只用SIP，拿不到就返回None，由调用方退回yfinance小时线兜底。

alpaca-py在函数内按需import，scheduler平时不加载它。
"""
import os

import pandas as pd
import pytz

ET = pytz.timezone("America/New_York")


def is_configured() -> bool:
    return bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))


def _keys() -> tuple:
    return os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]


def stock_data_client():
    if not is_configured():
        return None
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(*_keys())


def option_data_client():
    if not is_configured():
        return None
    from alpaca.data.historical import OptionHistoricalDataClient
    return OptionHistoricalDataClient(*_keys())


def paper_trading_client():
    """只返回模拟账户客户端（paper=True写死，不提供实盘入口）。"""
    if not is_configured():
        return None
    from alpaca.trading.client import TradingClient
    return TradingClient(*_keys(), paper=True)


def bars_to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    alpaca-py的bars.df（MultiIndex[symbol, timestamp(UTC)]，小写列名）→
    与yfinance一致的格式：ET时区的日期索引，Open/High/Low/Close/Volume列（纯函数）。
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)
    out = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})[["Open", "High", "Low", "Close", "Volume"]]
    idx = pd.DatetimeIndex(out.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    out.index = idx.tz_convert(ET).normalize()
    return out.astype(float)


def daily_bars(symbol: str, start, end=None):
    """SIP日线（已复权）。未配置/取数失败返回None。"""
    client = stock_data_client()
    if client is None:
        return None
    try:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start, end=end,
                               feed=DataFeed.SIP, adjustment=Adjustment.ALL)
        return bars_to_ohlcv(client.get_stock_bars(req).df)
    except Exception as e:
        print(f"[Alpaca] {symbol}日线取数失败：{e}")
        return None
