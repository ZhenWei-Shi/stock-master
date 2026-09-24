import yfinance as yf
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, nearest_workday, GoodFriday,
    USMartinLutherKingJr, USPresidentsDay, USMemorialDay, USLaborDay,
    USThanksgivingDay,
)


# Yahoo日线偶尔整天漏聚合（2026-09-22全市场缺失，同日1小时线完整），
# 只在最近这段窗口内检查缺口，缺了才多发一次小时线请求
GAP_CHECK_DAYS = 30


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """NYSE全天休市日（不含临时休市），用来排除缺口误报。"""
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def fill_daily_gaps(daily: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """用小时线聚合补日线缺失的交易日。

    只补"小时线有数据、日线没有"且不早于日线首日的日期；节假日小时线本身
    为空，不会被误补。补上的日期记在 attrs["filled_dates"]。
    """
    if daily.empty or hourly.empty:
        return daily
    tz = daily.index.tz
    have = {ts.date() for ts in daily.index}
    first = daily.index[0].date()
    rows = {}
    for day, g in hourly.groupby(hourly.index.date):
        if day in have or day < first:
            continue
        row = {c: 0.0 for c in daily.columns}   # Dividends/Stock Splits等补0
        row.update(Open=g["Open"].iloc[0], High=g["High"].max(),
                   Low=g["Low"].min(), Close=g["Close"].iloc[-1],
                   Volume=g["Volume"].sum())
        rows[pd.Timestamp(day).tz_localize(tz)] = row
    if not rows:
        return daily
    filled = pd.DataFrame.from_dict(rows, orient="index")[daily.columns]
    out = pd.concat([daily, filled]).sort_index()
    out.attrs["filled_dates"] = sorted(str(d.date()) for d in rows)
    return out


def _has_weekday_gap(daily: pd.DataFrame) -> bool:
    """最近GAP_CHECK_DAYS天内是否有NYSE交易日不在日线里。"""
    if daily.empty:
        return False
    last = daily.index[-1]
    start = max(last - pd.Timedelta(days=GAP_CHECK_DAYS), daily.index[0])
    holidays = NYSEHolidayCalendar().holidays(start.tz_localize(None).normalize(),
                                             last.tz_localize(None).normalize())
    recent = pd.bdate_range(start.tz_localize(None).normalize(),
                            last.tz_localize(None).normalize(),
                            freq="C", holidays=holidays)
    have = {ts.date() for ts in daily.index}
    return any(d.date() not in have for d in recent)


def daily_history(tk, period: str = "1y") -> pd.DataFrame:
    """日线 + 小时线兜底。tk 为 yf.Ticker；兜底失败时原样返回日线。"""
    daily = tk.history(period=period, interval="1d")
    if not _has_weekday_gap(daily):
        return daily
    try:
        hourly = tk.history(period="1mo", interval="1h")
        return fill_daily_gaps(daily, hourly)
    except Exception:
        return daily


def get_stock_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    return daily_history(yf.Ticker(ticker), period)


def get_stock_info(ticker: str) -> dict:
    info = yf.Ticker(ticker).info
    hist = get_stock_history(ticker, "5d")
    current = float(hist["Close"].iloc[-1]) if not hist.empty else None  # 昨日收盘价，非盘中实时
    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "—"),
        "industry": info.get("industry", "—"),
        "market_cap": info.get("marketCap", 0),
        "pe_ratio": info.get("trailingPE"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "current_price": current,  # 字段名沿用旧名；实为上一交易日收盘，非实时
        "institutional_ownership": info.get("institutionPercentHeld"),
        "short_ratio": info.get("shortRatio"),
        "avg_volume": info.get("averageVolume"),
    }


def get_options_chain(ticker: str, expiry: str = None):
    t = yf.Ticker(ticker)
    expirations = list(t.options) if t.options else []
    if not expirations:
        return None, None, None, []

    target = expiry if (expiry and expiry in expirations) else expirations[0]
    chain = t.option_chain(target)
    return chain.calls, chain.puts, target, expirations


def get_market_snapshot() -> dict:
    spy_hist = yf.Ticker("SPY").history(period="3mo")
    qqq_hist = yf.Ticker("QQQ").history(period="3mo")
    vix_hist = yf.Ticker("^VIX").history(period="5d")

    vix = float(vix_hist["Close"].iloc[-1]) if not vix_hist.empty else None
    spy_price = float(spy_hist["Close"].iloc[-1]) if not spy_hist.empty else None
    qqq_price = float(qqq_hist["Close"].iloc[-1]) if not qqq_hist.empty else None

    return {
        "spy_hist": spy_hist,
        "qqq_hist": qqq_hist,
        "vix": vix,
        "spy_price": spy_price,
        "qqq_price": qqq_price,
    }
