import yfinance as yf
import pandas as pd


def get_stock_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    return yf.Ticker(ticker).history(period=period)


def get_stock_info(ticker: str) -> dict:
    info = yf.Ticker(ticker).info
    hist = get_stock_history(ticker, "5d")
    current = float(hist["Close"].iloc[-1]) if not hist.empty else None
    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "—"),
        "industry": info.get("industry", "—"),
        "market_cap": info.get("marketCap", 0),
        "pe_ratio": info.get("trailingPE"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "current_price": current,
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
