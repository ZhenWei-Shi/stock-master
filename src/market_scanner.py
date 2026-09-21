"""
异动股票扫描（Market Scanner）

2026-09-21新增，直接解决的问题：build_dynamic_watchlist（sector_rotation.py）
只有在个股自己所属板块整体排进热门前3后，才会把它拉进扫描列表——而"板块
整体排名"要等成分股价格已经涨出规模才会跟着反映。META案例：META自己先大
涨，通信服务板块(XLC)才被带到第3名，中间隔了数小时（查服务器日志：板块
缓存13:29才显示XLC进前3，META当天的消息大概率盘前/早盘就出来了）。

这个模块反过来做：不等板块排名，直接对SECTOR_TICKERS里的候选股（~84只，
覆盖14个板块的高流动性代表股）批量检查"今天自己涨跌/放量是否异常"，独立
于所在板块排名，堵住这个"等板块追上"的滞后窗口。

范围说明：不是全市场/标普500扫描，是复用sector_rotation.py已有的精选
大盘股清单做的低成本扩展，不引入新的数据源或清单维护负担。
"""
from __future__ import annotations

import yfinance as yf
import pandas as pd

from .sector_rotation import SECTOR_TICKERS

MOVER_PCT_THRESHOLD = 5.0   # 单日涨跌幅超过±5%
MOVER_VOLUME_RATIO  = 2.0   # 成交量超过20日均量的2倍


def universe() -> list[str]:
    """SECTOR_TICKERS里全部候选股，保序去重（同一股票可能出现在多个板块，如NVDA在XLK和SMH都有）。"""
    seen: set[str] = set()
    out: list[str] = []
    for tickers in SECTOR_TICKERS.values():
        for t in tickers:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def is_mover(pct_change: float, volume_ratio: float,
             pct_threshold: float = MOVER_PCT_THRESHOLD,
             volume_ratio_threshold: float = MOVER_VOLUME_RATIO) -> bool:
    """纯逻辑：今天涨跌幅或量比是否达到异动标准。拆出来单独测试，不依赖网络。"""
    return abs(pct_change) >= pct_threshold or volume_ratio >= volume_ratio_threshold


def find_movers(top_n: int = 10) -> list[dict]:
    """
    对 universe() 做一次批量下载，独立于板块排名，直接筛出"今天自己涨跌
    或放量异常"的个股。联网请求编排层，不做单元测试（同run_scan()，见
    tests/test_trading_agent.py顶部说明）。

    返回 [{ticker, pct_change, volume_ratio}, ...]，按|pct_change|降序，
    失败或数据不足时返回空列表，不阻断调用方（build_dynamic_watchlist）
    的其余流程。
    """
    tickers = universe()
    try:
        raw = yf.download(" ".join(tickers), period="30d", interval="1d",
                           group_by="ticker", auto_adjust=True,
                           progress=False, threads=True)
    except Exception:
        return []

    movers = []
    for tk in tickers:
        try:
            if not isinstance(raw.columns, pd.MultiIndex) or tk not in raw.columns.get_level_values(0):
                continue
            df = raw[tk].dropna()
            if len(df) < 21:
                continue
            close, vol = df["Close"], df["Volume"]
            pct_change = round((float(close.iloc[-1]) / float(close.iloc[-2]) - 1) * 100, 2)
            avg_vol_20 = float(vol.iloc[-21:-1].mean())
            volume_ratio = round(float(vol.iloc[-1]) / avg_vol_20, 2) if avg_vol_20 > 0 else 0.0
            if is_mover(pct_change, volume_ratio):
                movers.append({"ticker": tk, "pct_change": pct_change, "volume_ratio": volume_ratio})
        except Exception:
            continue

    movers.sort(key=lambda m: abs(m["pct_change"]), reverse=True)
    return movers[:top_n]
