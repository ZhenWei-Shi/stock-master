# 四个固定日频因子（长表输出）。纯函数：不修改输入，不做前向填充。
# 注意：日期 t 的因子值使用 t 日收盘数据，回测中必须至少滞后一个交易日再使用。
from __future__ import annotations

import numpy as np
import pandas as pd

from .cache import validate_panel

FACTOR_NAMES = ('momentum_12_1', 'reversal_5', 'signed_rvol_20', 'low_vol_20')
OUTPUT_COLUMNS = ('date', 'symbol', 'factor', 'value')
BENCHMARK_SYMBOL = 'SPY'
MOMENTUM_SKIP = 21
MOMENTUM_LOOKBACK = 252
REVERSAL_WINDOW = 5
VOLUME_WINDOW = 20
VOLATILITY_WINDOW = 20
ANNUALIZATION_DAYS = 252


def _select_symbols(available, symbols):
    if symbols is None:
        chosen = sorted(s for s in available if s != BENCHMARK_SYMBOL)
        if not chosen:
            raise ValueError('no symbols left after excluding %s' % BENCHMARK_SYMBOL)
        return chosen
    if isinstance(symbols, (str, bytes)) or not hasattr(symbols, '__iter__'):
        raise ValueError('symbols must be an iterable of tickers, not %r' % (symbols,))
    chosen = list(symbols)
    if not chosen:
        raise ValueError('symbols must be non-empty')
    if not all(isinstance(s, str) for s in chosen):
        raise ValueError('symbols must all be strings')
    duplicates = sorted({s for s in chosen if chosen.count(s) > 1})
    if duplicates:
        raise ValueError('duplicate symbols requested: %s' % duplicates)
    unknown = sorted(set(chosen) - set(available))
    if unknown:
        raise ValueError('symbols not present in panel: %s' % unknown)
    return sorted(chosen)


def _calendar(panel):
    # 优先使用 SPY 的交易日；无 SPY 时退回全部日期并集
    is_benchmark = panel['symbol'] == BENCHMARK_SYMBOL
    dates = panel.loc[is_benchmark, 'date'] if is_benchmark.any() else panel['date']
    return pd.DatetimeIndex(dates.unique()).sort_values()


def _symbol_factors(part, calendar):
    # 对齐到全局日历，缺失交易日保持 NaN，滚动窗口因此不会跨越缺口
    frame = part.set_index('date').reindex(calendar)
    adj = frame['adj_close']
    volume = frame['volume']
    ret = adj / adj.shift(1) - 1.0
    momentum = adj.shift(MOMENTUM_SKIP) / adj.shift(MOMENTUM_LOOKBACK) - 1.0
    reversal = -(adj / adj.shift(REVERSAL_WINDOW) - 1.0)
    # 分母只用前 20 个交易日（shift(1)），不含当日；分母为 0 记为 NaN
    prior_volume = volume.shift(1).rolling(VOLUME_WINDOW, min_periods=VOLUME_WINDOW).mean()
    prior_volume = prior_volume.where(prior_volume > 0)
    signed_rvol = np.sign(ret) * (volume / prior_volume - 1.0)
    # 当日或前 20 日有拆股时原始成交量不可比，置 NaN
    split_flag = (frame['split'].fillna(0.0) > 0).astype('float64')
    embargo = split_flag.rolling(VOLUME_WINDOW + 1, min_periods=1).max() > 0
    signed_rvol = signed_rvol.mask(embargo)
    low_vol = -(ret.rolling(VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW).std(ddof=1)
                * np.sqrt(ANNUALIZATION_DAYS))
    out = pd.DataFrame({'momentum_12_1': momentum, 'reversal_5': reversal,
                        'signed_rvol_20': signed_rvol, 'low_vol_20': low_vol}, index=calendar)
    out = out.where(adj.notna(), axis=0)
    out = out.rename_axis('date')
    return out.loc[part['date'].min():part['date'].max()]


def compute_factors(panel, symbols=None):
    '''返回长表 date,symbol,factor,value；每个所选标的在其首末观测之间的每个日历日各 4 行。'''
    clean = validate_panel(panel)
    chosen = _select_symbols(set(clean['symbol']), symbols)
    calendar = _calendar(clean)
    frames = []
    for symbol in chosen:
        part = clean.loc[clean['symbol'] == symbol, ['date', 'adj_close', 'volume', 'split']]
        outside = ~part['date'].isin(calendar)
        if outside.any():
            raise ValueError('%s has %d session(s) outside the %s calendar, first %s'
                             % (symbol, int(outside.sum()), BENCHMARK_SYMBOL,
                                part.loc[outside, 'date'].iloc[0].date()))
        wide = _symbol_factors(part, calendar).reset_index()
        wide.insert(1, 'symbol', symbol)
        frames.append(wide)
    wide = pd.concat(frames, ignore_index=True)
    long = wide.melt(id_vars=['date', 'symbol'], value_vars=list(FACTOR_NAMES),
                     var_name='factor', value_name='value')
    rank = long['factor'].map({name: i for i, name in enumerate(FACTOR_NAMES)})
    long = (long.assign(_rank=rank)
            .sort_values(['date', 'symbol', '_rank'], kind='mergesort')
            .drop(columns='_rank')
            .reset_index(drop=True))
    long['value'] = long['value'].astype('float64')
    return long.loc[:, list(OUTPUT_COLUMNS)]
