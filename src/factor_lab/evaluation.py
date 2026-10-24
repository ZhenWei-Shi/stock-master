'''日频横截面因子研究评估（仅探索性筛选）。

显式金融假设：
1. 信号在交易日 t 收盘后可得；入场为 t+1 调整开盘价，出场为 t+1+horizon 调整开盘价
   （horizon=5 即 t+6 开盘）。调整开盘价 = open * adj_close / close（Yahoo 复权因子，
   已含拆股与分红调整；dividend/split 列只做校验，不重复调整）。
2. 交易日历 = 裁剪后面板全部 date 的有序去重集合（含 SPY）。某标的在入场到出场的任一
   日历交易日缺失有效开盘价，则该日整个比较不入样本，不利用未来缺数重新选股。
   信号日缺收盘价的股票当时不可用；不跨缺口平移、不填充。
3. 计算任何标签前，先把面板裁剪到 date <= end，end 之后保留的未来价格不能提供边界标签。
4. cohort 为固定 horizon 交易日的等权买入持有：组合收益 = 成员收益算术平均，
   不是每日自融资再平衡净值。相邻信号日持有期重叠，日度序列自相关，使用 HAC。
5. 成本为每个 cohort 固定往返 cost_bps（top、等权基准、随机组同等扣除）；压力情景
   stress_cost_bps 在完全相同样本上单独报告；不是基于换手的真实成交模型。
6. 股票池为当前存续标的，无 PIT 成分：结论一律 exploratory；t>2 不构成盈利结论。
'''
from __future__ import annotations

import hashlib
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import hac_mean, holm_adjust

PANEL_COLUMNS = ('date', 'symbol', 'open', 'high', 'low', 'close', 'adj_close',
                 'volume', 'dividend', 'split')
FACTOR_COLUMNS = ('date', 'symbol', 'factor', 'value')
PRICE_COLUMNS = ('open', 'high', 'low', 'close', 'adj_close')
MIN_NAMES = 10
SCREENING_STATUS = 'exploratory'
CAVEATS = (
    '股票池为当前数据中的存续标的，无时点(PIT)成分：存在幸存者偏差，结论仅为探索性筛选。',
    '相邻信号日的持有期重叠（horizon 个交易日），日度序列自相关；t 使用 Bartlett HAC，不是独立样本。',
    'cohort 为固定期限等权买入持有，不是每日自融资再平衡净值，不可直接拼接为资金曲线。',
    '成本为固定往返基点，对 top、等权基准和随机组同等扣除；不是基于换手和真实成交的成本模型。',
    't>2 或 Holm 调整后显著均不构成经济盈利结论；不得据此在测试期挑选因子。',
    '仅按信号时点数据排名；未来缺数使整个比较日缺失并单独计数，不能据此替换成员。',
)


def _num(x) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _mean(values) -> Optional[float]:
    arr = np.asarray(values, dtype=float)
    return _num(arr.mean()) if arr.size else None


def _check_int(value, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return int(value)


def _check_bps(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be numeric basis points')
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric basis points') from exc
    if not math.isfinite(v) or v < 0.0 or v >= 10000.0:
        raise ValueError(f'{name} must be finite and in [0, 10000) bps')
    return v


def _check_timestamp(value, name: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} is not a valid date') from exc
    if pd.isna(ts):
        raise ValueError(f'{name} is missing')
    if ts.tzinfo is not None:
        raise ValueError(f'{name} must be timezone-naive')
    if ts != ts.normalize():
        raise ValueError(f'{name} must be a calendar date without time')
    return ts


def _validate_date_column(series: pd.Series, name: str) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        raise ValueError(f'{name} must be timezone-naive')
    if not pd.api.types.is_datetime64_dtype(series.dtype):
        raise ValueError(f'{name} must be datetime64')
    if series.isna().any():
        raise ValueError(f'{name} contains NaT')
    if not (series == series.dt.normalize()).all():
        raise ValueError(f'{name} must be daily dates without time component')
    return series.astype('datetime64[ns]')


def _validate_labels(series: pd.Series, name: str) -> None:
    if not series.map(lambda v: isinstance(v, str) and v != '').all():
        raise ValueError(f'{name} must contain non-empty strings')


def _numeric(series: pd.Series, name: str) -> pd.Series:
    try:
        return pd.to_numeric(series, errors='raise').astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc


def _validate_panel(panel) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame):
        raise ValueError('panel must be a pandas DataFrame')
    missing = [c for c in PANEL_COLUMNS if c not in panel.columns]
    if missing:
        raise ValueError(f'panel missing columns: {missing}')
    df = panel.loc[:, list(PANEL_COLUMNS)].copy()
    df['date'] = _validate_date_column(df['date'], 'panel.date')
    _validate_labels(df['symbol'], 'panel.symbol')
    for col in PANEL_COLUMNS[2:]:
        df[col] = _numeric(df[col], f'panel.{col}')
    if np.isinf(df[list(PANEL_COLUMNS[2:])].to_numpy(dtype=float)).any():
        raise ValueError('panel contains infinite numbers')
    # NaN 表示缺失值（该标的该日不可用）；非正价格属于数据错误
    if (df[list(PRICE_COLUMNS)] <= 0).any().any():
        raise ValueError('panel prices must be positive (NaN marks a missing value)')
    if (df[['volume', 'dividend', 'split']] < 0).any().any():
        raise ValueError('panel volume/dividend/split must be non-negative')
    if df.duplicated(['date', 'symbol']).any():
        raise ValueError('panel has duplicate (date, symbol) rows')
    return df


def _validate_factors(factor_values, factor_names) -> Tuple[pd.DataFrame, list]:
    if not isinstance(factor_values, pd.DataFrame):
        raise ValueError('factor_values must be a pandas DataFrame')
    missing = [c for c in FACTOR_COLUMNS if c not in factor_values.columns]
    if missing:
        raise ValueError(f'factor_values missing columns: {missing}')
    df = factor_values.loc[:, list(FACTOR_COLUMNS)].copy()
    df['date'] = _validate_date_column(df['date'], 'factor_values.date')
    _validate_labels(df['symbol'], 'factor_values.symbol')
    _validate_labels(df['factor'], 'factor_values.factor')
    df['value'] = _numeric(df['value'], 'factor_values.value')
    if np.isinf(df['value'].to_numpy(dtype=float)).any():
        raise ValueError('factor_values contains infinite values')
    if df.duplicated(['date', 'symbol', 'factor']).any():
        raise ValueError('factor_values has duplicate (date, symbol, factor) rows')
    if factor_names is None:
        names = sorted(df['factor'].unique().tolist())
    else:
        if isinstance(factor_names, str):
            raise ValueError('factor_names must be a sequence of names')
        names = list(factor_names)
        if not all(isinstance(n, str) and n for n in names) or len(set(names)) != len(names):
            raise ValueError('factor_names must be unique non-empty strings')
        extra = sorted(set(df['factor'].unique()) - set(names))
        if extra:
            raise ValueError(f'factor_values contains unregistered factors: {extra}')
    if not names:
        raise ValueError('no prespecified factor candidates')
    return df, names


def _build_labels(pdf: pd.DataFrame, cal: pd.DatetimeIndex, symbols: Sequence[str],
                  horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    '''返回 (ret, ok)：ret[i, s] 为信号日 cal[i] 的固定期限调整开盘价收益。

    入场 cal[i+1]、出场 cal[i+1+horizon]（按日历行号，不按标的自身行号）；要求入场到出场
    每个日历交易日该标的都有有效调整开盘价，且信号日有收盘价。
    '''
    T, S = len(cal), len(symbols)
    ret = np.full((T, S), np.nan)
    ok = np.zeros((T, S), dtype=bool)
    m = T - horizon - 1
    if m <= 0 or S == 0:
        return ret, ok
    wide = pdf.set_index(['date', 'symbol'])[['open', 'close', 'adj_close']].unstack('symbol')

    def grid(col: str) -> np.ndarray:
        return wide[col].reindex(index=cal, columns=list(symbols)).to_numpy(dtype=float)

    op, cl, ac = grid('open'), grid('close'), grid('adj_close')
    with np.errstate(invalid='ignore', divide='ignore'):
        ao = op * ac / cl
    valid = np.isfinite(ao) & (ao > 0)
    has_close = np.isfinite(cl) & np.isfinite(ac)
    csum = np.vstack([np.zeros((1, S), dtype=np.int64), np.cumsum(valid, axis=0, dtype=np.int64)])
    # 窗口 [i+1, i+1+horizon] 内有效开盘价个数必须等于 horizon+1
    complete = (csum[horizon + 2:horizon + 2 + m] - csum[1:1 + m]) == horizon + 1
    with np.errstate(invalid='ignore', divide='ignore'):
        raw = ao[horizon + 1:horizon + 1 + m] / ao[1:1 + m] - 1.0
    good = complete & has_close[:m] & np.isfinite(raw)
    ok[:m] = good
    ret[:m] = np.where(good, raw, np.nan)
    return ret, ok


def _random_members(seed: int, date: pd.Timestamp, pool: Sequence[str], k: int) -> np.ndarray:
    '''匹配随机组：以 SHA-256(seed, 日期, 有序可选池) 为种子，稳定可复现；
    两个因子仅当同日可选池完全相同时才共享同一随机组。'''
    payload = '|'.join([str(seed), date.strftime('%Y-%m-%d'), chr(31).join(pool)])
    digest = hashlib.sha256(payload.encode('utf-8')).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))
    return np.sort(rng.choice(len(pool), size=k, replace=False))


def _row_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''逐日 Spearman：pandas rank（平均秩）后计算 Pearson 相关；常数截面返回 NaN。'''
    if x.shape[0] == 0:
        return np.empty(0)
    rx = pd.DataFrame(x).rank(axis=1, method='average').to_numpy(dtype=float)
    ry = pd.DataFrame(y).rank(axis=1, method='average').to_numpy(dtype=float)
    dx = rx - np.nanmean(rx, axis=1, keepdims=True)
    dy = ry - np.nanmean(ry, axis=1, keepdims=True)
    cov = np.nansum(dx * dy, axis=1)
    den = np.sqrt(np.nansum(dx * dx, axis=1) * np.nansum(dy * dy, axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 1e-12, cov / den, np.nan)


def _evaluate_factor(fv, ret, ok, signal_available, cal, symbols, *, n_quantiles, cost, stress, seed, lag) -> dict:
    # Rank using only signal-time availability, never future price coverage.
    elig = signal_available & np.isfinite(fv)
    counts = elig.sum(axis=1)
    candidate_rows = np.flatnonzero(counts >= MIN_NAMES)
    rows = np.array([i for i in candidate_rows if ok[i, elig[i]].all()], dtype=int)
    missing_outcome_dates = int(len(candidate_rows) - len(rows))
    skipped = int(np.count_nonzero((counts > 0) & (counts < MIN_NAMES)))
    missing_cells = int(np.count_nonzero(ok & np.isnan(fv)))
    nd = rows.size
    top, bottom, bench, rand = (np.empty(nd) for _ in range(4))
    for j, i in enumerate(rows):
        idx = np.flatnonzero(elig[i])
        k = math.ceil(idx.size / n_quantiles)
        v, r = fv[i, idx], ret[i, idx]
        # 值降序、代码升序（列已按代码排序）确定性打破并列
        top[j] = r[np.lexsort((idx, -v))[:k]].mean()
        bottom[j] = r[np.lexsort((idx, v))[:k]].mean()
        bench[j] = r.mean()
        rand[j] = r[_random_members(seed, cal[i], [symbols[c] for c in idx], k)].mean()
    ic = _row_spearman(np.where(elig[rows], fv[rows], np.nan),
                       np.where(elig[rows], ret[rows], np.nan))
    c, cs = cost / 1e4, stress / 1e4
    top_net, bench_net, rand_net = top - c, bench - c, rand - c
    ex_rand, ex_bench = top_net - rand_net, top_net - bench_net
    dates = cal[rows]
    hac = dict(hac_mean(pd.Series(ex_rand, index=dates), max_lag=lag))
    hac['max_lag'] = lag
    per_date = [
        {'date': d.strftime('%Y-%m-%d'), 'n': int(counts[i]), 'top_net': _num(top_net[j]),
         'benchmark_net': _num(bench_net[j]), 'random_net': _num(rand_net[j]),
         'excess_random': _num(ex_rand[j]), 'rank_ic': _num(ic[j])}
        for j, (i, d) in enumerate(zip(rows, dates))
    ]
    ic_ok = ic[np.isfinite(ic)]
    return {
        'screening_status': SCREENING_STATUS,
        'dates': int(nd),
        'first_date': per_date[0]['date'] if per_date else None,
        'last_date': per_date[-1]['date'] if per_date else None,
        'mean_net': _mean(top_net),
        'mean_benchmark_net': _mean(bench_net),
        'mean_random_net': _mean(rand_net),
        'mean_excess_random': _mean(ex_rand),
        'mean_excess_benchmark': _mean(ex_bench),
        'mean_net_stress': _mean(top - cs),
        'mean_benchmark_net_stress': _mean(bench - cs),
        'mean_random_net_stress': _mean(rand - cs),
        'mean_rank_ic': _mean(ic_ok),
        'rank_ic_dates': int(ic_ok.size),
        'top_bottom_gross': _mean(top - bottom),
        'positive_date_fraction': _mean(top_net > 0),
        'positive_excess_random_fraction': _mean(ex_rand > 0),
        'hac_excess_random': hac,
        'holm_p_excess_random': None,
        'skipped_dates_insufficient_names': skipped,
        'skipped_dates_missing_outcomes': missing_outcome_dates,
        'excluded_missing_factor_cells': missing_cells,
        'per_date': per_date,
    }


def evaluate_panel(panel, factor_values, *, start, end, horizon=5, cost_bps=10.0,
                   seed=20261022, n_quantiles=5, stress_cost_bps=20.0,
                   factor_names=None, hac_lag=None) -> dict:
    '''评估全部预设因子候选（全部报告，不做测试期选择）。

    start/end 为信号日区间（含端点）；面板先裁剪到 end 再构造日历与标签。horizon 为持有
    交易日数；cost_bps / stress_cost_bps 为每个 cohort 的固定往返成本（基点，10 = 0.001）。
    factor_names 可显式给出预先登记的候选（无数据的候选仍以零观测报告）；factor_values 中
    出现未登记因子时报错。hac_lag 默认等于 horizon（覆盖重叠持有期）。NaN 因子值（如预热期）
    显式排除并计数；无穷值报错。返回仅含 JSON 安全值（None 代替 NaN/inf），不含价格或 end
    之后的任何数据。
    '''
    horizon = _check_int(horizon, 'horizon', 1)
    n_quantiles = _check_int(n_quantiles, 'n_quantiles', 2)
    seed = _check_int(seed, 'seed', 0)
    lag = horizon if hac_lag is None else _check_int(hac_lag, 'hac_lag', 0)
    cost = _check_bps(cost_bps, 'cost_bps')
    stress = _check_bps(stress_cost_bps, 'stress_cost_bps')
    start_ts = _check_timestamp(start, 'start')
    end_ts = _check_timestamp(end, 'end')
    if start_ts > end_ts:
        raise ValueError('start must be <= end')
    pdf = _validate_panel(panel)
    fdf, names = _validate_factors(factor_values, factor_names)

    # 关键防泄漏步骤：先裁剪面板到 date <= end，再构造日历与任何标签
    pdf = pdf.loc[pdf['date'] <= end_ts]
    fdf = fdf.loc[(fdf['date'] >= start_ts) & (fdf['date'] <= end_ts)]
    cal = pd.DatetimeIndex(np.sort(pdf['date'].unique()))
    symbols = sorted(set(pdf['symbol'].unique()) & set(fdf['symbol'].unique()))
    ret, ok = _build_labels(pdf, cal, symbols, horizon)
    in_period = np.asarray((cal >= start_ts) & (cal <= end_ts), dtype=bool)
    ok &= in_period[:, None]
    current = (pdf.pivot(index='date', columns='symbol', values='close')
               .reindex(index=cal, columns=symbols).to_numpy(dtype=float))
    signal_available = np.isfinite(current) & in_period[:, None]
    signal_available[max(0, len(cal) - horizon - 1):] = False

    results: Dict[str, dict] = {}
    for name in names:
        sub = fdf.loc[fdf['factor'] == name]
        if sub.empty or len(cal) == 0 or not symbols:
            fv = np.full((len(cal), len(symbols)), np.nan)
        else:
            fv = (sub.pivot(index='date', columns='symbol', values='value')
                  .reindex(index=cal, columns=symbols).to_numpy(dtype=float))
        results[name] = _evaluate_factor(fv, ret, ok, signal_available, cal, symbols, n_quantiles=n_quantiles,
                                         cost=cost, stress=stress, seed=seed, lag=lag)
    holm = holm_adjust({n: results[n]['hac_excess_random']['p'] for n in names})
    for n in names:
        results[n]['holm_p_excess_random'] = holm[n]
    used = [d for n in names for d in (results[n]['first_date'], results[n]['last_date']) if d]
    return {
        'screening_status': SCREENING_STATUS,
        'candidates': list(names),
        'period': {
            'start': start_ts.strftime('%Y-%m-%d'),
            'end': end_ts.strftime('%Y-%m-%d'),
            'panel_cropped_to': end_ts.strftime('%Y-%m-%d'),
            'first_signal_date': min(used) if used else None,
            'last_signal_date': max(used) if used else None,
            'calendar_sessions_in_period': int(in_period.sum()),
        },
        'label': {
            'signal': 'close t',
            'entry': 'adjusted open t+1',
            'exit': f'adjusted open t+{horizon + 1}',
            'adjusted_open': 'open * adj_close / close',
            'horizon_sessions': horizon,
            'calendar': 'sorted unique panel dates (incl. SPY) after cropping to end',
            'missing_sessions': 'rank at signal time; incomplete future cohorts excluded whole, never reranked',
        },
        'cohorts': {
            'holding': 'fixed-horizon equal-weight buy-and-hold (not daily self-financing NAV)',
            'selection': f'top/bottom ceil(n/{n_quantiles}) by value desc/asc, symbol asc ties',
            'benchmark': 'equal weight of all eligible names on the date',
            'random': 'same size as top, drawn from same eligible pool, SHA-256 date-seeded',
            'min_names': MIN_NAMES,
            'n_quantiles': n_quantiles,
            'seed': seed,
        },
        'cost': {
            'roundtrip_bps': cost,
            'stress_roundtrip_bps': stress,
            'description': ('每个 cohort 扣一次固定往返成本（基点，10bp = 0.001），top、等权基准、'
                            '随机组同等扣除；压力情景同样本单独报告；非换手驱动的真实成交成本。'),
        },
        'statistics': {
            'test': 'one-sided Bartlett HAC t of mean(top_net - random_net) > 0',
            'hac_max_lag': lag,
            'multiple_testing': 'holm',
            'family_size': len(names),
            'missing_p_counted_as': 1.0,
        },
        'caveats': list(CAVEATS),
        'factors': results,
    }
