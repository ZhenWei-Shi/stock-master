'''统计工具：Bartlett Newey-West HAC 均值检验与 Holm 逐步调整。

显式假设：
- 输入序列按日期升序，每个日期至多一个观测（日期等权）。若传入带 DatetimeIndex 的
  pandas.Series，会按日期排序，并拒绝重复或缺失日期。
- p 值为单侧正尾、渐近正态近似，仅用于探索性筛选，不构成经济盈利结论。
- 非有限值直接抛出 ValueError，不静默丢弃；样本不足或零方差时 t/p 为 None。
- p 值设下限 P_FLOOR，避免正态尾部下溢为 0 而表达无限确定性。
'''
from __future__ import annotations

import math
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

P_FLOOR = 1e-15


def min_obs_for_t(max_lag: int) -> int:
    '''计算 t 统计量所需的最少日期数：max(20, 2*lag+2)。'''
    return max(20, 2 * int(max_lag) + 2)


def _validate_lag(max_lag) -> int:
    if isinstance(max_lag, bool) or not isinstance(max_lag, (int, np.integer)) or max_lag < 0:
        raise ValueError('max_lag must be a non-negative integer')
    return int(max_lag)


def _ordered_array(values) -> np.ndarray:
    '''转为有序 float 数组；带 DatetimeIndex 的 Series 按日期排序，拒绝重复/缺失日期。'''
    if isinstance(values, pd.Series):
        if isinstance(values.index, pd.DatetimeIndex):
            if values.index.hasnans or not values.index.is_unique:
                raise ValueError('date index must be unique and non-null')
            values = values.sort_index()
        values = values.to_numpy()
    try:
        arr = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(f'values must be numeric: {exc}') from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError('values contain non-finite entries; clean them explicitly')
    return arr


def hac_mean(values, max_lag: int = 5) -> Dict[str, Optional[float]]:
    '''均值的 Bartlett Newey-West HAC t 检验（单侧，H1: mean > 0）。

    返回 {'n', 'mean', 't', 'p'}。日期数少于 max(20, 2*max_lag+2)、零方差或长期方差
    估计非正时 t/p 为 None。p 为渐近正态正尾概率，下限 P_FLOOR。
    '''
    lag = _validate_lag(max_lag)
    arr = _ordered_array(values)
    n = int(arr.size)
    mean = float(arr.mean()) if n else None
    out = {'n': n, 'mean': mean, 't': None, 'p': None}
    if n < min_obs_for_t(lag):
        return out
    e = arr - mean
    gamma0 = float(e @ e) / n
    scale = float(np.max(np.abs(arr)))
    # 零方差（含浮点残差）不给出 t，避免无限确定性
    if gamma0 <= (1e-12 * scale) ** 2:
        return out
    s = gamma0
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1.0)
        s += 2.0 * weight * float(e[k:] @ e[:-k]) / n
    var_mean = s / n
    if not math.isfinite(var_mean) or var_mean <= 0.0:
        return out
    t = mean / math.sqrt(var_mean)
    if not math.isfinite(t):
        return out
    p = 0.5 * math.erfc(t / math.sqrt(2.0))
    out['t'] = float(t)
    out['p'] = float(min(1.0, max(P_FLOOR, p)))
    return out


def holm_adjust(p_values: Mapping[str, Optional[float]]) -> Dict[str, Optional[float]]:
    '''Holm 逐步调整。

    每个键都是预先登记的假设；None 表示缺失 p，结果保留 None，但计入假设总数 m
    （等价于 p=1，排在最后，不会降低其他假设的乘数）。NaN 或越界 p 报错。
    '''
    keys = list(p_values.keys())
    m = len(keys)
    present = []
    for key in keys:
        p = p_values[key]
        if p is None:
            continue
        if isinstance(p, bool):
            raise ValueError(f'p value for {key} must be numeric')
        try:
            pf = float(p)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'p value for {key} must be numeric') from exc
        if not math.isfinite(pf) or pf < 0.0 or pf > 1.0:
            raise ValueError(f'p value for {key} must be in [0, 1]; use None for missing')
        present.append((pf, str(key), key))
    present.sort(key=lambda item: (item[0], item[1]))
    out: Dict[str, Optional[float]] = {key: None for key in keys}
    running = 0.0
    for i, (pf, _, key) in enumerate(present):
        running = max(running, min(1.0, (m - i) * pf))
        out[key] = running
    return out
