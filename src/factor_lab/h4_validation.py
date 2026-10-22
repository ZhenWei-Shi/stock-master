'''H4 补充验证（只读 v2 交易记录，日期等权 signal - control 对比）。

显式假设：
- 仅读取 pool == 'v2' 的原账记录；v1 或无版本记录直接忽略（只计数）。
- 字段：date（信号日，YYYY-MM-DD 或日期对象）、ticker、group（signal/control）、pool、
  ret，可选 status。ret 为该笔交易已扣除登记成本后的小数净收益（0.01 = 1%）。
- 完成判定：存在 status 时 status == 'completed'；否则 ret 非 None。完成记录的 ret 必须
  为有限数值，否则报错；未完成记录只计入 pending，不参与任何统计。
- 同一日期多笔交易高度相关，不能视为独立样本：先按日期求组内均值，再对匹配日期等权，
  每个日期至多计一次；日期序列使用 HAC(Bartlett, lag=1)。
- 只有一个预先指定的对比（signal - control），不做 Holm 调整。
- supplemental_v2026-10-22 门槛是对已登记规则的补充，不替代登记规则；缺失数据不能通过。
- 不修改输入记录。
'''
from __future__ import annotations

import datetime as _dt
import math
from collections import defaultdict
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .metrics import hac_mean

GATE_VERSION = 'supplemental_v2026-10-22'
SOURCE_VERSION = 'v2'
VERSION_FIELD = 'pool'
VALID_GROUPS = ('signal', 'control')
HAC_LAG = 1
T_THRESHOLD = 2.0


def _num(x) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _mean(values) -> Optional[float]:
    return _num(np.mean(values)) if len(values) else None


def _check_int(value, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _check_bps(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be numeric basis points')
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric basis points') from exc
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f'{name} must be finite and non-negative')
    return v


def _parse_date(value, pos: int) -> str:
    if not isinstance(value, (str, _dt.date, np.datetime64)):
        raise ValueError(f'trade {pos}: date must be a date string or date object')
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'trade {pos}: invalid date {value!r}') from exc
    if pd.isna(ts) or ts.tzinfo is not None or ts != ts.normalize():
        raise ValueError(f'trade {pos}: date must be a naive calendar date')
    return ts.strftime('%Y-%m-%d')


def _req_str(rec: Mapping, field: str, pos: int) -> str:
    value = rec.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f'trade {pos}: {field} must be a non-empty string')
    return value


def _parse_ret(value, pos: int) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise ValueError(f'trade {pos}: completed ret must be numeric')
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f'trade {pos}: completed ret must be finite')
    return v


def _is_completed(rec: Mapping, pos: int) -> bool:
    if 'status' in rec:
        status = rec['status']
        if not isinstance(status, str):
            raise ValueError(f'trade {pos}: status must be a string')
        return status == 'completed'
    return rec.get('ret') is not None


def _legacy(rets: List[float]) -> dict:
    '''旧口径参考：逐笔等权、普通 iid t；同日交易相关，此 t 会高估证据强度。'''
    n = len(rets)
    t = None
    if n >= 2:
        arr = np.asarray(rets, dtype=float)
        sd = float(arr.std(ddof=1))
        if sd > 0:
            t = _num(arr.mean() / (sd / math.sqrt(n)))
    return {
        'label': 'legacy_trade_level_iid_reference_only',
        'note': '逐笔视为独立样本，未处理同日相关；不用于 supplemental gate。',
        'completed_signal': n,
        'mean_net_signal_trade_weighted': _mean(rets),
        't_iid': t,
    }


def evaluate_h4(trades: List[dict], *, min_trades=200, min_dates=30,
                cost_stress_bps=10.0) -> dict:
    '''只读评估 H4 v2 记录，返回 JSON 安全 dict（None 代替 NaN）。

    门槛（全部满足才 passed）：完成 signal 笔数 >= min_trades；匹配日期 >= min_dates；
    日期等权 signal 净收益再扣 cost_stress_bps 后 > 0；日期等权超额 > 0 且 HAC t > 2。
    '''
    min_trades = _check_int(min_trades, 'min_trades', 1)
    min_dates = _check_int(min_dates, 'min_dates', 1)
    stress_bps = _check_bps(cost_stress_bps, 'cost_stress_bps')
    if not isinstance(trades, (list, tuple)):
        raise ValueError('trades must be a list of dicts')
    seen = set()
    ignored = 0
    pending = {g: 0 for g in VALID_GROUPS}
    completed = {g: 0 for g in VALID_GROUPS}
    by_date: Dict[str, Dict[str, List[float]]] = {g: defaultdict(list) for g in VALID_GROUPS}
    signal_rets: List[float] = []
    for pos, rec in enumerate(trades):
        if not isinstance(rec, Mapping):
            raise ValueError(f'trade {pos}: record must be a mapping')
        if rec.get(VERSION_FIELD) != SOURCE_VERSION:
            ignored += 1
            continue
        date = _parse_date(rec.get('date'), pos)
        ticker = _req_str(rec, 'ticker', pos)
        pool = _req_str(rec, 'pool', pos)
        group = rec.get('group')
        if group not in VALID_GROUPS:
            raise ValueError(f'trade {pos}: group must be one of {VALID_GROUPS}')
        key = (date, ticker, group, pool)
        if key in seen:
            raise ValueError(f'trade {pos}: duplicate (date, ticker, group, pool) {key}')
        seen.add(key)
        if not _is_completed(rec, pos):
            pending[group] += 1
            continue
        ret = _parse_ret(rec.get('ret'), pos)
        completed[group] += 1
        by_date[group][date].append(ret)
        if group == 'signal':
            signal_rets.append(ret)

    sig, ctl = by_date['signal'], by_date['control']
    matched = sorted(set(sig) & set(ctl))
    signal_only = sorted(set(sig) - set(ctl))
    control_only = sorted(set(ctl) - set(sig))
    sig_mean = np.array([np.mean(sig[d]) for d in matched], dtype=float)
    ctl_mean = np.array([np.mean(ctl[d]) for d in matched], dtype=float)
    diff = sig_mean - ctl_mean
    hac = dict(hac_mean(pd.Series(diff, index=pd.DatetimeIndex(matched)), max_lag=HAC_LAG))
    hac['max_lag'] = HAC_LAG
    mean_sig = _mean(sig_mean)
    mean_sig_stressed = None if mean_sig is None else mean_sig - stress_bps / 1e4
    mean_excess = _mean(diff)
    conditions = {
        'completed_signal_ge_min': completed['signal'] >= min_trades,
        'matched_dates_ge_min': len(matched) >= min_dates,
        'net_positive_after_stress': mean_sig_stressed is not None and mean_sig_stressed > 0,
        'excess_positive': mean_excess is not None and mean_excess > 0,
        'hac_t_gt_threshold': hac['t'] is not None and hac['t'] > T_THRESHOLD,
    }
    return {
        'source_version': SOURCE_VERSION,
        'ignored_non_v2_records': ignored,
        'completed_signal': completed['signal'],
        'completed_control': completed['control'],
        'pending_signal': pending['signal'],
        'pending_control': pending['control'],
        'matched_dates': len(matched),
        'unmatched_completed_dates': len(signal_only) + len(control_only),
        'signal_only_dates': len(signal_only),
        'control_only_dates': len(control_only),
        'first_matched_date': matched[0] if matched else None,
        'last_matched_date': matched[-1] if matched else None,
        'mean_net_signal': mean_sig,
        'mean_net_control': _mean(ctl_mean),
        'mean_net_signal_after_stress': mean_sig_stressed,
        'mean_excess_control': mean_excess,
        'hac': hac,
        'weighting': 'date-equal: 每个匹配日期先取组内均值，日期至多计一次',
        'multiple_comparison': '单一预设对比 (signal - control)，不做 Holm 调整',
        'cost_stress_bps': stress_bps,
        'legacy': _legacy(signal_rets),
        'supplemental_gate': {
            'version': GATE_VERSION,
            'label': '补充门槛，不替代已登记规则',
            'thresholds': {'min_completed_signal': min_trades, 'min_matched_dates': min_dates,
                           'additional_stress_bps': stress_bps, 'hac_t': T_THRESHOLD},
            'conditions': conditions,
            'passed': all(conditions.values()),
        },
    }
