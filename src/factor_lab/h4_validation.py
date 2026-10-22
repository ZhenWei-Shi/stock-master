'''H4 补充验证（只读 v2 交易记录，日期等权 signal - control 对比）。

显式假设：
- 仅读取 pool == 'v2' 的原账记录；v1 或无版本记录直接忽略（只计数）。
- 字段：date（信号日，YYYY-MM-DD 或日期对象）、ticker、group（signal/control）、pool、
  ret，可选 status。ret 为该笔交易已扣除登记成本后的小数净收益（0.01 = 1%）。
- 完成判定：存在 status 时只允许 completed/pending/open，未知值直接报错；completed 须有
  有限数值 ret。无 status 时 ret 非 None 即完成，ret 同样须为有限数值。
- 可评价日期：两组均有完成记录，且该日没有任何已登记的未完成 v2 记录；同日部分完成则
  整日暂缓。较新的全未完成日期不影响历史日期。只依据已登记记录，无法发现源文件中未登记
  的遗漏。
- 同一日期多笔交易高度相关，不能视为独立样本：先按日期求组内均值，再对可评价日期等权，
  每个日期至多计一次；日期序列使用 HAC(Bartlett, lag=1)。笔数门槛、日期门槛、均值与
  HAC 使用同一可评价日期集合。
- 只有一个预先指定的对比（signal - control），不做 Holm 调整。
- supplemental_v2026-10-22-r2 门槛是对已登记规则的补充，不替代登记规则；缺失数据不能通过。
- 不修改输入记录。
'''
from __future__ import annotations

import datetime as _dt
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from .metrics import hac_mean

GATE_VERSION = 'supplemental_v2026-10-22-r2'
SOURCE_VERSION = 'v2'
VERSION_FIELD = 'pool'
VALID_GROUPS = ('signal', 'control')
VALID_STATUSES = ('completed', 'pending', 'open')
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
        if not isinstance(status, str) or status not in VALID_STATUSES:
            raise ValueError(
                f'trade {pos}: status must be one of {VALID_STATUSES}, got {status!r}')
        return status == 'completed'
    return rec.get('ret') is not None


def _count(by_date: Mapping[str, List[float]], ds: Iterable[str]) -> int:
    return sum(len(by_date.get(d, ())) for d in ds)


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

    门槛（全部满足才 passed）：可评价 signal 笔数 >= min_trades；可评价日期 >= min_dates；
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
    pending_dates = set()
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
            pending_dates.add(date)
            continue
        ret = _parse_ret(rec.get('ret'), pos)
        completed[group] += 1
        by_date[group][date].append(ret)
        if group == 'signal':
            signal_rets.append(ret)

    sig, ctl = by_date['signal'], by_date['control']
    sig_dates, ctl_dates = set(sig), set(ctl)
    matched = sorted(sig_dates & ctl_dates)
    signal_only = sorted(sig_dates - ctl_dates)
    control_only = sorted(ctl_dates - sig_dates)
    # 同日存在任一已登记未完成记录则整日暂缓；全未完成的较新日期不影响历史日期。
    evaluated = [d for d in matched if d not in pending_dates]
    partial = sorted((sig_dates | ctl_dates) & pending_dates)
    pending_only = sorted(pending_dates - sig_dates - ctl_dates)
    excl_signal_only = [d for d in signal_only if d not in pending_dates]
    excl_control_only = [d for d in control_only if d not in pending_dates]
    n_sig, n_ctl = _count(sig, evaluated), _count(ctl, evaluated)

    sig_mean = np.array([np.mean(sig[d]) for d in evaluated], dtype=float)
    ctl_mean = np.array([np.mean(ctl[d]) for d in evaluated], dtype=float)
    diff = sig_mean - ctl_mean
    hac = dict(hac_mean(pd.Series(diff, index=pd.DatetimeIndex(evaluated)), max_lag=HAC_LAG))
    hac['max_lag'] = HAC_LAG
    mean_sig = _mean(sig_mean)
    mean_sig_stressed = None if mean_sig is None else mean_sig - stress_bps / 1e4
    mean_excess = _mean(diff)
    conditions = {
        'evaluated_signal_ge_min': n_sig >= min_trades,
        'evaluated_dates_ge_min': len(evaluated) >= min_dates,
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
        'evaluated_signal': n_sig,
        'evaluated_control': n_ctl,
        'evaluated_dates': len(evaluated),
        'matched_dates': len(matched),
        'unmatched_completed_dates': len(signal_only) + len(control_only),
        'signal_only_dates': len(signal_only),
        'control_only_dates': len(control_only),
        'unmatched_note': ('matched/signal_only/control_only 按完成记录计数，含有未完成记录的日期；'
                           'excluded 为互斥原因，只有 evaluated 日期进入统计'),
        'excluded': {
            'label': '互斥原因；completed = evaluated + 各原因排除笔数',
            'date_counts': {'partial_pending': len(partial),
                            'signal_only': len(excl_signal_only),
                            'control_only': len(excl_control_only),
                            'pending_only': len(pending_only)},
            'completed_signal': {'partial_pending': _count(sig, partial),
                                 'signal_only': _count(sig, excl_signal_only)},
            'completed_control': {'partial_pending': _count(ctl, partial),
                                  'control_only': _count(ctl, excl_control_only)},
            'partial_pending_dates': partial,
        },
        'first_evaluated_date': evaluated[0] if evaluated else None,
        'last_evaluated_date': evaluated[-1] if evaluated else None,
        'mean_net_signal': mean_sig,
        'mean_net_control': _mean(ctl_mean),
        'mean_net_signal_after_stress': mean_sig_stressed,
        'mean_excess_control': mean_excess,
        'hac': hac,
        'weighting': 'date-equal: 每个可评价日期先取组内均值，日期至多计一次',
        'date_rule': '两组均有完成记录且该日无任何已登记未完成 v2 记录；未登记记录无法发现',
        'multiple_comparison': '单一预设对比 (signal - control)，不做 Holm 调整',
        'cost_stress_bps': stress_bps,
        'legacy': _legacy(signal_rets),
        'supplemental_gate': {
            'version': GATE_VERSION,
            'label': '补充门槛，不替代已登记规则',
            'thresholds': {'min_evaluated_signal': min_trades, 'min_evaluated_dates': min_dates,
                           'additional_stress_bps': stress_bps, 'hac_t': T_THRESHOLD},
            'conditions': conditions,
            'passed': all(conditions.values()),
        },
    }
