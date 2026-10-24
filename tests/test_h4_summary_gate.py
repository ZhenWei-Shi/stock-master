import copy
import datetime
import random

import pytest

from src import overnight_lab as ol


def rows(days, n, sig, ctl, sd=0.002, seed=7):
    rng = random.Random(seed)
    out = []
    for d in range(days):
        date = (datetime.date(2026, 1, 1) + datetime.timedelta(days=d)).isoformat()
        for g, mu in (('signal', sig), ('control', ctl)):
            for k in range(n):
                out.append({'date': date, 'ticker': f'{g[0]}{k:03d}', 'group': g, 'pool': 'v2',
                            'ret': mu + (rng.gauss(0, sd) if sd else 0.0)})
    return out


def S(trades):
    return ol.summarize({'trades': trades})


@pytest.fixture(autouse=True)
def no_load(monkeypatch):
    monkeypatch.setattr(ol, '_load', lambda: pytest.fail('不应读取私有文件'))


def test_enough_valid_passes():
    out = S(rows(40, 6, 0.004, 0.0))
    assert '✅ 按补充标准通过' in out and '仅参考' in out and '成立' not in out


def test_200_trades_under_30_dates_insufficient():
    out = S(rows(20, 10, 0.004, 0.0))
    assert '⏳ 样本不足' in out and '可评价日期<30天' in out and '✅' not in out


def test_signal_behind_control_fails():
    out = S(rows(40, 6, 0.003, 0.005))
    assert '❌' in out and '超额' in out and '✅' not in out


def test_negative_after_extra_10bp_fails():
    out = S(rows(40, 6, 0.0005, -0.003))
    assert '❌' in out and '追加10bp后净收益非正' in out


def test_partial_pending_dates_reduce_evaluable():
    tr = rows(40, 6, 0.004, 0.0)
    for d in range(15):
        date = (datetime.date(2026, 1, 1) + datetime.timedelta(days=d)).isoformat()
        tr.append({'date': date, 'ticker': 'p000', 'group': 'signal', 'pool': 'v2', 'ret': None})
    out = S(tr)
    assert '⏳ 样本不足' in out and '完整日期25天' in out


def test_hac_unknown_not_pass():
    out = S(rows(40, 6, 0.004, 0.0, sd=0))
    assert 'HAC t=未知' in out and '✅' not in out


def test_v1_does_not_affect_verdict():
    bad_v1 = [{'group': 'signal', 'ret': 'x'}, {'group': 'signal', 'ret': float('inf')},
              {'pool': 'v1', 'group': 'signal', 'ret': -0.5}]
    good_v1 = [{'pool': 'v1', 'group': 'signal', 'ret': 0.05 + i * 1e-3} for i in range(300)]
    out = S(rows(40, 6, 0.004, 0.0) + bad_v1)
    assert '✅' in out and '2笔ret缺失或非法未计' in out
    assert '✅' not in S(rows(40, 6, 0.003, 0.005) + good_v1)


def test_explicit_empty_dict_does_not_load():
    with pytest.raises(ValueError):
        ol.summarize({})


def test_none_calls_load(monkeypatch):
    monkeypatch.setattr(ol, '_load', lambda: {'trades': []})
    assert '样本不足' in ol.summarize(None)


def test_empty_trades_insufficient():
    assert '⏳ 样本不足' in S([]) and '⏳ 样本不足' in S(())


@pytest.mark.parametrize('state', [
    [], {'trades': {}}, {'trades': [1]}, {'x': []},
    {'trades': [{'date': '2026-01-02', 'group': 'signal', 'pool': 'v2', 'ret': 0.01}]},
    {'trades': [{'date': '2026-01-02', 'ticker': 'a', 'group': 'signal', 'pool': 'v2',
                 'ret': float('nan')}]},
])
def test_bad_input_raises(state):
    with pytest.raises(ValueError):
        ol.summarize(state)


def test_input_not_mutated():
    state = {'trades': rows(40, 6, 0.004, 0.0) + [{'pool': 'v1', 'group': 'signal', 'ret': 0.01}]}
    snap = copy.deepcopy(state)
    ol.summarize(state)
    assert state == snap
