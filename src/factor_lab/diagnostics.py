"""Registered descriptions of existing cohorts, never a new selection procedure."""
import datetime as dt
import math
import re

PERIODS = ('development', 'validation')
COSTS = [0.0, 10.0, 20.0, 40.0]
YEARS = list(range(2013, 2024))
RESERVED = '2024-01-01'
TOP_N = 5
DATE = re.compile('[0-9]{4}-[0-9]{2}-[0-9]{2}')
EXTRA_LIMITS = ['descriptive_not_new_oos', 'flat_common_cost_not_execution_model',
                'date_not_name_contribution']
FIELDS = ('top_net', 'benchmark_net', 'random_net', 'excess_random')


def _num(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError('non-finite or non-numeric: ' + name)
    return float(v)


def _date(v, name):
    if not isinstance(v, str) or not DATE.fullmatch(v):
        raise ValueError('invalid date: ' + name)
    dt.date.fromisoformat(v)
    return v


def _mean(xs):
    return math.fsum(xs) / len(xs)


def _check_registration(reg, exp_id):
    expect = {'schema_version': 1, 'source_experiment_id': exp_id,
              'allowed_periods': list(PERIODS), 'allowed_years': YEARS,
              'reserved_start': RESERVED, 'cost_scenarios_bps': COSTS,
              'concentration_top_dates': TOP_N, 'run_reserved': False,
              'new_hypotheses': False, 'select_candidates': False}
    optional = {'diagnostic_id', 'registered_on', 'purpose', 'expected_source_sha256',
                'expected_config_sha256', 'expected_cache_manifest_sha256',
                'expected_factors', 'concentration_definition'}
    required = set(expect) | {'limitations'}
    if not isinstance(reg, dict) or not required <= set(reg) or set(reg) - required - optional:
        raise ValueError('registration keys mismatch')
    for k, v in expect.items():
        if type(reg[k]) is not type(v) or reg[k] != v:
            raise ValueError('registration field out of scope: ' + k)
    if any(isinstance(c, bool) for c in reg['cost_scenarios_bps'] + reg['allowed_years']):
        raise ValueError('registration contains bool')
    lim = reg['limitations']
    if not isinstance(lim, list) or not all(isinstance(x, str) for x in lim):
        raise ValueError('limitations must be list of str')
    return lim


def _rows(rows, start, end, label):
    if not isinstance(rows, list) or not rows:
        raise ValueError('empty per_date: ' + label)
    out, prev = [], None
    for r in rows:
        if not isinstance(r, dict):
            raise ValueError('bad row: ' + label)
        d = _date(r.get('date'), label)
        if prev is not None and d <= prev:
            raise ValueError('dates not ascending/unique: ' + label)
        if not (start <= d <= end) or d >= RESERVED or int(d[:4]) not in YEARS:
            raise ValueError('date out of registered scope: ' + label + ' ' + d)
        v = {k: _num(r.get(k), label + '.' + k) for k in FIELDS}
        if not math.isclose(v['excess_random'], v['top_net'] - v['random_net'], rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError('excess_random inconsistent: ' + label + ' ' + d)
        v['date'] = d
        out.append(v)
        prev = d
    return out


def _summ(rows, cost):
    years = sorted({int(r['date'][:4]) for r in rows})
    annual = []
    for y in years:
        g = [r for r in rows if int(r['date'][:4]) == y]
        annual.append({'year': y, 'date_count': len(g),
                       'mean_net': _mean([r['top_net'] for r in g]),
                       'mean_benchmark_net': _mean([r['benchmark_net'] for r in g]),
                       'mean_random_net': _mean([r['random_net'] for r in g]),
                       'mean_excess_random': _mean([r['excess_random'] for r in g]),
                       'mean_excess_benchmark': _mean([r['top_net'] - r['benchmark_net'] for r in g])})
    mt = _mean([r['top_net'] for r in rows])
    mb = _mean([r['benchmark_net'] for r in rows])
    mr = _mean([r['random_net'] for r in rows])
    scen = []
    for c in COSTS:
        adj = cost / 10000.0 - c / 10000.0
        scen.append({'cost_bps': c, 'mean_top_net': mt + adj, 'mean_random_net': mr + adj,
                     'mean_benchmark_net': mb + adj, 'mean_excess_random': mt - mr,
                     'mean_excess_benchmark': mt - mb})
    pos = sorted(((r['excess_random'], r['date']) for r in rows if r['excess_random'] > 0), key=lambda t: (-t[0], t[1]))
    total = math.fsum(p[0] for p in pos)
    top = pos[:TOP_N]
    conc = {'top_dates': [{'date': d, 'excess_random': e} for e, d in top],
            'positive_sum': total,
            'top5_positive_share': (math.fsum(e for e, _ in top) / total) if pos else None}
    return {'date_count': len(rows), 'annual': annual, 'cost_scenarios': scen,
            'positive_date_concentration': conc}


def diagnose_result(source, experiment, registration):
    exp_id = experiment.get('experiment_id')
    if not isinstance(exp_id, str) or not exp_id:
        raise ValueError('experiment_id missing')
    lim = _check_registration(registration, exp_id)
    if experiment.get('reserved_start') != RESERVED:
        raise ValueError('experiment reserved_start mismatch')
    factors = experiment.get('factors')
    if not isinstance(factors, list) or not factors or len(set(factors)) != len(factors):
        raise ValueError('experiment factors invalid')
    cost = _num(experiment.get('roundtrip_cost_bps'), 'roundtrip_cost_bps')
    if cost < 0:
        raise ValueError('negative cost')
    if not isinstance(source, dict) or source.get('experiment_id') != exp_id:
        raise ValueError('source experiment_id mismatch')
    if source.get('reserved_evaluated') is not False:
        raise ValueError('reserved_evaluated must be strictly false')
    periods = source.get('periods')
    if not isinstance(periods, dict) or set(periods) != set(PERIODS):
        raise ValueError('periods must be exactly development/validation')
    out = {}
    for p in PERIODS:
        start = _date(experiment.get(p + '_start'), p + '_start')
        end = _date(experiment.get(p + '_end'), p + '_end')
        ps = periods[p]
        if not isinstance(ps, dict) or not isinstance(ps.get('period'), dict) or not start <= end < RESERVED:
            raise ValueError('invalid period metadata: ' + p)
        if any(ps['period'].get(key) != value for key, value in [('start', start), ('end', end)]):
            raise ValueError('period bounds mismatch: ' + p)
        c = _num((ps.get('cost') or {}).get('roundtrip_bps'), p + '.cost')
        if c < 0 or c != cost:
            raise ValueError('period cost mismatch: ' + p)
        fs = ps.get('factors')
        if not isinstance(fs, dict) or set(fs) != set(factors):
            raise ValueError('factor set mismatch: ' + p)
        out[p] = {f: _summ(_rows((fs[f] or {}).get('per_date'), start, end, p + '.' + f), cost) for f in factors}
    return {'experiment_id': exp_id, 'periods': out, 'cost_basis_bps': cost,
            'limitations': list(lim) + [x for x in EXTRA_LIMITS if x not in lim],
            'reserved_evaluated': False, 'candidate_selection': False}
