import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from factor_lab.evaluation import evaluate_panel  # noqa: E402
from factor_lab.metrics import hac_mean, holm_adjust  # noqa: E402

SYMS = [f'S{i:02d}' for i in range(12)]
FACTORS = ('f_up', 'f_down', 'f_flat', 'f_rand')


def make_panel(n_days=40, symbols=SYMS, spy=True):
    '''合成面板：S00 在第 20 日 2:1 拆股（原始开盘价减半，之前 adj 因子 0.5）。'''
    dates = pd.bdate_range('2021-01-04', periods=n_days)
    rows = []
    for j, sym in enumerate(list(symbols) + (['SPY'] if spy else [])):
        for d, dt in enumerate(dates):
            base = 40.0 + 3.0 * j + 0.2 * d * ((j % 4) + 1) + 0.3 * ((d * (j + 1)) % 5)
            pre_split = sym == 'S00' and d < 20
            raw_open = base * 2.0 if pre_split else base
            close = raw_open * (1.0 + 0.002 * ((d + j) % 3))
            rows.append({'date': dt, 'symbol': sym, 'open': raw_open, 'high': raw_open * 1.02,
                         'low': raw_open * 0.98, 'close': close,
                         'adj_close': close * (0.5 if pre_split else 1.0), 'volume': 1e6,
                         'dividend': 0.0, 'split': 2.0 if (sym == 'S00' and d == 20) else 0.0})
    return pd.DataFrame(rows)


def make_factors(panel, symbols=SYMS):
    rng = np.random.default_rng(7)
    rows = []
    for dt in sorted(panel['date'].unique()):
        for i, s in enumerate(symbols):
            rows.append({'date': dt, 'symbol': s, 'factor': 'f_up', 'value': float(i)})
            rows.append({'date': dt, 'symbol': s, 'factor': 'f_down', 'value': float(-i)})
            rows.append({'date': dt, 'symbol': s, 'factor': 'f_flat', 'value': 0.0})
            rows.append({'date': dt, 'symbol': s, 'factor': 'f_rand', 'value': float(rng.normal())})
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    return df


def calendar(panel):
    return pd.DatetimeIndex(sorted(panel['date'].unique()))


def iso(ts):
    return pd.Timestamp(ts).strftime('%Y-%m-%d')


def table(panel, adjusted=True):
    p = panel.copy()
    p['px'] = p['open'] * p['adj_close'] / p['close'] if adjusted else p['open']
    return p.pivot(index='date', columns='symbol', values='px')


def expected(px, cal, i, syms, h=5):
    return float(np.mean([px.loc[cal[i + 1 + h], s] / px.loc[cal[i + 1], s] - 1.0 for s in syms]))


def run(panel, factors, **kw):
    cal = calendar(panel)
    params = {'start': cal[0], 'end': cal[-1], 'cost_bps': 0.0}
    params.update(kw)
    return evaluate_panel(panel, factors, **params)


def by_date(res, factor):
    return {row['date']: row for row in res['factors'][factor]['per_date']}


class MetricsTest(unittest.TestCase):
    def test_holm_counts_missing_hypotheses(self):
        adj = holm_adjust({'a': 0.01, 'b': 0.04, 'c': None})
        self.assertAlmostEqual(adj['a'], 0.03)
        self.assertAlmostEqual(adj['b'], 0.08)
        self.assertIsNone(adj['c'])
        fewer = holm_adjust({'a': 0.01, 'b': 0.04})
        self.assertAlmostEqual(fewer['a'], 0.02)

    def test_holm_monotone_and_rejects_nan(self):
        adj = holm_adjust({'a': 0.04, 'b': 0.01, 'c': 0.03})
        self.assertAlmostEqual(adj['b'], 0.03)
        self.assertAlmostEqual(adj['c'], 0.06)
        self.assertAlmostEqual(adj['a'], 0.06)
        with self.assertRaises(ValueError):
            holm_adjust({'a': float('nan')})

    def test_hac_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            hac_mean([0.1] * 25 + [float('nan')])
        with self.assertRaises(ValueError):
            hac_mean([0.1] * 25 + [float('inf')])

    def test_hac_insufficient_and_constant(self):
        r = hac_mean([float(x) for x in range(19)], max_lag=5)
        self.assertEqual(r['n'], 19)
        self.assertIsNone(r['t'])
        self.assertIsNone(hac_mean([float(x) for x in range(21)], max_lag=10)['t'])
        self.assertIsNotNone(hac_mean([float(x) for x in range(22)], max_lag=10)['t'])
        const = hac_mean([0.01] * 30)
        self.assertIsNone(const['t'])
        self.assertIsNone(const['p'])

    def test_hac_overlap_inflates_standard_error(self):
        rng = np.random.default_rng(3)
        x = 0.05 + np.convolve(rng.normal(size=400), np.ones(5) / 5.0, mode='valid')
        self.assertLess(abs(hac_mean(x, max_lag=5)['t']), abs(hac_mean(x, max_lag=0)['t']))

    def test_hac_no_infinite_certainty(self):
        rng = np.random.default_rng(4)
        r = hac_mean(1.0 + 1e-9 * rng.normal(size=50), max_lag=1)
        self.assertTrue(math.isfinite(r['t']))
        self.assertGreater(r['p'], 0.0)
        self.assertLess(r['p'], 1e-10)

    def test_hac_rejects_duplicate_dates(self):
        idx = pd.DatetimeIndex(['2021-01-04'] * 2 + list(pd.bdate_range('2021-01-06', periods=20)))
        with self.assertRaises(ValueError):
            hac_mean(pd.Series(np.arange(22, dtype=float), index=idx))


class EvaluatePanelTest(unittest.TestCase):
    def setUp(self):
        self.panel = make_panel()
        self.factors = make_factors(self.panel)
        self.cal = calendar(self.panel)

    def test_label_adjusted_open_t1_to_t6(self):
        res = run(self.panel, self.factors)
        rows = by_date(res, 'f_up')
        ao = table(self.panel)
        self.assertEqual(res['factors']['f_up']['dates'], 34)
        for i in (0, 10, 33):
            row = rows[iso(self.cal[i])]
            self.assertAlmostEqual(row['top_net'], expected(ao, self.cal, i, ['S11', 'S10', 'S09']), places=12)
            self.assertAlmostEqual(row['benchmark_net'], expected(ao, self.cal, i, SYMS), places=12)
        self.assertNotIn(iso(self.cal[34]), rows)

    def test_split_uses_adjusted_open(self):
        res = run(self.panel, self.factors)
        row = by_date(res, 'f_down')[iso(self.cal[15])]
        top = ['S00', 'S01', 'S02']
        self.assertAlmostEqual(row['top_net'], expected(table(self.panel), self.cal, 15, top), places=12)
        raw = expected(table(self.panel, adjusted=False), self.cal, 15, top)
        self.assertGreater(abs(row['top_net'] - raw), 0.1)

    def test_crop_prevents_future_leakage(self):
        end = self.cal[25]
        res = run(self.panel, self.factors, end=end)
        dates = [r['date'] for r in res['factors']['f_up']['per_date']]
        self.assertEqual(dates[-1], iso(self.cal[19]))
        self.assertNotIn(iso(self.cal[20]), dates)
        shocked = self.panel.copy()
        shocked.loc[shocked['date'] > end, 'open'] *= 7.0
        self.assertEqual(res, run(shocked, self.factors, end=end))
        cropped = self.panel.loc[self.panel['date'] <= end]
        self.assertEqual(res, run(cropped, self.factors, start=self.cal[0], end=end))

    def test_missing_session_not_shifted(self):
        p = self.panel.loc[~((self.panel['symbol'] == 'S11') & (self.panel['date'] == self.cal[10]))]
        res = run(p, make_factors(p))
        rows = by_date(res, 'f_up')
        for i in range(4, 10):
            self.assertNotIn(iso(self.cal[i]), rows)
        self.assertEqual(rows[iso(self.cal[10])]['n'], 11)
        for i in (3, 11):
            self.assertEqual(rows[iso(self.cal[i])]['n'], 12)
        self.assertGreaterEqual(res['factors']['f_up']['skipped_dates_missing_outcomes'], 6)

    def test_cost_units_flat_roundtrip(self):
        a = run(self.panel, self.factors, cost_bps=0.0)['factors']['f_up']
        b = run(self.panel, self.factors, cost_bps=10.0)['factors']['f_up']
        for key in ('mean_net', 'mean_benchmark_net', 'mean_random_net'):
            self.assertAlmostEqual(a[key] - b[key], 0.001, places=12)
        for key in ('mean_excess_random', 'mean_excess_benchmark'):
            self.assertAlmostEqual(a[key], b[key], places=12)
        self.assertAlmostEqual(a['mean_net'] - a['mean_net_stress'], 0.002, places=12)
        self.assertAlmostEqual(a['mean_net_stress'], b['mean_net_stress'], places=12)
        self.assertAlmostEqual(a['per_date'][0]['top_net'] - b['per_date'][0]['top_net'], 0.001, places=12)

    def test_deterministic_ties_by_symbol(self):
        res = run(self.panel, self.factors)
        row = by_date(res, 'f_flat')[iso(self.cal[0])]
        self.assertAlmostEqual(row['top_net'], expected(table(self.panel), self.cal, 0, ['S00', 'S01', 'S02']), places=12)
        self.assertAlmostEqual(res['factors']['f_flat']['top_bottom_gross'], 0.0, places=12)
        self.assertIsNone(res['factors']['f_flat']['mean_rank_ic'])

    def test_random_group_deterministic_and_pool_shared(self):
        r1 = run(self.panel, self.factors)
        r2 = run(self.panel, self.factors)
        rand = lambda r, f: [x['random_net'] for x in r['factors'][f]['per_date']]
        self.assertEqual(rand(r1, 'f_up'), rand(r2, 'f_up'))
        self.assertEqual(rand(r1, 'f_up'), rand(r1, 'f_flat'))
        r3 = run(self.panel, self.factors, seed=1)
        self.assertNotEqual(rand(r1, 'f_up'), rand(r3, 'f_up'))

    def test_all_candidates_reported_and_json_safe(self):
        res = run(self.panel, self.factors, factor_names=FACTORS + ('f_missing',))
        self.assertEqual(res['candidates'], list(FACTORS) + ['f_missing'])
        miss = res['factors']['f_missing']
        self.assertEqual(miss['dates'], 0)
        self.assertIsNone(miss['mean_net'])
        self.assertIsNone(miss['hac_excess_random']['t'])
        self.assertIsNone(miss['holm_p_excess_random'])
        self.assertEqual(res['statistics']['family_size'], 5)
        for name, out in res['factors'].items():
            self.assertEqual(out['screening_status'], 'exploratory')
            p, adj = out['hac_excess_random']['p'], out['holm_p_excess_random']
            if p is not None:
                self.assertGreaterEqual(adj, p)
        json.dumps(res, allow_nan=False)

    def test_min_names(self):
        syms = SYMS[:9]
        p = make_panel(symbols=syms)
        res = run(p, make_factors(p, syms))
        self.assertEqual(res['factors']['f_up']['dates'], 0)
        self.assertEqual(res['factors']['f_up']['skipped_dates_insufficient_names'], 34)

    def test_nan_warmup_excluded_explicitly(self):
        f = self.factors.copy()
        f.loc[(f['factor'] == 'f_up') & (f['date'].isin(self.cal[:3])), 'value'] = np.nan
        out = run(self.panel, f)['factors']['f_up']
        self.assertEqual(out['dates'], 31)
        self.assertEqual(out['first_date'], iso(self.cal[3]))
        self.assertEqual(out['excluded_missing_factor_cells'], 36)

    def test_validation_errors(self):
        p, f, cal = self.panel, self.factors, self.cal
        tz = p.copy()
        tz['date'] = tz['date'].dt.tz_localize('UTC')
        bad_inf = f.copy()
        bad_inf.loc[0, 'value'] = np.inf
        cases = [
            lambda: run(pd.concat([p, p.iloc[[0]]], ignore_index=True), f),
            lambda: run(p, pd.concat([f, f.iloc[[0]]], ignore_index=True)),
            lambda: run(p, bad_inf),
            lambda: run(p, f, horizon=0),
            lambda: run(p, f, horizon=2.5),
            lambda: run(p, f, cost_bps=-1.0),
            lambda: run(p, f, cost_bps=float('nan')),
            lambda: run(p, f, start=cal[10], end=cal[5]),
            lambda: run(p.drop(columns='adj_close'), f),
            lambda: evaluate_panel(tz, f, start=cal[0], end=cal[-1]),
            lambda: run(p, f, factor_names=('f_up', 'f_down')),
        ]
        for case in cases:
            with self.assertRaises(ValueError):
                case()


if __name__ == '__main__':
    unittest.main()
