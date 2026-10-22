import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from factor_lab.h4_validation import GATE_VERSION, evaluate_h4  # noqa: E402


def trade(date, ticker, group, ret, pool='v2', status='completed'):
    return {'date': date, 'ticker': ticker, 'group': group,
            'pool': pool, 'status': status, 'ret': ret}


def pending(date, ticker, group, status='pending'):
    return trade(date, ticker, group, None, status=status)


def dates(n):
    return [d.strftime('%Y-%m-%d') for d in pd.bdate_range('2026-01-05', periods=n)]


def build(n_dates=40, n_signal=6, n_control=3, signal_mu=0.01, control_mu=0.0,
          noise=0.002, control_dates=None, seed=0):
    rng = np.random.default_rng(seed)
    ds = dates(n_dates)
    cset = set(ds if control_dates is None else control_dates)
    out = []
    for d in ds:
        for k in range(n_signal):
            out.append(trade(d, f'T{k:03d}', 'signal', float(signal_mu + rng.normal(0, noise))))
        if d in cset:
            for k in range(n_control):
                out.append(trade(d, f'C{k:03d}', 'control', float(control_mu + rng.normal(0, noise))))
    return out


class H4ValidationTest(unittest.TestCase):
    def assert_partition(self, res):
        ex = res['excluded']
        self.assertEqual(res['completed_signal'],
                         res['evaluated_signal'] + sum(ex['completed_signal'].values()))
        self.assertEqual(res['completed_control'],
                         res['evaluated_control'] + sum(ex['completed_control'].values()))
        if res['evaluated_dates']:
            self.assertEqual(res['hac']['n'], res['evaluated_dates'])

    def test_gate_passes_with_enough_independent_dates(self):
        res = evaluate_h4(build())
        self.assertEqual(res['completed_signal'], 240)
        self.assertEqual(res['completed_control'], 120)
        self.assertEqual(res['evaluated_signal'], 240)
        self.assertEqual(res['evaluated_control'], 120)
        self.assertEqual(res['matched_dates'], 40)
        self.assertEqual(res['evaluated_dates'], 40)
        self.assertGreater(res['hac']['t'], 2.0)
        self.assertEqual(res['hac']['max_lag'], 1)
        self.assert_partition(res)
        gate = res['supplemental_gate']
        self.assertEqual(gate['version'], GATE_VERSION)
        self.assertEqual(GATE_VERSION, 'supplemental_v2026-10-22-r2')
        self.assertEqual(gate['thresholds']['min_evaluated_signal'], 200)
        self.assertEqual(gate['thresholds']['min_evaluated_dates'], 30)
        self.assertTrue(all(gate['conditions'].values()))
        self.assertTrue(gate['passed'])

    def test_same_date_trades_not_independent(self):
        res = evaluate_h4(build(n_dates=10, n_signal=30))
        self.assertEqual(res['completed_signal'], 300)
        self.assertEqual(res['evaluated_signal'], 300)
        self.assertEqual(res['evaluated_dates'], 10)
        self.assertGreater(res['legacy']['t_iid'], 2.0)
        self.assertIsNone(res['hac']['t'])
        self.assertFalse(res['supplemental_gate']['conditions']['evaluated_dates_ge_min'])
        self.assertFalse(res['supplemental_gate']['passed'])

    def test_missing_controls_do_not_pass(self):
        res = evaluate_h4(build(control_dates=dates(40)[:15]))
        self.assertEqual(res['matched_dates'], 15)
        self.assertEqual(res['evaluated_dates'], 15)
        self.assertEqual(res['evaluated_signal'], 90)
        self.assertEqual(res['unmatched_completed_dates'], 25)
        self.assertEqual(res['signal_only_dates'], 25)
        self.assertEqual(res['excluded']['date_counts']['signal_only'], 25)
        self.assertEqual(res['excluded']['completed_signal']['signal_only'], 150)
        self.assert_partition(res)
        self.assertFalse(res['supplemental_gate']['passed'])
        none = evaluate_h4(build(control_dates=[]))
        self.assertEqual(none['matched_dates'], 0)
        self.assertEqual(none['evaluated_dates'], 0)
        self.assertEqual(none['evaluated_signal'], 0)
        self.assertIsNone(none['mean_excess_control'])
        self.assertIsNone(none['mean_net_signal'])
        self.assertIsNone(none['hac']['t'])
        self.assertFalse(none['supplemental_gate']['conditions']['excess_positive'])
        self.assertFalse(none['supplemental_gate']['passed'])

    def test_partial_pending_dates_deferred(self):
        base = build()
        ds = dates(40)
        extra = [
            pending(ds[0], 'P001', 'signal'),
            trade(ds[1], 'P002', 'signal', 0.5, status='open'),
            {'date': ds[2], 'ticker': 'P003', 'group': 'control', 'pool': 'v2', 'ret': None},
            pending('2026-12-31', 'P004', 'control'),
        ]
        res = evaluate_h4(base + extra)
        self.assertEqual(res['pending_signal'], 2)
        self.assertEqual(res['pending_control'], 2)
        self.assertEqual(res['completed_signal'], 240)
        self.assertEqual(res['completed_control'], 120)
        self.assertEqual(res['matched_dates'], 40)
        self.assertEqual(res['evaluated_dates'], 37)
        self.assertEqual(res['evaluated_signal'], 222)
        self.assertEqual(res['evaluated_control'], 111)
        ex = res['excluded']
        self.assertEqual(ex['partial_pending_dates'], ds[:3])
        self.assertEqual(ex['date_counts']['partial_pending'], 3)
        self.assertEqual(ex['date_counts']['pending_only'], 1)
        self.assertEqual(ex['completed_signal']['partial_pending'], 18)
        self.assertEqual(ex['completed_control']['partial_pending'], 9)
        self.assert_partition(res)
        expected = evaluate_h4([t for t in base if t['date'] not in ds[:3]])
        self.assertAlmostEqual(res['mean_excess_control'], expected['mean_excess_control'], places=12)
        self.assertAlmostEqual(res['hac']['t'], expected['hac']['t'], places=9)

    def test_review_repro_signals_without_control_fail(self):
        res = evaluate_h4(build(n_dates=200, n_signal=1, n_control=1,
                                control_dates=dates(200)[:30]))
        self.assertEqual(res['completed_signal'], 200)
        self.assertEqual(res['evaluated_signal'], 30)
        self.assertEqual(res['evaluated_control'], 30)
        self.assertEqual(res['evaluated_dates'], 30)
        self.assertEqual(res['signal_only_dates'], 170)
        self.assertEqual(res['excluded']['completed_signal']['signal_only'], 170)
        self.assert_partition(res)
        cond = res['supplemental_gate']['conditions']
        self.assertFalse(cond['evaluated_signal_ge_min'])
        self.assertTrue(cond['evaluated_dates_ge_min'])
        self.assertFalse(res['supplemental_gate']['passed'])

    def test_review_repro_partial_dates_fail(self):
        trades = build(n_dates=30, n_signal=7)
        trades += [pending(d, 'P000', 'signal') for d in dates(30)]
        res = evaluate_h4(trades)
        self.assertEqual(res['completed_signal'], 210)
        self.assertEqual(res['pending_signal'], 30)
        self.assertEqual(res['matched_dates'], 30)
        self.assertEqual(res['evaluated_signal'], 0)
        self.assertEqual(res['evaluated_control'], 0)
        self.assertEqual(res['evaluated_dates'], 0)
        self.assertEqual(res['excluded']['partial_pending_dates'], dates(30))
        self.assertEqual(res['excluded']['completed_signal']['partial_pending'], 210)
        self.assert_partition(res)
        self.assertIsNone(res['mean_excess_control'])
        self.assertIsNone(res['hac']['t'])
        self.assertFalse(res['supplemental_gate']['passed'])

    def test_latest_pending_dates_do_not_block(self):
        base = build()
        later = [pending('2026-12-31', 'P001', 'signal'),
                 pending('2026-12-31', 'P002', 'control', status='open'),
                 {'date': '2026-12-30', 'ticker': 'P003', 'group': 'signal',
                  'pool': 'v2', 'ret': None}]
        ref = evaluate_h4(base)
        res = evaluate_h4(base + later)
        self.assertEqual(res['pending_signal'], 2)
        self.assertEqual(res['pending_control'], 1)
        self.assertEqual(res['evaluated_signal'], 240)
        self.assertEqual(res['evaluated_dates'], 40)
        self.assertEqual(res['excluded']['date_counts']['pending_only'], 2)
        self.assertEqual(res['excluded']['date_counts']['partial_pending'], 0)
        self.assertEqual(res['mean_excess_control'], ref['mean_excess_control'])
        self.assertEqual(res['hac'], ref['hac'])
        self.assertTrue(res['supplemental_gate']['passed'])

    def test_unknown_status_raises(self):
        d = dates(1)[0]
        for status in ('closed', 'Completed', '', None, 1):
            with self.subTest(status=status), self.assertRaises(ValueError):
                evaluate_h4([trade(d, 'X', 'signal', 0.01, status=status)])

    def test_stress_cost_in_bps(self):
        trades = build(signal_mu=0.0005, control_mu=-0.01, noise=0.0001)
        res = evaluate_h4(trades)
        cond = res['supplemental_gate']['conditions']
        self.assertAlmostEqual(res['mean_net_signal'] - res['mean_net_signal_after_stress'], 0.001, places=12)
        self.assertTrue(cond['hac_t_gt_threshold'])
        self.assertFalse(cond['net_positive_after_stress'])
        self.assertFalse(res['supplemental_gate']['passed'])
        self.assertTrue(evaluate_h4(trades, cost_stress_bps=0.0)['supplemental_gate']['passed'])

    def test_v1_records_ignored(self):
        base = build()
        legacy = [trade(dates(40)[0], 'X', 'bogus', float('nan'), pool='v1'),
                  trade(dates(40)[0], 'X', 'bogus', float('nan'), pool='v1', status='weird'),
                  {'date': 'not-a-date', 'group': 'signal', 'ret': 1.0}]
        ref = evaluate_h4(base)
        res = evaluate_h4(base + legacy)
        self.assertEqual(res['ignored_non_v2_records'], 3)
        res.pop('ignored_non_v2_records')
        ref.pop('ignored_non_v2_records')
        self.assertEqual(res, ref)

    def test_invalid_records_raise(self):
        d = dates(1)[0]
        no_pool = trade(d, 'X', 'signal', 0.01)
        del no_pool['pool']
        cases = [
            [trade(d, 'X', 'signal', 0.01), trade(d, 'X', 'signal', 0.02)],
            [trade(d, 'X', 'treatment', 0.01)],
            [trade(d, 'X', 'signal', float('nan'))],
            [trade(d, 'X', 'signal', float('inf'))],
            [trade(d, 'X', 'signal', True)],
            [trade(d, 'X', 'signal', None)],
            [{'date': d, 'ticker': 'X', 'group': 'signal', 'pool': 'v2', 'ret': float('nan')}],
            [trade('2026-13-45', 'X', 'signal', 0.01)],
        ]
        for trades in cases:
            with self.assertRaises(ValueError):
                evaluate_h4(trades)
        for kwargs in ({'min_trades': 0}, {'min_dates': 0}, {'cost_stress_bps': -1.0}):
            with self.assertRaises(ValueError):
                evaluate_h4([], **kwargs)
        with self.assertRaises(ValueError):
            evaluate_h4('not a list')

    def test_each_date_counted_once_across_tickers(self):
        trades = []
        for d in dates(25):
            for suffix in ('A', 'B'):
                trades.append(trade(d, 'T' + suffix, 'signal', 0.01))
                trades.append(trade(d, 'C' + suffix, 'control', 0.0))
        res = evaluate_h4(trades)
        self.assertEqual(res['completed_signal'], 50)
        self.assertEqual(res['evaluated_signal'], 50)
        self.assertEqual(res['evaluated_dates'], 25)
        self.assertEqual(res['hac']['n'], 25)

    def test_actual_ledger_schema_without_status_or_version(self):
        records = [dict(date='2026-10-12', ticker='AAA', group='signal', pool='v2', ret=0.01),
                   dict(date='2026-10-12', ticker='BBB', group='control', pool='v2', ret=0.0),
                   dict(date='2026-10-13', ticker='AAA', group='signal', pool='v2', exit=None)]
        res = evaluate_h4(records)
        self.assertEqual(res['completed_signal'], 1)
        self.assertEqual(res['evaluated_signal'], 1)
        self.assertEqual(res['evaluated_dates'], 1)
        self.assertEqual(res['pending_signal'], 1)
        self.assertEqual(res['excluded']['date_counts']['pending_only'], 1)
        self.assertEqual(res['excluded']['date_counts']['partial_pending'], 0)

    def test_source_records_not_modified(self):
        base = build() + [pending(dates(40)[0], 'P001', 'signal'),
                          {'date': '2026-12-31', 'ticker': 'P002', 'group': 'control',
                           'pool': 'v2', 'ret': None}]
        snapshot = copy.deepcopy(base)
        evaluate_h4(base)
        self.assertEqual(base, snapshot)


if __name__ == '__main__':
    unittest.main()
