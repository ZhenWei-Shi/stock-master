import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from factor_lab.factors import FACTOR_NAMES, compute_factors

PERIODS = 300


def make_panel(symbols=('AAA', 'BBB', 'SPY'), periods=PERIODS, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2020-01-02', periods=periods)
    frames = []
    for k, symbol in enumerate(symbols):
        close = 50.0 * (k + 1) * np.cumprod(1.0 + rng.normal(0.0005, 0.01, periods))
        frames.append(pd.DataFrame({
            'date': dates, 'symbol': symbol, 'open': close, 'high': close * 1.01,
            'low': close * 0.99, 'close': close, 'adj_close': close * 0.97,
            'volume': rng.integers(1000, 5000, periods).astype('float64'),
            'dividend': 0.0, 'split': 0.0}))
    return pd.concat(frames, ignore_index=True)


def column(panel, symbol, name):
    part = panel[panel['symbol'] == symbol].sort_values('date')
    return part[name].to_numpy(dtype='float64')


def set_cell(panel, symbol, date, name, value):
    out = panel.copy()
    out.loc[(out['symbol'] == symbol) & (out['date'] == date), name] = value
    return out


class FactorTestCase(unittest.TestCase):
    def setUp(self):
        self.panel = make_panel()
        self.dates = pd.DatetimeIndex(self.panel['date'].unique()).sort_values()

    def value(self, out, i, factor, symbol='AAA'):
        hit = out[(out['date'] == self.dates[i]) & (out['symbol'] == symbol) & (out['factor'] == factor)]
        self.assertEqual(len(hit), 1)
        return float(hit['value'].iloc[0])

    def assertClose(self, actual, expected):
        self.assertTrue(np.isclose(actual, expected, rtol=1e-10, atol=1e-12), '%r != %r' % (actual, expected))

    def past(self, out, i):
        return out[out['date'] < self.dates[i]].reset_index(drop=True)


class SchemaTests(FactorTestCase):
    def test_long_output_has_four_rows_per_symbol_date(self):
        out = compute_factors(self.panel)
        self.assertEqual(list(out.columns), ['date', 'symbol', 'factor', 'value'])
        self.assertEqual(set(out['symbol']), {'AAA', 'BBB'})
        self.assertEqual(len(out), 2 * PERIODS * len(FACTOR_NAMES))
        self.assertEqual(set(out['factor']), set(FACTOR_NAMES))
        self.assertEqual(list(out['factor'].iloc[:4]), list(FACTOR_NAMES))
        self.assertTrue((out.groupby(['date', 'symbol']).size() == 4).all())
        self.assertFalse(out.duplicated(['date', 'symbol', 'factor']).any())
        self.assertEqual(out['value'].dtype, np.float64)

    def test_explicit_symbols(self):
        out = compute_factors(self.panel, symbols=['BBB', 'SPY'])
        self.assertEqual(set(out['symbol']), {'BBB', 'SPY'})

    def test_rejects_bad_symbol_arguments(self):
        for bad in ([], ['ZZZ'], ['AAA', 'AAA'], 'AAA', [1]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    compute_factors(self.panel, symbols=bad)

    def test_rejects_invalid_schema(self):
        cases = {
            'missing column': self.panel.drop(columns=['adj_close']),
            'empty': self.panel.iloc[0:0],
            'string dates': self.panel.assign(date=self.panel['date'].dt.strftime('%Y-%m-%d')),
            'nonpositive price': self.panel.assign(adj_close=0.0),
            'duplicate rows': pd.concat([self.panel, self.panel.iloc[[0]]], ignore_index=True),
        }
        for name, panel in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    compute_factors(panel)

    def test_does_not_mutate_input(self):
        before = self.panel.copy()
        compute_factors(self.panel)
        pd.testing.assert_frame_equal(self.panel, before)

    def test_sessions_outside_benchmark_calendar_rejected(self):
        drop = (self.panel['symbol'] == 'SPY') & (self.panel['date'] == self.dates[50])
        with self.assertRaises(ValueError):
            compute_factors(self.panel[~drop])


class DefinitionTests(FactorTestCase):
    def test_momentum_12_1_definition_and_warmup(self):
        out = compute_factors(self.panel)
        adj = column(self.panel, 'AAA', 'adj_close')
        self.assertTrue(np.isnan(self.value(out, 251, 'momentum_12_1')))
        self.assertClose(self.value(out, 252, 'momentum_12_1'), adj[231] / adj[0] - 1.0)
        self.assertClose(self.value(out, 299, 'momentum_12_1'), adj[278] / adj[47] - 1.0)

    def test_reversal_5_definition_and_warmup(self):
        out = compute_factors(self.panel)
        adj = column(self.panel, 'AAA', 'adj_close')
        self.assertTrue(np.isnan(self.value(out, 4, 'reversal_5')))
        for t in (5, 200):
            self.assertClose(self.value(out, t, 'reversal_5'), -(adj[t] / adj[t - 5] - 1.0))

    def test_low_vol_20_definition_and_warmup(self):
        out = compute_factors(self.panel)
        adj = column(self.panel, 'AAA', 'adj_close')
        ret = adj[1:] / adj[:-1] - 1.0
        self.assertTrue(np.isnan(self.value(out, 19, 'low_vol_20')))
        for t in (20, 150, 299):
            expected = -np.std(ret[t - 20:t], ddof=1) * np.sqrt(252.0)
            self.assertClose(self.value(out, t, 'low_vol_20'), expected)

    def test_signed_rvol_denominator_excludes_current_session(self):
        panel = set_cell(self.panel, 'AAA', self.dates[100], 'volume', 1e7)
        out = compute_factors(panel)
        adj = column(panel, 'AAA', 'adj_close')
        vol = column(panel, 'AAA', 'volume')
        self.assertTrue(np.isnan(self.value(out, 19, 'signed_rvol_20')))
        for t in (20, 100, 101):
            expected = np.sign(adj[t] / adj[t - 1] - 1.0) * (vol[t] / vol[t - 20:t].mean() - 1.0)
            self.assertClose(self.value(out, t, 'signed_rvol_20'), expected)
        self.assertGreater(abs(self.value(out, 100, 'signed_rvol_20')), 100.0)

    def test_zero_prior_volume_gives_nan(self):
        panel = self.panel.copy()
        window = (panel['symbol'] == 'AAA') & panel['date'].isin(self.dates[30:50])
        panel.loc[window, 'volume'] = 0.0
        out = compute_factors(panel)
        self.assertTrue(np.isnan(self.value(out, 50, 'signed_rvol_20')))
        self.assertFalse(np.isnan(self.value(out, 51, 'signed_rvol_20')))
        self.assertFalse(np.isnan(self.value(out, 50, 'reversal_5')))

    def test_split_embargo_covers_current_and_prior_20_sessions(self):
        panel = set_cell(self.panel, 'AAA', self.dates[150], 'split', 2.0)
        out = compute_factors(panel)
        for t in range(150, 171):
            self.assertTrue(np.isnan(self.value(out, t, 'signed_rvol_20')), t)
        for t in (149, 171):
            self.assertFalse(np.isnan(self.value(out, t, 'signed_rvol_20')), t)
        self.assertFalse(np.isnan(self.value(out, 160, 'reversal_5')))
        self.assertFalse(np.isnan(self.value(out, 160, 'low_vol_20')))


class TimingTests(FactorTestCase):
    def test_missing_sessions_stay_nan_and_windows_do_not_span_gaps(self):
        gaps = (100, 270)
        drop = (self.panel['symbol'] == 'AAA') & self.panel['date'].isin(self.dates[list(gaps)])
        out = compute_factors(self.panel[~drop])
        base = compute_factors(self.panel)
        adj = column(self.panel, 'AAA', 'adj_close')
        self.assertEqual(int((out['symbol'] == 'AAA').sum()), PERIODS * len(FACTOR_NAMES))
        for name in FACTOR_NAMES:
            for t in gaps:
                self.assertTrue(np.isnan(self.value(out, t, name)))
        self.assertClose(self.value(out, 104, 'reversal_5'), -(adj[104] / adj[99] - 1.0))
        self.assertTrue(np.isnan(self.value(out, 105, 'reversal_5')))
        self.assertTrue(np.isnan(self.value(out, 120, 'low_vol_20')))
        self.assertClose(self.value(out, 121, 'low_vol_20'), self.value(base, 121, 'low_vol_20'))
        self.assertTrue(np.isnan(self.value(out, 101, 'signed_rvol_20')))
        self.assertTrue(np.isnan(self.value(out, 120, 'signed_rvol_20')))
        self.assertClose(self.value(out, 121, 'signed_rvol_20'), self.value(base, 121, 'signed_rvol_20'))
        self.assertClose(self.value(out, 290, 'momentum_12_1'), self.value(base, 290, 'momentum_12_1'))
        self.assertTrue(np.isnan(self.value(out, 291, 'momentum_12_1')))
        other = out[out['symbol'] == 'BBB'].reset_index(drop=True)
        pd.testing.assert_frame_equal(other, base[base['symbol'] == 'BBB'].reset_index(drop=True))

    def test_future_rows_do_not_change_history(self):
        cutoff = self.dates[279]
        truncated = compute_factors(self.panel[self.panel['date'] <= cutoff])
        perturbed = self.panel.copy()
        future = perturbed['date'] > cutoff
        for name in ('open', 'high', 'low', 'close', 'adj_close'):
            perturbed.loc[future, name] *= 1.5
        perturbed.loc[future, 'volume'] *= 3.0
        perturbed.loc[future & (perturbed['symbol'] == 'AAA'), 'split'] = 2.0
        for panel in (self.panel, perturbed):
            out = compute_factors(panel)
            past = out[out['date'] <= cutoff].reset_index(drop=True)
            pd.testing.assert_frame_equal(past, truncated)

    def test_signal_at_t_uses_close_at_t_and_nothing_later(self):
        base = compute_factors(self.panel)
        adj = column(self.panel, 'AAA', 'adj_close')
        bumped = set_cell(self.panel, 'AAA', self.dates[200], 'adj_close', adj[200] * 1.1)
        out = compute_factors(bumped)
        self.assertClose(self.value(out, 200, 'reversal_5'), -(adj[200] * 1.1 / adj[195] - 1.0))
        self.assertNotAlmostEqual(self.value(out, 200, 'reversal_5'), self.value(base, 200, 'reversal_5'))
        pd.testing.assert_frame_equal(self.past(out, 200), self.past(base, 200))


if __name__ == '__main__':
    unittest.main()
