import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from factor_lab.cache import (MANIFEST_NAME, CacheError, quality_report, read_cache,
                              validate_panel, write_cache)


def make_panel(symbols=('AAA', 'SPY'), periods=10):
    dates = pd.bdate_range('2024-01-02', periods=periods)
    frames = []
    for k, symbol in enumerate(symbols):
        close = 100.0 + 10.0 * k + 0.5 * np.arange(periods)
        frames.append(pd.DataFrame({
            'date': dates, 'symbol': symbol, 'open': close - 0.25, 'high': close + 1.0,
            'low': close - 1.0, 'close': close, 'adj_close': close * 0.9,
            'volume': 1000.0 + np.arange(periods), 'dividend': 0.0, 'split': 0.0}))
    return pd.concat(frames, ignore_index=True)


def make_metadata(**overrides):
    meta = {'source': 'synthetic unit-test panel', 'requested_start': '2024-01-02',
            'requested_end': '2024-01-15', 'universe_status': 'fixed synthetic list',
            'limitations': ['synthetic data; no point-in-time universe']}
    meta.update(overrides)
    return meta


def with_cell(column, value, row=3):
    def mutate(frame):
        frame = frame.copy()
        frame.loc[row, column] = value
        return frame
    return mutate


# 第 3 行是 AAA：close=101.5, open=101.25, high=102.5, low=100.5
INVALID_CASES = [
    ('null close', with_cell('close', np.nan)),
    ('zero adj_close', with_cell('adj_close', 0.0)),
    ('negative open', with_cell('open', -1.0)),
    ('infinite volume', with_cell('volume', np.inf)),
    ('high below close', with_cell('high', 101.0)),
    ('low above open', with_cell('low', 101.3)),
    ('negative volume', with_cell('volume', -5.0)),
    ('negative dividend', with_cell('dividend', -0.1)),
    ('negative split', with_cell('split', -1.0)),
    ('lowercase symbol', with_cell('symbol', 'aaa')),
    ('illegal symbol char', with_cell('symbol', 'AA$')),
    ('symbol too long', with_cell('symbol', 'A' * 16)),
    ('missing symbol', with_cell('symbol', None)),
    ('intraday timestamp', lambda f: f.assign(date=f['date'] + pd.Timedelta(hours=16))),
    ('string dates', lambda f: f.assign(date=f['date'].dt.strftime('%Y-%m-%d'))),
    ('tz-aware dates', lambda f: f.assign(date=f['date'].dt.tz_localize('UTC'))),
    ('object volume', lambda f: f.assign(volume=f['volume'].astype(str))),
    ('bool split', lambda f: f.assign(split=False)),
    ('duplicate row', lambda f: pd.concat([f, f.iloc[[0]]], ignore_index=True)),
    ('missing column', lambda f: f.drop(columns=['split'])),
    ('empty', lambda f: f.iloc[0:0]),
]


class ValidatePanelTests(unittest.TestCase):
    def test_concurrent_writer_lock_is_not_removed_or_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / '.writer.lock'
            lock.write_text('existing writer')
            with self.assertRaises(CacheError):
                write_cache(make_panel(), root, make_metadata())
            self.assertEqual(lock.read_text(), 'existing writer')
            self.assertFalse((root / MANIFEST_NAME).exists())

    def test_returns_sorted_normalized_copy_without_mutation(self):
        panel = make_panel().iloc[::-1].reset_index(drop=True)
        before = panel.copy()
        out = validate_panel(panel)
        pd.testing.assert_frame_equal(panel, before)
        resorted = out.sort_values(['date', 'symbol']).reset_index(drop=True)
        pd.testing.assert_frame_equal(out, resorted)
        self.assertEqual(list(out.columns), list(before.columns))
        self.assertEqual(out['volume'].dtype, np.float64)

    def test_rejects_invalid_inputs(self):
        for name, mutate in INVALID_CASES:
            with self.subTest(name):
                with self.assertRaises(CacheError):
                    validate_panel(mutate(make_panel()))

    def test_accepts_raw_prices_with_divergent_adj_close_within_tolerance(self):
        panel = make_panel()
        panel['adj_close'] = panel['close'] * 0.25
        panel.loc[3, 'high'] = panel.loc[3, 'close'] - 5e-7
        out = validate_panel(panel)
        self.assertEqual(len(out), len(panel))


class CacheIoTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.root = self.base / 'run-001'
        self.panel = make_panel()

    def snapshot(self):
        return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in self.root.iterdir()}

    def rewrite_manifest(self, edit):
        path = self.root / MANIFEST_NAME
        manifest = json.loads(path.read_text(encoding='utf-8'))
        edit(manifest)
        path.write_text(json.dumps(manifest), encoding='utf-8')

    def test_round_trip_is_exact_and_manifest_complete(self):
        manifest = write_cache(self.panel, self.root, make_metadata())
        panel, loaded = read_cache(self.root)
        pd.testing.assert_frame_equal(panel, validate_panel(self.panel), check_exact=True)
        self.assertEqual(loaded, manifest)
        self.assertEqual(manifest['schema_version'], 1)
        self.assertEqual(manifest['row_count'], len(self.panel))
        self.assertEqual(manifest['metadata'], make_metadata())
        self.assertEqual([f['symbol'] for f in manifest['files']], ['AAA', 'SPY'])
        for entry in manifest['files']:
            data = (self.root / entry['path']).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry['sha256'])
            self.assertEqual(entry['rows'], 10)
            self.assertEqual((entry['first_date'], entry['last_date']), ('2024-01-02', '2024-01-15'))
        self.assertEqual({p.name for p in self.root.iterdir()}, {'AAA.csv', 'SPY.csv', MANIFEST_NAME})

    def test_existing_manifest_is_never_overwritten(self):
        write_cache(self.panel, self.root, make_metadata())
        before = self.snapshot()
        manifest_bytes = (self.root / MANIFEST_NAME).read_bytes()
        with self.assertRaises(FileExistsError):
            write_cache(make_panel(periods=5), self.root, make_metadata())
        self.assertEqual((self.root / MANIFEST_NAME).read_bytes(), manifest_bytes)
        self.assertEqual(self.snapshot(), before)
        read_cache(self.root)

    def test_invalid_inputs_leave_no_manifest(self):
        no_limitations = make_metadata()
        del no_limitations['limitations']
        cases = [
            ('missing limitations', self.panel, no_limitations),
            ('empty source', self.panel, make_metadata(source='')),
            ('nan in metadata', self.panel, make_metadata(extra=float('nan'))),
            ('invalid panel', INVALID_CASES[0][1](make_panel()), make_metadata()),
        ]
        for name, panel, meta in cases:
            with self.subTest(name):
                with self.assertRaises(CacheError):
                    write_cache(panel, self.root, meta)
                self.assertFalse((self.root / MANIFEST_NAME).exists())

    def test_tampered_or_missing_files_fail(self):
        write_cache(self.panel, self.root, make_metadata())
        aaa = self.root / 'AAA.csv'
        original = aaa.read_bytes()
        tampered = original.replace(b'1000.0', b'1001.0', 1)
        self.assertNotEqual(tampered, original)
        aaa.write_bytes(tampered)
        with self.assertRaises(CacheError):
            read_cache(self.root)
        aaa.write_bytes(original)
        read_cache(self.root)
        (self.root / 'SPY.csv').unlink()
        with self.assertRaises(CacheError):
            read_cache(self.root)

    def test_manifest_paths_must_stay_under_root(self):
        write_cache(self.panel, self.root, make_metadata())
        outside = self.base / 'outside.csv'
        shutil.copyfile(self.root / 'AAA.csv', outside)
        for bad in ('../outside.csv', str(outside), '..\\outside.csv', './AAA.csv'):
            with self.subTest(bad):
                self.rewrite_manifest(lambda m: m['files'][0].__setitem__('path', bad))
                with self.assertRaises(CacheError):
                    read_cache(self.root)

    def test_manifest_counts_and_schema_must_match(self):
        write_cache(self.panel, self.root, make_metadata())

        def bump_rows(manifest):
            manifest['files'][0]['rows'] += 1
            manifest['row_count'] += 1

        self.rewrite_manifest(bump_rows)
        with self.assertRaises(CacheError):
            read_cache(self.root)
        self.rewrite_manifest(lambda m: m.__setitem__('schema_version', 2))
        with self.assertRaises(CacheError):
            read_cache(self.root)

    def test_missing_manifest_fails(self):
        self.root.mkdir()
        with self.assertRaises(CacheError):
            read_cache(self.root)

    def test_read_has_no_file_side_effects(self):
        write_cache(self.panel, self.root, make_metadata())
        before = self.snapshot()
        read_cache(self.root)
        self.assertEqual(self.snapshot(), before)


class QualityReportTests(unittest.TestCase):
    def test_missing_sessions_only_within_observed_span(self):
        panel = make_panel()
        dates = pd.DatetimeIndex(panel['date'].unique()).sort_values()
        aaa = panel['symbol'] == 'AAA'
        drop = aaa & panel['date'].isin([dates[0], dates[1], dates[5], dates[9]])
        panel = panel[~drop].reset_index(drop=True)
        aaa = panel['symbol'] == 'AAA'
        panel.loc[aaa & (panel['date'] == dates[3]), 'split'] = 2.0
        panel.loc[aaa & (panel['date'] == dates[4]), 'dividend'] = 0.5
        report = quality_report(panel)
        entry = report['per_symbol']['AAA']
        self.assertEqual(entry['rows'], 6)
        self.assertEqual((entry['start'], entry['end']), ('2024-01-04', '2024-01-12'))
        self.assertEqual(entry['missing_reference_sessions_within_listing_span'], 1)
        self.assertEqual(entry['missing_dates'], ['2024-01-09'])
        self.assertEqual(entry['split_events'], 1)
        self.assertEqual(entry['split_dates'], ['2024-01-05'])
        self.assertEqual(entry['dividend_events'], 1)
        self.assertEqual(report['per_symbol']['SPY']['missing_reference_sessions_within_listing_span'], 0)
        self.assertEqual(report['symbols'], ['AAA', 'SPY'])
        self.assertEqual(report['rows'], 16)
        self.assertEqual((report['first_date'], report['last_date']), ('2024-01-02', '2024-01-15'))
        self.assertTrue(report['limitations'])

    def test_requires_benchmark(self):
        with self.assertRaises(CacheError):
            quality_report(make_panel(symbols=('AAA',)))
        report = quality_report(make_panel(symbols=('AAA', 'QQQ')), benchmark='QQQ')
        self.assertEqual(report['benchmark'], 'QQQ')


if __name__ == '__main__':
    unittest.main()
