# 日频价格面板缓存：严格校验、按运行目录不可变写入、先校验 SHA256 再解析的读取、覆盖质量报告。
# 只处理调用方提供的公开行情面板；本模块不抓取任何数据。
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
MANIFEST_NAME = 'manifest.json'
REQUIRED_COLUMNS = ('date', 'symbol', 'open', 'high', 'low', 'close',
                    'adj_close', 'volume', 'dividend', 'split')
NUMERIC_COLUMNS = REQUIRED_COLUMNS[2:]
PRICE_COLUMNS = ('open', 'high', 'low', 'close', 'adj_close')
NONNEGATIVE_COLUMNS = ('volume', 'dividend', 'split')
SYMBOL_PATTERN = re.compile('[A-Z][A-Z0-9.-]{0,14}')
SHA256_PATTERN = re.compile('[0-9a-f]{64}')
OHLC_TOLERANCE = 1e-6
DATE_FORMAT = '%Y-%m-%d'
REQUIRED_METADATA = ('source', 'requested_start', 'requested_end',
                     'universe_status', 'limitations')
MANIFEST_KEYS = ('schema_version', 'captured_at_utc', 'metadata', 'files', 'row_count')
FILE_KEYS = ('symbol', 'path', 'sha256', 'rows', 'first_date', 'last_date')
# Windows 保留设备名不能作为文件名主干（如 CON.csv），这类 symbol 加 '_' 前缀
_RESERVED_STEMS = frozenset(['CON', 'PRN', 'AUX', 'NUL']
                            + ['COM%d' % i for i in range(1, 10)]
                            + ['LPT%d' % i for i in range(1, 10)])
QUALITY_LIMITATIONS = (
    'reference calendar = sessions observed for the benchmark; gaps in the benchmark itself are not detectable',
    'gaps are only checked between each symbol first and last observed date; missing history outside that span is not detectable',
    'cannot prove point-in-time universe membership, delisting coverage or freedom from survivorship bias',
    'split and dividend events are reported as supplied by the source and are not cross-checked against adj_close',
)


class CacheError(ValueError):
    '''面板校验或缓存完整性失败。'''


def _iso(value):
    return pd.Timestamp(value).strftime(DATE_FORMAT)


def _utc_now():
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).strftime('%Y-%m-%dT%H:%M:%SZ')


def _is_count(value):
    return type(value) is int and value >= 0


def _fail_rows(message, frame, mask):
    mask = np.asarray(mask, dtype=bool)
    sample = frame.loc[mask, ['date', 'symbol']].head(5).to_dict('records')
    raise CacheError('%s: %d row(s), sample=%r' % (message, int(mask.sum()), sample))


def validate_panel(frame):
    '''返回按 date,symbol 排序的规范化副本；任何非法值直接报错，不丢弃、不强转。'''
    if not isinstance(frame, pd.DataFrame):
        raise CacheError('panel must be a pandas DataFrame')
    if frame.columns.duplicated().any():
        raise CacheError('panel has duplicate column names')
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise CacheError('panel is missing required columns: %s' % missing)
    if len(frame) == 0:
        raise CacheError('panel is empty')
    out = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    out.index = pd.RangeIndex(len(out))

    nulls = out.isna().any(axis=1).to_numpy()
    if nulls.any():
        _fail_rows('null values are not allowed', out, nulls)

    # 日期：只接受 naive datetime64，且必须是午夜；带时区的输入拒绝而不是猜测交易时段
    dates = out['date']
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        raise CacheError('timezone-aware dates are rejected; supply naive session dates')
    if not pd.api.types.is_datetime64_dtype(dates.dtype):
        raise CacheError('date column must have naive datetime64 dtype, got %s' % dates.dtype)
    try:
        dates = dates.astype('datetime64[ns]')
    except (ValueError, OverflowError) as exc:
        raise CacheError('date column cannot be represented as datetime64[ns]: %s' % exc) from exc
    off_midnight = (dates != dates.dt.normalize()).to_numpy()
    if off_midnight.any():
        _fail_rows('dates must be midnight session dates', out, off_midnight)
    out['date'] = dates

    symbol_ok = np.array([isinstance(v, str) and SYMBOL_PATTERN.fullmatch(v) is not None
                          for v in out['symbol']], dtype=bool)
    if not symbol_ok.all():
        _fail_rows('symbols must match [A-Z][A-Z0-9.-]{0,14}', out, ~symbol_ok)
    out['symbol'] = out['symbol'].astype(object)

    for column in NUMERIC_COLUMNS:
        dtype = out[column].dtype
        if (pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_complex_dtype(dtype)
                or not pd.api.types.is_numeric_dtype(dtype)):
            raise CacheError('column %r must have a real numeric dtype, got %s' % (column, dtype))
        out[column] = out[column].to_numpy(dtype='float64')

    values = out.loc[:, list(NUMERIC_COLUMNS)].to_numpy(dtype='float64')
    nonfinite = ~np.isfinite(values).all(axis=1)
    if nonfinite.any():
        _fail_rows('numeric values must be finite', out, nonfinite)
    nonpositive = (out.loc[:, list(PRICE_COLUMNS)].to_numpy() <= 0).any(axis=1)
    if nonpositive.any():
        _fail_rows('open/high/low/close/adj_close must be strictly positive', out, nonpositive)
    negative = (out.loc[:, list(NONNEGATIVE_COLUMNS)].to_numpy() < 0).any(axis=1)
    if negative.any():
        _fail_rows('volume/dividend/split must be non-negative', out, negative)

    op, hi, lo, cl = (out[name].to_numpy() for name in ('open', 'high', 'low', 'close'))
    bad_high = hi < np.maximum(np.maximum(op, cl), lo) - OHLC_TOLERANCE
    bad_low = lo > np.minimum(np.minimum(op, cl), hi) + OHLC_TOLERANCE
    if (bad_high | bad_low).any():
        _fail_rows('inconsistent OHLC range', out, bad_high | bad_low)

    duplicated = out.duplicated(['date', 'symbol'], keep=False).to_numpy()
    if duplicated.any():
        _fail_rows('duplicate (date, symbol) rows', out, duplicated)
    return out.sort_values(['date', 'symbol'], kind='mergesort').reset_index(drop=True)


def _check_metadata(metadata):
    if not isinstance(metadata, dict):
        raise CacheError('metadata must be a dict')
    missing = [k for k in REQUIRED_METADATA if k not in metadata]
    if missing:
        raise CacheError('metadata is missing keys: %s' % missing)
    try:
        frozen = json.loads(json.dumps(metadata, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise CacheError('metadata must be JSON-serialisable: %s' % exc) from exc
    empty = [k for k in REQUIRED_METADATA if frozen[k] in (None, '', [], {})]
    if empty:
        raise CacheError('metadata values must be non-empty: %s' % empty)
    return frozen


def _file_name(symbol):
    stem = symbol.split('.')[0]
    return ('_' + symbol if stem in _RESERVED_STEMS else symbol) + '.csv'


def _serialize(part):
    body = part.loc[:, list(REQUIRED_COLUMNS)].copy()
    body['date'] = body['date'].dt.strftime(DATE_FORMAT)
    return body.to_csv(index=False, lineterminator='\n').encode('utf-8')


def _atomic_write(path, data):
    # 先写同目录临时文件并 fsync，再 os.replace；失败时删除临时文件
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix='.tmp-', suffix='.part')
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_cache_locked(frame, root, metadata):
    '''校验后按 symbol 写 CSV，最后原子写 manifest；manifest 已存在则拒绝。'''
    meta = _check_metadata(metadata)
    panel = validate_panel(frame)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST_NAME
    if manifest_path.exists():
        raise FileExistsError('cache is immutable, manifest already exists: %s' % manifest_path)
    entries = []
    written = []
    try:
        for symbol, part in panel.groupby('symbol', sort=True):
            name = _file_name(symbol)
            data = _serialize(part)
            target = root / name
            _atomic_write(target, data)
            written.append(target)
            entries.append({'symbol': symbol, 'path': name,
                            'sha256': hashlib.sha256(data).hexdigest(),
                            'rows': int(len(part)),
                            'first_date': _iso(part['date'].min()),
                            'last_date': _iso(part['date'].max())})
        manifest = {'schema_version': SCHEMA_VERSION,
                    'captured_at_utc': _utc_now(),
                    'metadata': meta,
                    'files': entries,
                    'row_count': int(len(panel))}
        payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode('utf-8')
        if manifest_path.exists():
            raise FileExistsError('cache was finalised concurrently: %s' % manifest_path)
        _atomic_write(manifest_path, payload)
    except BaseException:
        # 失败时不留 manifest，清理本次写入的 CSV；若并发写者已定稿则不动其文件
        if not manifest_path.exists():
            for path in written:
                try:
                    path.unlink()
                except OSError:
                    pass
        raise
    return copy.deepcopy(manifest)


def write_cache(frame, root, metadata):
    """Publish once, with an exclusive writer lock; readers never acquire it."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.writer.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise CacheError('cache writer lock exists; inspect previous writer before retry') from exc
    try:
        os.close(fd)
        return _write_cache_locked(frame, root, metadata)
    finally:
        lock.unlink()


def _check_manifest(manifest):
    if not isinstance(manifest, dict):
        raise CacheError('manifest must be a JSON object')
    missing = [k for k in MANIFEST_KEYS if k not in manifest]
    if missing:
        raise CacheError('manifest is missing keys: %s' % missing)
    version = manifest['schema_version']
    if type(version) is not int or version != SCHEMA_VERSION:
        raise CacheError('unsupported manifest schema_version: %r' % (version,))
    if not isinstance(manifest['captured_at_utc'], str) or not manifest['captured_at_utc']:
        raise CacheError('manifest captured_at_utc must be a non-empty string')
    _check_metadata(manifest['metadata'])
    files = manifest['files']
    if not isinstance(files, list) or not files:
        raise CacheError('manifest files must be a non-empty list')
    if not _is_count(manifest['row_count']):
        raise CacheError('manifest row_count must be a non-negative integer')
    symbols, paths = set(), set()
    for entry in files:
        if not isinstance(entry, dict) or any(k not in entry for k in FILE_KEYS):
            raise CacheError('malformed manifest file entry: %r' % (entry,))
        symbol, path = entry['symbol'], entry['path']
        if not isinstance(symbol, str) or SYMBOL_PATTERN.fullmatch(symbol) is None:
            raise CacheError('invalid symbol in manifest: %r' % (symbol,))
        if not isinstance(path, str) or not path:
            raise CacheError('invalid path in manifest for %s' % symbol)
        if symbol in symbols or path.lower() in paths:
            raise CacheError('duplicate symbol or path in manifest: %s' % symbol)
        symbols.add(symbol)
        paths.add(path.lower())
        sha = entry['sha256']
        if not isinstance(sha, str) or SHA256_PATTERN.fullmatch(sha) is None:
            raise CacheError('invalid sha256 in manifest for %s' % symbol)
        if not _is_count(entry['rows']) or entry['rows'] == 0:
            raise CacheError('invalid rows in manifest for %s' % symbol)
        for key in ('first_date', 'last_date'):
            try:
                _dt.datetime.strptime(entry[key], DATE_FORMAT)
            except (TypeError, ValueError) as exc:
                raise CacheError('manifest %s for %s is not YYYY-MM-DD' % (key, symbol)) from exc
    if sum(e['rows'] for e in files) != manifest['row_count']:
        raise CacheError('manifest row_count does not equal the sum of file rows')


def _resolve_data_path(root, relative):
    # 只允许 root 下的 POSIX 相对路径：禁止绝对路径、盘符、反斜杠、. 与 ..
    if '\\' in relative or ':' in relative:
        raise CacheError('unsafe data path in manifest: %r' % relative)
    pieces = relative.split('/')
    if PurePosixPath(relative).is_absolute() or any(p in ('', '.', '..') for p in pieces):
        raise CacheError('unsafe data path in manifest: %r' % relative)
    candidate = root.joinpath(*pieces).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CacheError('data path escapes cache root: %r' % relative) from exc
    if not candidate.is_file():
        raise CacheError('data file missing: %r' % relative)
    return candidate


def _parse_csv(data, label):
    # na_filter=False：空字段不会变成 NaN，而是让列变为 object 后被 validate_panel 拒绝
    try:
        frame = pd.read_csv(io.BytesIO(data), dtype={'date': str, 'symbol': str},
                            na_filter=False, float_precision='round_trip', encoding='utf-8')
    except ValueError as exc:
        raise CacheError('cannot parse %r: %s' % (label, exc)) from exc
    if list(frame.columns) != list(REQUIRED_COLUMNS):
        raise CacheError('%r has unexpected columns %s' % (label, list(frame.columns)))
    try:
        frame['date'] = pd.to_datetime(frame['date'], format=DATE_FORMAT)
    except (TypeError, ValueError) as exc:
        raise CacheError('%r has invalid dates: %s' % (label, exc)) from exc
    return frame


def _check_part(part, entry):
    try:
        checked = validate_panel(part)
    except CacheError as exc:
        raise CacheError('%r failed validation: %s' % (entry['path'], exc)) from exc
    found = set(checked['symbol'])
    if found != {entry['symbol']}:
        raise CacheError('%r contains symbols %s, expected %s' % (entry['path'], sorted(found), entry['symbol']))
    if len(checked) != entry['rows']:
        raise CacheError('%r has %d rows, manifest says %d' % (entry['path'], len(checked), entry['rows']))
    span = (_iso(checked['date'].iloc[0]), _iso(checked['date'].iloc[-1]))
    if span != (entry['first_date'], entry['last_date']):
        raise CacheError('%r date span %s does not match manifest' % (entry['path'], span))
    return checked


def _reject_constant(name):
    raise ValueError('non-finite JSON constant %s' % name)


def read_cache(root):
    '''读取并完整校验缓存，返回 (panel, manifest)；不写任何文件。'''
    root = Path(root).resolve()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise CacheError('manifest not found: %s' % manifest_path)
    try:
        manifest = json.loads(manifest_path.read_bytes().decode('utf-8'),
                              parse_constant=_reject_constant)
    except (OSError, ValueError) as exc:
        raise CacheError('manifest is not readable UTF-8 JSON: %s' % exc) from exc
    _check_manifest(manifest)
    parts = []
    for entry in manifest['files']:
        path = _resolve_data_path(root, entry['path'])
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise CacheError('cannot read %r: %s' % (entry['path'], exc)) from exc
        # 先校验哈希，再解析同一份字节，避免检查与使用之间文件被替换
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise CacheError('sha256 mismatch for %r' % entry['path'])
        parts.append(_check_part(_parse_csv(data, entry['path']), entry))
    panel = validate_panel(pd.concat(parts, ignore_index=True))
    if len(panel) != manifest['row_count']:
        raise CacheError('row_count mismatch: %d != %d' % (len(panel), manifest['row_count']))
    return panel, manifest


def quality_report(frame, benchmark='SPY'):
    '''以基准实际交易日为参照，只在每个标的首末观测之间检查缺失；不做任何填充。'''
    panel = validate_panel(frame)
    if benchmark not in set(panel['symbol']):
        raise CacheError('benchmark %r not present in panel' % (benchmark,))
    reference = pd.DatetimeIndex(panel.loc[panel['symbol'] == benchmark, 'date']).unique().sort_values()
    per_symbol = {}
    for symbol, part in panel.groupby('symbol', sort=True):
        dates = pd.DatetimeIndex(part['date'])
        start, end = dates.min(), dates.max()
        span = reference[(reference >= start) & (reference <= end)]
        missing = span.difference(dates)
        extra = dates.difference(reference)
        splits = part.loc[part['split'] > 0, 'date']
        dividends = part.loc[part['dividend'] > 0, 'date']
        per_symbol[symbol] = {
            'rows': int(len(part)),
            'start': _iso(start),
            'end': _iso(end),
            'missing_reference_sessions_within_listing_span': int(len(missing)),
            'missing_dates': [_iso(d) for d in missing],
            'sessions_not_in_reference': [_iso(d) for d in extra],
            'split_events': int(len(splits)),
            'split_dates': [_iso(d) for d in splits],
            'dividend_events': int(len(dividends)),
            'dividend_dates': [_iso(d) for d in dividends],
        }
    return {
        'benchmark': benchmark,
        'symbols': sorted(per_symbol),
        'rows': int(len(panel)),
        'first_date': _iso(panel['date'].min()),
        'last_date': _iso(panel['date'].max()),
        'reference_sessions': int(len(reference)),
        'per_symbol': per_symbol,
        'limitations': list(QUALITY_LIMITATIONS),
    }
