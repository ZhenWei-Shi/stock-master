"""Canonical acquisition identity; experiment metadata remains separate provenance."""
import datetime as dt
import hashlib
import json
import re

SOURCE_PREFIX = 'Yahoo Finance via yfinance'
TICKER = re.compile('[A-Z][A-Z0-9.-]{0,14}')
DATE = re.compile('[0-9]{4}-[0-9]{2}-[0-9]{2}')


def _date(v, name):
    if not isinstance(v, str) or not DATE.fullmatch(v):
        raise ValueError('invalid date: ' + name)
    dt.date.fromisoformat(v)
    return v


def _symbols(v, name):
    if not isinstance(v, list) or not v:
        raise ValueError('empty or non-list: ' + name)
    for s in v:
        if not isinstance(s, str) or not TICKER.fullmatch(s):
            raise ValueError('unsafe ticker in ' + name)
    if len(set(v)) != len(v):
        raise ValueError('duplicate ticker in ' + name)
    return sorted(v)


def data_request(cfg):
    src = cfg.get('source')
    if not isinstance(src, str) or not src.startswith(SOURCE_PREFIX):
        raise ValueError('unsupported source')
    symbols = _symbols(cfg.get('symbols'), 'symbols')
    bench = cfg.get('benchmark')
    _symbols([bench], 'benchmark')
    if bench in symbols:
        raise ValueError('benchmark must not be in symbols')
    start = _date(cfg.get('download_start'), 'download_start')
    end = _date(cfg.get('download_end_exclusive'), 'download_end_exclusive')
    if start >= end:
        raise ValueError('download_start must be before download_end_exclusive')
    policy = {'provider': 'yahoo_finance_yfinance', 'interval': '1d', 'auto_adjust': False,
              'actions': True, 'repair': False, 'field_policy': 'raw_ohlcv_adjclose_actions_v1'}
    for key, value in policy.items():
        if key in cfg and (type(cfg[key]) is not type(value) or cfg[key] != value):
            raise ValueError('unsupported acquisition policy: ' + key)
    return {'provider': 'yahoo_finance_yfinance', 'interval': '1d', 'auto_adjust': False,
            'actions': True, 'repair': False, 'field_policy': 'raw_ohlcv_adjclose_actions_v1',
            'symbols': sorted([*symbols, bench]), 'start': start, 'end': end}


def request_sha256(req):
    blob = json.dumps(req, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def assert_cache_compatible(cfg, manifest, *, legacy_verified=False):
    req = data_request(cfg)
    digest = request_sha256(req)
    md = manifest.get('metadata') if isinstance(manifest, dict) else None
    if not isinstance(md, dict):
        raise ValueError('manifest metadata missing')
    src = md.get('source')
    if not isinstance(src, str) or not src.startswith(SOURCE_PREFIX):
        raise ValueError('cache source mismatch')
    r = md.get('request')
    if not isinstance(r, dict) or _symbols(r.get('symbols'), 'metadata.request.symbols') != req['symbols']:
        raise ValueError('cache request symbols mismatch')
    if r.get('start') != req['start'] or r.get('end') != req['end']:
        raise ValueError('cache request dates mismatch')
    if md.get('requested_start') != req['start'] or md.get('requested_end') != req['end']:
        raise ValueError('cache requested_start/end mismatch')
    files = manifest.get('files')
    if not isinstance(files, list):
        raise ValueError('manifest files must be a list')
    seen = []
    for f in files:
        if not isinstance(f, dict):
            raise ValueError('bad manifest file entry')
        seen.append(f.get('symbol'))
        first = _date(f.get('first_date'), 'first_date')
        last = _date(f.get('last_date'), 'last_date')
        if not (req['start'] <= first <= last < req['end']):
            raise ValueError('file dates outside request window')
    if _symbols(seen, 'manifest.files') != req['symbols']:
        raise ValueError('manifest file symbols mismatch')
    has_req, has_sha = 'data_request' in md, 'data_request_sha256' in md
    if has_req and has_sha:
        if md['data_request'] != req or md['data_request_sha256'] != digest:
            raise ValueError('data_request identity mismatch')
        if request_sha256(md['data_request']) != md['data_request_sha256']:
            raise ValueError('data_request_sha256 is not the real hash')
        mode = 'explicit_data_identity_v1'
    elif not has_req and not has_sha:
        if legacy_verified is not True:
            raise ValueError('legacy cache requires externally verified registry')
        mode = 'legacy_verified_request_v1'
    else:
        raise ValueError('partial data identity metadata')
    return {'data_request_sha256': digest, 'compatibility_mode': mode}
