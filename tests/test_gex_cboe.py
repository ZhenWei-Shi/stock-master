"""
gex_cboe.py 测试（用构造的CBOE格式数据，不联网）。
"""
from datetime import date

import pytest

from src.gex_cboe import parse_occ, compute_gex


class TestParseOcc:
    def test_parses_call(self):
        assert parse_occ("ASTS261002C00035000") == (date(2026, 10, 2), "C", 35.0)

    def test_parses_fractional_put(self):
        assert parse_occ("SPY261016P00512500") == (date(2026, 10, 16), "P", 512.5)

    def test_rejects_garbage(self):
        assert parse_occ("not-an-option") is None


def _opt(sym, oi, gamma, iv=0.7):
    return {"option": sym, "open_interest": oi, "gamma": gamma, "iv": iv}


def _data(options, spot=61.0):
    return {"current_price": spot, "iv30": 77.3, "options": options}


class TestComputeGex:
    ASOF = date(2026, 9, 29)

    def test_walls_and_sign_convention(self):
        d = _data([
            _opt("ASTS261016C00065000", 5000, 0.05),   # 大量Call → 正伽马墙
            _opt("ASTS261016P00055000", 6000, 0.04),   # 大量Put → 负伽马
            _opt("ASTS261016C00062000", 100, 0.08),
        ])
        r = compute_gex(d, asof=self.ASOF)
        assert r["call_wall"] == 65.0 and r["put_wall"] == 55.0
        assert r["gex_by_strike"][65.0] > 0 and r["gex_by_strike"][55.0] < 0
        assert r["source"] == "cboe" and r["iv30"] == 77.3

    def test_per_1pct_units(self):
        # 1000张 × gamma 0.1 × 100股 × 61² × 1% = 372,100美元 → 0.37（百万）
        r = compute_gex(_data([_opt("ASTS261016C00061000", 1000, 0.1)]), asof=self.ASOF)
        assert r["gex_by_strike"][61.0] == pytest.approx(0.37, abs=0.01)

    def test_excludes_expired_and_far_dated(self):
        d = _data([
            _opt("ASTS260925C00065000", 9999, 0.05),   # 已过期
            _opt("ASTS270115C00065000", 9999, 0.05),   # 超过45天
            _opt("ASTS261016C00063000", 10, 0.05),
        ])
        r = compute_gex(d, asof=self.ASOF)
        assert r["call_wall"] == 63.0

    def test_flip_between_put_heavy_below_and_call_heavy_above(self):
        d = _data([
            _opt("ASTS261016P00055000", 8000, 0.03, iv=0.6),
            _opt("ASTS261016C00067000", 8000, 0.03, iv=0.6),
        ])
        r = compute_gex(d, asof=self.ASOF)
        assert r["flip_strike"] is not None and 55.0 < r["flip_strike"] < 67.0

    def test_no_valid_options(self):
        assert "error" in compute_gex(_data([]), asof=self.ASOF)

    def test_no_spot(self):
        assert "error" in compute_gex({"options": []}, asof=self.ASOF)
