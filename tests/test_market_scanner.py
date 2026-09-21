"""
market_scanner.py 回归测试

覆盖范围：is_mover —— 纯逻辑判断，不依赖网络。
不覆盖 find_movers() 本身——联网请求编排层，超出"直接导入生产代码测
纯逻辑"这套测试哲学的适用范围（同run_scan()，见test_trading_agent.py）。
"""
from src.market_scanner import is_mover, universe, MOVER_PCT_THRESHOLD, MOVER_VOLUME_RATIO


class TestIsMover:
    def test_below_both_thresholds_not_a_mover(self):
        assert is_mover(pct_change=1.2, volume_ratio=1.1) is False

    def test_pct_change_alone_triggers(self):
        assert is_mover(pct_change=12.36, volume_ratio=1.0) is True

    def test_volume_ratio_alone_triggers(self):
        assert is_mover(pct_change=0.5, volume_ratio=3.0) is True

    def test_negative_pct_change_uses_absolute_value(self):
        # 大跌也是异动，不是只看大涨
        assert is_mover(pct_change=-8.0, volume_ratio=1.0) is True

    def test_exactly_at_threshold_counts(self):
        assert is_mover(pct_change=MOVER_PCT_THRESHOLD, volume_ratio=0.0) is True
        assert is_mover(pct_change=0.0, volume_ratio=MOVER_VOLUME_RATIO) is True

    def test_custom_thresholds_override_defaults(self):
        assert is_mover(pct_change=3.0, volume_ratio=1.0,
                         pct_threshold=2.5, volume_ratio_threshold=5.0) is True
        assert is_mover(pct_change=3.0, volume_ratio=1.0,
                         pct_threshold=10.0, volume_ratio_threshold=5.0) is False


class TestUniverse:
    def test_dedupes_tickers_across_sectors(self):
        # NVDA/AMD/AVGO 同时出现在 XLK 和 SMH，universe() 应该只保留一份
        u = universe()
        assert len(u) == len(set(u))
        assert u.count("NVDA") == 1

    def test_includes_meta_under_xlc(self):
        # 2026-09-21起 market_scanner 存在的直接原因：META要能被独立扫到
        assert "META" in universe()
