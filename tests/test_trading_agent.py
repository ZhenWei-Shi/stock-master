"""
trading_agent.py 回归测试

覆盖范围：_apply_conditional_sizing —— CONDITIONAL 信号减半仓位逻辑
（2026-08-13新增，修复debate.py"建议缩小仓位50%"从未真正影响下单量的问题）。

不覆盖 run_scan()/run_monitor() 本身——这两个是联网请求的编排层，
超出"直接导入生产代码测纯逻辑"这套测试哲学的适用范围。
"""
from src.trading_agent import _apply_conditional_sizing


class TestConditionalSizing:
    def test_conditional_halves_shares(self):
        assert _apply_conditional_sizing(4, "CONDITIONAL") == 2

    def test_go_keeps_full_shares(self):
        assert _apply_conditional_sizing(4, "GO") == 4

    def test_odd_shares_floor_division(self):
        # 3股减半后floor到1股，不是1.5股
        assert _apply_conditional_sizing(3, "CONDITIONAL") == 1

    def test_single_share_conditional_rounds_to_zero(self):
        # 1股减半后是0——调用方需要据此跳过整笔交易，而不是强制开1股
        assert _apply_conditional_sizing(1, "CONDITIONAL") == 0

    def test_none_conclusion_keeps_full_shares(self):
        # debate报错/未运行时 debate_conclusion 可能是 None，不应误判成CONDITIONAL
        assert _apply_conditional_sizing(4, None) == 4

    def test_wait_conclusion_keeps_full_shares(self):
        # WAIT不会进入go_signals（run_scan里已过滤），但函数本身对未知值要保守放行
        assert _apply_conditional_sizing(4, "WAIT") == 4
