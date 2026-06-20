"""
数学逻辑回归测试

所有公式函数直接从生产代码导入，而非内联副本。
若生产代码公式被改错，测试会真正失败（而非假绿）。

覆盖：
  - calc_surprise_pct (src/scraper.py)       P0回归：epsDifference×100错误
  - is_accelerating   (src/sector_rotation.py) P1回归：accel加速度逻辑
  - Kelly公式内联记录（debate.py/paper_trading.py三处相同公式，用内联+注释标记）
"""
import pytest
from src.scraper import calc_surprise_pct
from src.sector_rotation import is_accelerating


# ─────────────────────────────────────────────────────────────────────────────
# Kelly Criterion
# 公式在 debate.py:422 / paper_trading.py:548 / backtest.py:541 三处完全相同
# 因跨文件内联（非独立函数），此处用内联副本记录预期行为；
# 若需重构，提取为 src/utils.py::kelly() 后改为导入。
# ─────────────────────────────────────────────────────────────────────────────

def _kelly(W: float, R: float) -> float:
    """Kelly% = W - (1-W)/R；与三处生产代码完全一致。"""
    if R <= 0:
        return 0.0
    return W - (1 - W) / R


class TestKelly:
    def test_positive_edge(self):
        # 53%胜率、2:1盈亏比 → Kelly = 0.53 - 0.47/2 = 0.295
        k = _kelly(0.53, 2.0)
        assert k == pytest.approx(0.295)
        assert k / 2 < 0.25  # 半Kelly不超过25%

    def test_negative_edge(self):
        # 40%胜率、1:1盈亏比 → 期望为负，Kelly<0
        assert _kelly(0.40, 1.0) < 0

    def test_break_even(self):
        # 50%胜率、1:1盈亏比 → 期望为零
        assert _kelly(0.50, 1.0) == pytest.approx(0.0)

    def test_zero_rr_ratio_returns_zero(self):
        # rr_ratio=0 触发分母保护，返回0
        assert _kelly(0.60, 0.0) == 0.0

    def test_half_kelly_clips_negative(self):
        # max(0, kelly_f/2)：负期望时半Kelly必须为0，不能为负
        negative_k = _kelly(0.30, 0.5)  # W=0.30, R=0.5 → 负期望
        assert negative_k < 0           # 确认全Kelly为负
        assert max(0, negative_k / 2) == 0.0  # 半Kelly截断为0

    def test_high_win_rate_high_rr(self):
        # 70%胜率、3:1盈亏比 → Kelly = 0.70 - 0.30/3 = 0.60
        assert _kelly(0.70, 3.0) == pytest.approx(0.60)


# ─────────────────────────────────────────────────────────────────────────────
# calc_surprise_pct (src/scraper.py) —— 直接导入，测试真实生产代码
# ─────────────────────────────────────────────────────────────────────────────

class TestSurprisePct:
    def test_beat_10pct(self):
        assert calc_surprise_pct(1.10, 1.00) == pytest.approx(10.0)

    def test_miss_10pct(self):
        assert calc_surprise_pct(0.90, 1.00) == pytest.approx(-10.0)

    def test_negative_estimate_beat(self):
        # est=-0.10，act=-0.05（少亏），超预期 +50%
        assert calc_surprise_pct(-0.05, -0.10) == pytest.approx(50.0)

    def test_negative_estimate_miss(self):
        # est=-0.10，act=-0.15（多亏），低于预期 -50%
        assert calc_surprise_pct(-0.15, -0.10) == pytest.approx(-50.0)

    def test_zero_estimate_returns_none(self):
        assert calc_surprise_pct(0.10, 0.00) is None

    def test_exact_match_is_zero(self):
        assert calc_surprise_pct(1.00, 1.00) == pytest.approx(0.0)

    def test_regression_not_epsDifference_times_100(self):
        """
        P0回归：旧代码 surprise_pct = epsDifference * 100（epsDifference是美元差，不是%）。
        est=0.10，act=0.11（beat $0.01）：
          旧：0.01 × 100 = 1.0  ← 错误（无量纲）
          新：0.01 / 0.10 × 100 = 10.0%  ← 正确
        此测试导入真实 scraper.py::calc_surprise_pct，若代码回退会真正失败。
        """
        eps_act, eps_est = 0.11, 0.10
        result = calc_surprise_pct(eps_act, eps_est)
        old_bug_result = round((eps_act - eps_est) * 100, 1)  # 旧公式

        assert result == pytest.approx(10.0)       # 正确结果
        assert old_bug_result == pytest.approx(1.0)  # 旧 bug 结果
        assert result != pytest.approx(old_bug_result)  # 两者不同

    def test_regression_large_eps(self):
        """
        大盘股 EPS 时旧公式误差更严重：
        est=5.00，act=5.25（beat +5%）：
          旧：0.25 × 100 = 25.0  ← 严重虚报（放大5倍）
          新：0.25 / 5.00 × 100 = 5.0%  ← 正确
        """
        result = calc_surprise_pct(5.25, 5.00)
        old_bug = round((5.25 - 5.00) * 100, 1)

        assert result == pytest.approx(5.0)
        assert old_bug == pytest.approx(25.0)
        assert result != pytest.approx(old_bug)


# ─────────────────────────────────────────────────────────────────────────────
# is_accelerating (src/sector_rotation.py) —— 直接导入，测试真实生产代码
#
# 技术说明：`prior_2m_avg = (rs_3m - rs_1m)/2; rs_1m > prior_2m_avg`
# 与原始公式 `rs_1m > rs_3m/3` 代数等价（两者恒产生相同结果）。
# 提取为函数的价值在于：测试可导入真实代码，而非内联副本。
# ─────────────────────────────────────────────────────────────────────────────

class TestAccel:
    def test_genuine_acceleration(self):
        # 近3月=3%，近1月=4%（前2月均值=-0.5%），本月明显高于均值
        assert is_accelerating(rs_1m=4.0, rs_3m=3.0) is True

    def test_deceleration(self):
        # 近3月=6%，近1月=0.5%（前2月均值=2.75%），本月低于均值
        assert is_accelerating(rs_1m=0.5, rs_3m=6.0) is False

    def test_negative_this_month_always_false(self):
        # 本月为负，rs_1m > 0 条件不满足，无论前几个月多好
        assert is_accelerating(rs_1m=-1.0, rs_3m=5.0) is False
        assert is_accelerating(rs_1m=-0.01, rs_3m=-5.0) is False

    def test_momentum_turning_positive(self):
        # 3月=-3%，1月=+1%（前2月均值=-2%），动能转正
        assert is_accelerating(rs_1m=1.0, rs_3m=-3.0) is True

    def test_deceleration_strong_trend(self):
        # 近3月=9%（强势），近1月=1%（大幅减速）
        assert is_accelerating(rs_1m=1.0, rs_3m=9.0) is False

    def test_all_zero_is_false(self):
        # rs_1m=0 不满足 rs_1m > 0
        assert is_accelerating(rs_1m=0.0, rs_3m=0.0) is False
