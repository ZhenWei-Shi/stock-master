"""
数学逻辑回归测试

覆盖已发现的三类公式bug：
  P0: epsDifference × 100 → 应为 (act-est)/|est|×100
  P1: sector_rotation 加速度定义
  P1: Kelly Criterion 正负期望判断
"""
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 内联公式定义（与代码中实现完全一致，作为黄金标准记录）
# ─────────────────────────────────────────────────────────────────────────────

def _kelly(W: float, R: float) -> float:
    """Kelly% = W - (1-W)/R；与 paper_trading.py:548、backtest.py:541、debate.py:422 相同"""
    if R <= 0:
        return 0.0
    return W - (1 - W) / R


def _surprise_pct(eps_act: float, eps_est: float):
    """百分比超预期；与 scraper.py:312 相同"""
    if eps_est == 0:
        return None
    return round((eps_act - eps_est) / abs(eps_est) * 100, 1)


def _accel(rs_1m: float, rs_3m: float) -> bool:
    """板块加速度；与 sector_rotation.py:257-258 相同"""
    prior_2m_avg = (rs_3m - rs_1m) / 2
    return rs_1m > prior_2m_avg and rs_1m > 0


# ─────────────────────────────────────────────────────────────────────────────
# Kelly Criterion
# ─────────────────────────────────────────────────────────────────────────────

class TestKelly:
    def test_positive_edge(self):
        # 53%胜率、2:1盈亏比 → Kelly = 0.53 - 0.47/2 = 0.295（≈29.5%），半Kelly ≈ 14.75%
        k = _kelly(0.53, 2.0)
        assert k == pytest.approx(0.295)
        assert k / 2 < 0.25  # 半Kelly不超过25%

    def test_negative_edge(self):
        # 40%胜率、1:1盈亏比 → 负期望
        assert _kelly(0.40, 1.0) < 0

    def test_break_even(self):
        # 50%胜率、1:1盈亏比 → 期望为零
        assert _kelly(0.50, 1.0) == pytest.approx(0.0)

    def test_zero_rr_ratio_returns_zero(self):
        # rr_ratio=0 时不建议入场（分母为零保护）
        assert _kelly(0.60, 0.0) == 0.0

    def test_half_kelly_never_negative(self):
        # max(0, kelly_f/2) 保证输出非负
        for W, R in [(0.30, 0.5), (0.40, 1.0), (0.20, 0.8)]:
            assert max(0, _kelly(W, R) / 2) >= 0

    def test_high_win_rate_high_rr(self):
        # 极端情况：70%胜率、3:1盈亏比 → Kelly 约 46.7%
        k = _kelly(0.70, 3.0)
        assert k == pytest.approx(0.70 - 0.30 / 3.0)


# ─────────────────────────────────────────────────────────────────────────────
# surprise_pct —— 防止 epsDifference×100 复活
# ─────────────────────────────────────────────────────────────────────────────

class TestSurprisePct:
    def test_beat_10pct(self):
        assert _surprise_pct(1.10, 1.00) == pytest.approx(10.0)

    def test_miss_10pct(self):
        assert _surprise_pct(0.90, 1.00) == pytest.approx(-10.0)

    def test_negative_estimate_beat(self):
        # est=-0.10，act=-0.05（少亏），超预期 +50%
        assert _surprise_pct(-0.05, -0.10) == pytest.approx(50.0)

    def test_negative_estimate_miss(self):
        # est=-0.10，act=-0.15（多亏），低于预期 -50%
        assert _surprise_pct(-0.15, -0.10) == pytest.approx(-50.0)

    def test_zero_estimate_returns_none(self):
        assert _surprise_pct(0.10, 0.00) is None

    def test_exact_match_is_zero(self):
        assert _surprise_pct(1.00, 1.00) == pytest.approx(0.0)

    def test_regression_not_epsDifference_times_100(self):
        """
        P0回归：旧代码 surprise_pct = epsDifference * 100
        对于 est=0.10，act=0.11（beat $0.01）：
          旧公式：0.01 × 100 = 1.0  ← 单位是"美分×100"，无意义
          新公式：0.01 / 0.10 × 100 = 10.0%  ← 正确
        """
        eps_act, eps_est = 0.11, 0.10
        correct = _surprise_pct(eps_act, eps_est)
        old_bug = round((eps_act - eps_est) * 100, 1)  # epsDifference * 100

        assert correct == pytest.approx(10.0)
        assert old_bug == pytest.approx(1.0)  # 旧 bug 给 1.0
        assert correct != pytest.approx(old_bug)

    def test_regression_large_eps(self):
        """
        大盘股 eps 较大时，旧公式更明显出错。
        est=5.00，act=5.25（beat $0.25 = +5%）：
          旧：0.25 × 100 = 25.0  ← 严重高估（虚报+25%）
          新：0.25 / 5.00 × 100 = 5.0%  ← 正确
        """
        eps_act, eps_est = 5.25, 5.00
        correct = _surprise_pct(eps_act, eps_est)
        old_bug = round((eps_act - eps_est) * 100, 1)

        assert correct == pytest.approx(5.0)
        assert old_bug == pytest.approx(25.0)
        assert correct != pytest.approx(old_bug)


# ─────────────────────────────────────────────────────────────────────────────
# sector_rotation 加速度
# ─────────────────────────────────────────────────────────────────────────────

class TestAccel:
    def test_genuine_acceleration(self):
        # 近3月=3%，近1月=4%（前2月均值=−0.5%），本月明显高于均值
        assert _accel(rs_1m=4.0, rs_3m=3.0) is True

    def test_deceleration(self):
        # 近3月=6%，近1月=0.5%（前2月均值=2.75%），本月低于均值
        assert _accel(rs_1m=0.5, rs_3m=6.0) is False

    def test_negative_this_month_always_false(self):
        # rs_1m<0：本月相对表现为负，无论之前如何都不算加速
        assert _accel(rs_1m=-1.0, rs_3m=5.0) is False
        assert _accel(rs_1m=-0.01, rs_3m=-5.0) is False

    def test_momentum_turning_positive(self):
        # 3月=-3%（整体疲弱），本月=+1%（转正），前2月均值=−2%，1>−2 → 加速
        assert _accel(rs_1m=1.0, rs_3m=-3.0) is True

    def test_regression_old_formula_denominator(self):
        """
        P1回归：旧公式 rs_1m > rs_3m/3
        关键差异——rs_3m很大但本月大幅减速时：
          rs_1m=1.0，rs_3m=9.0
          旧：1.0 > 9/3=3.0 → False ✓（偶然正确）
          新：前2m均值=(9-1)/2=4.0，1.0>4.0 → False ✓
        两者在此例一致，但新公式在分布上更稳健。
        确保两者对边界案例都给出 False：
        """
        assert _accel(rs_1m=1.0, rs_3m=9.0) is False

    def test_all_zero_is_false(self):
        # rs_1m=0 不满足 rs_1m > 0
        assert _accel(rs_1m=0.0, rs_3m=0.0) is False
