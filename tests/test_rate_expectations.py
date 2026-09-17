"""
美联储加息/降息隐含概率测试

_calc_from_next_fomc() 直接从生产代码导入（src/rate_expectations.py），
只测试纯数学计算的边界，不依赖网络（yfinance ZQ=F）/文件IO/真实FOMC日历。
"""
from datetime import date

from src.rate_expectations import (
    _calc_from_next_fomc,
    calc_rate_probability,
    RATE_STEP,
)


def _implied_price(current_rate, implied_new_rate, as_of, next_fomc):
    """测试辅助：反向从"想要的隐含新利率"推算出对应的期货价格。"""
    import calendar
    n = calendar.monthrange(next_fomc.year, next_fomc.month)[1]
    d = next_fomc.day
    if d <= 1:
        implied_avg = implied_new_rate
    else:
        pre_days, post_days = d - 1, n - d + 1
        implied_avg = (pre_days * current_rate + post_days * implied_new_rate) / n
    return 100 - implied_avg


class TestCalcFromNextFomc:
    def test_futures_price_none_returns_not_ok(self):
        r = _calc_from_next_fomc(date(2026, 10, 1), date(2026, 10, 28), None, 3.875)
        assert r["ok"] is False
        assert "获取失败" in r["note"]

    def test_decision_not_in_current_month_degrades(self):
        # 查询日期在9月，但下次决议在10月 → front-month期货尚未反映该次决议
        r = _calc_from_next_fomc(date(2026, 9, 20), date(2026, 10, 28), 96.10, 3.875)
        assert r["ok"] is False
        assert "暂未反映" in r["note"]

    def test_full_hike_priced_in(self):
        # 构造场景：市场100%确定下次决议加息25bp
        as_of, next_fomc = date(2026, 10, 1), date(2026, 10, 28)
        current_rate, target_new_rate = 3.875, 4.125
        price = _implied_price(current_rate, target_new_rate, as_of, next_fomc)
        r = _calc_from_next_fomc(as_of, next_fomc, price, current_rate)
        assert r["ok"] is True
        # implied_avg_rate 中途 round(4位小数) 会被 n/post_days 放大误差，
        # 用小容差而非精确相等（clamp后的概率本身不受这点误差影响）
        assert abs(r["implied_new_rate"] - 4.125) < 0.001
        assert r["prob_hike"] == 1.0
        assert r["prob_cut"] == 0.0
        assert r["prob_hold"] == 0.0

    def test_full_cut_priced_in(self):
        as_of, next_fomc = date(2026, 10, 1), date(2026, 10, 28)
        current_rate, target_new_rate = 3.875, 3.625
        price = _implied_price(current_rate, target_new_rate, as_of, next_fomc)
        r = _calc_from_next_fomc(as_of, next_fomc, price, current_rate)
        assert r["ok"] is True
        assert abs(r["implied_new_rate"] - 3.625) < 0.001
        assert r["prob_cut"] == 1.0
        assert r["prob_hike"] == 0.0

    def test_no_change_priced_in(self):
        as_of, next_fomc = date(2026, 10, 1), date(2026, 10, 28)
        current_rate = 3.875
        price = _implied_price(current_rate, current_rate, as_of, next_fomc)
        r = _calc_from_next_fomc(as_of, next_fomc, price, current_rate)
        assert r["ok"] is True
        assert r["prob_hike"] == 0.0
        assert r["prob_cut"] == 0.0
        assert r["prob_hold"] == 1.0
        assert r["extreme"] is False

    def test_half_priced_hike(self):
        # 市场对加息25bp只price in一半概率
        as_of, next_fomc = date(2026, 10, 1), date(2026, 10, 28)
        current_rate = 3.875
        target_new_rate = current_rate + RATE_STEP * 0.5
        price = _implied_price(current_rate, target_new_rate, as_of, next_fomc)
        r = _calc_from_next_fomc(as_of, next_fomc, price, current_rate)
        assert r["ok"] is True
        assert abs(r["prob_hike"] - 0.5) < 0.01
        assert r["prob_cut"] == 0.0
        assert abs(r["prob_hold"] - 0.5) < 0.01

    def test_decision_on_first_of_month(self):
        # 决议在月初(d=1)：走近似分支，implied_new_rate 直接等于 implied_avg_rate
        as_of, next_fomc = date(2026, 11, 1), date(2026, 11, 1)
        current_rate, target_new_rate = 3.875, 4.125
        price = 100 - target_new_rate
        r = _calc_from_next_fomc(as_of, next_fomc, price, current_rate)
        assert r["ok"] is True
        assert r["implied_new_rate"] == target_new_rate
        assert r["prob_hike"] == 1.0

    def test_decision_on_last_day_of_month(self):
        # 决议在月末最后一天：post_days=1，权重最集中，数值稳定性边界
        as_of, next_fomc = date(2026, 9, 1), date(2026, 9, 30)
        current_rate, target_new_rate = 3.875, 4.125
        price = _implied_price(current_rate, target_new_rate, as_of, next_fomc)
        r = _calc_from_next_fomc(as_of, next_fomc, price, current_rate)
        assert r["ok"] is True
        assert abs(r["implied_new_rate"] - target_new_rate) < 0.01

    def test_extreme_flag_when_delta_exceeds_1_5_steps(self):
        # 隐含变动超过1.5档（37.5bp）应标记extreme，提示简化模型可能失真
        as_of, next_fomc = date(2026, 10, 1), date(2026, 10, 28)
        current_rate = 3.875
        target_new_rate = current_rate + RATE_STEP * 2   # 隐含50bp，超过1.5档阈值
        price = _implied_price(current_rate, target_new_rate, as_of, next_fomc)
        r = _calc_from_next_fomc(as_of, next_fomc, price, current_rate)
        assert r["extreme"] is True
        assert "失真" in r["note"]

    def test_prob_hike_and_cut_never_both_nonzero(self):
        # delta 只可能偏向一侧，不应同时给出加息和降息概率
        as_of, next_fomc = date(2026, 10, 1), date(2026, 10, 28)
        current_rate, target_new_rate = 3.875, 4.0
        price = _implied_price(current_rate, target_new_rate, as_of, next_fomc)
        r = _calc_from_next_fomc(as_of, next_fomc, price, current_rate)
        assert r["prob_hike"] == 0 or r["prob_cut"] == 0

    def test_probabilities_sum_to_one(self):
        as_of, next_fomc = date(2026, 10, 1), date(2026, 10, 28)
        current_rate, target_new_rate = 3.875, 4.05
        price = _implied_price(current_rate, target_new_rate, as_of, next_fomc)
        r = _calc_from_next_fomc(as_of, next_fomc, price, current_rate)
        total = r["prob_hike"] + r["prob_cut"] + r["prob_hold"]
        assert abs(total - 1.0) < 1e-6

    def test_note_mentions_fomc_date_and_days_away(self):
        as_of, next_fomc = date(2026, 10, 1), date(2026, 10, 28)
        r = _calc_from_next_fomc(as_of, next_fomc, 96.10, 3.875)
        assert "2026-10-28" in r["note"]
        assert r["days_away"] == 27


class TestCalcRateProbabilityWithRealCalendar:
    """calc_rate_probability() 走真实 macro_filter._ALL_FOMC 日历查表分支。"""

    def test_finds_next_real_fomc_date(self):
        # 2026-09-16 FOMC当天查询：_next_fomc_date 应含"今天"本身（>=判断）
        r = calc_rate_probability(date(2026, 9, 16), 96.10, 3.875)
        assert r["next_fomc_date"] == "2026-09-16"
        assert r["days_away"] == 0

    def test_finds_upcoming_fomc_after_today(self):
        r = calc_rate_probability(date(2026, 9, 17), 96.10, 3.875)
        assert r["next_fomc_date"] == "2026-10-28"

    def test_far_future_beyond_calendar_returns_not_ok(self):
        # FOMC日历只覆盖到2026年底，2027年查询应优雅降级而非抛异常
        r = calc_rate_probability(date(2027, 6, 1), 96.10, 3.875)
        assert r["ok"] is False
        assert "年度更新" in r["note"]
