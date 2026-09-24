"""
fetcher.py 回归测试

覆盖范围：fill_daily_gaps / _has_weekday_gap / daily_history 的兜底编排，
用假数据和假Ticker，不依赖网络。背景：2026-09-22 Yahoo日线全市场漏了
一天，同日1小时线完整。
"""
import pandas as pd

from src.fetcher import fill_daily_gaps, _has_weekday_gap, daily_history

TZ = "America/New_York"
COLS = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]


def _daily(dates):
    idx = pd.DatetimeIndex([pd.Timestamp(d).tz_localize(TZ) for d in dates])
    n = len(idx)
    return pd.DataFrame({"Open": [10.0] * n, "High": [11.0] * n, "Low": [9.0] * n,
                         "Close": [10.5] * n, "Volume": [1000] * n,
                         "Dividends": [0.0] * n, "Stock Splits": [0.0] * n}, index=idx)


def _hourly(day, bars):
    """bars: [(open, high, low, close, volume), ...]，从09:30起每小时一根"""
    idx = pd.date_range(f"{day} 09:30", periods=len(bars), freq="h", tz=TZ)
    return pd.DataFrame(bars, columns=["Open", "High", "Low", "Close", "Volume"], index=idx)


# 2026-09-22（周二）缺失，与实际事故一致
DAYS_WITH_GAP = ["2026-09-17", "2026-09-18", "2026-09-21", "2026-09-23"]


class TestFillDailyGaps:
    def test_missing_day_aggregated_from_hourly(self):
        hourly = _hourly("2026-09-22", [(61.97, 62.45, 60.99, 61.88, 100),
                                        (61.86, 65.41, 61.26, 64.85, 200),
                                        (64.48, 64.65, 63.64, 63.66, 300)])
        out = fill_daily_gaps(_daily(DAYS_WITH_GAP), hourly)
        row = out.loc[pd.Timestamp("2026-09-22").tz_localize(TZ)]
        assert (row.Open, row.High, row.Low, row.Close, row.Volume) == (61.97, 65.41, 60.99, 63.66, 600)
        assert row["Dividends"] == 0.0
        assert out.index.is_monotonic_increasing
        assert list(out.columns) == COLS
        assert out.attrs["filled_dates"] == ["2026-09-22"]

    def test_existing_days_not_overwritten(self):
        daily = _daily(DAYS_WITH_GAP)
        hourly = _hourly("2026-09-21", [(1, 2, 0.5, 1.5, 5)])
        out = fill_daily_gaps(daily, hourly)
        assert out is daily
        assert "filled_dates" not in out.attrs

    def test_days_before_daily_start_ignored(self):
        daily = _daily(DAYS_WITH_GAP)
        hourly = _hourly("2026-09-10", [(1, 2, 0.5, 1.5, 5)])
        assert len(fill_daily_gaps(daily, hourly)) == len(daily)

    def test_empty_inputs_return_daily(self):
        daily = _daily(DAYS_WITH_GAP)
        assert fill_daily_gaps(daily, pd.DataFrame()) is daily
        empty = pd.DataFrame(columns=COLS)
        assert fill_daily_gaps(empty, _hourly("2026-09-22", [(1, 2, 0.5, 1.5, 5)])) is empty


class TestHasWeekdayGap:
    def test_detects_missing_weekday(self):
        assert _has_weekday_gap(_daily(DAYS_WITH_GAP)) is True

    def test_complete_week_no_gap(self):
        assert _has_weekday_gap(_daily(pd.bdate_range("2026-09-14", "2026-09-23"))) is False

    def test_weekend_not_counted_as_gap(self):
        # 9/18周五 → 9/21周一，中间周末不算缺口
        assert _has_weekday_gap(_daily(["2026-09-17", "2026-09-18", "2026-09-21"])) is False

    def test_nyse_holiday_not_counted_as_gap(self):
        # 9/7劳动节休市：否则最近30天窗口内每次调用都会误发小时线请求
        days = [d for d in pd.bdate_range("2026-08-24", "2026-09-23") if str(d.date()) != "2026-09-07"]
        assert _has_weekday_gap(_daily(days)) is False

    def test_good_friday_not_counted_as_gap(self):
        # 耶稣受难日NYSE休市但不是联邦假日
        days = [d for d in pd.bdate_range("2026-03-30", "2026-04-08") if str(d.date()) != "2026-04-03"]
        assert _has_weekday_gap(_daily(days)) is False

    def test_empty(self):
        assert _has_weekday_gap(pd.DataFrame(columns=COLS)) is False


class _FakeTicker:
    def __init__(self, daily, hourly=None, hourly_raises=False):
        self.daily, self.hourly, self.hourly_raises = daily, hourly, hourly_raises
        self.calls = []

    def history(self, period, interval):
        self.calls.append(interval)
        if interval == "1d":
            return self.daily
        if self.hourly_raises:
            raise RuntimeError("rate limited")
        return self.hourly


class TestDailyHistory:
    def test_no_gap_skips_hourly_request(self):
        tk = _FakeTicker(_daily(pd.bdate_range("2026-09-14", "2026-09-23")))
        daily_history(tk, "1y")
        assert tk.calls == ["1d"]

    def test_gap_filled_from_hourly(self):
        tk = _FakeTicker(_daily(DAYS_WITH_GAP), _hourly("2026-09-22", [(1, 2, 0.5, 1.5, 5)]))
        out = daily_history(tk, "1y")
        assert tk.calls == ["1d", "1h"]
        assert out.attrs["filled_dates"] == ["2026-09-22"]

    def test_hourly_failure_falls_back_to_daily(self):
        daily = _daily(DAYS_WITH_GAP)
        tk = _FakeTicker(daily, hourly_raises=True)
        assert daily_history(tk, "1y") is daily
