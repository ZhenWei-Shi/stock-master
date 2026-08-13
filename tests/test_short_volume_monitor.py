"""
空头成交量参考信号（FINRA Daily Short Sale Volume）测试

_calc_signal() 是纯函数，直接测试统计边界，不依赖网络。
_fetch_finra_file() / check_ticker_short_volume() 用 monkeypatch 隔离
requests.get 和历史文件路径，不发真实网络请求、不碰 data/ 下的生产文件。
"""
import src.short_volume_monitor as svm
from src.short_volume_monitor import _calc_signal, MIN_BASELINE_DAYS


class TestCalcSignal:
    def test_insufficient_when_less_than_two_points(self):
        r = _calc_signal([57.0])
        assert r["flag"] == "insufficient"

    def test_insufficient_when_baseline_below_minimum(self):
        # 基线天数（不含最新一天）< MIN_BASELINE_DAYS
        ratios = [55.0] * (MIN_BASELINE_DAYS - 1) + [57.0]
        r = _calc_signal(ratios)
        assert r["flag"] == "insufficient"
        assert r["baseline_days"] == MIN_BASELINE_DAYS - 1

    def test_normal_when_within_one_std(self):
        # 基线均值57，标准差小，最新值贴近均值
        baseline = [56.0, 57.0, 58.0, 57.0, 56.0, 58.0, 57.0, 57.0, 56.0, 58.0, 57.0]
        r = _calc_signal(baseline + [57.2])
        assert r["flag"] == "normal"

    def test_elevated_high_between_one_and_two_std(self):
        baseline = [50.0] * 12
        # std=0基线是另一个分支，这里手工构造有波动的基线
        baseline = [48, 50, 52, 49, 51, 50, 48, 52, 51, 49, 50, 50]
        import statistics
        mean_b, std_b = statistics.mean(baseline), statistics.pstdev(baseline)
        latest = mean_b + 1.5 * std_b
        r = _calc_signal(baseline + [latest])
        assert r["flag"] == "elevated_high"

    def test_extreme_high_beyond_two_std(self):
        baseline = [48, 50, 52, 49, 51, 50, 48, 52, 51, 49, 50, 50]
        import statistics
        mean_b, std_b = statistics.mean(baseline), statistics.pstdev(baseline)
        latest = mean_b + 3 * std_b
        r = _calc_signal(baseline + [latest])
        assert r["flag"] == "extreme_high"
        assert "做市商" in r["note"]   # 异常时必须带局限性说明，不能只报数字

    def test_extreme_low_beyond_negative_two_std(self):
        baseline = [48, 50, 52, 49, 51, 50, 48, 52, 51, 49, 50, 50]
        import statistics
        mean_b, std_b = statistics.mean(baseline), statistics.pstdev(baseline)
        latest = mean_b - 3 * std_b
        r = _calc_signal(baseline + [latest])
        assert r["flag"] == "extreme_low"

    def test_normal_note_has_no_disclaimer(self):
        # 正常范围不需要"不等于看空押注"这句提醒，避免每次都刷屏同一句话
        baseline = [56.0, 57.0, 58.0, 57.0, 56.0, 58.0, 57.0, 57.0, 56.0, 58.0, 57.0]
        r = _calc_signal(baseline + [57.2])
        assert r["flag"] == "normal"
        assert "做市商" not in r["note"]

    def test_zero_variance_baseline_exact_match_is_normal(self):
        baseline = [55.0] * 11
        r = _calc_signal(baseline + [55.0])
        assert r["flag"] == "normal"

    def test_zero_variance_baseline_any_deviation_is_extreme(self):
        # 历史基线从未变化过（std=0），一旦当天数值不同就不能除0强判"正常"
        baseline = [55.0] * 11
        r = _calc_signal(baseline + [60.0])
        assert r["flag"] == "extreme_high"


class TestFetchFinraFile:
    def test_parses_pipe_delimited_response(self, monkeypatch):
        fake_text = (
            "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
            "20260812|ASTS|2759794.52|12155|4843978.69|B,Q,N\n"
            "20260812|NVDA|100.0|0|200.0|B,Q,N\n"
        )

        class FakeResp:
            status_code = 200
            text = fake_text

        monkeypatch.setattr(svm.requests, "get", lambda *a, **k: FakeResp())
        from datetime import date
        result = svm._fetch_finra_file(date(2026, 8, 12))
        assert result["ASTS"] == (2759794.52, 4843978.69)
        assert result["NVDA"] == (100.0, 200.0)

    def test_non_200_status_returns_none(self, monkeypatch):
        class FakeResp:
            status_code = 403
            text = ""

        monkeypatch.setattr(svm.requests, "get", lambda *a, **k: FakeResp())
        from datetime import date
        assert svm._fetch_finra_file(date(2026, 1, 1)) is None

    def test_network_error_returns_none(self, monkeypatch):
        def _raise(*a, **k):
            raise ConnectionError("boom")

        monkeypatch.setattr(svm.requests, "get", _raise)
        from datetime import date
        assert svm._fetch_finra_file(date(2026, 8, 12)) is None

    def test_malformed_line_skipped_not_crashed(self, monkeypatch):
        fake_text = (
            "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
            "20260812|BAD|not_a_number|0|100.0|B,Q,N\n"
            "20260812|OK|10.0|0|20.0|B,Q,N\n"
        )

        class FakeResp:
            status_code = 200
            text = fake_text

        monkeypatch.setattr(svm.requests, "get", lambda *a, **k: FakeResp())
        from datetime import date
        result = svm._fetch_finra_file(date(2026, 8, 12))
        assert "BAD" not in result
        assert result["OK"] == (10.0, 20.0)


class TestCheckTickerShortVolume:
    def test_no_history_returns_pass_true_with_note(self, tmp_path, monkeypatch):
        monkeypatch.setattr(svm, "_HISTORY_FILE", str(tmp_path / "short_volume_history.json"))
        r = svm.check_ticker_short_volume("ASTS")
        assert r["pass"] is True
        assert "尚未建立基线" in r["note"]

    def test_reads_history_and_always_passes(self, tmp_path, monkeypatch):
        import json
        hist_file = tmp_path / "short_volume_history.json"
        history = {"ASTS": {f"2026-07-{d:02d}": 57.0 for d in range(1, 15)}}
        hist_file.write_text(json.dumps(history), encoding="utf-8")
        monkeypatch.setattr(svm, "_HISTORY_FILE", str(hist_file))

        r = svm.check_ticker_short_volume("asts")   # 小写ticker也要能匹配
        assert r["pass"] is True   # 核心不变式：无论信号多异常，都不veto/不参与打分
        assert "ratio" in r

    def test_never_vetoes_even_when_extreme(self, tmp_path, monkeypatch):
        import json
        hist_file = tmp_path / "short_volume_history.json"
        baseline = {f"2026-07-{d:02d}": 50.0 for d in range(1, 13)}
        baseline["2026-07-13"] = 95.0   # 极端异常值
        hist_file.write_text(json.dumps({"ASTS": baseline}), encoding="utf-8")
        monkeypatch.setattr(svm, "_HISTORY_FILE", str(hist_file))

        r = svm.check_ticker_short_volume("ASTS")
        assert r["pass"] is True
        assert r.get("flag") == "extreme_high"


class TestRunShortVolumeMonitor:
    def test_empty_watchlist_skips(self):
        r = svm.run_short_volume_monitor([])
        assert r["ok"] is True
        assert "跳过" in r["note"]

    def test_none_watchlist_skips(self):
        r = svm.run_short_volume_monitor(None)
        assert r["ok"] is True


class TestBackfillHistory:
    """
    高层编排函数测试：monkeypatch _fetch_finra_file 模拟"每3天里有1天是
    非交易日（返回None）"的日历节奏，验证 backfill_history/refresh_latest
    真正走到底的日期计算+滚动窗口裁剪逻辑，而不只是测被它们调用的纯函数。
    """

    def _make_fake_fetch(self, monkeypatch, none_every=3):
        calls = {"n": 0}

        def _fake_fetch(d):
            calls["n"] += 1
            if calls["n"] % none_every == 0:
                return None   # 模拟周末/假日
            return {"ASTS": (100.0 + calls["n"], 200.0)}

        monkeypatch.setattr(svm, "_fetch_finra_file", _fake_fetch)
        return calls

    def test_fills_target_trading_days_skipping_none_days(self, tmp_path, monkeypatch):
        monkeypatch.setattr(svm, "_HISTORY_FILE", str(tmp_path / "history.json"))
        self._make_fake_fetch(monkeypatch, none_every=3)

        r = svm.backfill_history(["ASTS"], days=10)
        assert r["ok"] is True
        assert r["days_filled"] == 10   # 目标交易日数必须真正凑够，不因None日提前退出

        history = svm._load_history()
        assert len(history["ASTS"]) == 10   # 10个有效交易日各存一条，不含None日

    def test_trims_to_history_days_window(self, tmp_path, monkeypatch):
        monkeypatch.setattr(svm, "_HISTORY_FILE", str(tmp_path / "history.json"))
        monkeypatch.setattr(svm, "HISTORY_DAYS", 5)   # 缩小窗口方便测试
        self._make_fake_fetch(monkeypatch, none_every=100)   # 全是交易日，不跳过

        svm.backfill_history(["ASTS"], days=8)
        history = svm._load_history()
        assert len(history["ASTS"]) == 5   # 超过HISTORY_DAYS的部分被裁掉

    def test_stops_at_calendar_try_limit_even_if_target_not_reached(self, tmp_path, monkeypatch):
        monkeypatch.setattr(svm, "_HISTORY_FILE", str(tmp_path / "history.json"))
        monkeypatch.setattr(svm, "_fetch_finra_file", lambda d: None)   # 永远拿不到数据

        history, filled = svm._fill_history_backward(
            {}, {"ASTS"}, target_days=10, max_calendar_tries=7)
        assert filled == 0   # 不会死循环，尝试次数耗尽后老实返回0


class TestRefreshLatest:
    def test_appends_new_day_and_keeps_existing_history(self, tmp_path, monkeypatch):
        import json
        hist_file = tmp_path / "history.json"
        hist_file.write_text(
            json.dumps({"ASTS": {"2026-07-01": 50.0}}), encoding="utf-8")
        monkeypatch.setattr(svm, "_HISTORY_FILE", str(hist_file))

        calls = {"n": 0}

        def _fake_fetch(d):
            calls["n"] += 1
            return {"ASTS": (60.0, 100.0)}   # 第一次尝试就拿到（昨天）

        monkeypatch.setattr(svm, "_fetch_finra_file", _fake_fetch)
        r = svm.refresh_latest(["ASTS"])

        assert r["ok"] is True
        history = svm._load_history()
        assert "2026-07-01" in history["ASTS"]   # 旧记录保留
        assert len(history["ASTS"]) == 2         # 新增一条

    def test_returns_not_ok_when_no_file_available_within_retry_window(self, tmp_path, monkeypatch):
        monkeypatch.setattr(svm, "_HISTORY_FILE", str(tmp_path / "history.json"))
        monkeypatch.setattr(svm, "_fetch_finra_file", lambda d: None)

        r = svm.refresh_latest(["ASTS"])
        assert r["ok"] is False
        assert "未取到" in r["note"]

    def test_skips_ticker_not_present_in_finra_file(self, tmp_path, monkeypatch):
        # 停牌/退市/新上市股票当天文件里没有该ticker的行，不应崩溃，应正常跳过
        monkeypatch.setattr(svm, "_HISTORY_FILE", str(tmp_path / "history.json"))
        monkeypatch.setattr(svm, "_fetch_finra_file", lambda d: {"OTHER": (1.0, 2.0)})

        r = svm.refresh_latest(["ASTS"])
        assert r["ok"] is True
        history = svm._load_history()
        assert "ASTS" not in history
