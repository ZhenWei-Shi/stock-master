"""
发债/可转债公告事件门测试

grade_debt_event() 直接从生产代码导入（src/debt_event_monitor.py），
只测试纯函数评分边界，不依赖网络/EDGAR/文件IO。
"""
from src.debt_event_monitor import grade_debt_event, VETO_DAYS, WARN_DAYS


class TestGradeDebtEvent:
    def test_no_event_passes(self):
        r = grade_debt_event(None)
        assert r["pass"] is True

    def test_announcement_day_vetoes(self):
        # 公告当天（0天前）：一票否决
        r = grade_debt_event(0)
        assert r["pass"] is False

    def test_day_after_vetoes(self):
        # 公告次日（VETO_DAYS=1）：仍一票否决，覆盖ASTS盘后公告次日的场景
        r = grade_debt_event(VETO_DAYS)
        assert r["pass"] is False

    def test_boundary_after_veto_window_warns(self):
        # 否决期结束后第一天：进入警示期，不再一票否决
        r = grade_debt_event(VETO_DAYS + 1)
        assert r["pass"] == "warn"

    def test_within_warn_window_warns(self):
        r = grade_debt_event(WARN_DAYS)
        assert r["pass"] == "warn"

    def test_boundary_after_warn_window_passes(self):
        # 警示期结束后第一天：完全通过
        r = grade_debt_event(WARN_DAYS + 1)
        assert r["pass"] is True

    def test_far_past_passes(self):
        r = grade_debt_event(30)
        assert r["pass"] is True

    def test_negative_days_treated_as_no_event(self):
        # 理论上不应出现（file_date在未来），防御性处理为无事件
        r = grade_debt_event(-1)
        assert r["pass"] is True

    def test_note_always_present(self):
        for days in (None, 0, 1, 4, 8, 30):
            r = grade_debt_event(days)
            assert r.get("note")


class TestOfferingQueriesQuality:
    """查询短语的基本质量检查——避免误引入回购/赎回类措辞。"""

    def test_no_repurchase_or_redemption_keywords(self):
        from src.debt_event_monitor import _OFFERING_QUERIES
        banned = ("redeem", "repurchase", "redemption", "tender offer", "buyback")
        for query in _OFFERING_QUERIES:
            lowered = query.lower()
            for word in banned:
                assert word not in lowered, f"'{query}' 不应包含回购/赎回类措辞 '{word}'"

    def test_queries_non_empty(self):
        from src.debt_event_monitor import _OFFERING_QUERIES
        assert len(_OFFERING_QUERIES) > 0


class TestTickerExtraction:
    """EDGAR display_names → ticker 提取正则的边界检查。"""

    def test_extracts_single_ticker(self):
        from src.debt_event_monitor import _TICKER_IN_DISPLAY_NAME
        m = _TICKER_IN_DISPLAY_NAME.search(
            "AST SpaceMobile, Inc.  (ASTS)  (CIK 0001780312)")
        assert m and m.group(1) == "ASTS"

    def test_no_ticker_when_absent(self):
        # 部分申报人（如个人/无上市代码的机构）display_name不含ticker括号
        from src.debt_event_monitor import _TICKER_IN_DISPLAY_NAME
        m = _TICKER_IN_DISPLAY_NAME.search("Horsehead Holding Corp  (CIK 0001385544)")
        assert m is None

    def test_does_not_mistake_cik_for_ticker(self):
        from src.debt_event_monitor import _TICKER_IN_DISPLAY_NAME
        m = _TICKER_IN_DISPLAY_NAME.search("Some Corp  (CIK 0000012345)")
        assert m is None
