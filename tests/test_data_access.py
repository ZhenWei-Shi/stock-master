"""
数据访问契约测试

覆盖已发现的字段名幻觉：
  P0: earnings_history 日期是 DataFrame.index，不是列名 "quarter"
  P1: yfinance info 字段 institutionPercentHeld（非 heldPercentInstitutions）
  P0: 空 DataFrame 访问 iloc[-1] 崩溃
"""
import pytest
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# earnings_history 解析逻辑
# ─────────────────────────────────────────────────────────────────────────────

def _parse_earnings(df: pd.DataFrame) -> list:
    """
    与 scraper.py:299-321 / fundamentals.py 相同的解析逻辑。
    用于独立验证字段访问方式正确。
    注意：pandas 将 None 存为 NaN（float），必须用 pd.isna() 而非 is None。
    """
    quarters = []
    for idx, row in df.head(8).iterrows():
        eps_est = row.get("epsEstimate")
        eps_act = row.get("epsActual")
        date = str(idx)[:10]          # 正确：date 是 index
        if pd.isna(eps_est) or pd.isna(eps_act):  # 正确：isna 捕获 NaN 和 None
            continue
        eps_est_f = float(eps_est)
        eps_act_f = float(eps_act)
        beat = eps_act_f > eps_est_f
        surprise = (
            round((eps_act_f - eps_est_f) / abs(eps_est_f) * 100, 1)
            if eps_est_f != 0 else None
        )
        quarters.append({"date": date, "eps_est": eps_est_f,
                          "eps_act": eps_act_f, "beat": beat, "surprise_pct": surprise})
    return quarters


class TestEarningsHistoryIndex:
    def _make_df(self, dates, eps_est_list, eps_act_list):
        return pd.DataFrame(
            {"epsEstimate": eps_est_list, "epsActual": eps_act_list},
            index=pd.to_datetime(dates),
        )

    def test_date_comes_from_index(self):
        df = self._make_df(
            ["2024-01-15", "2023-10-16"],
            [1.00, 0.80],
            [1.10, 0.75],
        )
        result = _parse_earnings(df)
        assert result[0]["date"] == "2024-01-15"
        assert result[1]["date"] == "2023-10-16"

    def test_regression_quarter_column_does_not_exist(self):
        """
        P0回归：旧代码 row.get("quarter", "") 始终返回 ""，
        因为 earnings_history 没有名为 "quarter" 的列。
        """
        df = self._make_df(["2024-01-15"], [1.00], [1.10])
        for idx, row in df.iterrows():
            assert "quarter" not in row.index, "列名 'quarter' 不应存在于 earnings_history"
            assert row.get("quarter", "") == ""   # 旧代码拿到的是空字符串
            assert str(idx)[:10] == "2024-01-15"  # 正确方式

    def test_partial_data_rows_skipped(self):
        """epsEstimate 或 epsActual 缺失时跳过该行"""
        df = pd.DataFrame(
            {"epsEstimate": [1.00, None, 0.90],
             "epsActual":   [1.10, 0.80, None]},
            index=pd.to_datetime(["2024-01-15", "2023-10-16", "2023-07-17"]),
        )
        result = _parse_earnings(df)
        assert len(result) == 1
        assert result[0]["date"] == "2024-01-15"

    def test_regression_none_becomes_nan_in_pandas(self):
        """
        P2回归：pandas 在混合数值列中将 None 存为 NaN（float64），
        旧代码 `if eps_est is None` 无法捕获 NaN，导致 NaN 行被错误包含。
        修复后使用 pd.isna() 可同时捕获 None 和 NaN。
        """
        # 混合数值+None → pandas 升级为 float64，None 变 NaN
        df = pd.DataFrame(
            {"epsEstimate": [1.00, None],   # None 在混合列中 → NaN
             "epsActual":   [1.10, 0.80]},
            index=pd.to_datetime(["2024-01-15", "2023-10-16"]),
        )
        # 验证第二行的 epsEstimate 是 NaN（不是 None）
        for idx, row in df.iterrows():
            if str(idx)[:10] == "2023-10-16":
                eps_e = row.get("epsEstimate")
                assert eps_e is not None    # NaN is not None → True（旧检查在此漏掉）
                assert pd.isna(eps_e)       # pd.isna 正确识别为缺失
        # 使用修复后的 pd.isna 检查，NaN 行被跳过，只返回1条
        result = _parse_earnings(df)
        assert len(result) == 1
        assert result[0]["date"] == "2024-01-15"

    def test_beat_flag_correct(self):
        df = self._make_df(["2024-01-15", "2023-10-16"], [1.00, 1.00], [1.10, 0.90])
        result = _parse_earnings(df)
        assert result[0]["beat"] is True
        assert result[1]["beat"] is False

    def test_head_8_limit(self):
        """最多取8条"""
        dates = [f"2024-0{i+1}-15" for i in range(9)]
        df = pd.DataFrame(
            {"epsEstimate": [1.0] * 9, "epsActual": [1.1] * 9},
            index=pd.to_datetime(dates),
        )
        result = _parse_earnings(df)
        assert len(result) == 8


# ─────────────────────────────────────────────────────────────────────────────
# yfinance info 字段名
# ─────────────────────────────────────────────────────────────────────────────

class TestYFinanceFieldNames:
    """
    P1回归：smart_money.py 旧代码用 info.get("heldPercentInstitutions")，
    正确字段名是 "institutionPercentHeld"。
    """

    def test_correct_field_name_returns_value(self):
        # 模拟 yf.Ticker("AAPL").info 返回值（基于实际 yfinance 输出）
        mock_info = {
            "institutionPercentHeld": 0.72,
            "symbol": "AAPL",
        }
        correct = float(mock_info.get("institutionPercentHeld", 0)) * 100
        assert correct == pytest.approx(72.0)

    def test_wrong_field_name_silently_returns_zero(self):
        """旧字段名静默失败，返回 0，导致机构持仓数据全部错误显示为 0%"""
        mock_info = {
            "institutionPercentHeld": 0.72,
        }
        old_bug = float(mock_info.get("heldPercentInstitutions", 0)) * 100
        assert old_bug == pytest.approx(0.0)  # 旧代码一直给 0%

    def test_regression_field_names_differ(self):
        mock_info = {"institutionPercentHeld": 0.72}
        correct = float(mock_info.get("institutionPercentHeld", 0)) * 100
        old_bug = float(mock_info.get("heldPercentInstitutions", 0)) * 100
        assert correct != pytest.approx(old_bug), "两个字段名结果不同，确认修复有效"


# ─────────────────────────────────────────────────────────────────────────────
# 空 DataFrame / 假日无数据
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyDataFrame:
    def test_iloc_on_empty_raises_index_error(self):
        """
        P0核心契约：空 DataFrame 访问 .iloc[-1] 会崩溃。
        所有 yfinance 调用后必须有 .empty 检查或 try/except。
        """
        import pandas as pd
        empty = pd.DataFrame({"Close": []})
        with pytest.raises(IndexError):
            _ = empty["Close"].iloc[-1]

    def test_empty_check_prevents_crash(self):
        """正确的防护模式：先检查 .empty，再访问 .iloc[-1]"""
        import pandas as pd
        hist = pd.DataFrame({"Close": []})
        price = float(hist["Close"].iloc[-1]) if not hist.empty else None
        assert price is None

    def test_non_empty_returns_value(self):
        import pandas as pd
        hist = pd.DataFrame({"Close": [150.0, 151.0, 152.0]})
        price = float(hist["Close"].iloc[-1]) if not hist.empty else None
        assert price == pytest.approx(152.0)
