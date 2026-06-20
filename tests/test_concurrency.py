"""
并发安全测试

覆盖已发现的三类竞态条件：
  P0: log_execution 无锁 → 并发写入丢失记录
  P1: record_day_trade 无锁 → PDT 记录丢失
  P1: start_bot_thread 无重入保护 → 启动多个 poll_loop 线程
"""
import json
import threading
import pytest


class TestLogExecutionConcurrency:
    """
    P0回归：log_execution 原本不在 _PT_LOCK 中，
    scheduler（自动止损）与 Telegram /logexec 并发时可能丢失执行记录。
    修复：整个 read-modify-write 放入 with _PT_LOCK 块。
    """

    def test_no_records_lost_50_threads(self, tmp_path, monkeypatch):
        import src.paper_trading as pt
        # 将文件路径重定向到隔离的临时目录
        monkeypatch.setattr(pt, "_EXEC_LOG", str(tmp_path / "execution_log.json"))

        N = 50
        errors = []

        def call(i):
            try:
                result = pt.log_execution(
                    ticker="AAPL",
                    signal_price=100.0 + i * 0.01,
                    actual_price=100.05 + i * 0.01,
                    signal_time="10:30:00",
                    note=f"concurrent-test-{i}",
                )
                assert result["ok"] is True
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=call, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"线程中发生异常：{errors}"

        data = json.loads((tmp_path / "execution_log.json").read_text(encoding="utf-8"))
        actual = len(data["logs"])
        assert actual == N, (
            f"并发写入丢失记录：期望 {N} 条，实际 {actual} 条"
            f"（丢失 {N - actual} 条）"
        )

    def test_deviation_pct_calculated_correctly(self, tmp_path, monkeypatch):
        """同时验证 deviation_pct 计算逻辑"""
        import src.paper_trading as pt
        monkeypatch.setattr(pt, "_EXEC_LOG", str(tmp_path / "execution_log.json"))

        result = pt.log_execution(
            ticker="MSFT",
            signal_price=100.0,
            actual_price=101.0,
            signal_time="09:35:00",
        )
        assert result["deviation_pct"] == pytest.approx(1.0, abs=0.01)


class TestRecordDayTradeConcurrency:
    """
    P1回归：record_day_trade 加 _PDT_LOCK 前，并发调用可能丢失记录。
    20个并发调用必须全部落盘。
    """

    def test_no_records_lost_20_threads(self, tmp_path, monkeypatch):
        import src.pdt_guard as pdt
        monkeypatch.setattr(pdt, "_DB", str(tmp_path / "pdt_log.json"))

        N = 20
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
        errors = []

        def call(i):
            try:
                pdt.record_day_trade(tickers[i % len(tickers)])
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=call, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"线程中发生异常：{errors}"

        data = json.loads((tmp_path / "pdt_log.json").read_text(encoding="utf-8"))
        total = sum(len(v) for v in data.values())
        assert total == N, (
            f"并发写入丢失 PDT 记录：期望 {N} 条，实际 {total} 条"
        )

    def test_rolling_window_counts_correctly(self, tmp_path, monkeypatch):
        """_PDT_LOCK 保护后，get_rolling_day_trades 计数准确"""
        import src.pdt_guard as pdt
        monkeypatch.setattr(pdt, "_DB", str(tmp_path / "pdt_log.json"))

        for _ in range(3):
            pdt.record_day_trade("AAPL")

        count = pdt.get_rolling_day_trades()
        assert count == 3


class TestStartBotThreadReentrance:
    """
    P1回归：start_bot_thread 原本没有重入保护，
    Flask reloader（两次启动）会创建两个 poll_loop 线程，
    导致每条 Telegram 消息被处理两次。
    修复：_bot_started 标志 + _bot_start_lock 确保只启动一次。
    """

    def test_only_one_thread_starts_under_concurrent_calls(self, monkeypatch):
        import src.telegram_bot as bot

        # 阻止真正调用 poll_loop（避免网络请求）
        def fake_poll_loop():
            pass

        monkeypatch.setattr(bot, "poll_loop", fake_poll_loop)
        monkeypatch.setattr(bot, "_bot_started", False)  # 重置状态

        threads_returned = []
        lock = threading.Lock()

        def call():
            t = bot.start_bot_thread()
            with lock:
                threads_returned.append(t)

        threads = [threading.Thread(target=call) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        non_none = [t for t in threads_returned if t is not None]
        assert len(non_none) == 1, (
            f"期望只启动 1 个线程，实际启动了 {len(non_none)} 个"
            "（重入保护失效，会导致重复处理 Telegram 消息）"
        )

    def test_second_call_returns_none(self, monkeypatch):
        """第二次调用 start_bot_thread 应返回 None（而非新线程）"""
        import src.telegram_bot as bot

        monkeypatch.setattr(bot, "poll_loop", lambda: None)
        monkeypatch.setattr(bot, "_bot_started", False)

        t1 = bot.start_bot_thread()
        t2 = bot.start_bot_thread()

        assert t1 is not None
        assert t2 is None
