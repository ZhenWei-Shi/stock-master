"""telegram_bot指令精简后的行为测试（不联网，不发消息）。"""
import pytest

import src.telegram_bot as bot


@pytest.fixture
def sent(monkeypatch):
    out = []
    monkeypatch.setattr(bot, "send", lambda msg: out.append(msg))
    return out


def test_help_lists_only_current_commands(sent):
    bot.handle_command("/help")
    text = sent[0]
    for c in ("/list", "/add", "/remove", "/events", "/perf", "/risk", "/gex", "/uoa", "/status"):
        assert c in text
    for c in bot.RETIRED_COMMANDS:
        assert c + " " not in text and c + "\n" not in text


@pytest.mark.parametrize("cmd", sorted(bot.RETIRED_COMMANDS))
def test_retired_commands_say_so(sent, cmd):
    bot.handle_command(cmd + " NVDA")
    assert "已下线" in sent[0]


def test_unknown_command(sent):
    bot.handle_command("/nope")
    assert "未知指令" in sent[0]


def test_uoa_without_args_shows_watchlist(sent, monkeypatch):
    monkeypatch.setattr("src.smart_money.get_uoa_watchlist", lambda: ["NVDA", "ASTS"])
    bot.handle_command("/uoa")
    assert "UOA自动监控列表" in sent[0] and "NVDA, ASTS" in sent[0]


def test_status_shows_current_schedule(sent, monkeypatch):
    monkeypatch.setattr(bot, "read_watchlist", lambda: ["NVDA"])
    bot.handle_command("/status")
    assert "事件实验室" in sent[0] and "/hotlist" not in sent[0]
