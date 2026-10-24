from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from src import telegram_bot


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(telegram_bot, '_token', lambda: 'SYNTHETIC_SECRET')
    monkeypatch.setattr(telegram_bot, '_chat_id', lambda: 'SYNTHETIC_CHAT')


def response(status=200, body=None):
    return SimpleNamespace(status_code=status, json=lambda: body)


@pytest.mark.parametrize('http,body,expected', [
    (200, {'ok': True, 'result': {'message_id': 1}}, 'sent'),
    (200, {'ok': False, 'description': 'SYNTHETIC_SECRET'}, 'rejected'),
    (400, {'ok': False, 'description': 'SYNTHETIC_CHAT'}, 'rejected'),
    (500, {'ok': False}, 'unknown'),
    (400, {'ok': True, 'result': {}}, 'unknown'),
    (200, {'ok': 'true', 'result': {}}, 'unknown'),
    (200, {'ok': True, 'result': []}, 'unknown'),
])
def test_send_result_is_explicit(configured, monkeypatch, capsys, http, body, expected):
    post = Mock(return_value=response(http, body))
    monkeypatch.setattr(telegram_bot.requests, 'post', post)
    result = telegram_bot.send('SYNTHETIC_PRIVATE_TEXT')
    assert isinstance(result, dict) and result['status'] == expected
    post.assert_called_once()
    assert post.call_args.kwargs['json']['parse_mode'] == 'HTML'
    assert post.call_args.kwargs['timeout'] == 10
    output = capsys.readouterr().out + repr(result)
    for private in ['SYNTHETIC_SECRET', 'SYNTHETIC_CHAT', 'SYNTHETIC_PRIVATE_TEXT']:
        assert private not in output


def test_ambiguous_send_never_reposts_or_logs_exception(configured, monkeypatch, capsys):
    post = Mock(side_effect=requests.Timeout('SYNTHETIC_SECRET SYNTHETIC_CHAT SYNTHETIC_PRIVATE_TEXT'))
    monkeypatch.setattr(telegram_bot.requests, 'post', post)
    result = telegram_bot.send('SYNTHETIC_PRIVATE_TEXT')
    assert isinstance(result, dict) and result['status'] == 'unknown'
    post.assert_called_once()
    assert 'SYNTHETIC' not in capsys.readouterr().out + repr(result)


@pytest.mark.parametrize('token,chat', [('', 'CHAT'), ('TOKEN', '')])
def test_missing_configuration_does_not_send(monkeypatch, token, chat):
    monkeypatch.setattr(telegram_bot, '_token', lambda: token)
    monkeypatch.setattr(telegram_bot, '_chat_id', lambda: chat)
    post = Mock()
    monkeypatch.setattr(telegram_bot.requests, 'post', post)
    result = telegram_bot.send('TEXT')
    assert isinstance(result, dict) and result['status'] == 'disabled'
    post.assert_not_called()


@pytest.mark.parametrize('fake', [None, response(200, []), response(200, {})])
def test_malformed_response_is_unknown(configured, monkeypatch, fake):
    monkeypatch.setattr(telegram_bot.requests, 'post', Mock(return_value=fake))
    result = telegram_bot.send('TEXT')
    assert isinstance(result, dict) and result['status'] == 'unknown'
