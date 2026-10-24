import pytest

from src import telegram_bot, event_lab, overnight_lab, options_watch


# 合成 v2 记录；字段需与 tests/test_h4_summary_gate.py 既有夹具/evaluate_h4 契约对齐
def _t(group, date, ret, i):
    return {'pool': 'v2', 'group': group, 'date': date, 'ret': ret, 'ticker': f'{group}-{date}-{i}'}


SCENARIOS = {
    'all_pending': [_t('signal', '2026-10-01', None, 0), _t('control', '2026-10-01', None, 0)],
    'one_day_done': [_t('signal', '2026-10-01', 0.002, 0), _t('control', '2026-10-01', 0.001, 0)],
    'n200_same_day': ([_t('signal', '2026-10-01', 0.001, i) for i in range(200)]
                      + [_t('control', '2026-10-01', 0.0, i) for i in range(200)]),
}

EXPECT_IN = {
    'all_pending': ['可评价signal不足200笔', '可评价日期不足30天'],
    'one_day_done': ['可评价signal不足200笔', '可评价日期不足30天'],
    'n200_same_day': ['可评价日期不足30天'],
}

EXPECT_NOT_IN = {
    'all_pending': [],
    'one_day_done': [],
    'n200_same_day': ['可评价signal不足'],
}


@pytest.mark.parametrize('name', list(SCENARIOS))
def test_events_payload_html_safe(monkeypatch, name):
    captured = []
    saves = []

    def fake_post(url, json=None, timeout=None, **kw):
        captured.append({'url': url, 'json': json})

    monkeypatch.setattr(telegram_bot, '_token', lambda: 'SYNTHETIC_TOKEN')
    monkeypatch.setattr(telegram_bot, '_chat_id', lambda: 'SYNTHETIC_CHAT')
    monkeypatch.setattr(telegram_bot.requests, 'post', fake_post)
    monkeypatch.setattr(event_lab, 'summarize', lambda *a, **k: 'EVENT synthetic')
    monkeypatch.setattr(options_watch, 'status_line', lambda *a, **k: 'OPTIONS synthetic')
    monkeypatch.setattr(overnight_lab, '_load', lambda: {'trades': [dict(t) for t in SCENARIOS[name]]})
    monkeypatch.setattr(overnight_lab, '_save', lambda *a, **k: saves.append(a), raising=False)

    telegram_bot.handle_command('/events')

    assert len(captured) == 1
    payload = captured[0]['json']
    assert payload['parse_mode'] == 'HTML'
    assert payload['chat_id'] == 'SYNTHETIC_CHAT'
    text = payload['text']
    for ch in '<>&':
        assert ch not in text, (name, ch, text)
    assert 'EVENT synthetic' in text and 'OPTIONS synthetic' in text
    assert 'H4' in text and '⏳ 样本不足' in text and '✅' not in text
    for s in EXPECT_IN[name]:
        assert s in text, (name, s, text)
    for s in EXPECT_NOT_IN[name]:
        assert s not in text, (name, s, text)
    assert saves == []
