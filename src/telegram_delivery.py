import requests

API_URL = "https://api.telegram.org/bot{token}/sendMessage"


def _result(status, reason):
    return {"status": status, "reason": reason}


def send_message(text, token, chat_id, post=None):
    if not token or not chat_id:
        return _result("disabled", "missing_credentials")
    poster = post if post is not None else requests.post  # 调用时解析，便于 monkeypatch
    try:
        response = poster(
            API_URL.format(token=token),
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        return _result("unknown", "transport_error")
    if response is None:
        return _result("unknown", "no_response")
    code = getattr(response, "status_code", None)
    if isinstance(code, bool) or not isinstance(code, int) or not 100 <= code <= 599:
        return _result("unknown", "malformed_status")
    if code >= 500:
        return _result("unknown", "server_error")
    try:
        body = response.json()
    except Exception:
        return _result("unknown", "malformed_json")
    ok = body.get("ok") if isinstance(body, dict) else None
    if ok is True and 200 <= code < 300:
        if isinstance(body.get("result"), dict):
            return _result("sent", "api_accepted")
        return _result("unknown", "malformed_result")
    if ok is True:
        return _result("unknown", "contradictory_response")
    if ok is False:
        return _result("rejected", "api_rejected")
    return _result("unknown", "malformed_ok")


def notify(text, token, chat_id, post=None):
    outcome = send_message(text, token, chat_id, post=post)
    if outcome["status"] != "sent":
        print(f"telegram delivery: {outcome['status']} ({outcome['reason']})")
    return outcome
