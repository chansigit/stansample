import io
import json
import urllib.error
import urllib.request

import pytest

from stanmetacols.llm_client import (
    LLMUnavailable, extract_json, OpenAICompatClient, call_structured)


class _Stub:
    """Minimal .complete(system, user) -> str client."""
    def __init__(self, content=None, raise_exc=None):
        self._content = content
        self._raise = raise_exc
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        if self._raise is not None:
            raise self._raise
        return self._content


# --- extract_json ---
def test_extract_json_plain_object():
    assert json.loads(extract_json('{"a": 1}')) == {"a": 1}

def test_extract_json_strips_fences_and_prose():
    assert json.loads(extract_json('note:\n```json\n{"a": 1}\n```\nthx')) == {"a": 1}

def test_extract_json_array():
    assert json.loads(extract_json('[1, 2, 3]')) == [1, 2, 3]

def test_extract_json_empty_raises():
    with pytest.raises(LLMUnavailable):
        extract_json("")

def test_extract_json_no_json_raises():
    with pytest.raises(LLMUnavailable):
        extract_json("no json here")


# --- call_structured: schema-agnostic via a parse callable ---
class _Box:
    def __init__(self, items):
        self.items = items
    @classmethod
    def model_validate(cls, data):          # duck-types a pydantic model
        return cls(data["items"])

def test_call_structured_with_model_validate():
    client = _Stub(json.dumps({"items": [1, 2]}))
    box = call_structured(client, "sys", "usr", _Box.model_validate)
    assert box.items == [1, 2]
    assert client.calls == [("sys", "usr")]           # two-positional protocol

def test_call_structured_with_plain_callable():
    client = _Stub(json.dumps({"x": 5}))
    assert call_structured(client, "s", "u", lambda d: d["x"] * 2) == 10

def test_call_structured_list_key_wraps_bare_array():
    client = _Stub(json.dumps([1, 2, 3]))
    assert call_structured(client, "s", "u", lambda d: d["nums"], list_key="nums") == [1, 2, 3]

def test_call_structured_fenced_json():
    client = _Stub("```\n{\"x\": 1}\n```")
    assert call_structured(client, "s", "u", lambda d: d["x"]) == 1

def test_call_structured_non_json_raises():
    with pytest.raises(LLMUnavailable):
        call_structured(_Stub("sorry, no"), "s", "u", lambda d: d)

def test_call_structured_client_error_becomes_unavailable():
    with pytest.raises(LLMUnavailable):
        call_structured(_Stub(raise_exc=RuntimeError("net")), "s", "u", lambda d: d)

def test_call_structured_parse_error_becomes_unavailable():
    def boom(_):
        raise ValueError("bad shape")
    with pytest.raises(LLMUnavailable):
        call_structured(_Stub(json.dumps({"x": 1})), "s", "u", boom)


# --- OpenAICompatClient over a stubbed urlopen ---
class _Resp:
    def __init__(self, payload):
        self._p = payload
    def read(self):
        return json.dumps(self._p).encode("utf-8")
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False

def test_client_builds_url_auth_and_body(monkeypatch):
    captured = {}
    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["auth"] = req.headers.get("Authorization")
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp({"choices": [{"message": {"content": '{"ok": true}'}}]})
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatClient("https://api.example.com/v1", "KEY", "m",
                                temperature=0, max_tokens=64)
    assert client.complete("SYS", "USR") == '{"ok": true}'
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["auth"] == "Bearer KEY"
    assert captured["body"]["model"] == "m"
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["max_tokens"] == 64
    assert captured["body"]["messages"][0] == {"role": "system", "content": "SYS"}
    assert captured["body"]["messages"][1] == {"role": "user", "content": "USR"}

def test_client_full_endpoint_url_not_doubled(monkeypatch):
    captured = {}
    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _Resp({"choices": [{"message": {"content": "{}"}}]})
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    OpenAICompatClient("https://ark.example/api/v3/chat/completions", "K", "m").complete("s", "u")
    assert captured["url"] == "https://ark.example/api/v3/chat/completions"

def test_client_omits_none_knobs(monkeypatch):
    captured = {}
    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp({"choices": [{"message": {"content": "{}"}}]})
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    OpenAICompatClient("https://x/v1", "K", "m").complete("s", "u")
    assert "temperature" not in captured["body"]
    assert "max_tokens" not in captured["body"]

def test_client_tolerates_list_parts_content(monkeypatch):
    def fake_urlopen(req, timeout=None):
        return _Resp({"choices": [{"message": {"content": [
            {"type": "text", "text": "hel"}, {"type": "text", "text": "lo"}]}}]})
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert OpenAICompatClient("https://x/v1", "K", "m").complete("s", "u") == "hello"

def test_client_http_error_becomes_unavailable(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, io.BytesIO(b"server error"))
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(LLMUnavailable):
        OpenAICompatClient("https://x/v1", "K", "m").complete("s", "u")
