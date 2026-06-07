"""Tests for the OpenAI-compatible streaming client."""
import json

import pytest
import requests

from src import openai_client
from src.openai_client import RateLimitError


class _FakeResp:
    def __init__(self, pieces, status=200, headers=None):
        self.status_code = status
        self.headers = headers or {}
        self._lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": p}}]})
            for p in pieces
        ]
        self._lines.append("data: [DONE]")
        self.captured = None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def iter_lines(self, decode_unicode=True):
        yield from self._lines

    def close(self):
        pass


def test_chat_streams_content(monkeypatch):
    captured = {}

    def fake_post(url, **kw):
        captured["url"] = url
        captured["json"] = kw.get("json")
        captured["headers"] = kw.get("headers")
        return _FakeResp(["Xin ", "chào"])

    monkeypatch.setattr(openai_client._SESSION, "post", fake_post)
    out = openai_client.chat("https://api.x/v1", "KEY", "m", "hi")
    assert out == "Xin chào"
    assert captured["url"] == "https://api.x/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer KEY"
    assert captured["json"]["model"] == "m"
    assert captured["json"]["stream"] is True


def test_chat_forces_utf8_and_preserves_unicode(monkeypatch):
    resp = _FakeResp(["Bệnh nhân ", "sốt cao"])

    def fake_post(url, **kw):
        return resp

    monkeypatch.setattr(openai_client._SESSION, "post", fake_post)
    out = openai_client.chat("https://api.x/v1", "K", "m", "hi")
    assert out == "Bệnh nhân sốt cao"
    # Must pin UTF-8 so requests doesn't decode the SSE stream as latin-1.
    assert resp.encoding == "utf-8"


def test_chat_json_object_mode(monkeypatch):
    captured = {}

    def fake_post(url, **kw):
        captured["json"] = kw.get("json")
        return _FakeResp(['{"a":1}'])

    monkeypatch.setattr(openai_client._SESSION, "post", fake_post)
    openai_client.chat("https://api.x/v1", "K", "m", "hi", format_json=True)
    assert captured["json"]["response_format"] == {"type": "json_object"}


def test_chat_json_schema_mode(monkeypatch):
    captured = {}

    def fake_post(url, **kw):
        captured["json"] = kw.get("json")
        return _FakeResp(['{"a":1}'])

    monkeypatch.setattr(openai_client._SESSION, "post", fake_post)
    schema = {"type": "object", "properties": {"a": {"type": "number"}}}
    openai_client.chat("https://api.x/v1", "K", "m", "hi", json_schema=schema)
    rf = captured["json"]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == schema


def test_chat_429_raises_ratelimit(monkeypatch):
    def fake_post(url, **kw):
        return _FakeResp([], status=429, headers={"retry-after": "12"})

    monkeypatch.setattr(openai_client._SESSION, "post", fake_post)
    with pytest.raises(RateLimitError) as ei:
        openai_client.chat("https://api.x/v1", "K", "m", "hi")
    assert ei.value.retry_after == 12.0


def test_chat_retries_connection_error(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, **kw):
        calls["n"] += 1
        if calls["n"] < 2:
            raise requests.ConnectionError("down")
        return _FakeResp(["ok"])

    monkeypatch.setattr(openai_client._SESSION, "post", fake_post)
    monkeypatch.setattr(openai_client.time, "sleep", lambda *_: None)
    out = openai_client.chat("https://api.x/v1", "K", "m", "hi",
                             max_retries=2, retry_backoff=0)
    assert out == "ok"
    assert calls["n"] == 2
