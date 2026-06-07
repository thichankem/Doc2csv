"""Tests for connection-retry behaviour in the Ollama client."""
import json

import pytest
import requests

from src import ollama_client


class _FakeResp:
    def __init__(self, pieces):
        self._lines = [json.dumps({"response": p}) for p in pieces]
        self._lines.append(json.dumps({"done": True}))

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        yield from self._lines

    def close(self):
        pass


def test_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.ConnectionError("boom")
        return _FakeResp(["hel", "lo"])

    monkeypatch.setattr(ollama_client._SESSION, "post", fake_post)
    monkeypatch.setattr(ollama_client.time, "sleep", lambda *_: None)

    out = ollama_client.generate("m", "p", max_retries=3, retry_backoff=0)
    assert out == "hello"
    assert calls["n"] == 3


def test_gives_up_after_max_retries(monkeypatch):
    def fake_post(url, **kw):
        raise requests.ConnectionError("always down")

    monkeypatch.setattr(ollama_client._SESSION, "post", fake_post)
    monkeypatch.setattr(ollama_client.time, "sleep", lambda *_: None)

    with pytest.raises(requests.ConnectionError):
        ollama_client.generate("m", "p", max_retries=2, retry_backoff=0)


def test_stop_during_retry_returns_empty(monkeypatch):
    def fake_post(url, **kw):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(ollama_client._SESSION, "post", fake_post)
    monkeypatch.setattr(ollama_client.time, "sleep", lambda *_: None)

    out = ollama_client.generate(
        "m", "p", max_retries=5, retry_backoff=0, should_stop=lambda: True
    )
    assert out == ""
