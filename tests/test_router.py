"""Tests for the round-robin LLMRouter and provider config loading."""
import pytest

from src.llm_backends import LLMRouter, OpenAIBackend
from src.openai_client import RateLimitError
from src.providers import parse_backends


class FakeBackend:
    kind = "fake"

    def __init__(self, name, behavior="ok"):
        self.name = name
        self.behavior = behavior   # "ok" | "raise" | "ratelimit"
        self.calls = 0

    def generate(self, prompt, **kw):
        self.calls += 1
        if self.behavior == "raise":
            raise RuntimeError("boom")
        if self.behavior == "ratelimit":
            raise RateLimitError("429", retry_after=0.01)
        return f"{self.name}:reply"


def _gen(router, should_stop=None):
    return router.generate(
        "p", temperature=0.1, num_predict=128, num_ctx=2048,
        format_json=True, json_schema=None, keep_alive="30m",
        on_token=None, should_stop=should_stop,
    )


def test_round_robin_advances():
    a, b, c = FakeBackend("a"), FakeBackend("b"), FakeBackend("c")
    r = LLMRouter([a, b, c])
    assert _gen(r) == "a:reply"
    assert _gen(r) == "b:reply"
    assert _gen(r) == "c:reply"
    assert _gen(r) == "a:reply"  # wraps around
    assert r.last_backend == "a"


def test_failover_to_next_backend():
    bad, good = FakeBackend("bad", "raise"), FakeBackend("good")
    r = LLMRouter([bad, good])
    assert _gen(r) == "good:reply"
    assert bad.calls == 1 and good.calls == 1


def test_ratelimit_cools_down_and_rotates():
    rl, ok = FakeBackend("rl", "ratelimit"), FakeBackend("ok")
    r = LLMRouter([rl, ok], cooldown=0.01)
    # First call: rl rate-limited → falls over to ok.
    assert _gen(r) == "ok:reply"
    # Next call starts at rl again but it's cooling → straight to ok.
    assert _gen(r) == "ok:reply"


def test_all_fail_raises():
    r = LLMRouter([FakeBackend("x", "raise"), FakeBackend("y", "raise")])
    with pytest.raises(RuntimeError):
        _gen(r)


def test_stop_during_run_returns_empty():
    r = LLMRouter([FakeBackend("a")])
    assert _gen(r, should_stop=lambda: True) == ""


def test_empty_router_rejected():
    with pytest.raises(ValueError):
        LLMRouter([])


def test_parse_backends_expands_models(monkeypatch):
    monkeypatch.setenv("MYKEY", "secret-123")
    cfg = {
        "endpoints": [
            {
                "name": "prov",
                "base_url": "https://api.x/v1",
                "api_key": "env:MYKEY",
                "models": ["m1", "m2"],
            },
            {"name": "off", "base_url": "https://y/v1", "enabled": False,
             "models": ["z"]},
        ]
    }
    backends = parse_backends(cfg)
    assert [b.name for b in backends] == ["prov:m1", "prov:m2"]
    assert all(isinstance(b, OpenAIBackend) for b in backends)
    assert backends[0].api_key == "secret-123"


def test_parse_backends_literal_key_and_single_model():
    cfg = {"endpoints": [
        {"name": "p", "base_url": "https://a/v1", "api_key": "lit", "model": "solo"}
    ]}
    backends = parse_backends(cfg)
    assert len(backends) == 1
    assert backends[0].model == "solo"
    assert backends[0].api_key == "lit"
