"""Offline-path coverage for the local Ollama backend.

Focuses on the parts the existing suite leaves untested: the HTTP payload
`generate()` actually builds, mid-stream stop / malformed-line handling, the
`is_running` / `list_models` / `preload` helpers, image-OCR wiring, and the
no-router Pipeline path (preload failure, serial stop). No real Ollama needed —
the requests Session is monkeypatched.
"""
import json

import pytest
import requests

from src import ollama_client
from src import pipeline as pl
from src.extractors import image_extractor as ie


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _StreamResp:
    """Mimics a streaming requests.Response for /api/generate."""

    def __init__(self, lines):
        self._lines = list(lines)
        self.encoding = None
        self.closed = False

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        yield from self._lines

    def close(self):
        self.closed = True


def _stream_lines(pieces, done=True):
    lines = [json.dumps({"response": p}) for p in pieces]
    if done:
        lines.append(json.dumps({"done": True}))
    return lines


class _GetResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


# --------------------------------------------------------------------------- #
# generate(): payload building
# --------------------------------------------------------------------------- #
def _capture_post(monkeypatch, lines=None):
    """Patch the session POST to capture kwargs and return a stream response."""
    captured = {}

    def fake_post(url, **kw):
        captured["url"] = url
        captured["json"] = kw.get("json")
        captured["timeout"] = kw.get("timeout")
        captured["stream"] = kw.get("stream")
        return _StreamResp(lines if lines is not None else _stream_lines(["ok"]))

    monkeypatch.setattr(ollama_client._SESSION, "post", fake_post)
    return captured


def test_generate_payload_defaults(monkeypatch):
    cap = _capture_post(monkeypatch)
    ollama_client.generate("mymodel", "hello")
    body = cap["json"]
    assert body["model"] == "mymodel"
    assert body["prompt"] == "hello"
    assert body["stream"] is True
    assert body["options"]["temperature"] == 0.3
    assert body["options"]["num_ctx"] == 8192
    # No optional fields unless asked for.
    assert "format" not in body
    assert "images" not in body
    assert "keep_alive" not in body
    assert cap["stream"] is True


def test_generate_options_merge_and_override(monkeypatch):
    cap = _capture_post(monkeypatch)
    ollama_client.generate(
        "m", "p", temperature=0.9, num_ctx=2048,
        options={"num_predict": 256, "num_ctx": 4096},
    )
    opts = cap["json"]["options"]
    assert opts["temperature"] == 0.9
    assert opts["num_predict"] == 256
    # options dict wins over the num_ctx kwarg on conflict.
    assert opts["num_ctx"] == 4096


def test_generate_format_json_flag(monkeypatch):
    cap = _capture_post(monkeypatch)
    ollama_client.generate("m", "p", format_json=True)
    assert cap["json"]["format"] == "json"


def test_generate_schema_takes_priority_over_format_json(monkeypatch):
    cap = _capture_post(monkeypatch)
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    ollama_client.generate("m", "p", format_json=True, json_schema=schema)
    # Structured schema must be sent as the format, not the loose "json".
    assert cap["json"]["format"] == schema


def test_generate_keep_alive_and_images(monkeypatch):
    cap = _capture_post(monkeypatch)
    ollama_client.generate(
        "m", "p", keep_alive="30m", images=["BASE64DATA"],
    )
    assert cap["json"]["keep_alive"] == "30m"
    assert cap["json"]["images"] == ["BASE64DATA"]


def test_generate_timeout_tuple(monkeypatch):
    cap = _capture_post(monkeypatch)
    ollama_client.generate("m", "p", timeout=123, connect_timeout=4.0)
    assert cap["timeout"] == (4.0, 123)


# --------------------------------------------------------------------------- #
# generate(): streaming behaviour
# --------------------------------------------------------------------------- #
def test_generate_concatenates_stream(monkeypatch):
    _capture_post(monkeypatch, lines=_stream_lines(["he", "ll", "o"]))
    assert ollama_client.generate("m", "p") == "hello"


def test_generate_on_token_cumulative(monkeypatch):
    _capture_post(monkeypatch, lines=_stream_lines(["ab", "c"]))
    seen = []
    ollama_client.generate("m", "p", on_token=lambda piece, total: seen.append((piece, total)))
    assert seen == [("ab", 2), ("c", 3)]


def test_generate_on_token_exception_is_swallowed(monkeypatch):
    _capture_post(monkeypatch, lines=_stream_lines(["x", "y"]))

    def boom(piece, total):
        raise ValueError("callback blew up")

    # A throwing UI callback must never break generation.
    assert ollama_client.generate("m", "p", on_token=boom) == "xy"


def test_generate_skips_malformed_json_lines(monkeypatch):
    lines = [
        json.dumps({"response": "a"}),
        "this is not json",          # must be skipped, not crash
        "",                          # blank line skipped
        json.dumps({"response": "b"}),
        json.dumps({"done": True}),
    ]
    _capture_post(monkeypatch, lines=lines)
    assert ollama_client.generate("m", "p") == "ab"


def test_generate_stops_mid_stream_returns_partial(monkeypatch):
    _capture_post(monkeypatch, lines=_stream_lines(["a", "b", "c"]))
    calls = {"n": 0}

    def should_stop():
        # False for the first line, True afterwards → keep only "a".
        calls["n"] += 1
        return calls["n"] > 1

    out = ollama_client.generate("m", "p", should_stop=should_stop)
    assert out == "a"


def test_generate_closes_response(monkeypatch):
    resp = _StreamResp(_stream_lines(["x"]))
    monkeypatch.setattr(ollama_client._SESSION, "post", lambda url, **kw: resp)
    ollama_client.generate("m", "p")
    assert resp.closed is True


def test_generate_mid_stream_error_not_retried(monkeypatch):
    """A failure that happens *after* streaming started must propagate, not retry
    (retrying would risk duplicating already-emitted tokens)."""
    posts = {"n": 0}

    class _Boom:
        encoding = None

        def raise_for_status(self):
            pass

        def iter_lines(self, decode_unicode=True):
            yield json.dumps({"response": "partial"})
            raise requests.ConnectionError("dropped mid-stream")

        def close(self):
            pass

    def fake_post(url, **kw):
        posts["n"] += 1
        return _Boom()

    monkeypatch.setattr(ollama_client._SESSION, "post", fake_post)
    monkeypatch.setattr(ollama_client.time, "sleep", lambda *_: None)

    with pytest.raises(requests.ConnectionError):
        ollama_client.generate("m", "p", max_retries=3)
    assert posts["n"] == 1   # connected once; mid-stream error was NOT retried


# --------------------------------------------------------------------------- #
# is_running / list_models / preload
# --------------------------------------------------------------------------- #
def test_is_running_true(monkeypatch):
    monkeypatch.setattr(ollama_client._SESSION, "get",
                        lambda url, **kw: _GetResp(200))
    assert ollama_client.is_running() is True


def test_is_running_false_on_exception(monkeypatch):
    def boom(url, **kw):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(ollama_client._SESSION, "get", boom)
    assert ollama_client.is_running() is False


def test_is_running_false_on_non_200(monkeypatch):
    monkeypatch.setattr(ollama_client._SESSION, "get",
                        lambda url, **kw: _GetResp(503))
    assert ollama_client.is_running() is False


def test_list_models_parses_names(monkeypatch):
    payload = {"models": [{"name": "llama3"}, {"name": "qwen2.5"},
                          {"size": 1}]}  # entry without name is skipped
    monkeypatch.setattr(ollama_client._SESSION, "get",
                        lambda url, **kw: _GetResp(200, payload))
    assert ollama_client.list_models() == ["llama3", "qwen2.5"]


def test_list_models_empty_on_error(monkeypatch):
    def boom(url, **kw):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(ollama_client._SESSION, "get", boom)
    assert ollama_client.list_models() == []


def test_preload_posts_empty_prompt(monkeypatch):
    cap = {}

    def fake_post(url, **kw):
        cap["url"] = url
        cap["json"] = kw.get("json")
        return _GetResp(200)

    monkeypatch.setattr(ollama_client._SESSION, "post", fake_post)
    assert ollama_client.preload("mymodel", keep_alive="10m") is True
    assert cap["json"]["model"] == "mymodel"
    assert cap["json"]["prompt"] == ""
    assert cap["json"]["stream"] is False
    assert cap["json"]["keep_alive"] == "10m"


def test_preload_false_on_exception(monkeypatch):
    def boom(url, **kw):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(ollama_client._SESSION, "post", boom)
    assert ollama_client.preload("m") is False


# --------------------------------------------------------------------------- #
# image OCR wiring (offline vision)
# --------------------------------------------------------------------------- #
def test_extract_image_missing_file():
    with pytest.raises(FileNotFoundError):
        ie.extract_image("nope_does_not_exist.png", model="llava")


def test_extract_image_requires_model(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"not really a png but exists")
    with pytest.raises(RuntimeError):
        ie.extract_image(str(img), model="")


def test_extract_image_sends_image_to_generate(tmp_path, monkeypatch):
    img = tmp_path / "a.png"
    img.write_bytes(b"rawbytes")
    cap = {}

    def fake_generate(model, prompt, **kw):
        cap["model"] = model
        cap["images"] = kw.get("images")
        cap["temperature"] = kw.get("temperature")
        return "  transcribed text  "

    monkeypatch.setattr(ie, "generate", fake_generate)
    out = ie.extract_image(str(img), model="llava")
    assert out == "transcribed text"          # .strip() applied
    assert cap["model"] == "llava"
    assert isinstance(cap["images"], list) and len(cap["images"]) == 1
    assert cap["temperature"] == 0.0          # deterministic OCR


# --------------------------------------------------------------------------- #
# Pipeline offline path (router=None)
# --------------------------------------------------------------------------- #
def _txt(tmp_path, paras=6):
    doc = tmp_path / "doc.txt"
    doc.write_text("\n\n".join(f"Đoạn {i} có một ít nội dung ở đây." for i in range(paras)),
                   encoding="utf-8")
    return doc


def test_pipeline_offline_runs_despite_preload_failure(tmp_path, monkeypatch):
    """preload() returning False must only warn — the run still completes."""
    monkeypatch.setattr(pl, "preload", lambda *a, **k: False)
    monkeypatch.setattr(pl, "generate", lambda model, prompt, **kw: '{"s":"ok"}')
    logs = []
    out = tmp_path / "out.csv"
    p = pl.Pipeline(files=[str(_txt(tmp_path))], model="m", output_csv=str(out),
                    instruction="Tóm tắt", chunk_words=5, max_json_retries=0,
                    on_log=logs.append)
    stats = p.run()
    assert stats["chunks"] > 0
    assert stats["errors"] == 0
    assert any("Không preload" in m for m in logs)


def test_pipeline_offline_passes_model_and_url(tmp_path, monkeypatch):
    """The no-router path must call generate() with the configured model/url."""
    monkeypatch.setattr(pl, "preload", lambda *a, **k: True)
    cap = {}

    def fake_generate(model, prompt, **kw):
        cap["model"] = model
        cap["base_url"] = kw.get("base_url")
        cap["format_json"] = kw.get("format_json")
        return '{"s":"ok"}'

    monkeypatch.setattr(pl, "generate", fake_generate)
    out = tmp_path / "out.csv"
    p = pl.Pipeline(files=[str(_txt(tmp_path, paras=2))], model="mymodel",
                    output_csv=str(out), instruction="x", chunk_words=5,
                    ollama_url="http://host:9999", max_json_retries=0)
    p.run()
    assert cap["model"] == "mymodel"
    assert cap["base_url"] == "http://host:9999"
    assert cap["format_json"] is True


def test_pipeline_offline_serial_stop_midway(tmp_path, monkeypatch):
    """With Stop flagged, the serial offline loop must halt and write nothing."""
    monkeypatch.setattr(pl, "preload", lambda *a, **k: True)
    monkeypatch.setattr(pl, "generate", lambda model, prompt, **kw: '{"s":"ok"}')
    out = tmp_path / "out.csv"
    p = pl.Pipeline(files=[str(_txt(tmp_path, paras=10))], model="m",
                    output_csv=str(out), instruction="x", chunk_words=5,
                    max_json_retries=0, should_stop=lambda: True)
    stats = p.run()
    assert stats["chunks"] == 0


def test_pipeline_offline_raw_fallback_counted(tmp_path, monkeypatch):
    """Unparseable model output still lands as a {"raw": ...} row, counted as a
    raw fallback rather than lost."""
    monkeypatch.setattr(pl, "preload", lambda *a, **k: True)
    monkeypatch.setattr(pl, "generate", lambda model, prompt, **kw: "totally not json")
    out = tmp_path / "out.csv"
    p = pl.Pipeline(files=[str(_txt(tmp_path, paras=3))], model="m",
                    output_csv=str(out), instruction="x", chunk_words=5,
                    max_json_retries=0)
    stats = p.run()
    assert stats["chunks"] > 0
    assert stats["raw_fallbacks"] == stats["chunks"]
    assert stats["errors"] == 0


def test_pipeline_missing_instruction_is_rejected(tmp_path, monkeypatch):
    """Instruction is mandatory — an all-whitespace one aborts before any LLM
    call (the model would have nothing to do)."""
    called = {"n": 0}
    monkeypatch.setattr(pl, "preload", lambda *a, **k: True)

    def fake_generate(*a, **k):
        called["n"] += 1
        return "{}"

    monkeypatch.setattr(pl, "generate", fake_generate)
    p = pl.Pipeline(files=[str(_txt(tmp_path))], model="m",
                    output_csv=str(tmp_path / "out.csv"), instruction="   ")
    stats = p.run()
    assert stats["errors"] == 1
    assert stats["chunks"] == 0
    assert called["n"] == 0          # never reached the model


def test_pipeline_empty_file_skipped(tmp_path, monkeypatch):
    """A file that extracts to no words is logged and skipped, not errored."""
    monkeypatch.setattr(pl, "preload", lambda *a, **k: True)
    monkeypatch.setattr(pl, "generate", lambda model, prompt, **kw: '{"s":"ok"}')
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n\n  ", encoding="utf-8")
    logs = []
    p = pl.Pipeline(files=[str(empty)], model="m",
                    output_csv=str(tmp_path / "out.csv"), instruction="x",
                    chunk_words=5, on_log=logs.append)
    stats = p.run()
    assert stats["chunks"] == 0
    assert stats["errors"] == 0
    assert any("rỗng" in m for m in logs)


def test_pipeline_extraction_error_logged_not_fatal(tmp_path, monkeypatch):
    """If one file fails to extract, it's logged and the run continues with the
    rest rather than crashing the whole batch."""
    monkeypatch.setattr(pl, "preload", lambda *a, **k: True)
    monkeypatch.setattr(pl, "generate", lambda model, prompt, **kw: '{"s":"ok"}')
    good = _txt(tmp_path, paras=3)
    bad = tmp_path / "bad.txt"
    bad.write_text("content", encoding="utf-8")

    real_extract = pl.extract_text

    def flaky_extract(path, *a, **k):
        if path == str(bad):
            raise OSError("disk gremlin")
        return real_extract(path, *a, **k)

    monkeypatch.setattr(pl, "extract_text", flaky_extract)
    logs = []
    p = pl.Pipeline(files=[str(bad), str(good)], model="m",
                    output_csv=str(tmp_path / "out.csv"), instruction="x",
                    chunk_words=5, on_log=logs.append)
    stats = p.run()
    assert stats["chunks"] > 0                      # good file still processed
    assert any("disk gremlin" in m for m in logs)   # bad file surfaced


def test_pipeline_stop_during_json_retry(tmp_path, monkeypatch):
    """Stop set after the first bad answer must break the retry loop instead of
    burning through every retry attempt."""
    monkeypatch.setattr(pl, "preload", lambda *a, **k: True)
    p = pl.Pipeline(files=[], model="m", output_csv=str(tmp_path / "o.csv"),
                    instruction="x", max_json_retries=5)
    calls = {"n": 0}

    def fake_call(prompt, idx, total, temperature=None, num_predict=None):
        calls["n"] += 1
        return "not json"

    stop = {"flag": False}
    p._call_llm = fake_call
    p.should_stop = lambda: stop["flag"]

    # First call returns bad JSON; then Stop is pressed → no further attempts.
    def driver():
        out, ok = p._generate_chunk_json("prompt", 1, 1)
        return out, ok

    # Flip stop right after the first model call by wrapping it.
    orig = fake_call

    def wrapped(prompt, idx, total, temperature=None, num_predict=None):
        res = orig(prompt, idx, total, temperature, num_predict)
        stop["flag"] = True
        return res

    p._call_llm = wrapped
    out, ok = driver()
    assert ok is False
    assert calls["n"] == 1            # stopped before any retry
