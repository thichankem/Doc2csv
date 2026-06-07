"""Tests for normalize_json_output and schema inference."""
import json

from src.pipeline import (
    build_schema_from_template,
    normalize_json_output,
)


def _parse(s):
    return json.loads(s)


def test_normalize_strips_code_fence():
    out = normalize_json_output('```json\n{"a": 1}\n```')
    assert _parse(out) == {"a": 1}


def test_normalize_strips_think_block():
    out = normalize_json_output('<think>reasoning here</think>\n{"x": [1, 2]}')
    assert _parse(out) == {"x": [1, 2]}


def test_normalize_sorts_keys_and_compacts():
    out = normalize_json_output('{"b": 2, "a": 1}')
    assert out == '{"a":1,"b":2}'


def test_normalize_preserves_unicode():
    out = normalize_json_output('{"bệnh": "sốt"}')
    assert "bệnh" in out and "sốt" in out
    assert "\\u" not in out


def test_normalize_extracts_embedded_json():
    out = normalize_json_output('Sure! {"k": "v"} hope that helps')
    assert _parse(out) == {"k": "v"}


def test_normalize_garbage_falls_back_to_raw():
    out = normalize_json_output("totally not json")
    assert _parse(out) == {"raw": "totally not json"}


def test_normalize_empty_falls_back():
    out = normalize_json_output("")
    assert _parse(out) == {"raw": ""}


def test_build_schema_basic():
    schema = build_schema_from_template('{"bệnh": [], "tuổi": 0}')
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"bệnh", "tuổi"}
    assert schema["additionalProperties"] is False
    assert schema["properties"]["bệnh"] == {"type": "array", "items": {"type": "string"}}
    assert schema["properties"]["tuổi"] == {"type": "number"}


def test_build_schema_nested():
    schema = build_schema_from_template('{"qa": {"q": "", "a": ""}}')
    inner = schema["properties"]["qa"]
    assert inner["type"] == "object"
    assert set(inner["required"]) == {"q", "a"}


def test_build_schema_invalid_returns_none():
    assert build_schema_from_template("not json") is None
    assert build_schema_from_template("") is None
    assert build_schema_from_template("[1,2,3]") is None  # not an object
