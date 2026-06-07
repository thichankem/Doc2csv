"""Tests for JSON-retry, schema validation and resume in the pipeline."""
import csv
import json

from src.pipeline import Pipeline, parse_json_lenient


def _mk(tmp_path, **kw):
    return Pipeline(
        files=[],
        model="m",
        output_csv=str(tmp_path / "out.csv"),
        instruction="do it",
        **kw,
    )


def test_parse_json_lenient_variants():
    assert parse_json_lenient('{"a":1}') == {"a": 1}
    assert parse_json_lenient('```json\n{"a":1}\n```') == {"a": 1}
    assert parse_json_lenient("noise {\"a\":1} tail") == {"a": 1}
    assert parse_json_lenient("nope") is None
    assert parse_json_lenient("") is None


def test_retry_recovers_after_bad_output(tmp_path):
    p = _mk(tmp_path, max_json_retries=2)
    answers = iter(["not json at all", '{"k": "v"}'])
    p._call_llm = lambda prompt, idx, total, temperature=None: next(answers)
    out, ok = p._generate_chunk_json("prompt", 1, 1)
    assert ok is True
    assert json.loads(out) == {"k": "v"}


def test_retry_exhausted_returns_raw_fallback(tmp_path):
    p = _mk(tmp_path, max_json_retries=1)
    p._call_llm = lambda prompt, idx, total, temperature=None: "still not json"
    out, ok = p._generate_chunk_json("prompt", 1, 1)
    assert ok is False
    assert json.loads(out) == {"raw": "still not json"}


def test_schema_validation_triggers_retry(tmp_path):
    # Required keys come from the template; first answer misses one.
    p = _mk(tmp_path, output_template='{"q": "", "a": ""}', max_json_retries=2)
    answers = iter(['{"q": "only q"}', '{"q": "x", "a": "y"}'])
    p._call_llm = lambda prompt, idx, total, temperature=None: next(answers)
    out, ok = p._generate_chunk_json("prompt", 1, 1)
    assert ok is True
    assert json.loads(out) == {"a": "y", "q": "x"}


def test_no_schema_accepts_any_object(tmp_path):
    p = _mk(tmp_path, max_json_retries=0)
    p._call_llm = lambda prompt, idx, total, temperature=None: '{"anything": 1}'
    out, ok = p._generate_chunk_json("prompt", 1, 1)
    assert ok is True
    assert json.loads(out) == {"anything": 1}


def test_load_done_keys_resume(tmp_path):
    out = tmp_path / "out.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["instruction", "input", "output", "source", "chunk_id"],
            quoting=csv.QUOTE_ALL,
        )
        w.writeheader()
        w.writerow({"instruction": "i", "input": "x", "output": "{}",
                    "source": "a.txt", "chunk_id": "0"})
        w.writerow({"instruction": "i", "input": "y", "output": "{}",
                    "source": "a.txt", "chunk_id": "1"})

    p = Pipeline(files=[], model="m", output_csv=str(out),
                 instruction="do", resume=True)
    keys = p._load_done_keys()
    assert keys == {("a.txt", "0"), ("a.txt", "1")}


def test_load_done_keys_disabled_when_not_resume(tmp_path):
    out = tmp_path / "out.csv"
    out.write_text("x", encoding="utf-8")
    p = Pipeline(files=[], model="m", output_csv=str(out),
                 instruction="do", resume=False)
    assert p._load_done_keys() == set()
