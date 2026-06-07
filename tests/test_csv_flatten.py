"""Tests for JSON-output flattening + array explosion."""
import csv
import json

from src.csv_flatten import flatten_csv


def _write_src(path, output_jsons):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["instruction", "input", "output", "source", "chunk_id"],
            quoting=csv.QUOTE_ALL,
        )
        w.writeheader()
        for i, oj in enumerate(output_jsons):
            w.writerow({
                "instruction": "ins", "input": "in",
                "output": oj, "source": "s", "chunk_id": i,
            })


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_scalar_keys_become_columns(tmp_path):
    src = tmp_path / "src.csv"
    dst = tmp_path / "dst.csv"
    _write_src(src, [json.dumps({"bệnh": "sốt", "tuổi": 5}, ensure_ascii=False)])
    res = flatten_csv(str(src), str(dst), explode_arrays=True)
    rows = _read(dst)
    assert res["rows"] == 1
    assert rows[0]["bệnh"] == "sốt"
    assert rows[0]["tuổi"] == "5"


def test_array_explosion_cartesian(tmp_path):
    src = tmp_path / "src.csv"
    dst = tmp_path / "dst.csv"
    _write_src(src, [json.dumps(
        {"a": ["x", "y"], "b": [1, 2, 3]}, ensure_ascii=False)])
    res = flatten_csv(str(src), str(dst), explode_arrays=True)
    rows = _read(dst)
    assert len(rows) == 6  # 2 * 3 cartesian
    assert res["exploded_rows"] == 5


def test_array_no_explosion_when_disabled(tmp_path):
    src = tmp_path / "src.csv"
    dst = tmp_path / "dst.csv"
    _write_src(src, [json.dumps({"a": ["x", "y"]}, ensure_ascii=False)])
    flatten_csv(str(src), str(dst), explode_arrays=False)
    rows = _read(dst)
    assert len(rows) == 1
    assert rows[0]["a"] == '["x","y"]'  # stringified array


def test_parse_errors_become_raw(tmp_path):
    src = tmp_path / "src.csv"
    dst = tmp_path / "dst.csv"
    _write_src(src, ["this is not json"])
    res = flatten_csv(str(src), str(dst))
    rows = _read(dst)
    assert res["parse_errors"] == 1
    assert rows[0]["raw"] == "this is not json"


def test_nested_object_stringified(tmp_path):
    src = tmp_path / "src.csv"
    dst = tmp_path / "dst.csv"
    _write_src(src, [json.dumps({"meta": {"k": "v"}}, ensure_ascii=False)])
    flatten_csv(str(src), str(dst))
    rows = _read(dst)
    assert rows[0]["meta"] == '{"k":"v"}'
