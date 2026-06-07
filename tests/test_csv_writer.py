"""Tests for the append-mode Alpaca CSV writer."""
import csv

from src.csv_writer import AlpacaCSVWriter


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_writes_header_and_row(tmp_path):
    p = tmp_path / "out.csv"
    with AlpacaCSVWriter(str(p)) as w:
        n = w.write_samples(
            [{"instruction": "do", "input": "ctx", "output": "{}"}], "f.txt", 0
        )
    assert n == 1
    rows = _read(p)
    assert len(rows) == 1
    assert rows[0]["instruction"] == "do"
    assert rows[0]["source"] == "f.txt"
    assert rows[0]["chunk_id"] == "0"


def test_skips_rows_missing_instruction_or_output(tmp_path):
    p = tmp_path / "out.csv"
    with AlpacaCSVWriter(str(p)) as w:
        n = w.write_samples(
            [
                {"instruction": "", "output": "x"},
                {"instruction": "y", "output": ""},
                {"instruction": "ok", "output": "good"},
            ],
            "f.txt",
            1,
        )
    assert n == 1
    assert len(_read(p)) == 1


def test_append_does_not_duplicate_header(tmp_path):
    p = tmp_path / "out.csv"
    with AlpacaCSVWriter(str(p)) as w:
        w.write_samples([{"instruction": "a", "output": "1"}], "f", 0)
    with AlpacaCSVWriter(str(p)) as w:
        w.write_samples([{"instruction": "b", "output": "2"}], "f", 1)
    rows = _read(p)
    assert len(rows) == 2
    # Raw file should contain exactly one header line
    text = p.read_text(encoding="utf-8-sig")
    assert text.count("instruction") == 1


def test_utf8_bom_written(tmp_path):
    p = tmp_path / "out.csv"
    with AlpacaCSVWriter(str(p)) as w:
        w.write_samples([{"instruction": "tiếng việt", "output": "{}"}], "f", 0)
    raw = p.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM
