"""Tests for robust text-file reading across encodings."""
from src.pipeline import read_text_file


def test_reads_plain_utf8(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes("Tiếng Việt có dấu".encode("utf-8"))
    assert read_text_file(str(p)) == "Tiếng Việt có dấu"


def test_reads_utf8_bom(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes("﻿Xin chào".encode("utf-8"))
    assert read_text_file(str(p)) == "Xin chào"


def test_reads_utf16(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes("Nội dung UTF-16".encode("utf-16"))
    assert read_text_file(str(p)) == "Nội dung UTF-16"


def test_lossy_fallback_never_raises(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"\xff\x00\x81\xfevalid-ish")
    # Must return a string without raising, even on garbage bytes.
    assert isinstance(read_text_file(str(p)), str)
