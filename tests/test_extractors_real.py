"""Round-trip the real PDF/DOCX extractors over files we generate on the fly.

Unlike the mocked-LLM tests, these exercise pdfplumber / python-docx for real,
so the offline text-extraction path (PDF/DOCX → text → chunks) is covered
end-to-end without depending on any machine-specific files.
"""
import pytest

from src.extractors.docx_extractor import extract_docx
from src.extractors.pdf_extractor import extract_pdf
from src.pipeline import chunk_text, count_words, extract_text


# --------------------------------------------------------------------------- #
# DOCX (python-docx is a hard dependency of the extractor)
# --------------------------------------------------------------------------- #
docx = pytest.importorskip("docx", reason="python-docx not installed")


def _make_docx(path, paragraphs, table_rows=None):
    d = docx.Document()
    for para in paragraphs:
        d.add_paragraph(para)
    if table_rows:
        t = d.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for r, row in enumerate(table_rows):
            for c, val in enumerate(row):
                t.rows[r].cells[c].text = val
    d.save(str(path))
    return path


def test_docx_extracts_paragraphs(tmp_path):
    p = _make_docx(tmp_path / "a.docx",
                   ["Tiêu đề tài liệu", "Đoạn nội dung tiếng Việt có dấu.", "   "])
    text = extract_docx(str(p))
    assert "Tiêu đề tài liệu" in text
    assert "Đoạn nội dung tiếng Việt có dấu." in text
    # Blank paragraphs are dropped, not turned into empty blocks.
    assert "\n\n\n" not in text


def test_docx_extracts_table_cells(tmp_path):
    p = _make_docx(
        tmp_path / "t.docx",
        ["Bảng dữ liệu:"],
        table_rows=[["Họ tên", "Điểm"], ["An", "9"], ["Bình", "8"]],
    )
    text = extract_docx(str(p))
    # Cells of a row are joined with ' | '.
    assert "Họ tên | Điểm" in text
    assert "An | 9" in text


def test_docx_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_docx("definitely_not_here.docx")


def test_extract_text_dispatches_docx_and_chunks(tmp_path):
    body = ["Câu mở đầu của tài liệu này."] + [
        " ".join(["từ"] * 50) for _ in range(6)
    ]
    p = _make_docx(tmp_path / "d.docx", body)
    text = extract_text(str(p))           # goes through the .docx dispatch
    wc = count_words(text)
    assert wc > 0
    chunks = chunk_text(text, target_words=60)
    assert len(chunks) > 1                # long doc splits into several chunks


# --------------------------------------------------------------------------- #
# PDF (reportlab to author a real PDF; skip if it's not installed)
# --------------------------------------------------------------------------- #
def _make_pdf(path, lines):
    rl_canvas = pytest.importorskip(
        "reportlab.pdfgen.canvas", reason="reportlab not installed"
    )
    c = rl_canvas.Canvas(str(path))
    y = 800
    for line in lines:
        c.drawString(72, y, line)
        y -= 18
    c.showPage()
    c.save()
    return path


def test_pdf_extracts_text(tmp_path):
    p = _make_pdf(tmp_path / "a.pdf",
                  ["Project Milestone Instructions", "Section one line of text."])
    text = extract_pdf(str(p))
    assert "Project Milestone Instructions" in text
    assert "Section one line" in text


def test_pdf_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_pdf("nope_missing.pdf")


def test_extract_text_dispatches_pdf(tmp_path):
    p = _make_pdf(tmp_path / "b.pdf", [f"Line number {i} with content." for i in range(20)])
    text = extract_text(str(p))           # goes through the .pdf dispatch
    assert count_words(text) > 0


# --------------------------------------------------------------------------- #
# Unsupported format
# --------------------------------------------------------------------------- #
def test_extract_text_rejects_unknown_extension(tmp_path):
    f = tmp_path / "x.xyz"
    f.write_text("data", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_text(str(f))
