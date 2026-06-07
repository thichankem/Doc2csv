"""End-to-end Pipeline.run test with a mocked Ollama backend."""
import csv

from src import pipeline as pl
from src.llm_backends import LLMRouter


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _patch_ollama(monkeypatch, answer='{"summary": "ok"}'):
    monkeypatch.setattr(pl, "preload", lambda *a, **k: True)

    def fake_generate(model, prompt, **kw):
        return answer

    # Pipeline._call_llm imports generate from the module namespace.
    monkeypatch.setattr(pl, "generate", fake_generate)


def test_run_writes_one_row_per_chunk(tmp_path, monkeypatch):
    _patch_ollama(monkeypatch)
    doc = tmp_path / "doc.txt"
    doc.write_text("\n\n".join(f"Đoạn {i} có nội dung." for i in range(6)),
                   encoding="utf-8")
    out = tmp_path / "out.csv"

    p = pl.Pipeline(
        files=[str(doc)], model="m", output_csv=str(out),
        instruction="Tóm tắt", chunk_words=5, max_json_retries=0,
    )
    stats = p.run()

    rows = _read(out)
    assert stats["chunks"] == len(rows) > 0
    assert stats["errors"] == 0
    assert all(r["output"] == '{"summary":"ok"}' for r in rows)


def test_run_resume_skips_existing(tmp_path, monkeypatch):
    _patch_ollama(monkeypatch)
    doc = tmp_path / "doc.txt"
    # Each paragraph ~40 words; with target 120 / min 100 this yields several chunks.
    para = " ".join(["từ"] * 40) + "."
    doc.write_text("\n\n".join(para for _ in range(20)), encoding="utf-8")
    out = tmp_path / "out.csv"

    # Full first run, then simulate an interrupted run by truncating the CSV
    # to its header + first 2 data rows.
    p1 = pl.Pipeline(files=[str(doc)], model="m", output_csv=str(out),
                     instruction="x", chunk_words=120)
    p1.run()
    full_rows = len(_read(out))
    assert full_rows > 2

    # Rewrite the CSV keeping only the first 2 data rows (proper CSV parsing —
    # the `input` cells contain embedded newlines, so line slicing won't do).
    all_rows = _read(out)
    fields = ["instruction", "input", "output", "source", "chunk_id"]
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in all_rows[:2]:
            w.writerow({k: r[k] for k in fields})
    kept = len(_read(out))
    assert kept == 2

    # Resume run: skip the 2 kept chunks and finish the remainder.
    p2 = pl.Pipeline(files=[str(doc)], model="m", output_csv=str(out),
                     instruction="x", chunk_words=120, resume=True)
    stats = p2.run()
    assert stats["skipped"] == kept
    total_rows = len(_read(out))
    assert total_rows == full_rows  # back to the complete set


class _RRBackend:
    """Minimal backend recording which name served each chunk."""
    kind = "fake"

    def __init__(self, name):
        self.name = name

    def generate(self, prompt, **kw):
        return '{"ok": true}'


def test_run_with_router_rotates_and_writes(tmp_path):
    doc = tmp_path / "doc.txt"
    para = " ".join(["từ"] * 40) + "."
    doc.write_text("\n\n".join(para for _ in range(20)), encoding="utf-8")
    out = tmp_path / "out.csv"

    router = LLMRouter([_RRBackend("api-a"), _RRBackend("api-b")])
    logs = []
    p = pl.Pipeline(
        files=[str(doc)], model="", output_csv=str(out),
        instruction="x", chunk_words=120, router=router,
        on_log=logs.append,
    )
    stats = p.run()

    rows = _read(out)
    assert stats["chunks"] == len(rows) > 1
    assert all(r["output"] == '{"ok":true}' for r in rows)
    # Backend names appear in the per-chunk logs (rotation visible).
    joined = "\n".join(logs)
    assert "api-a" in joined and "api-b" in joined
