"""Continuous OFFLINE demo: run the real Pipeline on a PDF book via Ollama.

Picks a book from ~/Downloads, processes chunks continuously with a local model,
and writes an Alpaca CSV. Bounded to MAX_CHUNKS so it finishes in a few minutes
while still demonstrating sustained, error-free operation.
"""
import sys
import time
from pathlib import Path

from src.pipeline import Pipeline

DL = Path.home() / "Downloads"
BOOK = DL / "Truyen-nhiem-1.pdf"
MODEL = "granite4.1:8b"
MAX_CHUNKS = 25

out = Path.cwd() / "output" / "offline_book_demo.csv"
if out.exists():
    out.unlink()

logf = Path.cwd() / "_offline_book.log"
log_fp = open(logf, "w", encoding="utf-8")


def log(m: str) -> None:
    line = f"{m}"
    log_fp.write(line + "\n")
    log_fp.flush()


seen = {"n": 0}


def should_stop() -> bool:
    # Stop once MAX_CHUNKS rows are written (bounded continuous demo).
    return seen["n"] >= MAX_CHUNKS


def on_progress(cur, total, eta=None):
    seen["n"] = cur


t0 = time.time()
log(f"=== OFFLINE demo | book={BOOK.name} | model={MODEL} | cap={MAX_CHUNKS} chunks ===")
p = Pipeline(
    files=[str(BOOK)],
    model=MODEL,
    output_csv=str(out),
    instruction="Tóm tắt đoạn văn y khoa và trích các thực thể chính.",
    output_template='{"tom_tat":"", "thuat_ngu":[], "benh":[]}',
    chunk_words=180,
    num_predict=320,
    temperature=0.1,
    max_json_retries=2,
    on_log=log,
    on_progress=on_progress,
    should_stop=should_stop,
)
stats = p.run()
elapsed = time.time() - t0
log(f"\n=== DONE in {elapsed:.1f}s | stats={stats} ===")
if out.exists():
    rows = out.read_text(encoding="utf-8-sig").count("\n")
    log(f"CSV rows (incl header): {rows} | file={out}")
log_fp.close()
print("FINISHED", stats)
