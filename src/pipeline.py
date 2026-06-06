"""Pipeline: extract text → chunk → Ollama (JSON mode) → CSV.

Single mode only: user provides an instruction, each chunk emits ONE CSV row
where `output` is always a canonical JSON string. Optimized for fast iterative
testing: native Ollama JSON grammar, model kept in VRAM via keep_alive,
output length capped via num_predict.
"""
import json
import re
import time
from pathlib import Path
from typing import Callable, Optional

from .csv_writer import AlpacaCSVWriter
from .extractors.docx_extractor import extract_doc_legacy, extract_docx
from .extractors.image_extractor import IMAGE_EXTS, extract_image
from .extractors.pdf_extractor import extract_pdf
from .ollama_client import DEFAULT_BASE, generate, preload
from .text_chunker import chunk_text, count_words

PROMPT_TEMPLATE = """{instruction}
{schema_block}
INPUT:
{chunk}

Quy tắc trả lời:
- Trả về DUY NHẤT một đối tượng JSON hợp lệ, KHÔNG markdown / KHÔNG prose.
- Mọi giá trị PHẢI được trích từ INPUT (không bịa đặt, không suy diễn).
- Nếu trường nào không có dữ liệu, để [] (array rỗng) hoặc "" (string rỗng).
- TUYỆT ĐỐI không thêm / bớt / đổi tên key so với schema."""

_SCHEMA_PREAMBLE = (
    "\nBạn PHẢI tuân thủ tuyệt đối schema JSON sau "
    "(CÙNG keys, CÙNG kiểu, cho MỌI input):\n{template_json}\n"
)

_FENCE_OPEN = re.compile(r"^\s*```(?:\w+)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```\s*$")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _infer_schema(value):
    """Suy ra JSON Schema từ một giá trị ví dụ (dict/list/str/int/bool/None).

    - dict → object với tất cả keys required, không cho additionalProperties
    - list rỗng → array of string (mặc định an toàn cho keyword extraction)
    - list có item → infer từ phần tử đầu
    - str → string, int/float → number, bool → boolean, None → string
    """
    if isinstance(value, dict):
        props = {k: _infer_schema(v) for k, v in value.items()}
        return {
            "type": "object",
            "properties": props,
            "required": list(value.keys()),
            "additionalProperties": False,
        }
    if isinstance(value, list):
        if value:
            return {"type": "array", "items": _infer_schema(value[0])}
        return {"type": "array", "items": {"type": "string"}}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int) or isinstance(value, float):
        return {"type": "number"}
    # str, None, hoặc kiểu khác → string
    return {"type": "string"}


def build_schema_from_template(template_str: str) -> Optional[dict]:
    """Parse template JSON string → JSON Schema dict cho Ollama Structured Outputs.

    Ví dụ: '{"bệnh": [], "triệu chứng": []}'
      → {"type":"object", "properties":{"bệnh":{"type":"array","items":{"type":"string"}}, ...},
         "required":["bệnh","triệu chứng"], "additionalProperties": false}

    Trả về None nếu template không parse được hoặc rỗng.
    """
    if not template_str or not template_str.strip():
        return None
    try:
        obj = json.loads(template_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return _infer_schema(obj)


def extract_text(
    path: str,
    vision_model: Optional[str] = None,
    base_url: str = DEFAULT_BASE,
    keep_alive: str = "30m",
    on_token: Optional[Callable[[str, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return extract_pdf(path)
    if ext == ".docx":
        return extract_docx(path)
    if ext == ".doc":
        return extract_doc_legacy(path)
    if ext in (".txt", ".md"):
        return p.read_text(encoding="utf-8", errors="ignore")
    if ext in IMAGE_EXTS:
        return extract_image(
            path,
            model=vision_model or "",
            base_url=base_url,
            keep_alive=keep_alive,
            on_token=on_token,
            should_stop=should_stop,
        )
    raise ValueError(f"Định dạng không hỗ trợ: {ext} ({p.name})")


def _strip_artifacts(text: str) -> str:
    text = _THINK_BLOCK.sub("", text).strip()
    text = _FENCE_OPEN.sub("", text)
    text = _FENCE_CLOSE.sub("", text)
    return text.strip()


def normalize_json_output(text: str) -> str:
    """Return a canonical, compact JSON string.

    With Ollama's format=json the model already returns valid JSON, but we
    canonicalize (compact + sorted keys + Unicode preserved) so training data
    is byte-stable. Falls back to {"raw": <text>} for any unparseable cases.
    """
    text = _strip_artifacts(text)
    if not text:
        return json.dumps({"raw": ""}, ensure_ascii=False, separators=(",", ":"))
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return json.dumps({"raw": text}, ensure_ascii=False, separators=(",", ":"))
        else:
            return json.dumps({"raw": text}, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class Pipeline:
    def __init__(
        self,
        files: list[str],
        model: str,
        output_csv: str,
        instruction: str,
        chunk_words: int = 200,
        temperature: float = 0.1,
        num_ctx: int = 4096,
        num_predict: int = 512,
        keep_alive: str = "30m",
        ollama_url: str = "http://localhost:11434",
        vision_model: str = "",
        output_template: str = "",
        on_log: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[int, int, Optional[float]], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ):
        self.files = files
        self.model = model
        self.output_csv = output_csv
        self.instruction = instruction.strip()
        self.chunk_words = chunk_words
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self.ollama_url = ollama_url
        self.vision_model = (vision_model or "").strip()
        self.on_log = on_log or (lambda m: None)
        self.on_progress = on_progress or (lambda c, t, eta=None: None)
        self.on_status = on_status or (lambda s: None)
        self.should_stop = should_stop or (lambda: False)
        self._durations: list[float] = []

        # Schema cho Structured Outputs (Ollama ≥ 0.5).
        self.output_template = (output_template or "").strip()
        self.json_schema = build_schema_from_template(self.output_template)
        # Compact template để chèn vào prompt
        if self.output_template:
            try:
                obj = json.loads(self.output_template)
                self._template_compact = json.dumps(obj, ensure_ascii=False)
            except json.JSONDecodeError:
                self._template_compact = ""
        else:
            self._template_compact = ""

    def _log(self, m: str) -> None:
        self.on_log(m)

    def _status(self, m: str) -> None:
        self.on_status(m)

    def _record(self, cur: int, total: int, elapsed: float) -> None:
        self._durations.append(elapsed)
        window = self._durations[-5:]
        avg = sum(window) / len(window) if window else 0.0
        eta = avg * max(total - cur, 0)
        self.on_progress(cur, total, eta)

    def _extract(self, path: str, is_img: bool) -> str:
        """Đọc text từ 1 file. Ảnh được OCR bằng vision model, có cập nhật
        status theo số ký tự đọc được."""
        if not is_img:
            return extract_text(path)

        name = Path(path).name

        def on_tok(_piece: str, total_chars: int) -> None:
            self._status(f"🖼 OCR {name} · {total_chars:,} ký tự")

        return extract_text(
            path,
            vision_model=self.vision_model,
            base_url=self.ollama_url,
            keep_alive=self.keep_alive,
            on_token=on_tok,
            should_stop=self.should_stop,
        )

    def _build_chunks(self) -> list[tuple[str, int, str]]:
        all_chunks: list[tuple[str, int, str]] = []
        gid = 0
        for f in self.files:
            if self.should_stop():
                return all_chunks
            try:
                is_img = Path(f).suffix.lower() in IMAGE_EXTS
                if is_img:
                    self._status(f"🖼 Đọc ảnh (OCR): {Path(f).name}")
                    self._log(f"🖼 {f}")
                else:
                    self._status(f"Đang đọc: {Path(f).name}")
                    self._log(f"📄 {f}")
                text = self._extract(f, is_img)
                wc = count_words(text)
                if wc == 0:
                    self._log("   ⚠ rỗng, bỏ qua")
                    continue
                chunks = chunk_text(text, target_words=self.chunk_words)
                self._log(f"   {wc:,} từ → {len(chunks)} chunks")
                src = Path(f).name
                for c in chunks:
                    all_chunks.append((src, gid, c))
                    gid += 1
            except Exception as e:
                self._log(f"   ❌ {e}")
        return all_chunks

    def _call_llm(self, prompt: str, idx: int, total: int) -> str:
        t0 = time.time()
        last = [0.0]

        def on_tok(_piece: str, total_chars: int) -> None:
            now = time.time()
            if now - last[0] >= 0.2:
                last[0] = now
                self._status(
                    f"Chunk {idx}/{total} · {total_chars:,} ký tự · {now - t0:.1f}s"
                )

        return generate(
            model=self.model,
            prompt=prompt,
            base_url=self.ollama_url,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
            options={"num_predict": self.num_predict},
            format_json=True,
            json_schema=self.json_schema,
            keep_alive=self.keep_alive,
            on_token=on_tok,
            should_stop=self.should_stop,
        )

    def run(self) -> dict:
        stats = {"files": len(self.files), "chunks": 0, "samples": 0, "errors": 0}

        if not self.instruction:
            self._log("❌ Thiếu instruction — bắt buộc phải có để biết LLM cần làm gì.")
            stats["errors"] = 1
            return stats

        # Warm the model into VRAM before we start timing chunks - first call is
        # often 5-10s of model load that would skew ETA wildly.
        self._status(f"Đang nạp model {self.model} vào VRAM...")
        self._log(f"⚙ Preload model: {self.model} (keep_alive={self.keep_alive})")
        t_warm = time.time()
        if preload(self.model, base_url=self.ollama_url, keep_alive=self.keep_alive):
            self._log(f"   ✓ Sẵn sàng ({time.time() - t_warm:.1f}s)")
        else:
            self._log("   ⚠ Không preload được (sẽ load khi gọi chunk 1)")

        all_chunks = self._build_chunks()
        total = len(all_chunks)
        if total == 0:
            self._log("Không có chunk nào.")
            self._status("Không có chunk.")
            return stats

        preview = self.instruction[:120] + ("..." if len(self.instruction) > 120 else "")
        mode = "STRUCTURED (JSON Schema)" if self.json_schema else "JSON loose"
        self._log(
            f"🚀 Mode: {mode} | num_predict={self.num_predict} | "
            f"num_ctx={self.num_ctx} | temp={self.temperature}"
        )
        self._log(f"   Instruction: {preview}")
        if self._template_compact:
            tpl_preview = self._template_compact[:160] + (
                "..." if len(self._template_compact) > 160 else ""
            )
            self._log(f"   Schema mẫu: {tpl_preview}")
        elif self.output_template:
            self._log(
                "   ⚠ Schema mẫu không parse được JSON — sẽ chạy JSON loose mode."
            )
        self._log(f"   Chunks: {total}")
        self.on_progress(0, total, None)
        self._status(f"Đang xử lý 0/{total}...")

        schema_block = (
            _SCHEMA_PREAMBLE.format(template_json=self._template_compact)
            if self._template_compact
            else ""
        )

        with AlpacaCSVWriter(self.output_csv) as writer:
            for i, (src, cid, chunk) in enumerate(all_chunks, start=1):
                if self.should_stop():
                    self._log(f"⏹ Dừng ở chunk {i}/{total}.")
                    break
                t0 = time.time()
                try:
                    prompt = PROMPT_TEMPLATE.format(
                        instruction=self.instruction,
                        schema_block=schema_block,
                        chunk=chunk,
                    )
                    raw = self._call_llm(prompt, i, total)
                    output = normalize_json_output(raw)
                    n = writer.write_samples(
                        [{"instruction": self.instruction, "input": chunk, "output": output}],
                        src,
                        cid,
                    )
                    stats["samples"] += n
                    elapsed = time.time() - t0
                    self._log(f"   ✓ {i}/{total}: {elapsed:.1f}s · {len(output)} ký tự JSON")
                    stats["chunks"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    self._log(f"   ❌ {i}/{total}: {e}")
                self._record(i, total, time.time() - t0)

        self._log(
            f"✅ Xong! Chunks: {stats['chunks']} | Samples: {stats['samples']} | "
            f"Lỗi: {stats['errors']}"
        )
        self._log(f"📁 {self.output_csv}")
        self._status(
            f"Hoàn tất: {stats['chunks']} chunks · {stats['samples']} samples · "
            f"{stats['errors']} lỗi."
        )
        return stats
