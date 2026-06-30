"""Pipeline: extract text → chunk → Ollama (JSON mode) → CSV.

Single mode only: user provides an instruction, each chunk emits ONE CSV row
where `output` is always a canonical JSON string. Optimized for fast iterative
testing: native Ollama JSON grammar, model kept in VRAM via keep_alive,
output length capped via num_predict.
"""
import itertools
import json
import queue
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable, Optional

# Sentinel pushed onto the producer→consumer queue to signal "no more chunks".
_QUEUE_DONE = object()
# Sentinel for an empty work stream when peeking the first item.
_EMPTY = object()

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


def read_text_file(path: str) -> str:
    """Read a .txt/.md file robustly, trying common encodings before falling
    back to a lossy decode. Handles UTF-8 (with/without BOM), UTF-16 and the
    Vietnamese cp1258 code page rather than silently dropping bytes."""
    data = Path(path).read_bytes()
    # UTF-16 only when a BOM proves it — a blind utf-16 decode would happily
    # mangle plain UTF-8 of even length.
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass
    for enc in ("utf-8-sig", "cp1258", "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


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
        return read_text_file(path)
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


def parse_json_lenient(text: str):
    """Best-effort parse of a model answer into a Python object.

    Strips think-blocks / code fences, then tries a direct parse and finally a
    braces-slice ({...}) parse. Returns the parsed object, or None when nothing
    parseable was found (caller decides whether to retry or store as raw).
    """
    text = _strip_artifacts(text)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _canonical(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def normalize_json_output(text: str) -> str:
    """Return a canonical, compact JSON string.

    With Ollama's format=json the model already returns valid JSON, but we
    canonicalize (compact + sorted keys + Unicode preserved) so training data
    is byte-stable. Falls back to {"raw": <text>} for any unparseable cases.
    """
    obj = parse_json_lenient(text)
    if obj is None:
        return _canonical({"raw": _strip_artifacts(text)})
    return _canonical(obj)


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
        max_json_retries: int = 1,
        resume: bool = False,
        concurrency: int = 1,
        router: Optional[object] = None,
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
        self.max_json_retries = max(0, int(max_json_retries))
        self.resume = bool(resume)
        # >1 fans chunks out across threads. The big win is online/mix mode where
        # the router hands each thread a different endpoint; a single local Ollama
        # serialises on the GPU so concurrency there mostly just fills its queue.
        self.concurrency = max(1, int(concurrency))
        # Optional LLMRouter (online/mix). When None, the classic single local
        # Ollama model path is used (self.model + module-level generate()).
        self.router = router
        self.on_log = on_log or (lambda m: None)
        self.on_progress = on_progress or (lambda c, t, eta=None: None)
        self.on_status = on_status or (lambda s: None)
        self.should_stop = should_stop or (lambda: False)
        self._durations: list[float] = []
        # Streaming progress counters (set per run): chunks the producer has
        # enqueued, and chunks finalized by the consumer.
        self._discovered = 0
        self._n_done = 0

        # Schema cho Structured Outputs (Ollama ≥ 0.5).
        self.output_template = (output_template or "").strip()
        self.json_schema = build_schema_from_template(self.output_template)
        self._required_keys = (
            list(self.json_schema.get("required", []))
            if isinstance(self.json_schema, dict)
            else []
        )
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

    def _read_and_chunk(self, f: str, is_img: bool) -> tuple[int, list[str]]:
        """Extract one file's text and split it into chunks. Returns (word_count,
        chunks); (0, []) when empty. Pure enough to run in a worker thread for
        non-image files (chunk_text is pure); images still call the vision model."""
        text = self._extract(f, is_img)
        wc = count_words(text)
        if wc == 0:
            return 0, []
        return wc, chunk_text(text, target_words=self.chunk_words)

    def _iter_files_chunks(self):
        """Yield (source_name, [chunks]) per file, in input order.

        Non-image files are extracted+chunked on a small read-ahead thread pool
        (PDF/DOCX parsing is IO/CPU bound and releases the GIL), so later files
        are already being read while earlier ones are yielded. Images stay serial
        (shared single-GPU vision model). Emission order matches input order, so
        chunk ids are stable no matter which read finished first."""
        files = self.files
        n = len(files)
        readahead = min(8, n) or 1
        pool = ThreadPoolExecutor(max_workers=readahead)
        futs: dict[int, object] = {}
        next_submit = 0

        def prime(upto: int) -> None:
            nonlocal next_submit
            while next_submit < n and next_submit <= upto:
                j = next_submit
                if Path(files[j]).suffix.lower() not in IMAGE_EXTS:
                    futs[j] = pool.submit(self._read_and_chunk, files[j], False)
                next_submit += 1

        try:
            for i, f in enumerate(files):
                if self.should_stop():
                    return
                prime(i + readahead)  # keep reads running ahead of emission
                name = Path(f).name
                try:
                    if i in futs:
                        self._log(f"📄 {f}")
                        wc, chunks = futs.pop(i).result()
                    else:  # image → OCR now (serial)
                        self._status(f"🖼 Đọc ảnh (OCR): {name}")
                        self._log(f"🖼 {f}")
                        wc, chunks = self._read_and_chunk(f, is_img=True)
                    if not chunks:
                        self._log(f"   ⚠ {name}: rỗng, bỏ qua")
                        continue
                    self._log(f"   {name}: {wc:,} từ → {len(chunks)} chunks")
                    yield name, chunks
                except Exception as e:  # noqa: BLE001
                    self._log(f"   ❌ {e}")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _build_chunks(self) -> list[tuple[str, int, str]]:
        """Materialise every chunk as (source, chunk_id, text). chunk_id is a
        global running counter in input-file order (stable for resume)."""
        all_chunks: list[tuple[str, int, str]] = []
        gid = 0
        for src, chunks in self._iter_files_chunks():
            for c in chunks:
                all_chunks.append((src, gid, c))
                gid += 1
        return all_chunks

    def _call_llm(
        self, prompt: str, idx: int, total: int,
        temperature: Optional[float] = None, num_predict: Optional[int] = None,
    ) -> str:
        t0 = time.time()
        last = [0.0]

        def on_tok(_piece: str, total_chars: int) -> None:
            now = time.time()
            if now - last[0] >= 0.2:
                last[0] = now
                self._status(
                    f"Chunk {idx}/{total} · {total_chars:,} ký tự · {now - t0:.1f}s"
                )

        temp = self.temperature if temperature is None else temperature
        npredict = self.num_predict if num_predict is None else num_predict
        if self.router is not None:
            return self.router.generate(
                prompt,
                temperature=temp,
                num_predict=npredict,
                num_ctx=self.num_ctx,
                format_json=True,
                json_schema=self.json_schema,
                keep_alive=self.keep_alive,
                on_token=on_tok,
                should_stop=self.should_stop,
            )

        return generate(
            model=self.model,
            prompt=prompt,
            base_url=self.ollama_url,
            temperature=temp,
            num_ctx=self.num_ctx,
            options={"num_predict": npredict},
            format_json=True,
            json_schema=self.json_schema,
            keep_alive=self.keep_alive,
            on_token=on_tok,
            should_stop=self.should_stop,
        )

    def _schema_ok(self, obj) -> bool:
        """True if obj satisfies the required top-level keys of the template
        schema. When no schema is set, any parsed object is acceptable."""
        if not self._required_keys:
            return True
        return isinstance(obj, dict) and all(k in obj for k in self._required_keys)

    def _generate_chunk_json(self, prompt: str, idx: int, total: int) -> tuple[str, bool]:
        """Call the model, retrying on unparseable / schema-violating output.

        Returns (canonical_json_string, ok). `ok` is False when every attempt
        failed to yield usable JSON — in that case the string is a {"raw": ...}
        fallback so no chunk is ever lost.
        """
        last_raw = ""
        for attempt in range(self.max_json_retries + 1):
            if self.should_stop():
                break
            # On retries: nudge temperature up slightly to escape a bad groove,
            # AND raise num_predict — the #1 cause of unparseable JSON in
            # structured mode is the answer being truncated at the token cap.
            temp = self.temperature if attempt == 0 else min(self.temperature + 0.15 * attempt, 1.0)
            npredict = self.num_predict if attempt == 0 else min(self.num_predict * (attempt + 1), 4096)
            raw = self._call_llm(prompt, idx, total, temperature=temp, num_predict=npredict)
            last_raw = raw
            obj = parse_json_lenient(raw)
            if obj is not None and self._schema_ok(obj):
                return _canonical(obj), True
            if attempt < self.max_json_retries and not self.should_stop():
                self._log(
                    f"   ↻ {idx}/{total}: JSON chưa đạt, thử lại "
                    f"({attempt + 1}/{self.max_json_retries}, num_predict→{min(self.num_predict * (attempt + 2), 4096)})"
                )
        # Exhausted retries — keep whatever parsed, else raw fallback.
        obj = parse_json_lenient(last_raw)
        if obj is not None:
            return _canonical(obj), False
        return _canonical({"raw": _strip_artifacts(last_raw)}), False

    def _load_done_keys(self) -> set:
        """Read (source, chunk_id) pairs already present in the output CSV so a
        resumed run can skip them. Returns an empty set when not resuming or the
        file does not exist yet."""
        if not self.resume:
            return set()
        path = Path(self.output_csv)
        if not path.exists() or path.stat().st_size == 0:
            return set()
        import csv as _csv

        done: set = set()
        try:
            with open(path, encoding="utf-8-sig", newline="") as f:
                for row in _csv.DictReader(f):
                    src = (row.get("source") or "").strip()
                    cid = (row.get("chunk_id") or "").strip()
                    if cid != "":
                        done.add((src, cid))
        except Exception as e:
            self._log(f"   ⚠ Không đọc được file resume: {e}")
        return done

    def _process_chunk(self, item, schema_block) -> dict:
        """Run the LLM for one chunk (no I/O). Safe to call from a worker thread:
        it only reads shared state and calls the (thread-safe) backend."""
        i, src, cid, chunk = item
        t0 = time.time()
        res = {"i": i, "src": src, "cid": cid, "chunk": chunk,
               "output": "", "ok": False, "elapsed": 0.0,
               "backend": None, "error": None}
        try:
            prompt = PROMPT_TEMPLATE.format(
                instruction=self.instruction, schema_block=schema_block, chunk=chunk,
            )
            # Total is the count discovered so far by the producer (it runs ahead
            # of consumption), good enough for the "i/total" status line.
            output, ok = self._generate_chunk_json(prompt, i, max(self._discovered, i))
            res["output"], res["ok"] = output, ok
            if self.router is not None:
                res["backend"] = self.router.last_backend
        except Exception as e:  # noqa: BLE001 - surfaced to the writer thread
            res["error"] = str(e)
        res["elapsed"] = time.time() - t0
        return res

    def _commit_result(self, res, writer, stats) -> None:
        """Write one chunk's result + update stats/logs. Runs on one thread only
        (the main/writer thread) so the CSV writer and stats need no locking."""
        i = res["i"]
        # Don't persist a chunk that was cut short by Stop: a partial/raw row
        # would otherwise be treated as "done" and skipped on the next resume.
        if self.should_stop() and not res["ok"]:
            return
        total = max(self._discovered, self._n_done + 1)
        if res["error"] is not None:
            stats["errors"] += 1
            self._log(f"   ❌ {i}/{total}: {res['error']}")
        else:
            if not res["ok"]:
                stats["raw_fallbacks"] += 1
            n = writer.write_samples(
                [{"instruction": self.instruction, "input": res["chunk"], "output": res["output"]}],
                res["src"], res["cid"],
            )
            stats["samples"] += n
            flag = "" if res["ok"] else " ⚠ raw"
            bk = f" · {res['backend']}" if res["backend"] else ""
            self._log(
                f"   ✓ {i}/{total}: {res['elapsed']:.1f}s · "
                f"{len(res['output'])} ký tự JSON{flag}{bk}"
            )
            stats["chunks"] += 1
        self._n_done += 1
        # ETA: wall time per chunk overstates throughput by ~concurrency when
        # several run at once, so scale it down to reflect real pace.
        self._record(self._n_done, max(self._discovered, self._n_done),
                     res["elapsed"] / self.concurrency)

    def _put(self, out_q: "queue.Queue", item) -> bool:
        """Block until `item` is enqueued, bailing out promptly if Stop is hit
        (so the producer never deadlocks on a full queue after a stop)."""
        while True:
            if self.should_stop():
                return False
            try:
                out_q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue

    def _produce(self, out_q: "queue.Queue") -> None:
        """Background thread: read+chunk files and stream chunks onto the queue so
        the LLM consumer can start before all files are read. `self._discovered`
        runs ahead of consumption → meaningful ETA. Always ends with a sentinel."""
        try:
            for src, chunks in self._iter_files_chunks():
                for c in chunks:
                    if not self._put(out_q, (src, c)):
                        return
                    self._discovered += 1
        finally:
            # Sentinel must ALWAYS reach the consumer or it would block on get()
            # forever — so deliver it unconditionally (run() drains to unblock us
            # if the consumer already stopped reading).
            out_q.put(_QUEUE_DONE)

    def _drain_join(self, producer: threading.Thread, out_q: "queue.Queue") -> None:
        """Let the producer finish: pull anything left (so a producer blocked on a
        full queue after Stop can enqueue its sentinel and exit), then join."""
        while producer.is_alive():
            try:
                out_q.get(timeout=0.1)
            except queue.Empty:
                continue
        producer.join(timeout=2.0)

    def _work_items(self, out_q: "queue.Queue", done_keys: set, stats: dict):
        """Pull (src, chunk) off the queue, assign the global chunk id, drop
        resume-skipped ones, and yield (i, src, cid, chunk) for processing."""
        gid = 0
        while True:
            item = out_q.get()
            if item is _QUEUE_DONE:
                return
            src, chunk = item
            cid = gid
            gid += 1
            if done_keys and (src, str(cid)) in done_keys:
                stats["skipped"] += 1
                self._n_done += 1
                self.on_progress(self._n_done, max(self._discovered, self._n_done), None)
                continue
            yield (cid + 1, src, cid, chunk)

    def _run_stream(self, items, writer, schema_block, stats) -> None:
        """Consume work items, serially or across a thread pool. Commits (CSV +
        stats) always happen here on the single calling thread → no locks."""
        items = iter(items)
        if self.concurrency <= 1:
            for it in items:
                if self.should_stop():
                    self._log(f"⏹ Dừng ở chunk {it[0]}.")
                    break
                self._commit_result(self._process_chunk(it, schema_block), writer, stats)
            return

        self._log(f"   ⚡ Song song: {self.concurrency} chunk cùng lúc")
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            # Keep at most `concurrency` chunks in flight: prime the window, then
            # refill one-for-one as each finishes. On stop we quit refilling and
            # let the in-flight few drain (their workers short-circuit too).
            futures = set()
            for _ in range(self.concurrency):
                nxt = next(items, None)
                if nxt is None:
                    break
                futures.add(pool.submit(self._process_chunk, nxt, schema_block))
            while futures:
                completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                for fut in completed:
                    self._commit_result(fut.result(), writer, stats)
                if not self.should_stop():
                    for _ in range(len(completed)):
                        nxt = next(items, None)
                        if nxt is None:
                            break
                        futures.add(pool.submit(self._process_chunk, nxt, schema_block))
        if self.should_stop():
            self._log(f"⏹ Dừng ({self._n_done} chunk đã xử lý).")

    def run(self) -> dict:
        stats = {
            "files": len(self.files), "chunks": 0, "samples": 0,
            "errors": 0, "raw_fallbacks": 0, "skipped": 0,
        }

        if not self.instruction:
            self._log("❌ Thiếu instruction — bắt buộc phải có để biết LLM cần làm gì.")
            stats["errors"] = 1
            return stats

        # Warm the model into VRAM before we start timing chunks - first call is
        # often 5-10s of model load that would skew ETA wildly.
        if self.router is not None:
            self._status("Đang chuẩn bị backend...")
            self._log(f"⚙ Backends ({len(self.router)}): {', '.join(self.router.names())}")
            self.router.warmup(keep_alive=self.keep_alive)
        else:
            self._status(f"Đang nạp model {self.model} vào VRAM...")
            self._log(f"⚙ Preload model: {self.model} (keep_alive={self.keep_alive})")
            t_warm = time.time()
            if preload(self.model, base_url=self.ollama_url, keep_alive=self.keep_alive):
                self._log(f"   ✓ Sẵn sàng ({time.time() - t_warm:.1f}s)")
            else:
                self._log("   ⚠ Không preload được (sẽ load khi gọi chunk 1)")

        self._discovered = 0
        self._n_done = 0
        self._durations = []

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
        if self.max_json_retries:
            self._log(f"   Retry JSON: tối đa {self.max_json_retries} lần/chunk khi output chưa đạt")
        done_keys = self._load_done_keys()
        if done_keys:
            self._log(f"   ⏭ Resume: bỏ qua {len(done_keys)} chunk đã có trong file output")
        self.on_progress(0, 0, None)
        self._status("Đang đọc & xử lý...")

        schema_block = (
            _SCHEMA_PREAMBLE.format(template_json=self._template_compact)
            if self._template_compact
            else ""
        )

        # Producer reads+chunks files in the background and streams chunks onto a
        # bounded queue; the consumer below starts the LLM as soon as the first
        # chunk lands, so file-reading overlaps generation instead of blocking it.
        out_q: "queue.Queue" = queue.Queue(maxsize=max(64, self.concurrency * 8))
        producer = threading.Thread(target=self._produce, args=(out_q,), daemon=True)
        producer.start()

        gen = self._work_items(out_q, done_keys, stats)
        first = next(gen, _EMPTY)
        if first is _EMPTY:
            self._drain_join(producer, out_q)
            if self._discovered == 0:
                self._log("Không có chunk nào.")
                self._status("Không có chunk.")
                return stats
        else:
            with AlpacaCSVWriter(self.output_csv) as writer:
                self._run_stream(itertools.chain([first], gen), writer, schema_block, stats)
            self._drain_join(producer, out_q)

        extra = ""
        if stats["raw_fallbacks"]:
            extra += f" | JSON raw: {stats['raw_fallbacks']}"
        if stats["skipped"]:
            extra += f" | Bỏ qua (resume): {stats['skipped']}"
        self._log(
            f"✅ Xong! Chunks: {stats['chunks']} | Samples: {stats['samples']} | "
            f"Lỗi: {stats['errors']}{extra}"
        )
        self._log(f"📁 {self.output_csv}")
        self._status(
            f"Hoàn tất: {stats['chunks']} chunks · {stats['samples']} samples · "
            f"{stats['errors']} lỗi."
        )
        return stats
