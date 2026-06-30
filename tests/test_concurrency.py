"""Concurrency: thread-safe router fan-out + parallel Pipeline.run."""
import csv
import threading
import time

from src import pipeline as pl
from src.llm_backends import LLMRouter
from src.openai_client import RateLimitError


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class SlowBackend:
    """Backend that blocks for `delay` and records concurrent occupancy."""
    kind = "fake"

    def __init__(self, name, delay=0.05, state=None):
        self.name = name
        self.delay = delay
        self.calls = 0
        self._state = state  # shared {"now": int, "peak": int, "lock": Lock}

    def generate(self, prompt, **kw):
        self.calls += 1
        st = self._state
        if st is not None:
            with st["lock"]:
                st["now"] += 1
                st["peak"] = max(st["peak"], st["now"])
        try:
            time.sleep(self.delay)
            return f'{{"by":"{self.name}"}}'
        finally:
            if st is not None:
                with st["lock"]:
                    st["now"] -= 1


def _gen(router):
    return router.generate(
        "p", temperature=0.1, num_predict=128, num_ctx=2048,
        format_json=True, json_schema=None, keep_alive="30m",
        on_token=None, should_stop=None,
    )


def test_router_runs_backends_concurrently():
    """Two threads on a 2-backend router should occupy both at once."""
    state = {"now": 0, "peak": 0, "lock": threading.Lock()}
    a = SlowBackend("a", delay=0.1, state=state)
    b = SlowBackend("b", delay=0.1, state=state)
    r = LLMRouter([a, b])

    results = []
    threads = [threading.Thread(target=lambda: results.append(_gen(r))) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state["peak"] == 2          # both backends ran simultaneously
    assert a.calls == 1 and b.calls == 1   # distinct backends, no double-dip
    assert len(results) == 2


def test_router_concurrent_distinct_backends_no_overclaim():
    """N threads over N backends → each backend used exactly once (round-robin
    claim is atomic, so no two threads grab the same one)."""
    n = 6
    backs = [SlowBackend(f"b{i}", delay=0.03) for i in range(n)]
    r = LLMRouter(backs)
    threads = [threading.Thread(target=lambda: _gen(r)) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(b.calls == 1 for b in backs)


class _RouterBackend:
    """Router-style backend used through Pipeline (no model kw)."""
    kind = "fake"

    def __init__(self, name, delay=0.02):
        self.name = name
        self.delay = delay

    def generate(self, prompt, **kw):
        time.sleep(self.delay)
        return '{"ok": true}'


def _doc(tmp_path, paras=20):
    doc = tmp_path / "doc.txt"
    para = " ".join(["từ"] * 40) + "."
    doc.write_text("\n\n".join(para for _ in range(paras)), encoding="utf-8")
    return doc


def test_concurrent_pipeline_writes_every_chunk(tmp_path):
    doc = _doc(tmp_path)
    out = tmp_path / "out.csv"
    router = LLMRouter([_RouterBackend("a"), _RouterBackend("b"), _RouterBackend("c")])
    p = pl.Pipeline(
        files=[str(doc)], model="", output_csv=str(out),
        instruction="x", chunk_words=120, router=router, concurrency=3,
    )
    stats = p.run()
    rows = _read(out)
    assert stats["chunks"] == len(rows) > 1
    assert stats["errors"] == 0
    assert all(r["output"] == '{"ok":true}' for r in rows)
    # chunk_id set is exactly 0..N-1 — nothing dropped or duplicated.
    ids = sorted(int(r["chunk_id"]) for r in rows)
    assert ids == list(range(len(rows)))


def test_concurrent_is_faster_than_serial(tmp_path):
    """Wall-clock sanity: 3-wide should beat serial on slow backends."""
    out = tmp_path / "out.csv"

    def run(conc, doc):
        router = LLMRouter([_RouterBackend("a", 0.05), _RouterBackend("b", 0.05),
                            _RouterBackend("c", 0.05)])
        p = pl.Pipeline(files=[str(doc)], model="", output_csv=str(out),
                        instruction="x", chunk_words=120, router=router,
                        concurrency=conc, resume=False)
        t0 = time.time()
        p.run()
        return time.time() - t0

    da, db = tmp_path / "a", tmp_path / "b"
    da.mkdir()
    db.mkdir()
    serial = run(1, _doc(da))
    parallel = run(3, _doc(db))
    assert parallel < serial * 0.8   # comfortably faster


def test_concurrent_resume_skips_then_completes(tmp_path):
    doc = _doc(tmp_path)
    out = tmp_path / "out.csv"
    router = LLMRouter([_RouterBackend("a"), _RouterBackend("b")])
    pl.Pipeline(files=[str(doc)], model="", output_csv=str(out),
                instruction="x", chunk_words=120, router=router,
                concurrency=2).run()
    full = len(_read(out))

    # Truncate to first 2 data rows, then resume concurrently.
    all_rows = _read(out)
    fields = ["instruction", "input", "output", "source", "chunk_id"]
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in all_rows[:2]:
            w.writerow({k: r[k] for k in fields})

    router2 = LLMRouter([_RouterBackend("a"), _RouterBackend("b")])
    stats = pl.Pipeline(files=[str(doc)], model="", output_csv=str(out),
                        instruction="x", chunk_words=120, router=router2,
                        concurrency=2, resume=True).run()
    assert stats["skipped"] == 2
    assert len(_read(out)) == full


def test_parallel_file_extraction_concurrent_and_ordered(tmp_path, monkeypatch):
    """Multiple text files are read at once, yet chunk ids stay contiguous and
    grouped in input-file order regardless of which read finished first."""
    files = []
    for i in range(4):
        p = tmp_path / f"f{i}.txt"
        body = "\n\n".join(" ".join(["w"] * 30) + "." for _ in range(4))
        p.write_text(body, encoding="utf-8")
        files.append(str(p))

    state = {"now": 0, "peak": 0, "lock": threading.Lock()}

    def slow_extract(path, **kw):
        with state["lock"]:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        try:
            time.sleep(0.05)
            return pl.read_text_file(path)
        finally:
            with state["lock"]:
                state["now"] -= 1

    monkeypatch.setattr(pl, "extract_text", slow_extract)

    p = pl.Pipeline(files=files, model="m", output_csv=str(tmp_path / "o.csv"),
                    instruction="x", chunk_words=40)
    chunks = p._build_chunks()

    assert state["peak"] >= 2                       # genuinely concurrent reads
    ids = [c[1] for c in chunks]
    assert ids == list(range(len(chunks)))          # contiguous, no gaps/dupes
    srcs = [c[0] for c in chunks]
    first_seen = list(dict.fromkeys(srcs))
    assert first_seen == [f"f{i}.txt" for i in range(4)]   # input order kept


def test_streaming_overlaps_read_with_llm(tmp_path, monkeypatch):
    """The LLM must start before all files finish reading (producer/consumer
    overlap), otherwise the read phase is pure dead time."""
    n = 16
    files = []
    for i in range(n):
        p = tmp_path / f"f{i}.txt"
        p.write_text("\n\n".join(" ".join(["w"] * 30) + "." for _ in range(2)),
                     encoding="utf-8")
        files.append(str(p))

    last_read = [0.0]

    def slow_extract(path, **kw):
        time.sleep(0.05)
        last_read[0] = time.time()
        return pl.read_text_file(path)

    monkeypatch.setattr(pl, "extract_text", slow_extract)

    first_llm = [None]

    class B:
        kind = "fake"
        name = "b"

        def generate(self, prompt, **kw):
            if first_llm[0] is None:
                first_llm[0] = time.time()
            return '{"ok": true}'

    router = LLMRouter([B(), B(), B(), B()])
    p = pl.Pipeline(files=files, model="", output_csv=str(tmp_path / "o.csv"),
                    instruction="x", chunk_words=40, router=router, concurrency=4)
    stats = p.run()

    assert first_llm[0] is not None
    assert first_llm[0] < last_read[0]      # generation began before reads ended
    assert stats["chunks"] == len(_read(tmp_path / "o.csv")) > 1


def test_streaming_stop_before_start_does_not_hang(tmp_path):
    """Stop set before run() must return promptly, not deadlock on the queue."""
    files = []
    for i in range(8):
        p = tmp_path / f"f{i}.txt"
        p.write_text(" ".join(["w"] * 50) + ".", encoding="utf-8")
        files.append(str(p))

    router = LLMRouter([_RouterBackend("a")])
    p = pl.Pipeline(files=files, model="", output_csv=str(tmp_path / "o.csv"),
                    instruction="x", chunk_words=40, router=router, concurrency=2,
                    should_stop=lambda: True)
    t0 = time.time()
    stats = p.run()
    assert time.time() - t0 < 5.0          # returned, no hang
    assert stats["chunks"] == 0


def test_concurrent_handles_backend_errors_without_data_loss(tmp_path):
    """A flaky backend that sometimes raises must not lose chunks: the router
    fails over to a healthy one so every chunk still lands as a row."""
    doc = _doc(tmp_path)
    out = tmp_path / "out.csv"

    class Flaky:
        kind = "fake"
        name = "flaky"

        def generate(self, prompt, **kw):
            raise RateLimitError("429", retry_after=0.01)

    router = LLMRouter([Flaky(), _RouterBackend("good")], cooldown=0.01)
    stats = pl.Pipeline(files=[str(doc)], model="", output_csv=str(out),
                        instruction="x", chunk_words=120, router=router,
                        concurrency=2).run()
    rows = _read(out)
    assert stats["chunks"] == len(rows) > 1
    assert stats["errors"] == 0
