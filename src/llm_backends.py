"""LLM backend abstraction + round-robin router across local & online models.

A `Backend` exposes a uniform `generate(...)`. Two concrete kinds:

  * OllamaBackend  — a local model served by Ollama.
  * OpenAIBackend  — any OpenAI-compatible online endpoint (one per model).

`LLMRouter` holds a pool of backends and rotates through them round-robin. When
a backend hits a rate-limit (429) or errors, it is put on a short cooldown and
the router moves to the next one — so a long extraction run keeps going across
free-tier limits instead of stalling. This is what enables "full online" and
"mix" modes to run continuously.
"""
import threading
import time
from typing import Callable, Optional

from . import ollama_client, openai_client
from .openai_client import RateLimitError


class Backend:
    name = "backend"
    kind = "base"

    def warmup(self, keep_alive: str = "30m", on_log: Optional[Callable] = None) -> bool:
        return True

    def generate(self, prompt: str, **kw) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class OllamaBackend(Backend):
    kind = "ollama"

    def __init__(self, model: str, base_url: str = ollama_client.DEFAULT_BASE):
        self.model = model
        self.base_url = base_url
        self.name = f"ollama:{model}"

    def warmup(self, keep_alive: str = "30m", on_log: Optional[Callable] = None) -> bool:
        ok = ollama_client.preload(self.model, base_url=self.base_url, keep_alive=keep_alive)
        if on_log:
            on_log(f"   {'✓' if ok else '⚠'} Preload {self.name}")
        return ok

    def generate(
        self, prompt, *, temperature, num_predict, num_ctx,
        format_json, json_schema, keep_alive, on_token, should_stop,
    ) -> str:
        return ollama_client.generate(
            model=self.model,
            prompt=prompt,
            base_url=self.base_url,
            temperature=temperature,
            num_ctx=num_ctx,
            options={"num_predict": num_predict},
            format_json=format_json,
            json_schema=json_schema,
            keep_alive=keep_alive,
            on_token=on_token,
            should_stop=should_stop,
        )


class OpenAIBackend(Backend):
    kind = "openai"

    def __init__(
        self, name: str, base_url: str, api_key: str, model: str,
        use_schema: bool = False, extra_headers: Optional[dict] = None,
    ):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.use_schema = use_schema
        self.extra_headers = extra_headers

    def generate(
        self, prompt, *, temperature, num_predict, num_ctx,
        format_json, json_schema, keep_alive, on_token, should_stop,
    ) -> str:
        return openai_client.chat(
            self.base_url,
            self.api_key,
            self.model,
            prompt,
            temperature=temperature,
            max_tokens=num_predict,
            format_json=format_json,
            json_schema=json_schema if self.use_schema else None,
            on_token=on_token,
            should_stop=should_stop,
            extra_headers=self.extra_headers,
        )


class LLMRouter:
    """Round-robin pool of backends with per-backend cooldown on failure.

    Thread-safe: the cursor + cooldown table are guarded by a lock and a backend
    is *claimed* (cursor advanced) before its slow network call runs, so several
    threads calling :meth:`generate` concurrently each grab a distinct backend.
    This is what lets the pipeline fan chunks out across many online endpoints at
    once instead of one-at-a-time.
    """

    def __init__(
        self,
        backends: list,
        on_log: Optional[Callable[[str], None]] = None,
        cooldown: float = 30.0,
        max_cycles: int = 3,
        wait_cap: float = 15.0,
    ):
        if not backends:
            raise ValueError("Router cần ít nhất một backend.")
        self.backends = list(backends)
        self.on_log = on_log or (lambda m: None)
        self.cooldown = cooldown
        self.max_cycles = max_cycles
        self.wait_cap = wait_cap
        self._i = 0
        self._cool: dict[str, float] = {}
        self._lock = threading.Lock()
        self.last_backend: Optional[str] = None

    def __len__(self) -> int:
        return len(self.backends)

    def names(self) -> list[str]:
        return [b.name for b in self.backends]

    def warmup(self, keep_alive: str = "30m") -> None:
        seen: set[str] = set()
        for b in self.backends:
            if b.kind == "ollama" and getattr(b, "model", None) not in seen:
                seen.add(b.model)
                b.warmup(keep_alive=keep_alive, on_log=self.on_log)

    def _interruptible_sleep(self, seconds: float, should_stop) -> None:
        end = time.time() + seconds
        while time.time() < end:
            if should_stop and should_stop():
                return
            time.sleep(0.2)

    def _claim(self, exclude: set) -> tuple:
        """Atomically reserve the next ready backend, advancing the cursor past
        it. Returns (backend, None) when one is free, or (None, soonest) where
        ``soonest`` is the earliest cooldown-expiry among backends that are *not*
        in ``exclude`` (backends this caller already failed) — i.e. the earliest
        moment it could be worth waiting for. (None, None) means give up now."""
        with self._lock:
            n = len(self.backends)
            now = time.time()
            soonest: Optional[float] = None
            for step in range(n):
                idx = (self._i + step) % n
                b = self.backends[idx]
                until = self._cool.get(b.name, 0.0)
                if until <= now:
                    self._i = (idx + 1) % n
                    return b, None
                if b.name not in exclude:
                    soonest = until if soonest is None else min(soonest, until)
            return None, soonest

    def generate(
        self, prompt, *, temperature, num_predict, num_ctx,
        format_json, json_schema, keep_alive, on_token, should_stop,
    ) -> str:
        n = len(self.backends)
        errs: list[str] = []
        failed: set = set()   # backends this call already burned — don't wait on them
        cycles = 0
        while True:
            if should_stop and should_stop():
                return ""
            b, soonest = self._claim(failed)
            if b is not None:
                try:
                    text = b.generate(
                        prompt, temperature=temperature, num_predict=num_predict,
                        num_ctx=num_ctx, format_json=format_json, json_schema=json_schema,
                        keep_alive=keep_alive, on_token=on_token, should_stop=should_stop,
                    )
                    self.last_backend = b.name
                    return text
                except RateLimitError as e:
                    wait = e.retry_after or self.cooldown
                    with self._lock:
                        self._cool[b.name] = time.time() + wait
                    failed.add(b.name)
                    errs.append(f"{b.name}: 429")
                    self.on_log(f"   ⏳ {b.name}: rate-limit (nghỉ {wait:.0f}s) — xoay backend")
                except Exception as e:  # noqa: BLE001 - any failure → cooldown + rotate
                    with self._lock:
                        self._cool[b.name] = time.time() + self.cooldown
                    failed.add(b.name)
                    errs.append(f"{b.name}: {e}")
                    self.on_log(f"   ⚠ {b.name}: {e} — xoay backend")
                continue
            # No backend ready right now. Only wait if something is merely cooling
            # (not one we just burned) and we still have cycles left.
            if (soonest is not None and cycles < self.max_cycles
                    and not (should_stop and should_stop())):
                cycles += 1
                delay = max(0.0, min(soonest - time.time(), self.wait_cap))
                if delay > 0:
                    self.on_log(f"   ⏸ Mọi backend đang nghỉ — chờ {delay:.0f}s rồi thử lại")
                    self._interruptible_sleep(delay, should_stop)
                continue
            break
        raise RuntimeError("Tất cả backend đều lỗi/đang nghỉ: " + " | ".join(errs[-n:]))
