"""Minimal Ollama HTTP client with optional token streaming."""
import json
from typing import Callable, Optional

import requests

DEFAULT_BASE = "http://localhost:11434"
_SESSION = requests.Session()


def is_running(base_url: str = DEFAULT_BASE, timeout: float = 2.0) -> bool:
    try:
        r = _SESSION.get(f"{base_url}/api/tags", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def list_models(base_url: str = DEFAULT_BASE, timeout: float = 5.0) -> list[str]:
    """Return names of locally installed Ollama models, or [] if unreachable."""
    try:
        r = _SESSION.get(f"{base_url}/api/tags", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models", []) if "name" in m]
    except requests.RequestException:
        return []


def preload(model: str, base_url: str = DEFAULT_BASE,
            keep_alive: str = "30m", timeout: int = 60) -> bool:
    """Warm the model into VRAM without generating any output.
    Returns True on success. Big speedup when followed by many small calls."""
    try:
        r = _SESSION.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": "", "stream": False, "keep_alive": keep_alive},
            timeout=timeout,
        )
        return r.status_code == 200
    except requests.RequestException:
        return False


def generate(
    model: str,
    prompt: str,
    base_url: str = DEFAULT_BASE,
    temperature: float = 0.3,
    num_ctx: int = 8192,
    options: Optional[dict] = None,
    images: Optional[list[str]] = None,
    format_json: bool = False,
    json_schema: Optional[dict] = None,
    keep_alive: Optional[str] = None,
    timeout: int = 600,
    on_token: Optional[Callable[[str, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> str:
    """Call /api/generate with streaming. Returns the full response text.

    - on_token(chunk_text, total_chars): invoked for each streamed chunk; use it
      to update UI without blocking.
    - should_stop(): if it returns True, the stream is aborted and a partial
      response (whatever was generated so far) is returned.
    - images: list of base64-encoded images for vision models (llava,
      llama3.2-vision, minicpm-v, qwen2.5vl...). Gửi kèm field `images` của
      /api/generate — model PHẢI hỗ trợ ảnh, model text thường sẽ bỏ qua.
    - json_schema: if provided, sent as `format=<schema>` (Ollama Structured
      Outputs, ≥ 0.5). Cứng hơn `format_json` rất nhiều — model BẮT BUỘC dùng
      đúng keys & types. Khi có schema, `format_json` bị bỏ qua.
    """
    merged_options = {
        "temperature": temperature,
        "num_ctx": num_ctx,
    }
    if options:
        merged_options.update(options)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": merged_options,
    }
    if images:
        payload["images"] = images
    # Schema (dict) ưu tiên hơn "json" lỏng.
    if json_schema:
        payload["format"] = json_schema
    elif format_json:
        payload["format"] = "json"
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    parts: list[str] = []
    total = 0

    r = _SESSION.post(
        f"{base_url}/api/generate",
        json=payload,
        timeout=timeout,
        stream=True,
    )
    try:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if should_stop is not None and should_stop():
                break
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            piece = data.get("response", "")
            if piece:
                parts.append(piece)
                total += len(piece)
                if on_token is not None:
                    try:
                        on_token(piece, total)
                    except Exception:
                        pass
            if data.get("done"):
                break
    finally:
        r.close()

    return "".join(parts)
