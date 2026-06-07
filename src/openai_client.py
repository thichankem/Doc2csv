"""OpenAI-compatible Chat Completions client (online APIs).

A single client speaks to ANY provider that implements the OpenAI Chat
Completions API — which is the de-facto standard for free LLM endpoints:
OpenRouter, Groq, Cerebras, Mistral, Together, DeepInfra, and even Google
Gemini (via its `/v1beta/openai` compatibility endpoint) and local LM Studio.

JSON output defaults to `response_format={"type":"json_object"}` for the widest
compatibility — strict `json_schema` is unreliable across free models, so the
pipeline's retry + schema-key validation enforces the actual shape instead.
"""
import json
import time
from typing import Callable, Optional

import requests

_SESSION = requests.Session()

_DEFAULT_MAX_RETRIES = 1
_DEFAULT_RETRY_BACKOFF = 1.5
_DEFAULT_CONNECT_TIMEOUT = 10.0


class RateLimitError(RuntimeError):
    """Raised on HTTP 429 so the router can cool the endpoint down and rotate."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


def _auth_headers(api_key: str, extra_headers: Optional[dict]) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)
    return headers


def list_models(
    base_url: str,
    api_key: str = "",
    timeout: float = 8.0,
    extra_headers: Optional[dict] = None,
) -> list[str]:
    """Return model ids advertised at {base_url}/models, or [] on failure."""
    url = base_url.rstrip("/") + "/models"
    try:
        r = _SESSION.get(url, headers=_auth_headers(api_key, extra_headers), timeout=timeout)
        r.raise_for_status()
        data = r.json()
        items = data.get("data", data) if isinstance(data, dict) else data
        out = []
        for m in items or []:
            mid = m.get("id") if isinstance(m, dict) else None
            if mid:
                out.append(mid)
        return out
    except (requests.RequestException, ValueError):
        return []


def chat(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 512,
    format_json: bool = False,
    json_schema: Optional[dict] = None,
    system: Optional[str] = None,
    on_token: Optional[Callable[[str, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    timeout: int = 600,
    connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_backoff: float = _DEFAULT_RETRY_BACKOFF,
    extra_headers: Optional[dict] = None,
) -> str:
    """Stream a chat completion and return the full assistant text.

    Raises RateLimitError on 429 (so the caller can rotate), other
    requests exceptions on persistent connection failure.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = _auth_headers(api_key, extra_headers)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "extract", "schema": json_schema, "strict": False},
        }
    elif format_json:
        payload["response_format"] = {"type": "json_object"}

    attempt = 0
    while True:
        try:
            r = _SESSION.post(
                url, json=payload, headers=headers,
                timeout=(connect_timeout, timeout), stream=True,
            )
            if r.status_code == 429:
                ra = r.headers.get("retry-after")
                r.close()
                raise RateLimitError(
                    f"429 rate limit ({model})",
                    retry_after=float(ra) if ra and ra.replace(".", "", 1).isdigit() else None,
                )
            r.raise_for_status()
            # SSE responses often omit charset → requests would decode as
            # latin-1 and mangle UTF-8 (Vietnamese). Force UTF-8.
            r.encoding = "utf-8"
            break
        except RateLimitError:
            raise
        except requests.RequestException:
            if should_stop is not None and should_stop():
                return ""
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(retry_backoff * attempt)

    parts: list[str] = []
    total = 0
    try:
        for line in r.iter_lines(decode_unicode=True):
            if should_stop is not None and should_stop():
                break
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line or line == "[DONE]":
                if line == "[DONE]":
                    break
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            choices = data.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            # Streaming uses `delta`; some providers fall back to `message`.
            piece = ""
            delta = choice.get("delta")
            if isinstance(delta, dict):
                piece = delta.get("content") or ""
            if not piece:
                msg = choice.get("message")
                if isinstance(msg, dict):
                    piece = msg.get("content") or ""
            if piece:
                parts.append(piece)
                total += len(piece)
                if on_token is not None:
                    try:
                        on_token(piece, total)
                    except Exception:
                        pass
            if choice.get("finish_reason"):
                break
    finally:
        r.close()

    return "".join(parts)
