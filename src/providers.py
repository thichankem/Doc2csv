"""Load online API endpoints from a JSON config into rotating backends.

Config file (default `providers.json` in the working dir) shape:

{
  "endpoints": [
    {
      "name": "openrouter",
      "base_url": "https://openrouter.ai/api/v1",
      "api_key": "env:OPENROUTER_API_KEY",   # or a literal key
      "models": ["meta-llama/llama-3.3-70b-instruct:free", "google/gemma-2-9b-it:free"],
      "enabled": true,
      "structured": false,                    # true → send strict json_schema
      "headers": {"HTTP-Referer": "...", "X-Title": "Doc2CSV-AI"}
    }
  ]
}

Every (endpoint × model) pair becomes one OpenAIBackend in the rotation pool, so
listing several free models multiplies how long you can run before hitting limits.
`api_key` may be a literal string or "env:VARNAME" to read from the environment
(keeps secrets out of the committed file).
"""
import json
import os
from pathlib import Path
from typing import Callable, Optional

from .llm_backends import OpenAIBackend

DEFAULT_PATH = "providers.json"


def _resolve_key(val) -> str:
    if isinstance(val, str) and val.startswith("env:"):
        return os.environ.get(val[4:], "")
    return str(val or "")


def parse_backends(cfg: dict, on_log: Optional[Callable[[str], None]] = None) -> list:
    """Turn a parsed config dict into a list of OpenAIBackend rotation entries."""
    log = on_log or (lambda m: None)
    endpoints = cfg.get("endpoints", []) if isinstance(cfg, dict) else (cfg or [])
    backends: list = []
    skipped = 0
    for ep in endpoints:
        if not isinstance(ep, dict) or not ep.get("enabled", True):
            continue
        base = str(ep.get("base_url", "")).strip()
        if not base:
            continue
        key = _resolve_key(ep.get("api_key", ""))
        name = str(ep.get("name") or base).strip()
        headers = ep.get("headers") if isinstance(ep.get("headers"), dict) else None
        use_schema = bool(ep.get("structured", False))
        models = ep.get("models")
        if not models and ep.get("model"):
            models = [ep["model"]]
        for m in models or []:
            m = str(m).strip()
            if not m:
                continue
            if not key:
                skipped += 1
            backends.append(OpenAIBackend(
                name=f"{name}:{m}", base_url=base, api_key=key, model=m,
                use_schema=use_schema, extra_headers=headers,
            ))
    if skipped:
        log(f"   ⚠ {skipped} backend chưa có API key (đặt env hoặc điền 'api_key')")
    return backends


def load_backends(path: str = DEFAULT_PATH, on_log: Optional[Callable[[str], None]] = None) -> list:
    """Read a providers JSON file and return its rotation backends ([] if missing)."""
    log = on_log or (lambda m: None)
    p = Path(path)
    if not p.exists():
        return []
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log(f"⚠ Lỗi đọc {path}: {e}")
        return []
    backends = parse_backends(cfg, on_log=on_log)
    log(f"✓ Nạp {len(backends)} backend online từ {path}")
    return backends
