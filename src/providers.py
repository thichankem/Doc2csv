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

# Preset các nhà cung cấp API (đều OpenAI-compatible) + vài model free gợi ý.
# Dùng cho phần "chọn provider" trong GUI: chọn tên → tự điền base_url + models.
PRESETS: dict[str, dict] = {
    "OpenRouter (free)": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "key_url": "https://openrouter.ai/keys",
        "structured": False,
        "headers": {"HTTP-Referer": "https://localhost", "X-Title": "Doc2CSV-AI"},
        "models": [
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemma-2-9b-it:free",
            "qwen/qwen-2.5-72b-instruct:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
        ],
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "key_url": "https://console.groq.com/keys",
        "structured": False,
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "key_url": "https://aistudio.google.com/apikey",
        "structured": False,
        "models": ["gemini-2.0-flash", "gemini-2.0-flash-lite"],
    },
    "Cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "key_env": "CEREBRAS_API_KEY",
        "key_url": "https://cloud.cerebras.ai",
        "structured": False,
        "models": ["llama-3.3-70b", "llama3.1-8b"],
    },
    "Mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "key_env": "MISTRAL_API_KEY",
        "key_url": "https://console.mistral.ai/api-keys",
        "structured": False,
        "models": ["mistral-small-latest"],
    },
    "Ollama (local OpenAI API)": {
        "base_url": "http://localhost:11434/v1",
        "key_env": "",
        "key_url": "",
        "structured": False,
        "models": [],
    },
    "Tùy chỉnh (Custom)": {
        "base_url": "",
        "key_env": "",
        "key_url": "",
        "structured": False,
        "models": [],
    },
}


def preset_names() -> list[str]:
    return list(PRESETS.keys())


def load_config(path: str = DEFAULT_PATH) -> dict:
    """Read providers.json into a dict, or a fresh {'endpoints': []} if absent."""
    p = Path(path)
    if not p.exists():
        return {"endpoints": []}
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"endpoints": []}
    if isinstance(cfg, list):
        cfg = {"endpoints": cfg}
    cfg.setdefault("endpoints", [])
    return cfg


def save_config(cfg: dict, path: str = DEFAULT_PATH) -> None:
    Path(path).write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def add_endpoint(
    cfg: dict,
    name: str,
    base_url: str,
    api_key: str,
    models: list[str],
    *,
    structured: bool = False,
    headers: dict | None = None,
    store_key_in_env: bool = False,
) -> dict:
    """Append/replace an endpoint in the config. When store_key_in_env, the key
    is referenced as env:NAME_API_KEY instead of stored literally."""
    endpoints = [e for e in cfg.get("endpoints", []) if e.get("name") != name]
    api_key_field = api_key
    if store_key_in_env and api_key:
        env_name = name.upper().replace(" ", "_").replace("-", "_") + "_API_KEY"
        os.environ[env_name] = api_key
        api_key_field = f"env:{env_name}"
    ep = {
        "name": name,
        "base_url": base_url,
        "api_key": api_key_field,
        "enabled": True,
        "structured": structured,
        "models": [m for m in models if m.strip()],
    }
    if headers:
        ep["headers"] = headers
    endpoints.append(ep)
    cfg["endpoints"] = endpoints
    return cfg


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
