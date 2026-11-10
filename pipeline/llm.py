"""Single gateway to LLM: throttling, retry, disk cache, safe JSON.

Provider is chosen by LLM_PROVIDER in .env:
  - "gemini" (default): Google Gemini via google-genai
  - "groq":             Groq via groq-sdk (OpenAI-compatible)

Both providers return a parsed JSON dict. Cache is keyed by provider + model + system + prompt,
so switching providers does not collide with old cache entries.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MODEL = GROQ_MODEL if PROVIDER == "groq" else GEMINI_MODEL

OFFLINE = os.getenv("CLAIMCOURT_OFFLINE", "0") == "1"
MIN_INTERVAL_S = float(os.getenv("LLM_MIN_INTERVAL_S", "2" if PROVIDER == "groq" else "7"))
MAX_ATTEMPTS = 6
RETRYABLE_CODES = {429, 500, 502, 503, 504}
RETRYABLE_EXC = (ConnectionError, TimeoutError, OSError)

_gemini_client = None
_groq_client = None
_last_call_at = 0.0


class LLMError(RuntimeError):
    """Raised when the AI cannot give a usable answer. Message is safe to show the user."""


# ---------- client factories ----------

def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is missing. Add it to .env.")
        _gemini_client = genai.Client(api_key=key)
    return _gemini_client


def _get_groq():
    global _groq_client
    if _groq_client is None:
        try:
            from groq import Groq  # type: ignore
        except ImportError as exc:
            raise LLMError(
                "groq package is not installed. Run: pip install groq"
            ) from exc
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise LLMError("GROQ_API_KEY is missing. Add it to .env.")
        _groq_client = Groq(api_key=key)
    return _groq_client


# ---------- cache and helpers ----------

def _cache_path(system: str, prompt: str) -> Path:
    digest = hashlib.sha256(
        f"{PROVIDER}\n{MODEL}\n{system}\n{prompt}".encode("utf-8")
    ).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def _throttle() -> None:
    global _last_call_at
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.monotonic()


def _parse_json(text: str) -> Any:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    return json.loads(cleaned)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, RETRYABLE_EXC):
        return True
    if getattr(exc, "code", None) in RETRYABLE_CODES:
        return True
    if getattr(exc, "status_code", None) in RETRYABLE_CODES:
        return True
    return False


# ---------- provider-specific one-shot calls ----------

def _call_gemini(system: str, prompt: str) -> str:
    from google.genai import types
    client = _get_gemini()
    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        temperature=0,
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL, contents=prompt, config=config
    )
    return response.text or ""


def _call_groq(system: str, prompt: str) -> str:
    client = _get_groq()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return response.choices[0].message.content or ""


def _call_once(system: str, prompt: str) -> str:
    if PROVIDER == "groq":
        return _call_groq(system, prompt)
    return _call_gemini(system, prompt)


# ---------- public API ----------

def ask_json(system: str, prompt: str, *, use_cache: bool = True) -> dict[str, Any]:
    """Send one prompt and return the parsed JSON object.

    Order: cache hit -> offline guard -> throttled API call with retries.
    Only successful, parsed answers are cached.
    """
    path = _cache_path(system, prompt)
    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if OFFLINE:
        raise LLMError("Offline mode is on and this demo answer is not cached.")

    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        _throttle()
        try:
            raw = _call_once(system, prompt)
            data = _parse_json(raw)
            if not isinstance(data, dict):
                raise ValueError("Top-level JSON must be an object.")
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
        except Exception as exc:
            if not _is_retryable(exc):
                raise LLMError(f"{PROVIDER} call failed: {exc}") from exc
            last_error = exc
        else:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return data

        if attempt < MAX_ATTEMPTS:
            time.sleep(min(2 ** attempt, 60))

    raise LLMError(
        f"{PROVIDER} failed after {MAX_ATTEMPTS} tries. Last error: {last_error}. "
        f"If this is a 429, wait and try again, or switch LLM_PROVIDER in .env."
    )