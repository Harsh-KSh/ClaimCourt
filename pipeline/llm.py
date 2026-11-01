"""Single gateway to Gemini: throttling, retry, disk cache, safe JSON."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")  # confirm name in AI Studio
OFFLINE = os.getenv("CLAIMCOURT_OFFLINE", "0") == "1"
MIN_INTERVAL_S = float(os.getenv("GEMINI_MIN_INTERVAL_S", "7"))  # keeps you under free-tier RPM
MAX_ATTEMPTS = 4
RETRYABLE_CODES = {429, 500, 503, 504}

_client: genai.Client | None = None
_last_call_at = 0.0


class LLMError(RuntimeError):
    """Raised when the AI cannot give a usable answer. Message is safe to show the user."""


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is missing. Add it to your .env file.")
        _client = genai.Client(api_key=key)
    return _client


def _cache_path(system: str, prompt: str) -> Path:
    digest = hashlib.sha256(f"{MODEL}\n{system}\n{prompt}".encode("utf-8")).hexdigest()
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

    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        temperature=0,
    )
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        _throttle()
        try:
            response = _get_client().models.generate_content(
                model=MODEL, contents=prompt, config=config
            )
            data = _parse_json(response.text or "")
            if not isinstance(data, dict):
                raise ValueError("Top-level JSON must be an object.")
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc  # invalid output: retry
        except Exception as exc:  # SDK errors carry an HTTP-style .code
            if getattr(exc, "code", None) not in RETRYABLE_CODES:
                raise LLMError(f"Gemini call failed: {exc}") from exc
            last_error = exc
        else:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return data

        time.sleep(min(2 ** attempt, 30))

    raise LLMError(
        f"AI service busy or returned invalid output after {MAX_ATTEMPTS} tries: {last_error}"
    )