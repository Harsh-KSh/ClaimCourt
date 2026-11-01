"""AI Call 1: extract rejection reasons from the rejection letter. Returns validated dicts."""
from __future__ import annotations

from typing import Any

from pipeline.llm import LLMError, ask_json

CATEGORIES = {
    "pre_existing",
    "non_disclosure",
    "exclusion",
    "waiting_period",
    "documentation",
    "other",
}

SYSTEM = """You read a health insurance claim rejection letter and list the reasons given for rejection.
Rules:
- Use ONLY the letter text provided. Do not add reasons it does not state.
- insurer_words must be copied exactly from the letter (a short phrase, under 30 words).
- category must be one of: pre_existing, non_disclosure, exclusion, waiting_period, documentation, other.
- Do not give legal advice.
Return JSON only, in this exact shape:
{"claim_type": "string", "reasons": [{"id": "1", "category": "string", "insurer_words": "string", "plain_english": "string"}]}
"""


def _clean_reason(raw: Any, index: int) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    insurer_words = str(raw.get("insurer_words", "")).strip()
    plain = str(raw.get("plain_english", "")).strip()
    if not plain:
        return None
    category = str(raw.get("category", "other")).strip()
    if category not in CATEGORIES:
        category = "other"
    return {
        "id": str(raw.get("id") or index),
        "category": category,
        "insurer_words": insurer_words,
        "plain_english": plain,
    }


def extract_reasons(letter_text: str) -> dict[str, Any]:
    """Return {'claim_type': str, 'reasons': [...]}. Raises LLMError on unusable output."""
    letter_text = letter_text.strip()
    if not letter_text:
        raise LLMError("The rejection letter has no text to analyse.")

    data = ask_json(SYSTEM, f"REJECTION LETTER:\n{letter_text}")

    raw_reasons = data.get("reasons")
    if not isinstance(raw_reasons, list):
        raise LLMError("AI returned no reasons list. Try again.")

    reasons = []
    for i, raw in enumerate(raw_reasons, start=1):
        cleaned = _clean_reason(raw, i)
        if cleaned:
            reasons.append(cleaned)

    return {
        "claim_type": str(data.get("claim_type", "")).strip(),
        "reasons": reasons,
    }