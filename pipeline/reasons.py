"""AI Call 1: extract rejection reasons from the rejection letter. Returns validated dicts.

The AI's insurer_words is verified against the letter text using the same strict
word-run matching as verify.py. A hallucinated or paraphrased quote is kept (for
transparency) but flagged insurer_words_verified=False so it can never appear as
a verified quote in the UI or the appeal.
"""
from __future__ import annotations

from typing import Any

from pipeline.llm import LLMError, ask_json
from pipeline.verify import normalize

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


def _contains_word_run(haystack: list[str], needle: list[str]) -> bool:
    """True only if `needle` appears as a whole, consecutive run of words."""
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    first = needle[0]
    for i in range(len(haystack) - n + 1):
        if haystack[i] == first and haystack[i:i + n] == needle:
            return True
    return False


def _insurer_words_in_letter(insurer_words: str, letter_text: str) -> bool:
    """True if the insurer's quoted wording appears in the letter as an exact word run."""
    needle = normalize(insurer_words).split()
    if not needle:
        return False
    return _contains_word_run(normalize(letter_text).split(), needle)


def _clean_reason(raw: Any, index: int, letter_text: str) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    insurer_words = str(raw.get("insurer_words", "")).strip()
    plain = str(raw.get("plain_english", "")).strip()
    if not plain:
        return None
    category = str(raw.get("category", "other")).strip()
    if category not in CATEGORIES:
        category = "other"
    verified = _insurer_words_in_letter(insurer_words, letter_text) if insurer_words else False
    return {
        "id": str(raw.get("id") or index),
        "category": category,
        "insurer_words": insurer_words,
        "insurer_words_verified": verified,
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
        cleaned = _clean_reason(raw, i, letter_text)
        if cleaned:
            reasons.append(cleaned)

    return {
        "claim_type": str(data.get("claim_type", "")).strip(),
        "reasons": reasons,
    }