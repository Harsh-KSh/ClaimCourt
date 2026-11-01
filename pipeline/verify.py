"""Quote checker: confirms every AI-quoted policy sentence exists in the source PDF.

Pure logic, no network. Unverified quotes downgrade the verdict to 'unclear'.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable

from pipeline.extract import Page

FUZZY_THRESHOLD = 0.98  # high on purpose: near-exact only
MIN_QUOTE_WORDS = 4     # shorter quotes are too easy to match by accident
UNVERIFIED_VERDICT = "unclear"

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class QuoteCheck:
    verified: bool
    page: int | None
    score: float
    reason: str  # short machine-readable note, shown in debug view


def normalize(text: str) -> str:
    """Lowercase, fold punctuation and line breaks into single spaces."""
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return _NON_ALNUM.sub(" ", text).strip()


def verify_quote(quote: str, pages: Iterable[Page]) -> QuoteCheck:
    """Find `quote` in policy pages. Exact match first, then fuzzy window match."""
    norm_quote = normalize(quote)
    words = norm_quote.split()
    if len(words) < MIN_QUOTE_WORDS:
        return QuoteCheck(False, None, 0.0, "quote_too_short")

    page_list = list(pages)

    # 1. Exact match after normalization.
    for page in page_list:
        if norm_quote in normalize(page.text):
            return QuoteCheck(True, page.number, 1.0, "exact")

    # 2. Fuzzy match: slide a window of the same word count over each page.
    best_score, best_page = 0.0, None
    n = len(words)
    for page in page_list:
        page_words = normalize(page.text).split()
        for size in (n - 1, n, n + 1):  # tolerate one-word differences
            if size < 1 or size > len(page_words):
                continue
            for start in range(len(page_words) - size + 1):
                window = " ".join(page_words[start:start + size])
                score = SequenceMatcher(None, norm_quote, window).ratio()
                if score > best_score:
                    best_score, best_page = score, page.number

    if best_score >= FUZZY_THRESHOLD:
        return QuoteCheck(True, best_page, round(best_score, 3), "fuzzy")
    return QuoteCheck(False, None, round(best_score, 3), "not_found")


def apply_verification(verdicts: list[dict[str, Any]], pages: list[Page]) -> list[dict[str, Any]]:
    """Check each verdict's quote. Downgrade and flag any that fail.

    Expects keys: reason_id, verdict, policy_quote. Adds: quote_verified,
    quote_page, quote_score, quote_check. Returns new dicts; input is not mutated.
    """
    checked: list[dict[str, Any]] = []
    for item in verdicts:
        quote = (item.get("policy_quote") or "").strip()
        result = verify_quote(quote, pages) if quote else QuoteCheck(False, None, 0.0, "no_quote")

        new = dict(item)
        new["quote_verified"] = result.verified
        new["quote_page"] = result.page if result.verified else item.get("page")
        new["quote_score"] = result.score
        new["quote_check"] = result.reason
        if not result.verified:
            new["verdict"] = UNVERIFIED_VERDICT
        checked.append(new)
    return checked