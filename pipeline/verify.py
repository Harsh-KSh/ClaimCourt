"""Quote checker: confirms every AI-quoted policy sentence appears word-for-word in the source PDF.

Strict: only harmless formatting differences are ignored (case, punctuation, line breaks,
ligatures). No fuzzy matching. A quote that is not found is never treated as evidence.

Also checks the claimed policy clause number (e.g. "3.7A") against the matched page.
A quote that verifies but whose claimed clause is not on that page is downgraded to unclear.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

from pipeline.extract import Page

MIN_QUOTE_WORDS = 4
UNVERIFIED_VERDICT = "unclear"

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_QUOTE_MARKS = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-",
}


@dataclass(frozen=True)
class QuoteCheck:
    verified: bool
    page: int | None
    score: float  # 1.0 if verified, else 0.0
    reason: str   # "exact", "exact_multiple_pages", "quote_too_short", "not_found", "no_quote"


def normalize(text: str) -> str:
    """Fold harmless formatting differences into single-spaced lowercase words.

    Does not change word order, words, or digit sequences. "136" stays different from "36".
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    for src, dst in _QUOTE_MARKS.items():
        text = text.replace(src, dst)
    return " ".join(_NON_ALNUM.sub(" ", text).split())


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


def verify_quote(quote: str, pages: Iterable[Page]) -> QuoteCheck:
    """Find `quote` as an exact word sequence on a single policy page.

    Quotes that span two pages are reported as not found. That is intentional:
    the reviewer must see the exact page, not a guess.
    """
    if not isinstance(quote, str):
        return QuoteCheck(False, None, 0.0, "no_quote")
    needle = normalize(quote).split()
    if len(needle) < MIN_QUOTE_WORDS:
        return QuoteCheck(False, None, 0.0, "quote_too_short")

    matched_pages = [
        page.number
        for page in pages
        if _contains_word_run(normalize(page.text).split(), needle)
    ]

    if not matched_pages:
        return QuoteCheck(False, None, 0.0, "not_found")
    if len(matched_pages) == 1:
        return QuoteCheck(True, matched_pages[0], 1.0, "exact")
    return QuoteCheck(True, matched_pages[0], 1.0, "exact_multiple_pages")


def _clause_present(clause: str, page_text: str) -> bool:
    """True if the clause token appears on the page as a standalone token.

    Compares normalized forms so "3.7 A" matches "3.7A". Whole-token match only,
    so "3.7A" does not accidentally match "13.7A" or "3.7AB".
    """
    c = normalize(clause).replace(" ", "")
    if not c:
        return False
    page_norm = normalize(page_text)
    for token in page_norm.split():
        if token == c:
            return True
    # Also allow clause tokens that the normalizer split, e.g. "3 7a" -> "37a".
    compact = page_norm.replace(" ", "")
    return c in compact


def apply_verification(verdicts: list[dict[str, Any]], pages: list[Page]) -> list[dict[str, Any]]:
    """Check each verdict's quote and clause. Downgrade and flag any that fail.

    Adds: quote_verified, quote_page, quote_score, quote_check,
          clause_verified, clause_check.
    Returns new dicts; input is not mutated.
    """
    page_by_number = {p.number: p for p in pages}
    checked: list[dict[str, Any]] = []

    for item in verdicts:
        if not isinstance(item, dict):
            continue

        verdict = item.get("verdict")
        if verdict not in {"rejection_contradicted", "unclear", "rejection_supported"}:
            verdict = UNVERIFIED_VERDICT

        quote = item.get("policy_quote")
        if not isinstance(quote, str):
            quote = ""
        quote = quote.strip()
        result = verify_quote(quote, pages) if quote else QuoteCheck(False, None, 0.0, "no_quote")

        clause = item.get("clause")
        if not isinstance(clause, str):
            clause = None
        clause = clause.strip() if clause else None

        clause_verified = False
        clause_check = "no_clause"
        if result.verified and clause:
            page_text = page_by_number[result.page].text if result.page in page_by_number else ""
            if _clause_present(clause, page_text):
                clause_verified = True
                clause_check = "exact"
            else:
                clause_check = "not_found"

        new = dict(item)
        new["verdict"] = verdict
        new["quote_verified"] = result.verified
        new["quote_page"] = result.page if result.verified else None
        new["quote_score"] = result.score
        new["quote_check"] = result.reason
        new["clause"] = clause
        new["clause_verified"] = clause_verified
        new["clause_check"] = clause_check

        if not result.verified:
            new["verdict"] = UNVERIFIED_VERDICT
        elif clause and not clause_verified:
            # Quote is real but the claimed clause number is not on that page.
            new["verdict"] = UNVERIFIED_VERDICT

        checked.append(new)
    return checked