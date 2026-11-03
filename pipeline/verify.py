"""Quote checker: confirms every AI-quoted policy sentence appears word-for-word in the source PDF.

Strict: only harmless formatting differences are ignored (case, punctuation, line breaks,
ligatures). No fuzzy matching. A quote that is not found is never treated as evidence.

Clause numbers are verified positionally: the claimed clause (e.g. "3.7A") must appear
within CLAUSE_WINDOW_CHARS characters before the quote on the same page. A clause that
merely appears somewhere on the page is not accepted. This prevents a hallucinated clause
from riding along with a real quote from a different clause on the same page.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

from pipeline.extract import Page

MIN_QUOTE_WORDS = 4
CLAUSE_WINDOW_CHARS = 200
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
    score: float
    reason: str
    position: int | None  # character offset of the quote in the normalized page, else None


def normalize(text: str) -> str:
    """Fold harmless formatting differences into single-spaced lowercase words.

    Does not change word order, words, or digit sequences. "136" stays different from "36".
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    for src, dst in _QUOTE_MARKS.items():
        text = text.replace(src, dst)
    return " ".join(_NON_ALNUM.sub(" ", text).split())


def _find_word_run(haystack: list[str], needle: list[str]) -> int | None:
    """Return start index of `needle` as a whole, consecutive run of words, else None."""
    n = len(needle)
    if n == 0 or n > len(haystack):
        return None
    first = needle[0]
    for i in range(len(haystack) - n + 1):
        if haystack[i] == first and haystack[i:i + n] == needle:
            return i
    return None


def verify_quote(quote: str, pages: Iterable[Page]) -> QuoteCheck:
    """Find `quote` as an exact word sequence on a single policy page.

    Quotes that span two pages are reported as not found.
    """
    if not isinstance(quote, str):
        return QuoteCheck(False, None, 0.0, "no_quote", None)
    needle = normalize(quote).split()
    if len(needle) < MIN_QUOTE_WORDS:
        return QuoteCheck(False, None, 0.0, "quote_too_short", None)

    matched_pages: list[tuple[int, int]] = []
    for page in pages:
        page_words = normalize(page.text).split()
        idx = _find_word_run(page_words, needle)
        if idx is not None:
            matched_pages.append((page.number, idx))

    if not matched_pages:
        return QuoteCheck(False, None, 0.0, "not_found", None)
    if len(matched_pages) == 1:
        page_no, word_idx = matched_pages[0]
        return QuoteCheck(True, page_no, 1.0, "exact", word_idx)
    page_no, word_idx = matched_pages[0]
    return QuoteCheck(True, page_no, 1.0, "exact_multiple_pages", word_idx)


def _clause_near_quote(clause: str, page_text: str, quote_word_idx: int | None) -> bool:
    """True if the clause token appears within CLAUSE_WINDOW_CHARS chars before the quote.

    Compares on the normalized page text so "3.7 A" matches "3.7A".
    """
    if quote_word_idx is None:
        return False
    c = normalize(clause).replace(" ", "")
    if not c:
        return False
    norm_page = normalize(page_text)
    norm_words = norm_page.split()
    if quote_word_idx > len(norm_words):
        return False
    # Character offset of the quote start, in the normalized page text.
    quote_char = len(" ".join(norm_words[:quote_word_idx]))
    window_start = max(0, quote_char - CLAUSE_WINDOW_CHARS)
    window = norm_page[window_start:quote_char]
    window_compact = window.replace(" ", "")
    return c in window_compact


def apply_verification(verdicts: list[dict[str, Any]], pages: list[Page]) -> list[dict[str, Any]]:
    """Check each verdict's quote and clause. Downgrade and flag any that fail.

    Adds: quote_verified, quote_page, quote_score, quote_check, quote_position,
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
        result = verify_quote(quote, pages) if quote else QuoteCheck(False, None, 0.0, "no_quote", None)

        clause = item.get("clause")
        if not isinstance(clause, str):
            clause = None
        clause = clause.strip() if clause else None

        clause_verified = False
        clause_check = "no_clause"
        if result.verified and clause:
            page_text = page_by_number[result.page].text if result.page in page_by_number else ""
            if _clause_near_quote(clause, page_text, result.position):
                clause_verified = True
                clause_check = "exact_near_quote"
            else:
                clause_check = "not_near_quote"

        new = dict(item)
        new["verdict"] = verdict
        new["quote_verified"] = result.verified
        new["quote_page"] = result.page if result.verified else None
        new["quote_score"] = result.score
        new["quote_check"] = result.reason
        new["quote_position"] = result.position
        new["clause"] = clause
        new["clause_verified"] = clause_verified
        new["clause_check"] = clause_check

        if not result.verified:
            new["verdict"] = UNVERIFIED_VERDICT
        elif clause and not clause_verified:
            new["verdict"] = UNVERIFIED_VERDICT

        checked.append(new)
    return checked