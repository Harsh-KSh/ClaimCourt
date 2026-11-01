"""Section finding (keyword scoring, no AI) + AI Call 2: judge each reason against policy and verified rules."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pipeline.extract import Page
from pipeline.llm import LLMError, ask_json
from pipeline.verify import apply_verification

ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "data" / "rules.json"
TOP_K = 4
CHUNK_CHARS = 700

KEYWORDS = {
    "pre_existing": ["pre-existing", "pre existing", "ped", "disease", "prior"],
    "waiting_period": ["waiting", "period", "months", "continuous", "months"],
    "non_disclosure": ["disclos", "declar", "misrepresent", "continuous", "renew"],
    "exclusion": ["exclu", "not covered", "permanent", "cosmetic", "shall not"],
    "documentation": ["document", "proof", "submit", "bill", "discharge"],
    "other": [],
}

SYSTEM = """You judge health insurance claim rejection reasons against a policy and a short list of rules.
Rules:
- Use ONLY the policy excerpts and rules provided. Do not use outside knowledge.
- policy_quote must be copied EXACTLY (character for character) from a policy excerpt, and include its page number. If no exact quote supports your answer, set verdict to "unclear" and policy_quote to "".
- verdict must be one of: rejection_contradicted, unclear, rejection_supported.
  - rejection_contradicted: policy or a provided rule shows the rejection is not valid.
  - rejection_supported: policy clearly supports the rejection (for example a clear permanent exclusion).
  - unclear: evidence is missing or ambiguous.
- rule_ids_used lists only ids from the provided rules.
- explanation: 2 short sentences, simple language.
- facts_needed_from_user: list of short strings, empty if none.
- Do not give legal advice.
Return JSON only, in this exact shape:
{"verdicts": [{"reason_id": "string", "verdict": "string", "policy_quote": "string", "page": 1, "rule_ids_used": ["R1"], "explanation": "string", "facts_needed_from_user": []}]}
"""


def load_verified_rules() -> list[dict[str, Any]]:
    """Only rules marked verified=true reach the AI."""
    data = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    return [r for r in data.get("rules", []) if r.get("verified") is True]


def chunk_pages(pages: list[Page]) -> list[dict[str, Any]]:
    """Split each page into paragraph-sized chunks, keeping the page number."""
    chunks: list[dict[str, Any]] = []
    for page in pages:
        paras = [p.strip() for p in re.split(r"\n\s*\n|\n", page.text) if p.strip()]
        buf = ""
        for para in paras:
            if buf and len(buf) + len(para) > CHUNK_CHARS:
                chunks.append({"page": page.number, "text": buf})
                buf = para
            else:
                buf = f"{buf} {para}".strip()
        if buf:
            chunks.append({"page": page.number, "text": buf})
    return chunks


def _score_chunk(text: str, keywords: list[str]) -> int:
    lower = text.lower()
    return sum(lower.count(k) for k in set(keywords))


def pick_chunks(reason: dict[str, Any], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Top-K chunks by keyword hits for the reason's category plus its plain-English words."""
    words = [w for w in re.findall(r"[a-z]{5,}", reason.get("plain_english", "").lower())]
    keywords = KEYWORDS.get(reason.get("category", "other"), []) + words
    ranked = sorted(chunks, key=lambda c: _score_chunk(c["text"], keywords), reverse=True)
    picked = [c for c in ranked[:TOP_K] if _score_chunk(c["text"], keywords) > 0]
    return picked or ranked[:1]


def judge_reasons(
    reasons: list[dict[str, Any]],
    pages: list[Page],
) -> list[dict[str, Any]]:
    """AI Call 2. Returns verdicts with quote verification applied (verify.py).

    Raises LLMError if the AI output is unusable.
    """
    if not reasons:
        return []

    rules = load_verified_rules()
    chunks = chunk_pages(pages)

    blocks = []
    for reason in reasons:
        excerpts = pick_chunks(reason, chunks)
        excerpt_text = "\n".join(f"[page {c['page']}] {c['text']}" for c in excerpts)
        blocks.append(
            f"REASON id={reason['id']} category={reason['category']}\n"
            f"Insurer said: {reason['insurer_words']}\n"
            f"Plain meaning: {reason['plain_english']}\n"
            f"POLICY EXCERPTS:\n{excerpt_text}"
        )

    rules_text = "\n".join(f"{r['id']}: {r['text']}" for r in rules) or "(no verified rules)"
    prompt = "RULES:\n" + rules_text + "\n\n" + "\n\n".join(blocks)

    data = ask_json(SYSTEM, prompt)
    raw = data.get("verdicts")
    if not isinstance(raw, list):
        raise LLMError("AI returned no verdicts list. Try again.")

    valid_rule_ids = {r["id"] for r in rules}
    cleaned: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict", "unclear"))
        if verdict not in {"rejection_contradicted", "unclear", "rejection_supported"}:
            verdict = "unclear"
        cleaned.append({
            "reason_id": str(item.get("reason_id", "")),
            "verdict": verdict,
            "policy_quote": str(item.get("policy_quote", "")).strip(),
            "page": item.get("page"),
            "rule_ids_used": [r for r in item.get("rule_ids_used", []) if r in valid_rule_ids],
            "explanation": str(item.get("explanation", "")).strip(),
            "facts_needed_from_user": [str(f) for f in item.get("facts_needed_from_user", [])],
        })

    return apply_verification(cleaned, pages)