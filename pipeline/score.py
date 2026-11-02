"""Assessment status: plain code, never AI. Input is verified verdicts from verify.py.

No numeric score. No confidence percentage. Status is derived only from evidence
that passed verification. A claim is never 'contradicted' or 'supported' on an
unverified quote.
"""
from __future__ import annotations

from typing import Any

STATUS_SUPPORTED = "Not supported by policy"      # insurer's rejection contradicts the policy
STATUS_REJECTION_VALID = "Supported by policy"    # policy supports the insurer's rejection
STATUS_INSUFFICIENT = "Insufficient information"  # not enough verified evidence to decide


def _effective_verdict(item: dict[str, Any]) -> str:
    """A verdict only counts if its quote is verified. Otherwise it is 'unclear'."""
    verdict = item.get("verdict", "unclear")
    if verdict in {"rejection_contradicted", "rejection_supported"} and not item.get("quote_verified"):
        return "unclear"
    if verdict not in {"rejection_contradicted", "rejection_supported", "unclear"}:
        return "unclear"
    return verdict


def _status_from_verdicts(effective: list[str]) -> str:
    if not effective:
        return STATUS_INSUFFICIENT
    contradicted = sum(1 for v in effective if v == "rejection_contradicted")
    supported = sum(1 for v in effective if v == "rejection_supported")
    total = len(effective)

    # Every reason contradicted, all verified -> claim not supported by the policy.
    if contradicted == total:
        return STATUS_SUPPORTED
    # Every reason supported, all verified -> rejection looks valid.
    if supported == total:
        return STATUS_REJECTION_VALID
    # Any unresolved reason -> we do not have enough to commit.
    if contradicted < total and supported < total:
        return STATUS_INSUFFICIENT
    # Mixed contradicted and supported with no unclear: still insufficient to commit
    # to a single overall label; the user must read the per-reason lines.
    return STATUS_INSUFFICIENT


def score_claim(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Return per-reason lines and one overall status.

    Each verdict needs at least: reason_id, verdict. quote_verified is read if present;
    a verdict without verified quote counts as unclear for scoring.
    """
    if not verdicts:
        return {
            "status": STATUS_INSUFFICIENT,
            "reasons": [],
            "message": "No rejection reasons were found, so no assessment is available.",
        }

    per_reason = []
    for item in verdicts:
        effective = _effective_verdict(item)
        per_reason.append({
            "reason_id": item.get("reason_id"),
            "verdict": effective,
            "raw_verdict": item.get("verdict", "unclear"),
            "quote_verified": bool(item.get("quote_verified")),
            "explanation": item.get("explanation", ""),
        })

    status = _status_from_verdicts([r["verdict"] for r in per_reason])

    message = ""
    if status == STATUS_REJECTION_VALID:
        message = (
            "Based on the uploaded policy, this rejection appears valid. "
            "You may still ask the insurer for a written review."
        )

    return {
        "status": status,
        "reasons": per_reason,
        "message": message,
    }