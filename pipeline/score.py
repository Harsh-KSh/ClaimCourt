"""Strength score: plain code, never AI. Input is verified verdicts from verify.py."""
from __future__ import annotations

from typing import Any

POINTS = {
    "rejection_contradicted": 100,
    "unclear": 40,
    "rejection_supported": 0,
}
WEAK_CAP = 20  # applied when the only reason is rejection_supported
LABELS = ((70, "Strong"), (40, "Medium"), (0, "Weak"))
WEAK_MESSAGE = (
    "Based on your policy, this rejection looks valid. "
    "You may still ask the insurer for a review."
)


def _label(score: int) -> str:
    for threshold, label in LABELS:
        if score >= threshold:
            return label
    return "Weak"


def score_claim(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Return score, label, per-reason lines, and the honesty message if Weak.

    Each verdict needs: reason_id, verdict. Verdicts must already pass verify.py.
    """
    if not verdicts:
        return {
            "score": 0,
            "label": "Weak",
            "reasons": [],
            "message": "No rejection reasons were found, so no strength score is available.",
        }

    per_reason = []
    for item in verdicts:
        verdict = item.get("verdict", "unclear")
        points = POINTS.get(verdict, POINTS["unclear"])
        per_reason.append({
            "reason_id": item.get("reason_id"),
            "verdict": verdict,
            "points": points,
            "explanation": item.get("explanation", ""),
        })

    score = round(sum(r["points"] for r in per_reason) / len(per_reason))

    only_one = len(per_reason) == 1
    if only_one and per_reason[0]["verdict"] == "rejection_supported":
        score = min(score, WEAK_CAP)

    label = _label(score)
    message = WEAK_MESSAGE if label == "Weak" else ""

    return {
        "score": score,
        "label": label,
        "reasons": per_reason,
        "message": message,
    }