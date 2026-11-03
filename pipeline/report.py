"""Evidence report PDF: a printable, court-style summary of every verified check.

Separate from the appeal letter. Intended as the document a consumer-rights volunteer
reads before meeting the policyholder. Contains no claims that are not in the verdicts
already; this module only formats them.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fpdf import FPDF

FOOTER = "ClaimCourt evidence report. Decision support, not legal advice."


def _sanitize(text: str) -> str:
    table = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u20b9": "Rs ", "\u2022": "-",
    }
    for src, dst in table.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


def _yes_no(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if value is None:
        return "n/a"
    return str(value)


def _build_lines(state: dict[str, Any]) -> list[tuple[str, str]]:
    """Return a list of (style, text) where style is 'h1', 'h2', 'body', 'kv'.

    'kv' lines look like 'key: value' and are rendered slightly indented.
    """
    lines: list[tuple[str, str]] = []
    score = state.get("score", {})
    status = score.get("status", "Insufficient information")
    claim_type = state.get("claim_type", "")
    details = state.get("details", {}) or {}

    lines.append(("h1", "ClaimCourt - Evidence Report"))
    lines.append(("body", f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"))
    lines.append(("body", f"Overall assessment: {status}"))
    if claim_type:
        lines.append(("body", f"Claim type: {claim_type}"))
    for k, label in (
        ("insured_name", "Insured name"),
        ("policy_number", "Policy number"),
        ("claim_number", "Claim number"),
        ("insurer_name", "Insurer"),
    ):
        v = details.get(k) or "(not provided)"
        lines.append(("kv", f"{label}: {v}"))
    if score.get("message"):
        lines.append(("body", f"Note: {score['message']}"))

    reasons_by_id = {r["id"]: r for r in state.get("reasons", [])}
    for idx, v in enumerate(state.get("verdicts", []), start=1):
        reason = reasons_by_id.get(v.get("reason_id"))
        if reason is None:
            continue
        lines.append(("h2", f"Reason {idx} - id={reason.get('id')} ({reason.get('category')})"))
        lines.append(("kv", f"Insurer wording: {reason.get('insurer_words', '')!r}"))
        lines.append(("kv", f"Insurer wording verified: {_yes_no(reason.get('insurer_words_verified'))}"))
        lines.append(("kv", f"Plain meaning: {reason.get('plain_english', '')}"))
        lines.append(("kv", f"Verdict: {v.get('verdict')}"))
        lines.append(("kv", f"Policy quote: {v.get('policy_quote', '')!r}"))
        lines.append(("kv", f"Quote verified: {_yes_no(v.get('quote_verified'))}"))
        lines.append(("kv", f"Quote check: {v.get('quote_check')}"))
        lines.append(("kv", f"Quote page: {v.get('quote_page')}"))
        lines.append(("kv", f"Clause: {v.get('clause')!r}"))
        lines.append(("kv", f"Clause verified near quote: {_yes_no(v.get('clause_verified'))}"))
        lines.append(("kv", f"Clause check: {v.get('clause_check')}"))
        if v.get("explanation"):
            lines.append(("body", f"Explanation: {v['explanation']}"))
        facts = v.get("facts_needed_from_user") or []
        if facts:
            lines.append(("body", "Facts needed from the policyholder:"))
            for f in facts:
                lines.append(("kv", f"- {f}"))
        lines.append(("body", ""))

    appeal = state.get("appeal", {}) or {}
    has_appeal = bool(appeal.get("pdf_bytes")) and bool(appeal.get("letter"))
    lines.append(("h2", "Appeal"))
    if has_appeal:
        lines.append(("kv", f"Evidence items used: {appeal.get('evidence_count', 0)}"))
        lines.append(("kv", f"Insurer quotation check: {appeal.get('insurer_quote_check', 'n/a')}"))
        lines.append(("body", "The appeal letter PDF is provided separately."))
    else:
        lines.append(("kv", "No verified evidence supports an appeal."))
    lines.append(("body", ""))
    lines.append(("body", "This report lists only what the app checked. It is not legal advice."))
    return lines


def build_report_pdf(state: dict[str, Any]) -> bytes:
    """Return PDF bytes. Never raises for content issues; only for fpdf2-level failures."""
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(left=20, top=20, right=20)

    def footer() -> None:
        pdf.set_y(-15)
        pdf.set_font("Helvetica", size=8)
        pdf.cell(w=0, h=5, text=FOOTER, align="C")

    pdf.footer = footer
    pdf.add_page()

    for style, text in _build_lines(state):
        text = _sanitize(text)
        if style == "h1":
            pdf.set_font("Helvetica", "B", 15)
            pdf.multi_cell(w=0, h=8, text=text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif style == "h2":
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(w=0, h=7, text=text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif style == "kv":
            pdf.set_font("Helvetica", size=10)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(w=0, h=5, text=text, new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("Helvetica", size=10)
            pdf.multi_cell(w=0, h=5, text=text, new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())