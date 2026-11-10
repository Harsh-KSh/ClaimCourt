"""ClaimCourt Streamlit UI. Thin layer: all logic lives in pipeline/."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from pipeline.appeal import AppealError, generate_appeal
from pipeline.extract import ExtractError, extract_pdf
from pipeline.llm import LLMError
from pipeline.match import judge_reasons
from pipeline.reasons import extract_reasons
from pipeline.report import build_report_pdf
from pipeline.score import score_claim

ROOT = Path(__file__).resolve().parent
CASES_DIR = ROOT / "data" / "cases"
STATUS_COLOURS = {
    "Not supported by policy": "green",
    "Supported by policy": "red",
    "Insufficient information": "orange",
}
BADGE = {
    "rejection_contradicted": ("Contradicted by policy", "green"),
    "unclear": ("Unclear", "orange"),
    "rejection_supported": ("Supported by policy", "red"),
}
EMPTY_APPEAL = {"letter": "", "pdf_bytes": b"", "checklist": [], "evidence_count": 0}

st.set_page_config(page_title="ClaimCourt", layout="wide")


def demo_cases() -> dict[str, tuple[Path, Path]]:
    cases = {}
    for letter in sorted(CASES_DIR.glob("*_letter.pdf")):
        policy = letter.with_name(letter.name.replace("_letter.pdf", "_policy.pdf"))
        if policy.exists():
            cases[letter.name.replace("_letter.pdf", "").replace("_", " ").title()] = (policy, letter)
    return cases


def _tick(ok: bool, yes: str, no: str) -> str:
    return f":green[✓ {yes}]" if ok else f":red[✗ {no}]"


def run_pipeline(policy_bytes: bytes, letter_bytes: bytes, details: dict[str, str]) -> dict:
    with st.status("Working...", expanded=True) as status:
        status.write("Reading documents...")
        policy = extract_pdf(policy_bytes, name="policy.pdf")
        letter = extract_pdf(letter_bytes, name="letter.pdf")

        status.write("Finding rejection reasons...")
        parsed = extract_reasons(letter.full_text)
        reasons = parsed["reasons"]
        if not reasons:
            raise LLMError("No rejection reasons were found in the letter.")

        status.write("Checking policy and verifying quotes...")
        verdicts = judge_reasons(reasons, policy.pages)

        status.write("Scoring and writing appeal...")
        score = score_claim(verdicts)
        try:
            appeal = generate_appeal(reasons, verdicts, details, letter_text=letter.full_text)
        except AppealError:
            appeal = dict(EMPTY_APPEAL)

        status.update(label="Done", state="complete", expanded=False)

    return {
        "claim_type": parsed["claim_type"],
        "reasons": reasons,
        "verdicts": verdicts,
        "score": score,
        "appeal": appeal,
        "details": details,
        "letter_pages": [{"number": p.number, "text": p.text} for p in letter.pages],
        "policy_pages": [{"number": p.number, "text": p.text} for p in policy.pages],
    }


def _render_trust_panel(reason: dict, verdict: dict) -> None:
    insurer_ok = bool(reason.get("insurer_words_verified"))
    quote_ok = bool(verdict.get("quote_verified"))
    clause_claimed = verdict.get("clause")
    clause_ok = bool(verdict.get("clause_verified")) if clause_claimed else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Insurer quote**")
        st.markdown(_tick(insurer_ok, "verbatim in letter", "not verified"))
    with c2:
        st.markdown("**Policy quote**")
        st.markdown(_tick(quote_ok, "verbatim in policy", "not verified"))
    with c3:
        st.markdown("**Clause**")
        if clause_claimed is None:
            st.markdown(":gray[— none claimed]")
        else:
            st.markdown(_tick(clause_ok, "verified near quote", "not near quote"))
    with c4:
        st.markdown("**IRDAI rule**")
        st.markdown(":gray[— not in scope]")


def _render_audit_trail(reason: dict, verdict: dict, policy_pages: list[dict]) -> None:
    lines = []
    lines.append(f"Extracted reason id={reason.get('id')} category={reason.get('category')}")
    lines.append(f"  insurer_words: {reason.get('insurer_words', '')!r}")
    lines.append(f"  insurer_words_verified: {reason.get('insurer_words_verified')}")

    q = verdict.get("policy_quote") or ""
    lines.append(f"policy_quote: {q!r}")
    lines.append(f"  quote_verified: {verdict.get('quote_verified')}")
    lines.append(f"  quote_check: {verdict.get('quote_check')}")
    lines.append(f"  quote_page: {verdict.get('quote_page')}")
    lines.append(f"  quote_position: {verdict.get('quote_position')}")

    lines.append(f"clause: {verdict.get('clause')!r}")
    lines.append(f"  clause_verified: {verdict.get('clause_verified')}")
    lines.append(f"  clause_check: {verdict.get('clause_check')}")

    lines.append(f"raw AI verdict: {verdict.get('verdict')}")
    lines.append(f"explanation: {verdict.get('explanation', '')}")

    facts = verdict.get("facts_needed_from_user") or []
    if facts:
        lines.append("facts_needed_from_user:")
        for f in facts:
            lines.append(f"  - {f}")

    st.code("\n".join(lines), language="text")


def render_results(state: dict) -> None:
    score = state["score"]
    status = score["status"]
    colour = STATUS_COLOURS.get(status, "orange")

    st.subheader("Assessment")
    st.markdown(f"### :{colour}[{status}]")
    if state["claim_type"]:
        st.caption(f"Claim type: {state['claim_type']}")
    for line in score["reasons"]:
        st.write(f"- Reason {line['reason_id']}: {line['explanation'] or line['verdict']}")
    if score.get("message"):
        st.warning(score["message"])

    st.subheader("Reasons and evidence")
    policy_pages = state.get("policy_pages", [])
    for v in state["verdicts"]:
        reason = next((r for r in state["reasons"] if r["id"] == v["reason_id"]), None)
        if reason is None:
            continue
        label, badge_colour = BADGE.get(v["verdict"], BADGE["unclear"])
        with st.container(border=True):
            left, right = st.columns(2)
            with left:
                st.markdown("**What the insurer said**")
                st.write(reason["insurer_words"] or "(no direct quote)")
                if not reason.get("insurer_words_verified") and reason["insurer_words"]:
                    st.caption(":red[Insurer wording could not be verified against the letter.]")
                st.caption(reason["plain_english"])
            with right:
                st.markdown("**Policy wording**")
                if v.get("policy_quote"):
                    st.write(f"\u201c{v['policy_quote']}\u201d")
                    page = v.get("quote_page") or v.get("page") or "?"
                    clause = v.get("clause")
                    if clause and v.get("clause_verified"):
                        st.caption(f"Page {page} · Clause {clause}")
                    elif clause:
                        st.caption(f"Page {page} · Clause claimed: {clause} (not verified)")
                    else:
                        st.caption(f"Page {page}")
                else:
                    st.write("No policy wording found.")
                st.markdown(f":{badge_colour}[{label}]")
                if v.get("quote_verified"):
                    st.markdown(":green[Verified quote]")
                else:
                    st.markdown(
                        ":red[Quote could not be verified, removed from the appeal.]"
                        if v.get("policy_quote") else ":gray[No quote to verify]"
                    )
            if v.get("facts_needed_from_user"):
                st.info("Needed to decide: " + "; ".join(v["facts_needed_from_user"]))

            st.markdown("**Trust checks**")
            _render_trust_panel(reason, v)

            with st.expander("How we reached this"):
                _render_audit_trail(reason, v, policy_pages)

    st.subheader("Next steps")
    appeal = state["appeal"]
    has_appeal = bool(appeal.get("pdf_bytes")) and bool(appeal.get("letter"))

    try:
        report_bytes = build_report_pdf(state)
    except Exception:
        report_bytes = b""

    if not has_appeal:
        st.info("No verified evidence supports an appeal. The assessment above still stands.")
    else:
        if appeal.get("insurer_quote_check") == "not_found_removed":
            st.warning(
                "The AI's quotation of the insurer's wording did not match the rejection letter, "
                "so that block was removed from the appeal letter."
            )
        st.download_button(
            "Download appeal letter (PDF)",
            data=appeal["pdf_bytes"],
            file_name="claimcourt_appeal.pdf",
            mime="application/pdf",
        )
        with st.expander("Appeal letter text"):
            st.text(appeal["letter"])

    if report_bytes:
        st.download_button(
            "Download evidence report (PDF)",
            data=report_bytes,
            file_name="claimcourt_evidence_report.pdf",
            mime="application/pdf",
        )
        st.caption("Full check-by-check receipt for the assessment above. Not legal advice.")

    for step in appeal.get("checklist", []):
        st.checkbox(step, key=f"step_{hash(step)}")


def main() -> None:
    st.title("ClaimCourt")
    st.caption(
        "Decision-support tool. Not legal advice. Demo uses synthetic documents. "
        "Upload text-based PDFs only."
    )

    cases = demo_cases()
    with st.container(border=True):
        st.markdown("**Load a demo case** (works offline if cached)")
        choice = st.selectbox("Demo case", ["(none)"] + list(cases.keys()))

        c1, c2 = st.columns(2)
        policy_file = c1.file_uploader("Insurance policy (PDF)", type="pdf")
        letter_file = c2.file_uploader("Rejection letter (PDF)", type="pdf")

        with st.expander("Optional details for the letter"):
            insured_name = st.text_input("Your name")
            policy_number = st.text_input("Policy number")
            claim_number = st.text_input("Claim number")
            insurer_name = st.text_input("Insurer name")

        analyse = st.button("Analyse claim", type="primary")

    if analyse:
        try:
            if choice != "(none)" and not (policy_file and letter_file):
                policy_path, letter_path = cases[choice]
                policy_bytes, letter_bytes = policy_path.read_bytes(), letter_path.read_bytes()
            elif policy_file and letter_file:
                policy_bytes, letter_bytes = policy_file.getvalue(), letter_file.getvalue()
            else:
                st.error("Upload both PDFs, or pick a demo case.")
                return
            details = {
                "insured_name": insured_name,
                "policy_number": policy_number,
                "claim_number": claim_number,
                "insurer_name": insurer_name,
            }
            st.session_state["state"] = run_pipeline(policy_bytes, letter_bytes, details)
        except ExtractError as exc:
            st.session_state.pop("state", None)
            st.error(str(exc))
        except (LLMError, AppealError) as exc:
            st.session_state.pop("state", None)
            st.error(str(exc))

    if "state" in st.session_state:
        render_results(st.session_state["state"])


main()