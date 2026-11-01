"""ClaimCourt Streamlit UI. Thin layer: all logic lives in pipeline/."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from pipeline.appeal import AppealError, generate_appeal
from pipeline.extract import ExtractError, extract_pdf
from pipeline.llm import LLMError
from pipeline.match import judge_reasons
from pipeline.reasons import extract_reasons
from pipeline.score import score_claim

ROOT = Path(__file__).resolve().parent
CASES_DIR = ROOT / "data" / "cases"
LABEL_COLOURS = {"Strong": "green", "Medium": "orange", "Weak": "red"}
BADGE = {
    "rejection_contradicted": ("Contradicted by policy", "green"),
    "unclear": ("Unclear", "orange"),
    "rejection_supported": ("Supported by policy", "red"),
}

st.set_page_config(page_title="ClaimCourt", layout="wide")


def demo_cases() -> dict[str, tuple[Path, Path]]:
    """Pair case_X_policy.pdf with case_X_letter.pdf."""
    cases = {}
    for letter in sorted(CASES_DIR.glob("*_letter.pdf")):
        policy = letter.with_name(letter.name.replace("_letter.pdf", "_policy.pdf"))
        if policy.exists():
            cases[letter.name.replace("_letter.pdf", "").replace("_", " ").title()] = (policy, letter)
    return cases


def run_pipeline(policy_bytes: bytes, letter_bytes: bytes, details: dict[str, str]) -> dict:
    """Run every stage. Returns the state dict stored in session_state."""
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
        appeal = generate_appeal(reasons, verdicts, details)

        status.update(label="Done", state="complete", expanded=False)

    return {
        "claim_type": parsed["claim_type"],
        "reasons": reasons,
        "verdicts": verdicts,
        "score": score,
        "appeal": appeal,
    }


def render_results(state: dict) -> None:
    score = state["score"]
    colour = LABEL_COLOURS[score["label"]]

    st.subheader("Appeal strength")
    st.markdown(f"### :{colour}[{score['label']}]  ·  {score['score']} / 100")
    if state["claim_type"]:
        st.caption(f"Claim type: {state['claim_type']}")
    for line in score["reasons"]:
        st.write(f"- Reason {line['reason_id']}: {line['explanation'] or line['verdict']}")
    if score["message"]:
        st.warning(score["message"])

    st.subheader("Reasons and evidence")
    reasons_by_id = {r["reason_id"]: r for r in state["reasons"]}
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
                st.caption(reason["plain_english"])
            with right:
                st.markdown("**Policy wording**")
                if v.get("policy_quote"):
                    st.write(f"\u201c{v['policy_quote']}\u201d")
                    st.caption(f"Page {v.get('quote_page') or v.get('page') or '?'}")
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

    st.subheader("Next steps")
    appeal = state["appeal"]
    if appeal["evidence_count"] == 0:
        st.info("No verified evidence supports an appeal. Follow the steps below to escalate if needed.")
    st.download_button(
        "Download appeal letter (PDF)",
        data=appeal["pdf_bytes"],
        file_name="claimcourt_appeal.pdf",
        mime="application/pdf",
    )
    with st.expander("Appeal letter text"):
        st.text(appeal["letter"])
    for step in appeal["checklist"]:
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
            if choice != "(none)":
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