"""Resolv demo — the support-intake pipeline, live.

Run:  venv\\Scripts\\streamlit run app.py

A customer types a messy complaint. The extractor (LLM) turns it into a typed claim; the harness
then VERIFIES that claim against the real order record — deterministically, no LLM — and decides
the action. The whole point on screen: the model reads, the harness decides.
"""
import asyncio

import streamlit as st

from agents.extractor import extractor_agent
from agents.runner_utils import run_agent_once
from harness.validity import verify_claim
from schemas import CustomerClaim

st.set_page_config(page_title="Resolv", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stMarkdown, textarea, input, button {
        font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif !important;
    }
    :root { --ink:#1f2937; --accent:#4338ca; --muted:#6b7280; --line:#e5e7eb; --panel:#f8fafc; }
    .block-container { max-width: 820px; padding-top: 2.2rem; }
    .r-title { font-size: 2.1rem; font-weight: 700; color: var(--ink); letter-spacing: -0.02em; }
    .r-sub { color: var(--muted); font-size: 0.95rem; margin: 0.1rem 0 1.4rem; }
    .r-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
               color: var(--muted); margin: 1rem 0 0.3rem; }
    .r-card { border: 1px solid var(--line); border-radius: 10px; padding: 0.9rem 1.1rem; background: #fff;
              margin-bottom: 0.4rem; color: var(--ink); }
    .r-k { color: var(--muted); }
    .r-accent { color: var(--accent); font-weight: 700; }
    .stButton>button { background: var(--accent); color:#fff; border:none; border-radius:8px;
                       font-weight:600; padding:0.5rem 1.4rem; }
    .stButton>button:hover { background:#3730a3; color:#fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='r-title'>Resolv</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='r-sub'>Support intake that takes action — the LLM reads, the harness verifies and decides.</div>",
    unsafe_allow_html=True,
)

SAMPLE = (
    "hi so i ordered something like ord-1-0-0-7 or maybe ord-1-0-0-9 i dont know i have the email "
    "somewhere... it was supposed to arrive weeks ago and it turned up way late, i paid like 25 bucks. "
    "by the way my other order ord-5021 still hasnt been refunded either. can someone sort this out?"
)

st.markdown("<div class='r-label'>Customer message</div>", unsafe_allow_html=True)
msg = st.text_area("msg", SAMPLE, height=150, label_visibility="collapsed")
run = st.button("Process complaint")


def _decision(f) -> tuple[str, str]:
    if f.claim_true is True:
        amt = f" — approve credit of ${f.amount_usd:,.2f}" if f.amount_usd else ""
        return "auto_resolve", f"Claim verified{amt}."
    if f.claim_true is False:
        return "reject", "Record does not support the claim — send a polite explanation."
    return "need_info", "Cannot identify the order — ask the customer to confirm the number."


if run:
    with st.spinner("Extracting and verifying..."):
        raw = asyncio.run(run_agent_once(extractor_agent, msg, "customer_claim"))
        claim = CustomerClaim.model_validate(raw)
        finding = verify_claim(claim)
    action, reason = _decision(finding)

    st.markdown("<div class='r-label'>1 &nbsp;·&nbsp; Extracted claim (LLM)</div>", unsafe_allow_html=True)
    stated = f" &nbsp;&nbsp; <span class='r-k'>stated:</span> ~${claim.stated_amount_usd:,.0f}" if claim.stated_amount_usd else ""
    st.markdown(
        f"<div class='r-card'><span class='r-k'>order:</span> <b>{claim.order_id or '(none given)'}</b>"
        f" &nbsp;&nbsp; <span class='r-k'>type:</span> <b>{claim.claim_type}</b>{stated}"
        f"<br><span class='r-k' style='font-size:0.8rem'>pulled from a free-form message — order numbers may be "
        f"garbled, spelled out, or decoys</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='r-label'>2 &nbsp;·&nbsp; Verification against the record (deterministic)</div>",
        unsafe_allow_html=True,
    )
    verdict = {True: "TRUE", False: "FALSE", None: "UNVERIFIABLE"}[finding.claim_true]
    amt_line = ""
    if claim.stated_amount_usd and finding.amount_usd and abs(claim.stated_amount_usd - finding.amount_usd) > 0.5:
        amt_line = (
            f"<br><span class='r-k'>customer said</span> ~${claim.stated_amount_usd:,.0f}"
            f" &nbsp;·&nbsp; <span class='r-k'>record shows</span> <b>${finding.amount_usd:,.2f}</b>"
            f" &nbsp;<span class='r-accent'>&larr; the harness uses the real figure, not the customer's guess</span>"
        )
    st.markdown(
        f"<div class='r-card'><span class='r-k'>order found:</span> <b>{finding.order_found}</b>"
        f" &nbsp;&nbsp; <span class='r-k'>claim:</span> <span class='r-accent'>{verdict}</span>"
        f"<br><span class='r-k'>{finding.reason}</span>{amt_line}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='r-label'>3 &nbsp;·&nbsp; Decision (harness)</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='r-card'><span class='r-accent'>{action}</span><br>"
        f"<span class='r-k'>{reason}</span></div>",
        unsafe_allow_html=True,
    )
