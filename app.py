"""Minimal Streamlit demo for Resolv.

Run:  venv\\Scripts\\streamlit run app.py

Paste a raw exception event, pick which model drafts the email (Groq 70B or the local
fine-tuned GRPO adapter), and watch the full pipeline run: detection, revenue math, SLA
clause + retrieved contract text, the drafted email, the deterministic guardrail
verification, and the auto/human/escalate decision.
"""
import asyncio
import json
import os

import streamlit as st

from agents.orchestrator import process_exception

st.set_page_config(page_title="Resolv", layout="centered")

# --- styling: one accent (indigo), ink text, neutral greys. No emojis. ---
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
    .r-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em;
               text-transform: uppercase; color: var(--muted); margin-bottom: 0.3rem; }
    .r-card { border: 1px solid var(--line); border-radius: 10px; padding: 1rem 1.2rem;
              background: #fff; margin-bottom: 0.9rem; }
    .r-email { border: 1px solid var(--line); border-left: 3px solid var(--accent);
               border-radius: 10px; padding: 1rem 1.2rem; background: var(--panel);
               white-space: pre-wrap; color: var(--ink); line-height: 1.5; }
    .r-metric-v { font-size: 1.35rem; font-weight: 700; color: var(--ink); }
    .r-metric-l { font-size: 0.72rem; color: var(--muted); text-transform: uppercase;
                  letter-spacing: 0.05em; }
    .r-action { font-weight: 700; color: var(--accent); }
    .r-pass { color: var(--muted); font-weight: 600; }
    .r-fail { color: var(--accent); font-weight: 700; }
    .stButton>button { background: var(--accent); color: #fff; border: none; border-radius: 8px;
                       font-weight: 600; padding: 0.5rem 1.4rem; }
    .stButton>button:hover { background: #3730a3; color: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='r-title'>Resolv</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='r-sub'>Autonomous supply-chain exception manager — the LLM proposes, the harness disposes.</div>",
    unsafe_allow_html=True,
)

DEFAULT_EVENT = {
    "id": "exc-acme-1001",
    "source": "tms-webhook",
    "supplier_id": "acme-logistics",
    "order_ids": ["ORD-1001", "ORD-1002"],
    "raw_text": (
        "Shipment for orders ORD-1001 and ORD-1002 from acme-logistics is 6 days past "
        "the estimated delivery date. Multiple customer escalations received."
    ),
}

st.markdown("<div class='r-label'>Raw exception event</div>", unsafe_allow_html=True)
raw = st.text_area("event", json.dumps(DEFAULT_EVENT, indent=2), height=210, label_visibility="collapsed")

backend = st.radio(
    "Drafter model",
    ["Groq llama-3.3-70b", "Fine-tuned Qwen2.5-1.5B (RLVR)"],
    horizontal=True,
)
run = st.button("Process exception")


def _check_line(label, ok):
    cls = "r-pass" if ok else "r-fail"
    word = "pass" if ok else "fail"
    return f"<div>{label}: <span class='{cls}'>{word}</span></div>"


if run:
    os.environ["DRAFTER_BACKEND"] = "finetuned" if "Fine-tuned" in backend else "groq"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()

    with st.spinner("Running the pipeline..."):
        record = asyncio.run(process_exception(payload))

    impact = record["impact"]
    sla = record["sla_findings"]
    draft = record["draft"]
    decision = record["decision"]
    checks = record["draft_verification"]["checks"]

    # key facts
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f"<div class='r-metric-l'>Exception</div><div class='r-metric-v'>{record['type']}</div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"<div class='r-metric-l'>Revenue at risk</div><div class='r-metric-v'>${impact['revenue_at_risk_usd']:,.0f}</div>",
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"<div class='r-metric-l'>Urgency</div><div class='r-metric-v'>{record['urgency']}</div>",
        unsafe_allow_html=True,
    )

    # SLA
    clause = sla.get("clause_id") or "none on file"
    clause_text = sla.get("clause_text") or "(no contract text retrieved above the confidence threshold)"
    st.markdown("<div class='r-label' style='margin-top:1.2rem'>SLA finding</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='r-card'><b>Clause {clause}</b> &nbsp;·&nbsp; penalty ${sla['penalty_usd']:,.0f}"
        f"<br><span style='color:#6b7280'>{clause_text}</span></div>",
        unsafe_allow_html=True,
    )

    # drafted email
    st.markdown("<div class='r-label'>Drafted email</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='r-email'><b>Subject:</b> {draft['subject']}\n\n{draft['body']}</div>",
        unsafe_allow_html=True,
    )

    # guardrail verification + decision
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("<div class='r-label'>Guardrail verification</div>", unsafe_allow_html=True)
        rows = "".join(_check_line(k.replace("_", " "), ok) for k, ok in checks.items())
        st.markdown(f"<div class='r-card'>{rows}</div>", unsafe_allow_html=True)
    with v2:
        st.markdown("<div class='r-label'>Decision</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='r-card'><span class='r-action'>{decision['action']}</span>"
            f" &nbsp;(confidence {decision['confidence']})"
            f"<br><span style='color:#6b7280'>{decision['reason']}</span></div>",
            unsafe_allow_html=True,
        )
