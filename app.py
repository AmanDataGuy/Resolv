"""Resolv demo — the agent and the harness, side by side.

Run:  venv\\Scripts\\streamlit run app.py

A customer types a messy complaint. The LEFT panel is what the customer experiences: the agent's
reply and the email it sends. The RIGHT panel is what actually happened underneath: every tool
call the model proposed, and the policy verdict the harness returned on each one. The split is
the whole thesis on one screen — the model proposes, the harness disposes, and the two are never
the same column.

This runs the real pipeline (agents/loop.py -> harness/routing.py -> integrations/notify.py) at
temperature 0.0, so what you see is what the API would do.
"""
import asyncio
import uuid

import streamlit as st

from agents.loop import run_case
from harness import audit, routing
from integrations import notify

st.set_page_config(page_title="Resolv", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stMarkdown, textarea, input, button {
        font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif !important;
    }
    :root { --ink:#1f2937; --accent:#4338ca; --muted:#6b7280; --line:#e5e7eb;
            --ok:#047857; --deny:#b91c1c; --esc:#b45309; }
    .block-container { max-width: 1150px; padding-top: 2rem; }
    .r-title { font-size: 2.1rem; font-weight: 700; color: var(--ink); letter-spacing: -0.02em; }
    .r-sub { color: var(--muted); font-size: 0.95rem; margin: 0.1rem 0 1.2rem; }
    .r-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
               color: var(--muted); margin: 1rem 0 0.4rem; }
    .r-card { border: 1px solid var(--line); border-radius: 10px; padding: 0.8rem 1rem; background: #fff;
              margin-bottom: 0.5rem; color: var(--ink); font-size: 0.9rem; }
    .r-k { color: var(--muted); }
    .r-badge { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
               padding: 0.1rem 0.5rem; border-radius: 6px; color: #fff; }
    .b-allow { background: var(--ok); } .b-deny { background: var(--deny); } .b-esc { background: var(--esc); }
    .b-read { background: var(--muted); }
    .r-reply { border-left: 3px solid var(--accent); padding: 0.6rem 0.9rem; background: #f8fafc;
               border-radius: 0 8px 8px 0; color: var(--ink); }
    .r-tid { font-weight: 700; color: var(--accent); }
    .stButton>button { background: var(--accent); color:#fff; border:none; border-radius:8px;
                       font-weight:600; padding:0.5rem 1.4rem; }
    .stButton>button:hover { background:#3730a3; color:#fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='r-title'>Resolv</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='r-sub'>The model proposes, the harness disposes. Left: what the customer sees. "
    "Right: what policy actually enforced.</div>",
    unsafe_allow_html=True,
)

SAMPLE = (
    "hi so i ordered something like ord-1-0-0-0, it turned up way late and honestly ruined a gift. "
    "i paid like 900 dollars for it and i want a full refund for the trouble. this is unacceptable."
)

msg = st.text_area("Customer message", SAMPLE, height=120)
run = st.button("Resolve complaint")

_ACTION_BADGE = {"allow": "b-allow", "deny": "b-deny", "escalate": "b-esc"}


def _badge(rec: dict) -> str:
    # Reads (lookup_order) carry a synthetic allow but aren't a decision — show them as neutral.
    if rec["tool"] == "lookup_order":
        return "b-read"
    return _ACTION_BADGE.get(rec["action"], "b-read")


if run and msg.strip():
    case_id = f"demo-{uuid.uuid4().hex[:8]}"
    audit.clear(case_id)
    with st.spinner("Running the agent against the policy harness..."):
        result = asyncio.run(run_case(case_id, msg, temperature=0.0))
        ticket = routing.route(case_id, result["trail"], result["claim"])
        notify.send(ticket)

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("<div class='r-label'>What the customer sees</div>", unsafe_allow_html=True)
        claim = result["claim"]
        st.markdown(
            f"<div class='r-card'><span class='r-k'>intake read:</span> order "
            f"<b>{claim.get('order_id') or '(none)'}</b>, claim <b>{claim.get('claim_type') or '(unclear)'}</b>"
            f"<br><span class='r-k' style='font-size:0.8rem'>a hint the agent must verify, not trust</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='r-label'>Agent reply</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='r-reply'>{result['reply']}</div>", unsafe_allow_html=True)

        st.markdown("<div class='r-label'>Email sent to customer</div>", unsafe_allow_html=True)
        badge = {"resolved": "b-allow", "escalated": "b-esc", "denied": "b-deny"}[ticket.outcome]
        team = f" &nbsp;→&nbsp; <b>{ticket.team}</b> team" if ticket.team else ""
        st.markdown(
            f"<div class='r-card'><span class='r-tid'>{ticket.ticket_id}</span> "
            f"<span class='r-badge {badge}'>{ticket.outcome}</span>{team}"
            f"<br><br>{ticket.customer_message}</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("<div class='r-label'>Audit trail — every action, policy-checked</div>", unsafe_allow_html=True)
        for rec in result["trail"]:
            args = ""
            if rec["tool"] == "issue_refund":
                args = f" &nbsp;<span class='r-k'>${rec['args'].get('amount_usd', 0):.2f}</span>"
            st.markdown(
                f"<div class='r-card'><b>{rec['tool']}</b>{args} "
                f"<span class='r-badge {_badge(rec)}'>{rec['rule_id']}</span>"
                f"<br><span class='r-k'>{rec['reason']}</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<div class='r-sub' style='margin-top:0.6rem'>Nothing on the left happened without a "
            "row on the right. That's the point — the reply is prose, the trail is the truth.</div>",
            unsafe_allow_html=True,
        )