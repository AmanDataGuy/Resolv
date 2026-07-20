"""Resolv demo — the customer's view and the machinery, side by side.

Run it with:

    streamlit run app.py

WHAT YOU SEE. Two columns, and that split IS the project:

    LEFT  — what the customer sees: the agent's reply and the ticket they get back.
    RIGHT — what happened inside: every tool call and the policy verdict on each.

Nothing on the left can happen without a row on the right, because the harness checks every
action before it runs. So the two columns cannot disagree, and a refusal is visible as it
happens rather than inferred from the prose.

TYPE SCALE — five levels, each differing in SIZE, not merely weight, so the hierarchy survives a
squint. Bold-at-body-size was the earlier mistake: it reads as emphasis, not as structure.

    st.title      ~2.25rem   page identity, used exactly once
    st.subheader  ~1.50rem   the two column headers
    #### (h4)     ~1.25rem   labels inside a column
    body           1.00rem   content — replies, ticket text, reasons
    st.caption    ~0.875rem  meta only: the legend, the intake note

All five are native Streamlit — no custom CSS, no emoji, no decorative rules. Light background
and the stock sans font are pinned in .streamlit/config.toml so the page renders identically
regardless of the viewer's browser theme.

Colour does exactly one job: carry the policy verdict on an action. It is information, not
decoration, which is why it appears only on the trail rows and the ticket.

This runs the real pipeline (agents/loop.py -> harness/routing.py -> integrations/notify.py) at
temperature 0.0, so what you see here is what api/main.py would return for the same message.
"""
import asyncio
import uuid

import streamlit as st

from agents.loop import run_case
from harness import audit, routing
from integrations import notify

# `layout="wide"` gives the two columns room. Theme and font come from .streamlit/config.toml.
st.set_page_config(page_title="Resolv", layout="wide")

# Level 1 — page identity, once.
st.title("Resolv")
# Level 5 — one line of context, deliberately not a pitch.
st.caption("A refund agent that can only act inside a policy harness.")

# A messy, realistic complaint. It overstates the amount and mangles the order number, so the
# default run exercises the two rules worth watching: the lookup and the cap.
SAMPLE = (
    "hi so i ordered something like ord-1-0-0-0, it turned up way late and honestly ruined a "
    "gift. i paid like 900 dollars for it and i want a full refund for the trouble."
)

# Level 3 — the section label. The widget's own label is collapsed so the same words don't also
# appear at body size directly beneath it.
st.markdown("#### Customer message")
message = st.text_area("Customer message", SAMPLE, height=120, label_visibility="collapsed")
go = st.button("Resolve complaint")


def show_trail_row(record: dict) -> None:
    """Draw one line of the audit trail, coloured by what policy decided.

    Streamlit's built-in status boxes carry the colour, so there is no custom CSS: green means
    policy allowed it, red means it was refused, orange means it went to a human, and blue marks
    a read — looking an order up is not a decision, and colouring it like one would overstate it.
    """
    tool = record["tool"]

    # The amount is the point on a refund attempt, and noise on anything else.
    amount = ""
    if tool == "issue_refund":
        amount = f" — ${record['args'].get('amount_usd', 0):.2f}"

    line = f"**{tool}**{amount}  ·  `{record['rule_id']}`\n\n{record['reason']}"

    if tool == "lookup_order":
        st.info(line)
    elif record["action"] == "allow":
        st.success(line)
    elif record["action"] == "deny":
        st.error(line)
    else:
        st.warning(line)


# Only do work when the button is pressed and there's actually a message.
if go and message.strip():
    # A fresh case id per run, so each demo starts from an empty audit trail.
    case_id = f"demo-{uuid.uuid4().hex[:8]}"
    audit.clear(case_id)

    with st.spinner("Running the agent against the policy harness..."):
        # The real pipeline, the same three calls the API makes.
        result = asyncio.run(run_case(case_id, message, temperature=0.0))
        ticket = routing.route(case_id, result["trail"], result["claim"])
        notify.send(ticket)

    left, right = st.columns(2, gap="large")

    # ---- LEFT: what the customer experiences -------------------------------------------------
    with left:
        st.subheader("What the customer sees")                       # level 2

        st.markdown("#### Intake")                                   # level 3
        claim = result["claim"]
        st.write(                                                    # level 4
            f"Read as order {claim.get('order_id') or '(none)'}, "
            f"claim {claim.get('claim_type') or '(unclear)'}."
        )
        st.caption("A hint the agent must verify against the record, not trust.")   # level 5

        st.markdown("#### Agent reply")
        st.info(result["reply"])

        st.markdown("#### Ticket")
        summary = f"{ticket.ticket_id} · {ticket.outcome}"
        if ticket.team:
            summary += f" · routed to {ticket.team}"
        body = f"{summary}\n\n{ticket.customer_message}"
        # Same colour language as the trail, so an outcome reads identically on both sides.
        if ticket.outcome == "resolved":
            st.success(body)
        elif ticket.outcome == "escalated":
            st.warning(body)
        else:
            st.error(body)

    # ---- RIGHT: what actually happened underneath --------------------------------------------
    with right:
        st.subheader("What happened inside")                         # level 2

        st.markdown("#### Audit trail")                              # level 3
        for record in result["trail"]:
            show_trail_row(record)

        st.caption(                                                  # level 5 — legend, not slogan
            f"{len(result['trail'])} actions · "
            "green allowed · red denied · orange sent to a human · blue a lookup"
        )
