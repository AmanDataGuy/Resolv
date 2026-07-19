"""Resolv demo — the customer's view and the machinery, side by side.

Run it with:

    streamlit run app.py

WHAT YOU SEE. The screen is split into two columns, and that split IS the whole idea of the
project:

    LEFT  — "What the customer sees": the agent's reply and the ticket/email they get back.
    RIGHT — "What happened inside": every tool the agent called and the policy verdict on each.

The customer only ever sees polite prose. The right column is the truth underneath it — and
nothing on the left can happen without a row on the right, because the harness checks every
action before it runs. So the two columns can never disagree, and you can watch the agent get
told "no" by policy in real time.

STYLE. Deliberately plain: Streamlit's default light theme and font, native components, no custom
CSS. The point is the flow, not the decoration.

This runs the real pipeline (agents/loop.py -> harness/routing.py -> integrations/notify.py) at
temperature 0.0, so what you see here is exactly what the API in api/main.py would return.
"""
import asyncio
import uuid

import streamlit as st

from agents.loop import run_case
from harness import audit, routing
from integrations import notify

# `layout="wide"` gives the two columns room to breathe. Everything else is Streamlit's default
# light theme and standard sans-serif font — no styling of our own.
st.set_page_config(page_title="Resolv", layout="wide")

st.title("Resolv")
st.caption(
    "A refund agent that can only act inside a policy harness. "
    "Left: what the customer sees. Right: what policy actually enforced."
)

# A messy, realistic complaint to start with. It lies about the amount ($900) and mangles the
# order number — exactly the kind of input the harness is built to handle safely.
SAMPLE = (
    "hi so i ordered something like ord-1-0-0-0, it turned up way late and honestly ruined a "
    "gift. i paid like 900 dollars for it and i want a full refund for the trouble."
)

message = st.text_area("Customer message", SAMPLE, height=120)
go = st.button("Resolve complaint")


def show_trail_row(record: dict) -> None:
    """Draw one line of the audit trail, colour-coded by what policy decided.

    We reuse Streamlit's built-in coloured boxes instead of custom CSS: green = allowed,
    red = denied, orange = sent to a human, blue = a read (looking an order up isn't a decision).
    """
    tool = record["tool"]
    rule = record["rule_id"]
    reason = record["reason"]

    # A refund attempt is worth showing the amount for; other tools aren't.
    amount = ""
    if tool == "issue_refund":
        amount = f" — ${record['args'].get('amount_usd', 0):.2f}"

    line = f"**{tool}**{amount}  ·  `{rule}`\n\n{reason}"

    if tool == "lookup_order":
        st.info(line)                       # a read — neutral, not a decision
    elif record["action"] == "allow":
        st.success(line)                    # policy let it through
    elif record["action"] == "deny":
        st.error(line)                      # policy refused it
    else:
        st.warning(line)                    # policy sent it to a human (escalate)


# Only do work when the button is pressed and there's actually a message.
if go and message.strip():
    # A fresh case id per run, so each demo starts from an empty audit trail.
    case_id = f"demo-{uuid.uuid4().hex[:8]}"
    audit.clear(case_id)

    with st.spinner("Running the agent against the policy harness..."):
        # The real pipeline, same three calls the API makes.
        result = asyncio.run(run_case(case_id, message, temperature=0.0))
        ticket = routing.route(case_id, result["trail"], result["claim"])
        notify.send(ticket)

    left, right = st.columns(2, gap="large")

    # ---- LEFT: what the customer experiences -------------------------------------------------
    with left:
        st.subheader("What the customer sees")

        # What intake THOUGHT the message meant. It's only a hint — the agent still has to look
        # the order up and verify it, which is the whole reason a wrong guess here is harmless.
        claim = result["claim"]
        st.write(
            f"Intake read this as: order **{claim.get('order_id') or '(none)'}**, "
            f"claim **{claim.get('claim_type') or '(unclear)'}** — a hint the agent must verify, "
            "not trust."
        )

        st.markdown("**Agent reply**")
        st.info(result["reply"])

        st.markdown("**Ticket / email sent to the customer**")
        # Colour the ticket the same way as a decision: resolved = green, escalated = orange,
        # denied = red.
        summary = f"{ticket.ticket_id} · {ticket.outcome}"
        if ticket.team:
            summary += f" · routed to the {ticket.team} team"
        body = f"{summary}\n\n{ticket.customer_message}"
        if ticket.outcome == "resolved":
            st.success(body)
        elif ticket.outcome == "escalated":
            st.warning(body)
        else:
            st.error(body)

    # ---- RIGHT: what actually happened underneath --------------------------------------------
    with right:
        st.subheader("What happened inside")
        st.caption("Every action the agent took, and the policy verdict on each.")
        for record in result["trail"]:
            show_trail_row(record)
        st.caption(
            "Nothing on the left happened without a row here. The reply is prose; the trail is "
            "the truth."
        )
