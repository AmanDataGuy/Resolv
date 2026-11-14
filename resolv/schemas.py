"""
# resolv/schemas.py

**What:** the shared Pydantic models: CustomerClaim (what the extractor reads), ClaimFinding (the harness's
verdict), Ticket (the hand-off at the end of a case), and the ClaimType / Team vocabularies.
**Why:** the types encode the split between model and harness. The model's output carries only what a customer
actually said (which order, what kind of problem); amounts, dates and verdicts live on ClaimFinding, which only
the harness builds. One module, so anything can import a schema without circular imports.
**Used by:** resolv/agents/extractor.py, resolv/harness/validity.py, resolv/harness/policy.py, resolv/harness/routing.py, resolv/integrations/notify.py.
"""
from typing import Literal

from pydantic import BaseModel

# The three claim types, each backed by a real Olist order status: delivered late, shipped but never delivered,
# canceled. item_unavailable was dropped: Olist has only 6 such orders, too few to train on or measure.
ClaimType = Literal["late_delivery", "never_arrived", "order_canceled"]


# What the extractor pulls from a message. Both fields are optional: None is the honest answer when the customer
# gave no order number or stated no problem yet (a strict provider schema rejects null otherwise).
# stated_amount_usd is kept only to cross-check against the record, never trusted.
class CustomerClaim(BaseModel):
    order_id: str | None = None
    claim_type: ClaimType | None = None
    stated_amount_usd: float | None = None


# The deterministic verdict on a claim. claim_true None means "can't judge" (order not found), not "false".
class ClaimFinding(BaseModel):
    order_found: bool
    claim_true: bool | None
    reason: str
    order_id: str | None = None
    amount_usd: float | None = None


# Which human queue an escalation lands in; "general" is where a case goes when the trail can't route it better.
Team = Literal["billing", "logistics", "general"]


# The record handed off at the end of a case, derived from the audit trail by resolv/harness/routing.py.
# outcome: resolved (refunded, closed) | escalated (open on `team`) | denied (nothing owed, closed).
class Ticket(BaseModel):
    ticket_id: str
    case_id: str
    order_id: str | None
    outcome: Literal["resolved", "escalated", "denied"]
    team: Team | None  # set only when outcome == "escalated"
    status: Literal["open", "closed"]
    customer_message: str
    escalation_id: str | None = None  # the ledger escalation row a human works from, when there is one
