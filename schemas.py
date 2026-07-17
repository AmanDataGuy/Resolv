"""Shared Pydantic models — the typed contract between the model and the harness.

Centralized in one module so the agent, the harness, and the tests can import any schema
without circular imports. These types ARE the interface: the extractor emits CustomerClaim via
ADK's output_schema, the deterministic harness answers with ClaimFinding, and every boundary
between them is validated here — a malformed hand-off fails loudly at the seam instead of
silently downstream.

THE SPLIT THESE TYPES ENCODE. The model's output type carries only what a model can honestly
know: which order, what kind of problem — things a customer actually said. It carries no dollar
amounts, no real dates, no verdicts. Those live on ClaimFinding, which only the harness ever
constructs. The boundary isn't a convention someone has to remember; it's the type system.
"""
from typing import Literal

from pydantic import BaseModel

# The three claim types, each backed by a real Olist order_status (nothing invented):
#   late_delivery   — delivered after the promised date (or the customer believes so)
#   never_arrived   — shipped but never delivered
#   order_canceled  — the order was canceled
# A fourth, item_unavailable, was dropped: Olist has only 6 such orders — too few to train on
# and too few to measure. See SAMPLES in scripts/gen_complaint_cases.py.
ClaimType = Literal["late_delivery", "never_arrived", "order_canceled"]


class CustomerClaim(BaseModel):
    """What the extractor must pull out of a messy customer message.

    Deliberately narrow: WHICH order and WHAT KIND of problem. It does NOT ask the
    model for the amount or the real dates — customers misremember those, so the
    harness looks them up from the order record instead. Same split as everywhere
    else here: the model reads, the harness knows the numbers.

    order_id is optional on purpose: if the customer never gives one, the correct
    answer is None ("unknown"), not a hallucinated order.
    stated_amount_usd is captured only to cross-check against the record — it is
    never trusted as fact.
    """

    order_id: str | None = None
    claim_type: ClaimType
    stated_amount_usd: float | None = None


class ClaimFinding(BaseModel):
    """The deterministic verdict on a customer claim — the production gate AND the RLVR
    reward signal, computed by harness/validity.py against the order record. No LLM.

    claim_true is None (not False) when the order can't be found: "can't judge" is a
    distinct outcome from "the claim is false", and they drive different actions
    (ask-for-info vs reject).
    """

    order_found: bool
    claim_true: bool | None
    reason: str
    order_id: str | None = None
    amount_usd: float | None = None


class ComplaintCase(BaseModel):
    """One generated case: a messy message plus its exact answer key.

    Built by scripts/gen_complaint_cases.py from a REAL late Olist delivery, so the
    ground truth is known before the message exists — which is what makes the
    extraction reward genuinely verifiable rather than judged.
    """

    case_id: str
    channel: Literal["chat", "email", "transcript"]
    difficulty: Literal["easy", "medium", "hard"]
    message: str
    ground_truth: CustomerClaim
