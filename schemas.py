"""Shared Pydantic models — the typed contract between every stage of the pipeline.

Centralized in one module (rather than one schema per agent file) so the
orchestrator, the harness, and the test suite can all import any schema without
circular imports. These types ARE the interface: the LLM agents emit ExceptionEvent
and ResolutionDraft via ADK's output_schema, the deterministic harness produces
ImpactAssessment / SLAFindings / EscalationDecision, and every boundary between
them is validated against the models here — a malformed hand-off fails loudly at
the seam instead of silently downstream.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

ExceptionType = Literal[
    "late_shipment",
    "stockout",
    "customs_hold",
    "quality_rejection",
    "supplier_failure",
    "price_dispute",
]

Urgency = Literal["critical", "high", "medium", "low"]


class ExceptionEvent(BaseModel):
    exception_id: str
    exception_type: ExceptionType
    order_ids: list[str]
    supplier_id: str
    product_ids: list[str] = []
    urgency: Urgency
    raw_context: str
    detected_at: datetime
    source_system: str


class ImpactAssessment(BaseModel):
    revenue_at_risk_usd: float
    orders_affected: int
    customers_affected: int
    estimated_delay_days: int
    sla_breach_risk: bool
    sla_penalty_usd: float
    severity_label: Literal["low", "medium", "high", "critical"]
    recommended_action: str


class SupplierContext(BaseModel):
    contact_found: bool
    name: str | None = None
    contact_email: str | None = None
    contact_phone: str | None = None
    account_manager: str | None = None
    product_categories: list[str] = []
    risk_score: float | None = None
    alternate_suppliers: list[dict] = []


class SLAFindings(BaseModel):
    clause_id: str | None = None
    clause_text: str | None = None  # actual contract prose, from rag/contract_search.py
    violation_description: str
    penalty_usd: float
    notification_deadline_hours: float
    is_urgent: bool


class ResolutionDraft(BaseModel):
    subject: str
    body: str
    recipient_type: Literal["supplier", "customer", "internal"]
    tone: Literal["firm", "empathetic", "neutral", "urgent"]
    action_requested: str
    follow_up_in_hours: int


class EscalationDecision(BaseModel):
    action: Literal["auto_resolve", "human_approval", "escalate"]
    confidence: float
    reason: str
    notify_channels: list[Literal["email", "sms", "voice", "slack"]]


# --- Customer-complaint intake (chat / email / transcript) ---------------------
# The intake side of the pipeline: a real person describes a problem in their own
# words, and the extractor turns that into a typed claim the harness can verify.

# The four claim types, each backed by a real Olist order_status (no fabricated types):
#   late_delivery   — delivered after the promised date (or the customer believes so)
#   never_arrived   — shipped but never delivered
#   order_canceled  — the order was canceled
#   item_unavailable— the item became unavailable after ordering
ClaimType = Literal["late_delivery", "never_arrived", "order_canceled", "item_unavailable"]


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
