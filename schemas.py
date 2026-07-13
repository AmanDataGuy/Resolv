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
