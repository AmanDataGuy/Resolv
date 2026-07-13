"""Tests for the deterministic harness/ modules.

No API key, no network call, no mocking needed — every function here is plain
Python over data/mock_orders.json and data/mock_suppliers.json. This is the whole
point of moving impact/SLA/escalation logic out of LLM agents: it becomes this
easy to test.
"""
import pytest

from harness.escalation import decide_escalation
from harness.guardrails import apply_draft_guardrail, verify_draft
from harness.impact import assess_impact
from harness.sla import evaluate_sla
from schemas import ExceptionEvent, ImpactAssessment, ResolutionDraft, SLAFindings


@pytest.mark.asyncio
async def test_assess_impact_sums_revenue_and_flags_breach():
    impact = await assess_impact("acme-logistics", ["ORD-1001", "ORD-1002"])
    assert impact.revenue_at_risk_usd == 42000.0
    assert impact.orders_affected == 2
    assert impact.customers_affected == 2
    assert impact.sla_penalty_usd == 840.0  # 42000 * 2.0%
    assert impact.sla_breach_risk is True  # both mock orders are "delayed"
    assert impact.severity_label == "critical"


@pytest.mark.asyncio
async def test_assess_impact_unknown_supplier_is_zero():
    impact = await assess_impact("no-such-supplier", ["ORD-9999"])
    assert impact.revenue_at_risk_usd == 0
    assert impact.orders_affected == 0


@pytest.mark.asyncio
async def test_evaluate_sla_finds_clause_and_penalty():
    findings = await evaluate_sla(
        supplier_id="acme-logistics",
        exception_type="late_shipment",
        detected_at="2026-07-01T00:00:00+00:00",
        delay_days=5,
        revenue_at_risk_usd=42000.0,
    )
    assert findings.clause_id == "3.2"
    assert findings.penalty_usd == 840.0
    assert isinstance(findings.is_urgent, bool)
    # RAG-retrieved contract prose for the same clause (see rag/contract_search.py)
    assert findings.clause_text is not None
    assert "3.2" in findings.clause_text


@pytest.mark.asyncio
async def test_evaluate_sla_no_contact_supplier_has_no_clause():
    findings = await evaluate_sla(
        supplier_id="no-contact-supplier",
        exception_type="late_shipment",
        detected_at="2026-07-01T00:00:00+00:00",
        delay_days=5,
        revenue_at_risk_usd=1000.0,
    )
    assert findings.clause_id is None


def test_escalation_never_auto_resolves_quality_rejection():
    decision = decide_escalation(
        exception_type="quality_rejection",
        revenue_at_risk_usd=500,
        urgency="low",
        contact_found=True,
        clause_id="3.2",
        sla_is_urgent=False,
    )
    assert decision.action == "human_approval"


def test_escalation_never_auto_resolves_supplier_failure():
    decision = decide_escalation(
        exception_type="supplier_failure",
        revenue_at_risk_usd=100,
        urgency="low",
        contact_found=True,
        clause_id="3.2",
        sla_is_urgent=False,
    )
    assert decision.action == "human_approval"


def test_escalation_escalates_when_no_contact():
    decision = decide_escalation(
        exception_type="late_shipment",
        revenue_at_risk_usd=100,
        urgency="low",
        contact_found=False,
        clause_id=None,
        sla_is_urgent=False,
    )
    assert decision.action == "escalate"


def test_escalation_escalates_over_50k_revenue():
    decision = decide_escalation(
        exception_type="late_shipment",
        revenue_at_risk_usd=75000,
        urgency="low",
        contact_found=True,
        clause_id="3.2",
        sla_is_urgent=False,
    )
    assert decision.action == "escalate"


def test_escalation_human_approval_missing_clause_lowers_confidence():
    decision = decide_escalation(
        exception_type="late_shipment",
        revenue_at_risk_usd=500,
        urgency="low",
        contact_found=True,
        clause_id=None,  # confidence drops to 0.8, below the 0.85 threshold
        sla_is_urgent=False,
    )
    assert decision.action == "human_approval"
    assert decision.confidence == 0.8


def test_escalation_auto_resolves_when_all_conditions_met():
    decision = decide_escalation(
        exception_type="late_shipment",
        revenue_at_risk_usd=500,
        urgency="low",
        contact_found=True,
        clause_id="3.2",
        sla_is_urgent=False,
    )
    assert decision.action == "auto_resolve"
    assert decision.confidence == 1.0


def _sample_context():
    event = ExceptionEvent(
        exception_id="exc-1",
        exception_type="late_shipment",
        order_ids=["ORD-1001", "ORD-1002"],
        supplier_id="acme-logistics",
        urgency="high",
        raw_context="{}",
        detected_at="2026-07-01T00:00:00+00:00",
        source_system="test",
    )
    impact = ImpactAssessment(
        revenue_at_risk_usd=42000.0,
        orders_affected=2,
        customers_affected=2,
        estimated_delay_days=5,
        sla_breach_risk=True,
        sla_penalty_usd=840.0,
        severity_label="critical",
        recommended_action="notify",
    )
    sla_findings = SLAFindings(
        clause_id="3.2",
        violation_description="late delivery",
        penalty_usd=840.0,
        notification_deadline_hours=10.0,
        is_urgent=False,
    )
    return event, impact, sla_findings


def test_verify_draft_passes_when_everything_matches():
    event, impact, sla_findings = _sample_context()
    draft = ResolutionDraft(
        subject="SLA notice",
        body="Orders ORD-1001 and ORD-1002 are delayed. Revenue at risk: $42,000. See clause 3.2.",
        recipient_type="supplier",
        tone="firm",
        action_requested="acknowledge",
        follow_up_in_hours=24,
    )
    result = verify_draft(event, impact, sla_findings, draft)
    assert result["passed"] is True


def test_verify_draft_fails_when_order_id_missing():
    event, impact, sla_findings = _sample_context()
    draft = ResolutionDraft(
        subject="SLA notice",
        body="Order ORD-1001 is delayed. Revenue at risk: $42,000. See clause 3.2.",  # missing ORD-1002
        recipient_type="supplier",
        tone="firm",
        action_requested="acknowledge",
        follow_up_in_hours=24,
    )
    result = verify_draft(event, impact, sla_findings, draft)
    assert result["passed"] is False
    assert result["checks"]["order_ids_present"] is False


def test_apply_draft_guardrail_downgrades_failed_auto_resolve():
    event, impact, sla_findings = _sample_context()
    bad_draft = ResolutionDraft(
        subject="SLA notice",
        body="Something vague with no specifics.",
        recipient_type="supplier",
        tone="firm",
        action_requested="acknowledge",
        follow_up_in_hours=24,
    )
    verification = verify_draft(event, impact, sla_findings, bad_draft)
    decision = decide_escalation(
        exception_type="late_shipment",
        revenue_at_risk_usd=500,
        urgency="low",
        contact_found=True,
        clause_id="3.2",
        sla_is_urgent=False,
    )
    assert decision.action == "auto_resolve"  # would have auto-sent...
    guarded = apply_draft_guardrail(decision, verification)
    assert guarded.action == "human_approval"  # ...but the bad draft blocks it
