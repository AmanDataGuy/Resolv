"""Tests pipeline/grpo_reward.py — no API key, no network call: this reward
function is deterministic by design (it reuses harness/guardrails.py's exact
verifier), so it's just as cheap to test as any other harness function despite
living in the "fine-tuning" part of the project.
"""
from pipeline.grpo_reward import compute_reward
from schemas import ExceptionEvent, ImpactAssessment, SLAFindings

_EVENT = ExceptionEvent(
    exception_id="exc-1",
    exception_type="late_shipment",
    order_ids=["ORD-1001", "ORD-1002"],
    supplier_id="acme-logistics",
    urgency="high",
    raw_context="{}",
    detected_at="2026-07-01T00:00:00+00:00",
    source_system="test",
)
_IMPACT = ImpactAssessment(
    revenue_at_risk_usd=42000.0,
    orders_affected=2,
    customers_affected=2,
    estimated_delay_days=5,
    sla_breach_risk=True,
    sla_penalty_usd=840.0,
    severity_label="critical",
    recommended_action="notify",
)
_SLA = SLAFindings(
    clause_id="3.2",
    violation_description="late delivery",
    penalty_usd=840.0,
    notification_deadline_hours=10.0,
    is_urgent=False,
)


def test_reward_is_1_when_all_facts_present():
    body = "Orders ORD-1001 and ORD-1002 are delayed. Revenue at risk: $42,000. See clause 3.2."
    assert compute_reward(_EVENT, _IMPACT, _SLA, body) == 1.0


def test_reward_drops_when_order_id_missing():
    body = "Order ORD-1001 is delayed. Revenue at risk: $42,000. See clause 3.2."  # missing ORD-1002
    reward = compute_reward(_EVENT, _IMPACT, _SLA, body)
    assert 0.0 < reward < 1.0


def test_reward_is_0_for_a_vague_draft_with_no_facts():
    body = "Something is wrong with your shipment, please respond."
    assert compute_reward(_EVENT, _IMPACT, _SLA, body) == 0.0
