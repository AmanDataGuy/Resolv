"""Confirms the deterministic harness handles all 6 exception types correctly,
not just late_shipment. No LLM call needed — detector/drafter classification is
exercised separately (see notes.md); this file only proves harness/impact.py and
harness/escalation.py behave correctly across every type using the mock orders
added for stockout, customs_hold, quality_rejection, supplier_failure, and
price_dispute (see data/mock_orders.json: ORD-3001, ORD-4001, ORD-5001, ORD-6001,
ORD-7001).
"""
import pytest

from harness.escalation import decide_escalation
from harness.impact import assess_impact

# (exception_type, supplier_id, order_ids)
ALL_TYPES = [
    ("late_shipment", "acme-logistics", ["ORD-1001", "ORD-1002"]),
    ("stockout", "northwind-parts", ["ORD-3001"]),
    ("customs_hold", "acme-logistics", ["ORD-4001"]),
    ("quality_rejection", "acme-logistics", ["ORD-5001"]),
    ("supplier_failure", "northwind-parts", ["ORD-6001"]),
    ("price_dispute", "acme-logistics", ["ORD-7001"]),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("exception_type,supplier_id,order_ids", ALL_TYPES)
async def test_impact_assessment_works_for_every_type(exception_type, supplier_id, order_ids):
    impact = await assess_impact(supplier_id, order_ids)
    assert impact.revenue_at_risk_usd > 0
    assert impact.orders_affected == len(order_ids)
    assert impact.severity_label in {"low", "medium", "high", "critical"}


@pytest.mark.parametrize("exception_type,supplier_id,order_ids", ALL_TYPES)
def test_quality_and_supplier_failure_never_auto_resolve(exception_type, supplier_id, order_ids):
    decision = decide_escalation(
        exception_type=exception_type,
        revenue_at_risk_usd=500,  # deliberately low — would auto-resolve for other types
        urgency="low",
        contact_found=True,
        clause_id="3.2",
        sla_is_urgent=False,
    )
    if exception_type in {"quality_rejection", "supplier_failure"}:
        assert decision.action == "human_approval"
    else:
        assert decision.action == "auto_resolve"
