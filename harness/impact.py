"""Impact assessment — plain arithmetic, not an LLM agent.

The original design (resolv.md) made this an LlmAgent that called tools to fetch
orders and SLA terms, then asked Gemini to "calculate revenue_at_risk_usd = sum of
shipment_value_usd". Summing a column and multiplying by a percentage is not a
language understanding task — it's arithmetic a calculator does perfectly, for
free, with zero hallucination risk. Wrapping it in an LLM call added latency,
API cost, and a new way to get the math wrong, for no benefit. See
agents/detector.py and agents/drafter.py for the two steps that actually need
an LLM (unstructured classification, natural language generation).
"""
from harness.orders import query_orders_by_supplier
from harness.suppliers import get_sla_terms
from schemas import ImpactAssessment


def _severity(revenue_at_risk_usd: float, sla_breach_risk: bool) -> str:
    if sla_breach_risk or revenue_at_risk_usd > 50_000:
        return "critical"
    if revenue_at_risk_usd > 10_000:
        return "high"
    if revenue_at_risk_usd > 1_000:
        return "medium"
    return "low"


async def assess_impact(supplier_id: str, order_ids: list[str]) -> ImpactAssessment:
    """Computes revenue at risk, affected counts, and SLA penalty for one exception."""
    all_supplier_orders = await query_orders_by_supplier(supplier_id)
    affected = [o for o in all_supplier_orders if o["order_id"] in order_ids]

    revenue_at_risk_usd = sum(o["shipment_value_usd"] for o in affected)
    customers_affected = len({o["customer_id"] for o in affected})

    sla = await get_sla_terms(supplier_id)
    penalty_pct = sla.get("late_delivery_penalty_pct", 0)
    sla_penalty_usd = round(revenue_at_risk_usd * penalty_pct / 100, 2)
    sla_breach_risk = any(o["status"] == "delayed" for o in affected)

    return ImpactAssessment(
        revenue_at_risk_usd=revenue_at_risk_usd,
        orders_affected=len(affected),
        customers_affected=customers_affected,
        estimated_delay_days=0,  # mock fixtures carry no actual_delivery date to diff against
        sla_breach_risk=sla_breach_risk,
        sla_penalty_usd=sla_penalty_usd,
        severity_label=_severity(revenue_at_risk_usd, sla_breach_risk),
        recommended_action=f"Notify {supplier_id} of {len(affected)} affected order(s).",
    )
