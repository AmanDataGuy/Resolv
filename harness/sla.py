"""SLA findings — plain lookups and date math, not an LLM agent.

Same reasoning as harness/impact.py: "find the notification deadline" and
"calculate the penalty" are formula lookups against structured data, not
language understanding. The original sla_agent LlmAgent called three tools and
then asked Gemini to combine their outputs into prose — here the three tool
calls ARE the answer, just returned directly as schemas.SLAFindings.

clause_text is the one addition that uses rag/contract_search.py: the penalty
clause_id and dollar amount always come from the structured, deterministic
harness/suppliers.py lookup — RAG is never allowed to touch the money math,
since a retrieval score is a similarity estimate, not a verified fact. RAG only
supplies the actual contract PROSE for that same clause, so the drafter can
quote real contract language instead of just stating a bare clause number. If
retrieval doesn't find a confident match (score below _MIN_RAG_SCORE), clause_text
stays None and the draft falls back to citing the clause number alone.
"""
from harness.suppliers import calculate_penalty, check_notification_deadline, get_sla_terms
from rag.contract_search import search as search_contracts
from schemas import SLAFindings

_MIN_RAG_SCORE = 0.3

# What to search the contract text for, per exception type — these are natural-
# language questions, not clause numbers, because that's what semantic search
# is actually good at matching against contract prose.
_CLAUSE_QUERIES = {
    "late_shipment": "penalty for a late delivery shipment",
    "stockout": "supplier capacity shortfall, cannot fulfill order",
    "supplier_failure": "supplier capacity shortfall, cannot fulfill order",
    "customs_hold": "shipment held at customs due to missing paperwork",
    "quality_rejection": "customer rejected goods due to quality defects",
    "price_dispute": "invoice does not match the purchase order price",
}


async def evaluate_sla(
    supplier_id: str,
    exception_type: str,
    detected_at: str,
    delay_days: int,
    revenue_at_risk_usd: float,
) -> SLAFindings:
    """Returns the applicable clause, penalty, and notification deadline urgency."""
    sla = await get_sla_terms(supplier_id)
    deadline = await check_notification_deadline(supplier_id, exception_type, detected_at)
    penalty = await calculate_penalty(supplier_id, exception_type, delay_days, revenue_at_risk_usd)

    clause_id = penalty.get("clause_id")
    if clause_id:
        violation_description = f"Clause {clause_id}: late delivery penalty applies."
    elif sla:
        violation_description = "SLA terms on file, but no specific penalty clause matched."
    else:
        violation_description = "No SLA terms on file for this supplier."

    clause_text = None
    query = _CLAUSE_QUERIES.get(exception_type)
    if query:
        results = search_contracts(query, supplier_id=supplier_id, top_k=1)
        if results and results[0]["score"] >= _MIN_RAG_SCORE:
            clause_text = results[0]["text"]

    return SLAFindings(
        clause_id=clause_id,
        clause_text=clause_text,
        violation_description=violation_description,
        penalty_usd=penalty["penalty_usd"],
        notification_deadline_hours=deadline["hours_remaining"],
        is_urgent=deadline["is_urgent"],
    )
