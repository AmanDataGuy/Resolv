"""Resolv's exception pipeline.

Only two real LLM agents remain: detector_agent (classifies raw event text) and
drafter_agent (writes the email). Everything else — impact math, supplier lookup,
SLA calculation, and the escalation decision — is deterministic Python in
harness/, called directly in a fixed order. See each harness/*.py module's
docstring for why it used to be an LlmAgent and isn't anymore.

This function is still plain async Python, not an ADK sub_agents-routed root
agent: sub_agents is for LLM-driven delegation, and this pipeline's step order is
fixed business logic, not something to leave to a model's judgment (see
agents/runner_utils.py's docstring for the original reasoning, which still holds).
"""
from datetime import datetime, timezone

from agents.detector import detector_agent
from agents.drafter import drafter_agent
from agents.runner_utils import run_agent_once, to_prompt
from comms import post_slack, send_email
from db import update_exception_state
from harness.escalation import decide_escalation
from harness.guardrails import apply_draft_guardrail, verify_draft
from harness.impact import assess_impact
from harness.sla import evaluate_sla
from harness.suppliers import get_supplier_context
from schemas import ExceptionEvent, ResolutionDraft, SupplierContext


async def process_exception(raw_event: dict) -> dict:
    """Runs one raw event through the full Resolv pipeline and returns the final
    exception record (also persisted to db.py under exception_id).
    """
    event_data = await run_agent_once(detector_agent, to_prompt(raw_event), "exception_event")
    # detected_at is never the LLM's responsibility — it has no way to know the
    # actual current time and will happily hallucinate a plausible-looking one
    # (observed: it produced "2024-03-16" for an event with no timestamp field,
    # instead of the real current date). Use the raw event's own timestamp if it
    # provided one, otherwise the real wall-clock time, always overriding whatever
    # the model guessed.
    event_data["detected_at"] = (
        raw_event.get("detected_at") or raw_event.get("timestamp") or datetime.now(timezone.utc).isoformat()
    )
    exception_event = ExceptionEvent.model_validate(event_data)

    impact = await assess_impact(exception_event.supplier_id, exception_event.order_ids)

    supplier_data = await get_supplier_context(exception_event.supplier_id, exception_event.exception_type)
    supplier_context = SupplierContext.model_validate(supplier_data)

    sla_findings = await evaluate_sla(
        supplier_id=exception_event.supplier_id,
        exception_type=exception_event.exception_type,
        detected_at=exception_event.detected_at.isoformat(),
        delay_days=impact.estimated_delay_days,
        revenue_at_risk_usd=impact.revenue_at_risk_usd,
    )

    draft_prompt = to_prompt(
        {
            **exception_event.model_dump(mode="json"),
            **impact.model_dump(mode="json"),
            **supplier_context.model_dump(mode="json"),
            **sla_findings.model_dump(mode="json"),
        }
    )
    draft_data = await run_agent_once(drafter_agent, draft_prompt, "resolution_draft")
    draft = ResolutionDraft.model_validate(draft_data)

    decision = decide_escalation(
        exception_type=exception_event.exception_type,
        revenue_at_risk_usd=impact.revenue_at_risk_usd,
        urgency=exception_event.urgency,
        contact_found=supplier_context.contact_found,
        clause_id=sla_findings.clause_id,
        sla_is_urgent=sla_findings.is_urgent,
    )

    verification = verify_draft(exception_event, impact, sla_findings, draft)
    decision = apply_draft_guardrail(decision, verification)

    status = "escalated" if decision.action == "escalate" else "pending_approval"
    if decision.action == "auto_resolve" and supplier_context.contact_email:
        await send_email(to=supplier_context.contact_email, subject=draft.subject, body=draft.body)
        status = "sent"

    await post_slack(
        channel="",
        message=(
            f"[{exception_event.urgency.upper()}] {exception_event.exception_type} "
            f"for {exception_event.supplier_id}: ${impact.revenue_at_risk_usd:,.0f} at risk. "
            f"Decision: {decision.action} ({decision.reason})"
        ),
    )

    record = {
        "exception_id": exception_event.exception_id,
        "type": exception_event.exception_type,
        "supplier_id": exception_event.supplier_id,
        "urgency": exception_event.urgency,
        "status": status,
        "impact": impact.model_dump(mode="json"),
        "supplier_context": supplier_context.model_dump(mode="json"),
        "sla_findings": sla_findings.model_dump(mode="json"),
        "draft": draft.model_dump(mode="json"),
        "decision": decision.model_dump(mode="json"),
        "draft_verification": verification,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    await update_exception_state(exception_event.exception_id, record)
    return record
