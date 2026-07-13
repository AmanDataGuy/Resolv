"""Escalation decision — plain threshold logic, not an LLM agent.

This merges what used to be TWO things: agents/escalation_agent.py (an LlmAgent
whose "instructions" were already a hardcoded threshold table — confidence >= 0.85
AND revenue < $10,000 -> auto_resolve) and agents/guardrails.py (plain Python that
re-checked the LLM's proposal afterward and overrode it if it broke a hard rule).

Once you notice the LLM's own rules were already an exact threshold table, there
was never a judgment call for it to make — guardrails.py's job was to catch the
cases where the model didn't apply its own stated rule correctly. Implementing
the table once, in Python, removes that failure mode entirely instead of
detecting it after the fact. There is exactly one copy of this logic now.

confidence is no longer an LLM's self-reported number (research on RLVR/agent
evaluation flags self-reported LLM confidence as unreliable and ungameable-to-verify).
It's computed from how complete the upstream data was — missing a supplier contact
or SLA clause should make the system less sure of itself, and that's checkable.
"""
from config import AUTO_RESOLVE_CONFIDENCE_THRESHOLD, AUTO_RESOLVE_MAX_RISK_USD, NEVER_AUTO_RESOLVE_TYPES
from schemas import EscalationDecision


def compute_confidence(contact_found: bool, clause_id: str | None) -> float:
    """Data-completeness score, not a model's opinion. 1.0 = nothing was missing."""
    confidence = 1.0
    if not contact_found:
        confidence -= 0.5
    if not clause_id:
        confidence -= 0.2
    return round(max(confidence, 0.0), 2)


def decide_escalation(
    exception_type: str,
    revenue_at_risk_usd: float,
    urgency: str,
    contact_found: bool,
    clause_id: str | None,
    sla_is_urgent: bool,
) -> EscalationDecision:
    """Returns the final auto_resolve/human_approval/escalate decision. Rules apply
    in order — first match wins, same as resolv.md's original table.
    """
    confidence = compute_confidence(contact_found, clause_id)

    if not contact_found:
        return EscalationDecision(
            action="escalate",
            confidence=confidence,
            reason="No supplier contact on file — cannot proceed unattended.",
            notify_channels=["voice", "sms", "slack"],
        )

    if exception_type in NEVER_AUTO_RESOLVE_TYPES:
        return EscalationDecision(
            action="human_approval",
            confidence=confidence,
            reason=f"{exception_type} always requires human sign-off.",
            notify_channels=["voice", "slack"],
        )

    if revenue_at_risk_usd > 50_000:
        return EscalationDecision(
            action="escalate",
            confidence=confidence,
            reason=f"Revenue at risk (${revenue_at_risk_usd:,.0f}) exceeds $50,000.",
            notify_channels=["voice", "sms", "slack"],
        )

    needs_human = (
        revenue_at_risk_usd >= AUTO_RESOLVE_MAX_RISK_USD
        or confidence < AUTO_RESOLVE_CONFIDENCE_THRESHOLD
        or urgency == "critical"
        or sla_is_urgent
    )
    if needs_human:
        return EscalationDecision(
            action="human_approval",
            confidence=confidence,
            reason="One or more auto-resolve conditions not met (revenue, confidence, urgency, or SLA deadline).",
            notify_channels=["voice", "slack"],
        )

    return EscalationDecision(
        action="auto_resolve",
        confidence=confidence,
        reason="All auto-resolve conditions met.",
        notify_channels=["email", "slack"],
    )
