"""Draft content verification — plain string/number matching, not an LLM judge.

This is a new responsibility, not a rename of the old agents/guardrails.py (which
used to re-check the escalation decision — that job moved into
harness/escalation.py, see its docstring). This file checks something different:
did the drafted email actually say what the data says, or did the model drop an
order ID / invent a dollar figure / cite the wrong SLA clause?

Every check here is exact-match, not "ask an LLM if this looks right" — this
matters for two reasons:
  1. As a guardrail: a hallucinated dollar amount in a supplier-facing legal
     notice is a real liability. Catching it needs certainty, not a judge call.
  2. As the RLVR reward: RLVR (reinforcement learning with verifiable rewards)
     specifically means the reward comes from a deterministic verifier, not a
     model's opinion — these exact same functions are what the fine-tuning reward
     (pipeline/grpo_reward.py) is built from. One set of checks, two uses.
"""
from schemas import EscalationDecision, ExceptionEvent, ImpactAssessment, ResolutionDraft, SLAFindings


def verify_draft(
    exception_event: ExceptionEvent,
    impact: ImpactAssessment,
    sla_findings: SLAFindings,
    draft: ResolutionDraft,
) -> dict:
    """Returns {"passed": bool, "checks": {...}} — every check is a binary,
    reproducible fact about the draft text, never a subjective quality judgment.
    """
    body = draft.body

    order_ids_present = all(order_id in body for order_id in exception_event.order_ids)

    amount_strings = {
        f"{impact.revenue_at_risk_usd:,.0f}",
        f"{impact.revenue_at_risk_usd:.2f}",
        f"{sla_findings.penalty_usd:,.0f}",
        f"{sla_findings.penalty_usd:.2f}",
    }
    amounts_correct = any(amount in body for amount in amount_strings)

    clause_cited_correctly = sla_findings.clause_id in body if sla_findings.clause_id else True

    checks = {
        "order_ids_present": order_ids_present,
        "amounts_correct": amounts_correct,
        "clause_cited_correctly": clause_cited_correctly,
    }
    return {"passed": all(checks.values()), "checks": checks}


def apply_draft_guardrail(decision: EscalationDecision, verification: dict) -> EscalationDecision:
    """If the draft failed verification, force human_approval no matter what the
    escalation decision said — a bad draft should never auto-send.
    """
    if verification["passed"] or decision.action != "auto_resolve":
        return decision

    return EscalationDecision(
        action="human_approval",
        confidence=decision.confidence,
        reason=f"Draft failed content verification: {verification['checks']}",
        notify_channels=["voice", "slack"],
    )
