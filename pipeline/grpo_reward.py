"""GRPO reward function — this IS harness/guardrails.py's verifier, not a
separate reimplementation of similar-looking logic.

This is the concrete answer to "what makes this RLVR and not LLM-as-judge":
the reward for a generated draft comes from exact-match checks on its content
(order IDs present, dollar amounts correct, SLA clause cited) — the same
function that gates auto-send in the live pipeline (harness/guardrails.py).
There is exactly one definition of "did this draft get the facts right" in the
whole codebase; training and production both call it.

The live GRPO training loop is scripts/train_grpo.py, which uses TRL's
GRPOTrainer. TRL scores raw generated text, not a ResolutionDraft object, so that
script carries a text-level twin of this reward (same exact-match checks on order
ID and dollar amount). This module is the schema-level form of that reward: it
wraps a generated body in a minimal ResolutionDraft and runs the exact same
harness/guardrails.py verifier the production pipeline uses — keeping "did the
draft get the facts right" defined once, whichever caller needs it.
"""
from harness.guardrails import verify_draft
from schemas import ExceptionEvent, ImpactAssessment, ResolutionDraft, SLAFindings


def compute_reward(
    exception_event: ExceptionEvent,
    impact: ImpactAssessment,
    sla_findings: SLAFindings,
    generated_body: str,
) -> float:
    """Scores one GRPO rollout in [0.0, 1.0] — fraction of harness/guardrails.py's
    checks the generated draft body passes. GRPO only needs the body text (it's
    scoring free-form generation, not a full ADK output_schema object), so this
    wraps it in a minimal ResolutionDraft — subject/tone/action_requested aren't
    part of the reward, only body content is checked.
    """
    draft = ResolutionDraft(
        subject="",
        body=generated_body,
        recipient_type="supplier",
        tone="firm",
        action_requested="",
        follow_up_in_hours=24,
    )
    verification = verify_draft(exception_event, impact, sla_findings, draft)
    checks = verification["checks"]
    return sum(checks.values()) / len(checks)
