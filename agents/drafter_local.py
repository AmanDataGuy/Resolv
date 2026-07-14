"""Local fine-tuned drafter — runs the GRPO/RLVR adapter on this machine instead of
calling Groq.

The production drafter (agents/drafter.py) is a Groq-backed ADK LlmAgent. This module
is the alternative backend that actually *uses the fine-tuned model* the training
pipeline produced: it loads base Qwen2.5-1.5B-Instruct + the GRPO LoRA adapter and
generates the email body directly.

Train/serve note: the model was fine-tuned on raw email text, not the ResolutionDraft
JSON schema. So the model produces the body, and the deterministic fields (subject,
tone, recipient_type, follow_up) are filled from the harness facts here — the same
split as everywhere else in Resolv (the model writes prose; code owns structure). The
guardrail (harness/guardrails.py) then verifies the body regardless of which backend
wrote it.

Loaded lazily and cached, so importing this module is free and the ~3 GB model only
loads the first time a fine-tuned draft is actually requested.
"""
from schemas import ExceptionEvent, ImpactAssessment, ResolutionDraft, SLAFindings

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "models/adapters/grpo/latest"

# ResolutionDraft.tone / recipient_type are constrained Literals — map each exception
# type to a valid value (mirrors the tone guidance in agents/drafter.py's instruction).
_TONE = {
    "late_shipment": "firm",
    "stockout": "urgent",
    "customs_hold": "neutral",
    "quality_rejection": "firm",
    "supplier_failure": "urgent",
    "price_dispute": "neutral",
}
_RECIPIENT = {"stockout": "internal"}  # everything else addresses the supplier

_model = None
_tok = None


def _ensure_model():
    """Load base + adapter once; subsequent calls reuse them."""
    global _model, _tok
    if _model is not None:
        return
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="cuda")
    _model = PeftModel.from_pretrained(base, ADAPTER)
    _model.eval()


def _generate(prompt: str) -> str:
    import torch

    ids = _tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
    ).to(_model.device)
    with torch.no_grad():
        out = _model.generate(ids, max_new_tokens=220, do_sample=False)
    return _tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def draft_local(
    exception_event: ExceptionEvent,
    impact: ImpactAssessment,
    sla_findings: SLAFindings,
) -> ResolutionDraft:
    """Generate a ResolutionDraft with the fine-tuned model for the body and the
    harness facts for the structured fields."""
    _ensure_model()

    orders = ", ".join(exception_event.order_ids) or "N/A"
    clause = f" | SLA clause: {sla_findings.clause_id}" if sla_findings.clause_id else ""
    prompt = (
        f"Exception type: {exception_event.exception_type} | Orders: {orders} | "
        f"Revenue at risk: ${impact.revenue_at_risk_usd:.2f}{clause}\n"
        "Draft a professional resolution email."
    )
    body = _generate(prompt)

    etype = exception_event.exception_type
    first_order = exception_event.order_ids[0] if exception_event.order_ids else "N/A"
    return ResolutionDraft(
        subject=f"{etype.replace('_', ' ').title()} — order {first_order}",
        body=body,
        recipient_type=_RECIPIENT.get(etype, "supplier"),
        tone=_TONE.get(etype, "neutral"),
        action_requested="Review and respond to the exception above.",
        follow_up_in_hours=4 if exception_event.urgency == "critical" else 24,
    )
