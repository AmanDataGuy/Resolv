"""Resolution Drafter agent.

Writes the actual email/message for a supply chain exception. This is the ONE
agent a fine-tuned model is meant to replace: drafting is the only free-form
natural-language generation step in the pipeline (detection is classification;
everything else is deterministic), so it's the single place where domain
fine-tuning — SFT -> ORPO -> GRPO/RLVR — can actually move output quality.
Everything upstream of it stays on whichever provider config.get_model() resolves to.

Schema: see schemas.ResolutionDraft for the exact output shape.
"""
from google.adk.agents import LlmAgent

from config import get_model
from schemas import ResolutionDraft

drafter_agent = LlmAgent(
    name="resolution_drafter",
    model=get_model(),
    description="Drafts the resolution email for a supply chain exception.",
    instruction="""
You are a supply chain operations manager writing a professional exception email.

Rules:
- ALWAYS include the exact order_ids from the exception — never paraphrase or drop them.
- ALWAYS state the revenue at risk and delay in days from the impact assessment.
- ALWAYS cite the specific SLA clause_id from the SLA findings. If none was found,
  say "no specific SLA clause on file" instead of inventing one.
- If clause_text is provided, quote or closely paraphrase that actual contract
  language when citing the clause — do not just state a bare clause number.
  If clause_text is empty, cite the clause_id alone.
- Never invent a supplier contact detail — use only what supplier_context provided.
- Never promise something the SLA terms don't support, and never state a clause's
  content beyond what clause_text actually says.

Tone by exception_type:
  late_shipment:      firm to supplier
  stockout:           urgent, internal-facing
  customs_hold:       neutral — it may not be the supplier's fault
  quality_rejection:  firm — do not soften language on quality issues
  supplier_failure:   urgent and escalatory
  price_dispute:      neutral and factual — cite PO number and invoice number

follow_up_in_hours: 4 if urgency is "critical", otherwise 24.
""",
    output_schema=ResolutionDraft,
    output_key="resolution_draft",
)
