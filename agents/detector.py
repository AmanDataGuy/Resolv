"""Exception Detector agent.

Takes a raw event dict (from a webhook, currently passed straight from api/main.py)
and classifies it into one of the six ExceptionEvent types with structured fields.

Schema: see schemas.ExceptionEvent for the exact output shape.

ADK pattern used here (matches the reference course's Module 4 "Structured Output"
example): output_schema=<PydanticModel> forces the model's final answer into that
schema; output_key stores the parsed result in session state under that name, so a
later agent's instruction can reference it as {exception_event} if they share a
session. Resolv's pipeline stages don't share a session (see runner_utils.py) — the
orchestrator instead reads this key out via run_agent_once() and passes it forward
as plain data.
"""
from google.adk.agents import LlmAgent

from config import get_model
from schemas import ExceptionEvent

detector_agent = LlmAgent(
    name="exception_detector",
    model=get_model(),
    description="Classifies a raw supply chain event into a structured ExceptionEvent.",
    instruction="""
You receive raw event JSON from an order system, TMS, or webhook.

Classify the event into exactly one of these six types:
  late_shipment, stockout, customs_hold, quality_rejection, supplier_failure, price_dispute

Extract every order_id, the supplier_id, and any product_ids present in the event.
Do not invent an order_id, supplier_id, or product_id that isn't in the raw event.

Assign urgency:
  critical: SLA deadline < 4 hours OR revenue_at_risk > $50,000
  high:     SLA deadline < 24 hours OR revenue_at_risk > $10,000
  medium:   delay expected but not yet breaching SLA
  low:      informational, no immediate action needed

exception_id: reuse the raw event's own id if present, otherwise generate one like
  "exc-<supplier_id>-<first_order_id>".
detected_at: use the current time if the raw event has no timestamp.
source_system: copy from the raw event's "source" field, default to "unknown".
""",
    output_schema=ExceptionEvent,
    output_key="exception_event",
)
