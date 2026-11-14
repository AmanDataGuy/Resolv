"""
# resolv/agents/extractor.py

**What:** turns a messy customer message into a typed CustomerClaim {order_id, claim_type}, with three backends:
`adk` (ADK agent with output_schema, the default), `json` (one structured-output completion) and `legacy_line`
(the GRPO adapter's one-line "order=... type=..." format on the completions endpoint).
**Why:** reading the order number out of a ramble is the one hard language task, and the fine-tune target. The
model is trusted only for reading: resolv/harness/validity.py checks the result against the order record, and no
usable order number must give order_id=None, never a guess. All backends share one instruction text.
**Used by:** resolv/agents/loop.py (extract), evals/extractor.py.
"""
import re
from functools import lru_cache

from google.adk.agents import LlmAgent

from resolv import config
from resolv.agents.llm import acomplete
from resolv import observability
from resolv.schemas import CustomerClaim

EXTRACTOR_INSTRUCTION = """
You read a customer's support message and extract two things.

order_id: the order number the customer refers to, in the form "ORD-####".
  - The customer may spell it out ("oh-arr-dee one zero zero seven"), fumble digits, or bury it
    mid-sentence. Normalise it to "ORD-1007".
  - They may mention MORE THAN ONE order number — pick the one their complaint is actually about,
    not an aside about a different order.
  - If they give NO usable order number, set order_id to null. NEVER invent or guess one.

claim_type: what the customer is complaining about — exactly one of:
  late_delivery     — it arrived (or they believe it arrived) later than promised
  never_arrived     — it shipped but never showed up; they're still waiting
  order_canceled    — their order was canceled
  - If the message describes NO problem yet ("hi", "quick question", a greeting with nothing
    else), set claim_type to null. NEVER force it into one of the three types just to fill the
    field — that produces a confident guess about a complaint that was never made.

Do not judge whether the claim is true — only extract what the customer is asserting. The
verification happens elsewhere.
"""

# Must match scripts/train_extractor.py's INSTRUCTION byte for byte: the adapter was trained on this raw prompt.
CLAIM_TYPES = ["late_delivery", "never_arrived", "order_canceled"]
LEGACY_PROMPT = (
    "Extract the order number and the claim type from this customer message.\n"
    "Reply on ONE line exactly as: order=<ORD-#### or none> type=<one of: "
    + ", ".join(CLAIM_TYPES) + ">\n\nMessage:\n"
)

observability.register_prompt("extractor", "extractor", EXTRACTOR_INSTRUCTION)


# Build the ADK extractor on first use, not at import, so importing never needs a provider key.
@lru_cache(maxsize=1)
def get_extractor_agent() -> LlmAgent:
    return LlmAgent(
        name="claim_extractor",
        model=config.get_model("extractor"),
        description="Extracts a structured claim (order id + claim type) from a messy customer message.",
        instruction=EXTRACTOR_INSTRUCTION,
        output_schema=CustomerClaim,
        output_key="customer_claim",
    )


# Validate a model's JSON text as a CustomerClaim dict; raises ValueError on anything invalid.
def parse_json_claim(text: str) -> dict:
    return CustomerClaim.model_validate_json(text or "").model_dump()


# Parse the legacy one-line format; "none", garbage or a missing field all map to None.
def parse_legacy_line(text: str) -> dict:
    order = re.search(r"ORD[-\s]?(\d{3,5})", text or "", re.I)
    lowered = (text or "").lower()
    claim_type = next((ct for ct in CLAIM_TYPES if ct in lowered), None)
    return {"order_id": f"ORD-{order.group(1)}" if order else None, "claim_type": claim_type, "stated_amount_usd": None}


# json backend: one temperature-0 completion constrained to the CustomerClaim schema.
async def extract_json(message: str) -> dict:
    schema = {"name": "CustomerClaim", "schema": CustomerClaim.model_json_schema()}
    resp = await acomplete(
        trace_role="extractor",
        model=config.EXTRACTOR_MODEL,
        messages=[{"role": "system", "content": EXTRACTOR_INSTRUCTION}, {"role": "user", "content": message}],
        temperature=0,
        response_format={"type": "json_schema", "json_schema": schema},
    )
    return parse_json_claim(resp.choices[0].message.content)


# legacy_line backend: raw prompt on the completions endpoint, greedy, 40 tokens, as the adapter was trained.
async def extract_legacy_line(message: str) -> dict:
    resp = await acomplete(
        trace_role="extractor",
        model=config.EXTRACTOR_MODEL,
        prompt=LEGACY_PROMPT + message,
        temperature=0,
        max_tokens=40,
    )
    return parse_legacy_line(resp.choices[0].text)
