"""Claim Extractor agent — the intake side of the pipeline, and the RLVR fine-tune target.

Turns a messy free-form customer message (chat / email / transcript) into a typed
CustomerClaim{order_id, claim_type}. This is the ONE genuinely hard language task in the
system: the order number may be spelled out, typo'd, buried, or absent, and the claim type has
to be inferred from an emotional ramble. That difficulty is deliberate — it's what gives
fine-tuning real headroom (unlike the old drafter, which was handed the facts).

Its output is verified downstream by harness/validity.py against the order record, so the model
is never trusted for the numbers — only for reading. If the customer gives no usable order
number, the correct output is order_id = null, NOT an invented one.
"""
from google.adk.agents import LlmAgent

from config import get_model
from schemas import CustomerClaim

extractor_agent = LlmAgent(
    name="claim_extractor",
    model=get_model(),
    description="Extracts a structured claim (order id + claim type) from a messy customer message.",
    instruction="""
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
  item_unavailable  — the item was unavailable / out of stock after they ordered

Do not judge whether the claim is true — only extract what the customer is asserting. The
verification happens elsewhere.
""",
    output_schema=CustomerClaim,
    output_key="customer_claim",
)
