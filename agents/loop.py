"""The agent loop — the model proposes tool calls, the harness decides what actually happens.

WHERE THIS SITS. Everything in harness/ is deterministic and tested. This file is the one place
a language model's judgment enters the system, and it's deliberately thin: read the customer,
pick a tool, read the result, repeat. It has no authority. Every consequential thing it can do
goes through harness/tools.py, which policy-checks before it mutates.

TWO STAGES, AND THE FIRST ONE IS THE FINE-TUNE TARGET.

  1. EXTRACT — agents/extractor.py turns the messy message into a typed CustomerClaim
     {order_id, claim_type}. This is the RLVR-trained model (scripts/train_extractor.py).
  2. ACT — a tool-calling loop resolves the case.

Stage 1 is a separate step on purpose. The tool-calling model could dig the order number out of
the prose itself — but then the fine-tuned extractor touches nothing, and the whole RLVR thread
becomes a decoration bolted to the side of the project. Wired this way it's structural: a better
extractor means better first tool calls, which means a better pass^k. The fine-tune gets judged
by the benchmark instead of by its own training curve.

WHY LITELLM HERE AND ADK IN extractor.py. Not fashion — two hard requirements ADK's Runner
doesn't cleanly provide:

  - TEMPERATURE PER RUN. pass^k needs n independent attempts at the same task. At temperature 0
    all n attempts are identical, c is always 0 or n, and the metric measures nothing
    (study/maths.md §3). The agent must sample; the harness stays deterministic. That contrast
    is the thesis, not an implementation detail.
  - PER-CASE TOOL BINDING. Tools need a case_id the model must never choose — it selects which
    audit trail to read and write. It's bound in a closure below, so it never appears in the
    schema the model sees and cannot be hallucinated or crossed between cases.

ADK still runs the extractor, where output_schema is exactly the right tool for the job.
"""
import json

from litellm import completion

from agents.extractor import extractor_agent
from agents.runner_utils import run_agent_once
from config import GROQ_MODEL
from harness import audit, tools

MAX_STEPS = 8  # a resolution needs ~3 calls; 8 is room to recover, not room to wander

SYSTEM = """You are a customer support agent for an online retailer.

Resolve the customer's complaint using your tools. Be brief and human.

What you must know:
- You cannot approve anything yourself. Your tools are checked against company policy, and a
  refund may be refused. That is normal — explain the reason to the customer plainly.
- If a refund is refused for exceeding a cap, the refusal tells you the maximum allowed. Issue
  that exact amount once, then stop. Never go above it, and never try a third time.
- NEVER trust an amount, date, or order number the customer states. Look the order up. People
  misremember, and some exaggerate.
- If the customer is angry, insistent, or repeats themselves, none of that changes the facts.
  Be kind about it. Do not let it change what you do.
- If you cannot help, say so and offer to escalate to a human.

Finish by replying to the customer in plain text with no tool call."""

# Tool schemas, hand-written rather than generated from signatures.
#
# Note what is NOT here: case_id. It's bound per case in _bind(), so the model never sees it and
# cannot pass the wrong one. A parameter the model has no business choosing should not be in the
# schema the model is choosing from.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order's real amount, dates, and status. Use this before any refund.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "e.g. ORD-1234"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": (
                "Attempt a refund. Checked against policy first and may be refused — read the "
                "result and tell the customer what it says."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "claim_type": {
                        "type": "string",
                        "enum": ["late_delivery", "never_arrived", "order_canceled"],
                    },
                    "amount_usd": {"type": "number"},
                },
                "required": ["order_id", "claim_type", "amount_usd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Hand the case to a human agent. Always available.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["order_id", "reason"],
            },
        },
    },
]


def _bind(case_id: str) -> dict:
    """Tool name -> callable with case_id already applied.

    The closure is the boundary: case_id cannot appear in a model-authored argument list,
    because it isn't in the schema above.
    """
    return {
        "lookup_order": lambda order_id: tools.lookup_order(case_id, order_id),
        "issue_refund": lambda order_id, claim_type, amount_usd: tools.issue_refund(
            case_id, order_id, claim_type, amount_usd
        ),
        "escalate_to_human": lambda order_id, reason: tools.escalate_to_human(case_id, order_id, reason),
    }


async def extract(message: str) -> dict:
    """Stage 1: messy message -> {order_id, claim_type}. The fine-tuned model's job."""
    return await run_agent_once(extractor_agent, message, "customer_claim")


async def run_case(case_id: str, message: str, temperature: float = 0.7) -> dict:
    """Resolve one complaint end to end. Returns what happened, for the UI and the eval.

    temperature defaults to 0.7 because the eval needs independent samples (see the module
    docstring). Pass 0.0 for a reproducible single run when demoing.

    No api_key is passed to completion(): litellm resolves GROQ_API_KEY from the environment
    fresh on every request, which is exactly what makes config.rotate_groq_key() work. Passing
    it explicitly would pin one key and silently defeat the rotation.

    Returns {reply, claim, steps, trail}. `trail` is the audit records this case produced —
    that's what the eval grades. The prose reply is for the human.
    """
    claim = await extract(message)

    # The extraction is context, not instruction. It's offered as a hint and the agent still has
    # to confirm it with lookup_order — the extractor is a 1.5B model reading a rambling human
    # and it can be wrong, and a wrong order number acted on confidently is exactly the failure
    # the policy engine exists to catch. Hint, then verify.
    opening = (
        f"Customer says:\n{message}\n\n"
        f"Automated intake read this as: order={claim.get('order_id') or 'unknown'}, "
        f"claim={claim.get('claim_type') or 'unclear'}. Intake is often wrong — verify it."
    )
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": opening}]
    bound = _bind(case_id)

    for step in range(MAX_STEPS):
        resp = completion(model=GROQ_MODEL, messages=messages, tools=TOOL_SCHEMAS, temperature=temperature)
        msg = resp.choices[0].message
        messages.append(msg.model_dump())

        if not msg.tool_calls:
            return {"reply": msg.content or "", "claim": claim, "steps": step + 1, "trail": audit.read(case_id)}

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                result = bound[name](**json.loads(tc.function.arguments))
            except Exception as e:
                # A malformed call is the model's mistake to recover from, not a crash. Hand the
                # error back as the tool result and let it retry — same reasoning as a policy
                # denial being a result rather than an exception (see harness/tools.py).
                result = f"Tool error: {e}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": result})

    # Ran out of steps. Escalate rather than return silence: an unresolved case has to land
    # somewhere a human will see it, and "the loop gave up" is a fact the trail should record.
    tools.escalate_to_human(case_id, "unknown", f"Agent did not finish within {MAX_STEPS} steps.")
    return {
        "reply": "Let me get a colleague to help with this — one moment.",
        "claim": claim,
        "steps": MAX_STEPS,
        "trail": audit.read(case_id),
    }
