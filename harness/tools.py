"""The tools the agent can call — and the only place in this system that changes anything.

THE ONE IDEA. Every mutating tool calls policy.check_refund() BEFORE it mutates, and there is
no second write path. Not "the agent is instructed to check policy first" — instructed is a
suggestion, and a model under pressure from an angry customer is precisely where suggestions
lose. The check sits on the inside of the only door.

That's what makes "the agent does not decide what is allowed" testable rather than
aspirational: breaking it would require adding a second write path, which is a reviewable
change to this file — not something a prompt can talk its way into.

WHAT THE MODEL SEES. A denial comes back as an ordinary tool RESULT, not an exception. This is
deliberate. The model needs to read "Refused: $900 exceeds the $47.09 cap" and go explain that
to the customer — refusing with a reason is the correct behaviour, not an error. Raising would
abort the turn and strand the customer mid-conversation, converting a well-handled refusal into
a crash. A good harness lets the agent fail gracefully; it doesn't punish it for asking.

STATE. Refunds are recorded to the audit trail and nowhere else — there is no separate mutable
DB. The trail IS the state: rule 4 reads it to answer "already refunded?", the eval scores from
it, the UI renders it. One source of truth, append-only, and what the eval grades is the same
thing production would act on.
"""
from harness import audit
from harness.policy import Decision, check_refund
from harness.validity import get_order

# Every tool returns a plain string. Tool results land back in a language model's context, so a
# sentence it can act on beats a dict it has to interpret — and the structured version of the
# same event is already in the audit trail, which is where machines read it.


def lookup_order(case_id: str, order_id: str) -> str:
    """Read an order. The only tool with no policy check — it mutates nothing.

    Deliberately returns the REAL amount and dates. This is what lets the agent catch a customer
    who states a wrong figure: the facts are one tool call away, so there is never a reason to
    take a stated number on trust.
    """
    order = get_order(order_id)
    if not order:
        return f"Order {order_id} not found. Ask the customer to re-check the number."
    return (
        f"Order {order_id}: paid ${order['amount_usd']:.2f}, "
        f"promised {order['promised_date']}, "
        f"delivered {order['delivered_date'] or 'never'}, "
        f"status {order['status']}."
    )


def issue_refund(case_id: str, order_id: str, claim_type: str, amount_usd: float) -> str:
    """Refund money. Every rule in policy.py stands between this call and the mutation.

    Check, then record, then act — and the record is written for denials and escalations too.
    An attempt that was stopped is the most informative line in the trail.
    """
    history = audit.read(case_id)
    decision = check_refund(order_id, claim_type, amount_usd, history)

    ok = decision.action == "allow"
    if ok:
        result = f"Refunded ${amount_usd:.2f} on {order_id}."
    elif decision.action == "escalate":
        result = f"Not refunded — sent to a human for approval. {decision.reason}"
    else:
        result = f"Refused: {decision.reason}"

    audit.append(
        case_id, "issue_refund", order_id,
        {"amount_usd": amount_usd, "claim_type": claim_type},
        decision, ok, result,
    )
    return result


def escalate_to_human(case_id: str, order_id: str, reason: str) -> str:
    """Hand the case to a person. Always allowed, and never a failure.

    No policy check, because no rule could deny it: escalating is the safe action by
    construction. Worth stating plainly, since the instinct is to gate everything — gating the
    escape hatch is how an agent ends up with nowhere to go and starts improvising.

    Recorded with a synthetic allow decision so it lands in the same trail with the same shape.
    Escalation rate is a metric we care about (an agent that escalates everything is useless in
    a different way than one that refunds everything), and it can't be measured if it isn't
    written down.
    """
    decision = Decision(action="allow", rule_id="escalation_always_permitted", reason=reason)
    result = f"Escalated {order_id} to a human agent: {reason}"
    audit.append(case_id, "escalate_to_human", order_id, {"reason": reason}, decision, True, result)
    return result


# The agent's entire surface area. If it isn't in this list, the agent cannot do it.
TOOLS = [lookup_order, issue_refund, escalate_to_human]


def demo() -> None:
    """Prove the properties the rest of the system assumes: no bypass, denials are recorded,
    and a denial reads as a result rather than an exception.
    """
    from datetime import date

    from config import AUTO_APPROVE_MAX_USD, CLAIM_WINDOW_DAYS, REFUND_CAP_FRACTION
    from harness.policy import NOW

    case = "_demo_tools"
    audit.clear(case)

    orders = [o for o in (get_order(f"ORD-{i}") for i in range(1000, 1460)) if o]
    fresh = next(
        o for o in orders
        if o["situation"] == "late" and (NOW - date.fromisoformat(o["promised_date"])).days <= CLAIM_WINDOW_DAYS
    )
    oid = fresh["order_id"]

    # lookup exposes the true amount — the antidote to whatever figure the customer states
    assert f"${fresh['amount_usd']:.2f}" in lookup_order(case, oid)

    # An over-cap refund is refused, and the refusal is a STRING the model can act on.
    out = issue_refund(case, oid, "late_delivery", fresh["amount_usd"])
    assert out.startswith("Refused:"), out
    assert audit.read(case)[-1]["rule_id"] == "refund_exceeds_cap"
    assert audit.read(case)[-1]["ok"] is False  # the attempt is recorded even though it failed

    # A within-cap refund succeeds...
    cap = round(fresh["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    amount = min(cap, AUTO_APPROVE_MAX_USD)
    assert issue_refund(case, oid, "late_delivery", amount).startswith("Refunded")

    # ...and the SAME call a second time is denied off the trail alone. Nothing about the
    # request changed — only the past did. This is rule 4 reading what tools.py wrote.
    assert "already refunded" in issue_refund(case, oid, "late_delivery", amount).lower()

    # Escalation is never denied.
    assert escalate_to_human(case, oid, "customer asked for a manager").startswith("Escalated")

    trail = audit.read(case)
    assert len(trail) == 4, f"expected 4 recorded attempts, got {len(trail)}"
    audit.clear(case)
    print(f"tools demo OK — {len(TOOLS)} tools, every mutation policy-checked, 4 attempts recorded")


if __name__ == "__main__":
    demo()
