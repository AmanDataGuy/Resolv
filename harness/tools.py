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
import threading

from harness import audit
from harness.policy import Decision, check_refund
from harness.validity import get_order

# Guards the read-history -> check_refund -> append sequence in issue_refund(). Without it,
# concurrent calls on the same case can all read the trail before any of them appends, so more
# than one clears rule 4 — reproduced directly in the 2026-08-15 audit (7 of 8 concurrent calls
# refunded the same order once each). A single process-wide lock, not a per-case_id registry:
# at this scale the throughput cost is unmeasurable and a global lock has no lifecycle to manage.
# This protects one process only — see plan_ahead.md Priority 2 for why a multi-instance
# deployment needs a real datastore instead, which this lock does not and cannot provide.
_refund_lock = threading.Lock()

# Every tool returns a plain string. Tool results land back in a language model's context, so a
# sentence it can act on beats a dict it has to interpret — and the structured version of the
# same event is already in the audit trail, which is where machines read it.


def lookup_order(case_id: str, order_id: str) -> str:
    """Read an order. The only tool with no policy check — it mutates nothing.

    Deliberately returns the REAL amount and dates. This is what lets the agent catch a customer
    who states a wrong figure: the facts are one tool call away, so there is never a reason to
    take a stated number on trust.

    Recorded to the trail even though it's a read, which is a departure from "the trail is for
    mutations". Whether the agent LOOKED BEFORE IT ACTED is the single most valuable fact about
    a trajectory — an agent that refunds a correct amount without checking got lucky, and one
    that checked first is doing the job. Both look identical if reads aren't written down. The
    first end-to-end run made this obvious: the agent caught a customer's inflated $900 claim by
    looking up the real $639.43, and the trail showed no evidence it had happened.
    """
    order = get_order(order_id)
    found = bool(order)
    if found:
        result = (
            f"Order {order_id}: paid ${order['amount_usd']:.2f}, "
            f"promised {order['promised_date']}, "
            f"delivered {order['delivered_date'] or 'never'}, "
            f"status {order['status']}."
        )
    else:
        result = f"Order {order_id} not found. Ask the customer to re-check the number."

    # A synthetic decision so reads share the trail's shape. rule_id records what the read
    # found, because "the agent looked up an order that doesn't exist" is worth being able to
    # count. Rule 4 filters on tool == "issue_refund", so these rows can never affect it.
    decision = Decision(
        action="allow",
        rule_id="lookup_hit" if found else "lookup_miss",
        reason=f"Read order {order_id}.",
    )
    audit.append(case_id, "lookup_order", order_id, {}, decision, found, result)
    return result


def issue_refund(case_id: str, order_id: str, claim_type: str, amount_usd: float, caller_id: str | None) -> str:
    """Refund money. Every rule in policy.py stands between this call and the mutation.

    Check, then record, then act — and the record is written for denials and escalations too.
    An attempt that was stopped is the most informative line in the trail.

    HISTORY IS THE UNION OF TWO SCOPES. audit.read(case_id) is what happened in THIS conversation;
    audit.order_history(order_id) is every successful refund this order has EVER received, in any
    conversation. Rule 5 needs both — a customer who contacts support twice, in two separate,
    honest cases, must not be paid twice. The whole read-check-append sequence runs under a lock:
    without it, concurrent calls on the same case can all read the history before any of them
    appends, and more than one clears rule 5 (see _refund_lock above).

    caller_id has no default here on purpose — every call site (agents/loop.py's closure, this
    file's own demo()) must say explicitly who is asking, even if the answer is None. See
    harness/policy.py rule 2 for what None means: not "skip the check," but "no identity claimed."
    """
    with _refund_lock:
        history = audit.read(case_id) + audit.order_history(order_id)
        decision = check_refund(order_id, claim_type, amount_usd, history, caller_id)

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
            decision, ok, result, caller_id,
        )

        # An escalate verdict must reach a human whether or not the model remembers to say so
        # itself. Nothing in policy.py's rule 7 (over_auto_approve_limit) instructs the MODEL to
        # follow up with escalate_to_human -- and 2 of 3 imperfect eval runs (eval_report.md §2)
        # were exactly this: policy correctly said escalate, the agent never made the separate
        # call. routing.py already treats a policy-level escalate the same as an explicit one;
        # this makes that guarantee structural instead of relying on model compliance for it, the
        # same reasoning MAX_STEPS exhaustion and repeated guardrail failures already use in
        # agents/loop.py. Guarded so a model that DOES call escalate_to_human itself right after
        # doesn't produce two records for the same order.
        if decision.action == "escalate" and not any(
            r["tool"] == "escalate_to_human" and r["order_id"] == order_id for r in history
        ):
            escalate_to_human(case_id, order_id, decision.reason)
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
    owner = fresh["customer_id"]
    # This demo issues a real refund on a real order below (line ~170) — the order index (which
    # rule 4 now reads across cases, 2026-08-15 fix) must be reset before AND after, or this demo
    # leaks a permanent "already refunded" record into a real order's history, and running it
    # twice — or running it before anything else that touches this same order — fails for a
    # reason that has nothing to do with what this demo is checking.
    audit.clear_order(oid)

    # lookup exposes the true amount — the antidote to whatever figure the customer states
    assert f"${fresh['amount_usd']:.2f}" in lookup_order(case, oid)
    assert audit.read(case)[-1]["rule_id"] == "lookup_hit"  # reads are evidence; they get recorded
    assert "not found" in lookup_order(case, "ORD-999999")
    assert audit.read(case)[-1]["rule_id"] == "lookup_miss"

    # A caller with no stated connection to the order is refused before the cap is even checked.
    assert issue_refund(case, oid, "late_delivery", 1.0, "someone_else").startswith("Refused:")
    assert audit.read(case)[-1]["rule_id"] == "caller_not_order_owner"

    # An over-cap refund from the real owner is refused, and the refusal is a STRING the model
    # can act on.
    out = issue_refund(case, oid, "late_delivery", fresh["amount_usd"], owner)
    assert out.startswith("Refused:"), out
    assert audit.read(case)[-1]["rule_id"] == "refund_exceeds_cap"
    assert audit.read(case)[-1]["ok"] is False  # the attempt is recorded even though it failed

    # A within-cap refund from the real owner succeeds...
    cap = round(fresh["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    amount = min(cap, AUTO_APPROVE_MAX_USD)
    assert issue_refund(case, oid, "late_delivery", amount, owner).startswith("Refunded")

    # ...and the SAME call a second time is denied off the trail alone. Nothing about the
    # request changed — only the past did. This is rule 5 reading what tools.py wrote.
    assert "already refunded" in issue_refund(case, oid, "late_delivery", amount, owner).lower()

    # Escalation is never denied.
    assert escalate_to_human(case, oid, "customer asked for a manager").startswith("Escalated")

    # 2 lookups + 4 refund attempts + 1 escalation. Every call the agent could make, recorded —
    # including the three that were refused, which are the ones worth reading.
    trail = audit.read(case)
    assert len(trail) == 7, f"expected 7 recorded attempts, got {len(trail)}"
    audit.clear(case)
    audit.clear_order(oid)
    print(f"tools demo OK — {len(TOOLS)} tools, every mutation policy-checked, {len(trail)} attempts recorded")


if __name__ == "__main__":
    demo()
