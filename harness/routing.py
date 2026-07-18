"""Post-resolution routing — turns a finished case into a ticket, deterministically.

WHERE THIS SITS. The agent loop ends; the audit trail now holds everything that happened. This
module reads that trail and answers the two questions a support system asks next: what do we tell
the customer, and who (if anyone) picks this up? Both answers are DERIVED from the trail with no
LLM — the same principle as the rest of the harness. What the customer is emailed and which team
is paged are facts about what the agent actually did, not a second opinion a model could get
wrong on the way out the door.

WHY NOT IN agents/loop.py. The loop is where the model's judgment lives; this is where it
doesn't. Routing is a pure function of the trail — same trail, same ticket, always — which is
what makes it testable and what keeps the customer-facing message honest. Keeping it out of the
loop also keeps it out of the eval's hot path: the benchmark scores the trail, and the ticket is
a product concern the eval never needs to pay for.

THREE OUTCOMES, READ OFF THE TRAIL:
  a successful refund        -> resolved, ticket CLOSED, customer told the amount
  an escalation              -> escalated, ticket OPEN, routed to a team, customer told to wait
  nothing owed / no action   -> denied, ticket CLOSED, customer told why

"Escalated" covers both ways a case reaches a human: the agent calling escalate_to_human, and an
over-limit refund that policy sent to a human (rule 6, recorded as an issue_refund with
action="escalate"). Either way a person must act, so the ticket stays OPEN.
"""
import hashlib

from schemas import Team, Ticket


def _ticket_id(case_id: str) -> str:
    """A stable, human-looking ticket id. Deterministic from case_id so re-routing the same case
    yields the same ticket — routing must be idempotent, or a retry pages a second human.
    """
    return f"TKT-{hashlib.sha1(case_id.encode()).hexdigest()[:6].upper()}"


def _team_for(trail: list[dict], claim: dict) -> Team:
    """Which queue an escalation lands on. First match wins, most specific first.

    Money authority beats domain: a refund over the auto-approve limit is a billing sign-off no
    matter what the delivery problem was, so it goes to billing even for a never_arrived order.
    Below that, route by the kind of problem — a still-missing package is logistics' to chase.
    """
    if any(r.get("rule_id") == "over_auto_approve_limit" for r in trail):
        return "billing"
    claim_type = claim.get("claim_type")
    if claim_type == "never_arrived":
        return "logistics"
    if claim_type in ("late_delivery", "order_canceled"):
        return "billing"
    return "general"


def route(case_id: str, trail: list[dict], claim: dict) -> Ticket:
    """Read a finished case's trail and produce its ticket. Pure — no I/O, no model."""
    ticket_id = _ticket_id(case_id)
    order_id = claim.get("order_id")
    order_ref = order_id or "your order"

    refunds = [r for r in trail if r["tool"] == "issue_refund" and r["ok"]]
    escalated = any(r["tool"] == "escalate_to_human" for r in trail) or any(
        r.get("action") == "escalate" for r in trail
    )

    if refunds:
        amount = refunds[-1]["args"]["amount_usd"]
        return Ticket(
            ticket_id=ticket_id, case_id=case_id, order_id=order_id,
            outcome="resolved", team=None, status="closed",
            customer_message=(
                f"Good news about {order_ref}: we've approved a refund of ${amount:.2f}, which "
                f"you'll see in a few business days. Your reference is {ticket_id}. "
                "Thanks for your patience."
            ),
        )

    if escalated:
        team = _team_for(trail, claim)
        return Ticket(
            ticket_id=ticket_id, case_id=case_id, order_id=order_id,
            outcome="escalated", team=team, status="open",
            customer_message=(
                f"Thanks for reaching out about {order_ref}. We've logged this as {ticket_id} and "
                f"passed it to our {team} team to review — someone will be in touch shortly. We "
                "wanted a person to look at this properly rather than rush it."
            ),
        )

    # No refund, no escalation: the claim wasn't owed. Surface the policy's reason if the trail
    # recorded one — a denial the customer can't understand is a denial they'll dispute.
    denials = [r for r in trail if r["tool"] == "issue_refund" and not r["ok"]]
    reason = denials[-1]["reason"] if denials else "we weren't able to find a refund owed on this order"
    return Ticket(
        ticket_id=ticket_id, case_id=case_id, order_id=order_id,
        outcome="denied", team=None, status="closed",
        customer_message=(
            f"Thanks for contacting us about {order_ref}. After checking the order, {reason}. "
            f"If you think this is a mistake, reply with {ticket_id} and we'll take another look."
        ),
    )


def demo() -> None:
    """Every outcome and every team, from hand-built trails — no network, no real cases."""
    late = {"order_id": "ORD-1000", "claim_type": "late_delivery"}
    missing = {"order_id": "ORD-1200", "claim_type": "never_arrived"}

    # resolved: a successful refund closes the ticket and states the amount
    trail = [{"tool": "issue_refund", "order_id": "ORD-1000", "ok": True, "action": "allow",
              "args": {"amount_usd": 159.86}, "rule_id": "within_policy", "reason": "ok"}]
    t = route("case-a", trail, late)
    assert t.outcome == "resolved" and t.status == "closed" and t.team is None
    assert "159.86" in t.customer_message and t.ticket_id in t.customer_message

    # idempotent: same case_id -> same ticket_id (a retry must not page a second human)
    assert route("case-a", trail, late).ticket_id == t.ticket_id

    # escalated via rule 6 -> billing, ticket OPEN, regardless of the delivery problem
    over = [{"tool": "issue_refund", "order_id": "ORD-1200", "ok": False, "action": "escalate",
             "args": {"amount_usd": 500.0}, "rule_id": "over_auto_approve_limit", "reason": "over limit"}]
    t = route("case-b", over, missing)
    assert t.outcome == "escalated" and t.status == "open" and t.team == "billing"

    # escalate_to_human on a never_arrived case -> logistics
    esc = [{"tool": "escalate_to_human", "order_id": "ORD-1200", "ok": True, "action": "allow",
            "args": {"reason": "customer asked for a manager"}, "rule_id": "escalation_always_permitted",
            "reason": "manager"}]
    assert route("case-c", esc, missing).team == "logistics"

    # denied: the policy reason is carried into the customer message
    deny = [{"tool": "issue_refund", "order_id": "ORD-1000", "ok": False, "action": "deny",
             "args": {"amount_usd": 10.0}, "rule_id": "outside_claim_window",
             "reason": "Order is 200 days old; claims close after 90 days"}]
    t = route("case-d", deny, late)
    assert t.outcome == "denied" and t.status == "closed" and "90 days" in t.customer_message

    # no order id -> general team, and the message still reads
    t = route("case-e", esc, {"order_id": None, "claim_type": None})
    assert t.team == "general" and "your order" in t.customer_message

    print("routing demo OK — resolved/escalated/denied, 3 teams, idempotent ticket ids")


if __name__ == "__main__":
    demo()
