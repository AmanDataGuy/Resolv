"""
# resolv/harness/routing.py

**What:** turns a finished case's audit trail into a Ticket: what the customer is told and which team (if any)
picks it up.
**Why:** both answers are derived from the trail with no LLM, so the email and the page are facts about what the
agent actually did. A pure function (same trail, same ticket) is testable and stays out of the eval's hot path.
Outcomes: a successful refund resolves and closes; any escalation (the agent's own, a rule 7 over-limit refund, a
declined or unconfirmed payment) stays open on a team's queue; anything else is denied with the policy's reason.
An escalated ticket links its ledger escalation id, read from the escalate_to_human record's args, so routing
stays free of I/O.
**Used by:** resolv/api/main.py, app.py.
"""
import hashlib

from resolv.schemas import Team, Ticket


# A stable ticket id from the case id, so re-routing a retried case never pages a second human.
def _ticket_id(case_id: str) -> str:
    return f"TKT-{hashlib.sha1(case_id.encode()).hexdigest()[:6].upper()}"


# Which queue an escalation lands on: money authority first, then the kind of problem.
def _team_for(trail: list[dict], claim: dict) -> Team:
    if any(r.get("rule_id") in ("over_auto_approve_limit", "payment_unknown") for r in trail):
        return "billing"
    claim_type = claim.get("claim_type")
    if claim_type == "never_arrived":
        return "logistics"
    if claim_type in ("late_delivery", "order_canceled"):
        return "billing"
    return "general"


# The newest ledger escalation id in the trail; tools._escalate writes it in the same transaction as the row.
def _escalation_id(trail: list[dict]) -> str | None:
    ids = [(r.get("args") or {}).get("escalation_id") for r in trail if r["tool"] == "escalate_to_human"]
    return next((i for i in reversed(ids) if i), None)


# Read a finished case's trail and produce its ticket. Pure: no I/O, no model.
def route(case_id: str, trail: list[dict], claim: dict) -> Ticket:
    ticket_id = _ticket_id(case_id)
    order_id = claim.get("order_id")
    order_ref = order_id or "your order"

    refunds = [r for r in trail if r["tool"] == "issue_refund" and r["ok"]]
    escalated = any(r["tool"] == "escalate_to_human" for r in trail) or any(
        r.get("action") == "escalate" or r.get("rule_id") == "payment_unknown" for r in trail
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
            outcome="escalated", team=team, status="open", escalation_id=_escalation_id(trail),
            customer_message=(
                f"Thanks for reaching out about {order_ref}. We've logged this as {ticket_id} and "
                f"passed it to our {team} team to review — someone will be in touch shortly. We "
                "wanted a person to look at this properly rather than rush it."
            ),
        )

    # Nothing owed: surface the policy's reason, because a denial the customer can't understand gets disputed.
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


# Every outcome and every team, from hand-built trails.
def demo() -> None:
    late = {"order_id": "ORD-1000", "claim_type": "late_delivery"}
    missing = {"order_id": "ORD-1200", "claim_type": "never_arrived"}

    trail = [{"tool": "issue_refund", "order_id": "ORD-1000", "ok": True, "action": "allow",
              "args": {"amount_usd": 159.86}, "rule_id": "within_policy", "reason": "ok"}]
    t = route("case-a", trail, late)
    assert t.outcome == "resolved" and t.status == "closed" and "159.86" in t.customer_message
    assert route("case-a", trail, late).ticket_id == t.ticket_id

    # Rule 7 over-limit refund: billing, ticket open, whatever the delivery problem was.
    over = [{"tool": "issue_refund", "order_id": "ORD-1200", "ok": False, "action": "escalate",
             "args": {"amount_usd": 500.0}, "rule_id": "over_auto_approve_limit", "reason": "over limit"}]
    t = route("case-b", over, missing)
    assert t.outcome == "escalated" and t.status == "open" and t.team == "billing"

    esc = [{"tool": "escalate_to_human", "order_id": "ORD-1200", "ok": True, "action": "allow",
            "args": {"reason": "manager", "escalation_id": "ESC-1"}, "rule_id": "escalation_always_permitted",
            "reason": "manager"}]
    assert route("case-c", esc, missing).team == "logistics"
    assert route("case-c", esc, missing).escalation_id == "ESC-1"
    assert t.escalation_id is None

    deny = [{"tool": "issue_refund", "order_id": "ORD-1000", "ok": False, "action": "deny",
             "args": {"amount_usd": 10.0}, "rule_id": "outside_claim_window",
             "reason": "Order is 200 days old; claims close after 90 days"}]
    t = route("case-d", deny, late)
    assert t.outcome == "denied" and "90 days" in t.customer_message

    unknown = [{"tool": "issue_refund", "order_id": "ORD-1000", "ok": False, "action": "allow",
                "args": {"amount_usd": 10.0}, "rule_id": "payment_unknown", "reason": "TimeoutError"}]
    assert route("case-e", unknown, late).outcome == "escalated"
    assert route("case-f", esc, {"order_id": None, "claim_type": None}).team == "general"
    print("routing demo OK: resolved/escalated/denied, 3 teams, idempotent ticket ids")


if __name__ == "__main__":
    demo()
