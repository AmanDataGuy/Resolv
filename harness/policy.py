"""The policy engine — every business rule in the system, in one file, as plain Python.

This file was harness/escalation.py (`git log --follow` shows the lineage). That version's core
insight is still the reason this one exists, so it's worth restating: the LlmAgent it replaced
had "instructions" that were already an exact threshold table. Once you notice the model's own
rules leave it no judgment to exercise, asking a model to apply them is pure downside — it can
only get the table wrong. Write the table in Python and that failure mode is gone, rather than
detected after the fact.

TWO THINGS CHANGED, AND THEY'RE WHY THIS IS A NEW FILE RATHER THAN A RENAME.

1. It ENFORCES instead of ADVISING. escalation.py returned a decision the orchestrator then
   chose to honour — a caller could ignore it, or forget to call it at all. check_refund() is
   called from inside harness/tools.py, before the mutation, and tools.py is the only write
   path in the system. There is no route around this function. That is the difference between
   a rule and a suggestion.

2. It reads HISTORY, not just the current request. Every parameter of the old
   decide_escalation() described one event in isolation. Rule 4 below asks "has this order
   already been refunded?" — a question no stateless check can answer. It's the rule that
   catches a customer who asks twice: politely the first time, insistently the second.

HOW TO READ THE RULES. check_refund() runs them in order, first match wins, and each returns a
rule_id so the audit log and the UI can name exactly which line stopped a call. They're ordered
cheapest and most fundamental first — existence, then ownership, then truth, then limits. Every
denial carries a reason a human can read, because a policy engine nobody can debug gets switched
off within a week.

CALLER IDENTITY (rule 2, added after tests/test_hallucination_collision.py proved the gap real).
Every rule below this line asks "is the CLAIM true?" — none of them asked "does the CALLER own
the order?" Before this rule existed, a caller with zero stated connection to an order got paid
its refund as long as the claim about it happened to be true, because nothing compared the caller
to the order's customer_id (present in the record all along, read by nothing until now).
caller_id has no bypassable default: passing None is a valid call, but it means "no identity
claimed," and that denies exactly like a real mismatch would. A check that can be skipped by
omitting an argument is the escalation.py mistake this whole file exists to avoid repeating.

WHAT THIS DELIBERATELY ISN'T. No Rego, no OPA, no external policy service. Seven rules over the
order record and a cap table is a function, and reaching for a policy DSL here would be
resume-driven architecture.
The point is that a reviewer reads this in a minute and can say exactly what the system will
and won't do.
"""
from datetime import date
from typing import Literal

from pydantic import BaseModel

from config import AUTO_APPROVE_MAX_USD, CLAIM_WINDOW_DAYS, REFUND_CAP_FRACTION
from harness.validity import get_order, verify_claim
from schemas import CustomerClaim

# The demo clock. Olist is real 2016-2018 data, so wall-clock time would put every order years
# out of window and rule 3 would deny everything — a demo where nothing is ever allowed proves
# nothing. Pinned just after the dataset's last order. This is config, not a fudge: the eval and
# the UI both display it, and the claim-window rule stays genuinely exercised because the data
# spans ~2 years, so plenty of orders still fall outside 90 days of this date.
NOW = date(2018, 10, 20)

Action = Literal["allow", "deny", "escalate"]


class Decision(BaseModel):
    """The policy verdict on one proposed tool call.

    `rule_id` isn't decoration — it's what makes a denial auditable. "Denied by
    refund_exceeds_cap" is a fact you can test, count, and argue with. "The agent decided not
    to" is none of those things.

        allow    -> tools.py performs the mutation
        deny     -> tools.py refuses, and returns the reason to the model as a tool error
        escalate -> tools.py refuses the mutation and files it for a human instead

    deny and escalate must never be collapsed into one: deny means "this must not happen";
    escalate means "this may well be right, but not on the agent's authority."
    """

    action: Action
    rule_id: str
    reason: str


def _decide(action: Action, rule_id: str, reason: str) -> Decision:
    return Decision(action=action, rule_id=rule_id, reason=reason)


def check_refund(
    order_id: str, claim_type: str, amount_usd: float, history: list[dict], caller_id: str | None = None,
) -> Decision:
    """Should this refund be issued? The single authority on that question.

    `history` is the audit trail so far (harness/audit.py records) — the prior tool calls for
    this case. It's passed in rather than read from disk here so this function stays pure: same
    inputs, same verdict, always. That's what makes it testable, and what makes a pass^k eval
    reproducible instead of dependent on whatever happens to be on disk.

    `caller_id` is who is asking — the customer_id a real deployment would attach to an
    authenticated session, threaded here from api/main.py. Defaulting to None is not "skip the
    check": None means no identity was claimed, and rule 2 denies that exactly like a mismatch.
    """
    # --- Rule 0: the amount must be a positive number. --------------------------------------
    # Input validation at the trust boundary, before any rule that needs the order record. A
    # $0.00 refund is a no-op that still burns the order's one allowed refund — rule 4 would then
    # deny the real one — and a NEGATIVE refund is a charge to the customer wearing a refund's
    # name. Neither is a judgment call, so neither belongs below a lookup.
    #
    # Found by the degenerate-input tests in tests/test_policy.py, not in production: the agent
    # never proposes a negative amount, so no sweep would ever have surfaced it. That is the
    # argument for testing boundaries rather than only the path the happy case walks.
    if amount_usd <= 0:
        return _decide("deny", "invalid_amount", f"${amount_usd:.2f} is not a positive refund amount.")

    # --- Rule 1: the order must exist. ------------------------------------------------------
    # First, because every rule below reads the order record. A customer citing an order we've
    # never heard of is the most common adversarial opening — they misremember, or they're
    # fishing. Either way there's nothing to reason about yet.
    order = get_order(order_id)
    if not order:
        return _decide("deny", "unknown_order", f"Order {order_id} is not in our records.")

    # --- Rule 2: the caller must be the order's owner. --------------------------------------
    # Ownership, not merit — checked before the claim is even read. A caller with no stated
    # connection to this order has no standing to discuss it, true claim or not (see
    # tests/test_hallucination_collision.py: an extractor that hallucinates a real order ID whose
    # true situation happens to match the guessed claim type used to pay out to whoever asked,
    # because nothing here compared the CALLER to the order, only the CLAIM to the record).
    if caller_id != order.get("customer_id"):
        return _decide(
            "deny", "caller_not_order_owner",
            "This order does not belong to the caller on record.",
        )

    # --- Rule 3: the claim must be TRUE against the record. ---------------------------------
    # Delegates to the same verify_claim() that gates the UI and scores the RLVR reward: one
    # definition of "did this actually happen", used in training, in production, and here. This
    # is the rule an insistent customer hits — saying "it was late" firmly, twice, does not move
    # a delivery date.
    finding = verify_claim(CustomerClaim(order_id=order_id, claim_type=claim_type))
    if not finding.claim_true:
        return _decide("deny", "claim_not_supported", finding.reason)

    # --- Rule 4: the claim must be inside the window. ---------------------------------------
    # Checked even when the claim is TRUE. A real late delivery from two years ago is still out
    # of window — merit and eligibility are different questions, and conflating them is exactly
    # how "but it REALLY was late" talks a system into paying.
    age_days = (NOW - date.fromisoformat(order["promised_date"])).days
    if age_days > CLAIM_WINDOW_DAYS:
        return _decide(
            "deny",
            "outside_claim_window",
            f"Order is {age_days} days old; claims close after {CLAIM_WINDOW_DAYS} days.",
        )

    # --- Rule 5: never refund the same order twice. -----------------------------------------
    # The stateless blind spot, and the reason `history` exists at all. Nothing about the second
    # request is wrong on its face: same order, same true claim, same permissible amount. Only
    # the past makes it wrong. An agent with no memory of its own actions pays twice without
    # hesitating, and a customer who asks twice is not hypothetical.
    prior = [h for h in history if h["tool"] == "issue_refund" and h["order_id"] == order_id and h["ok"]]
    if prior:
        paid = prior[0]["args"].get("amount_usd", 0.0)
        return _decide("deny", "already_refunded", f"Order {order_id} was already refunded (${paid:.2f}).")

    # --- Rule 6: the amount must be within the cap for this claim type. ---------------------
    # The cap is a fraction of what the customer actually PAID, read from the order record —
    # never from the amount they stated, which is precisely the number an adversarial user
    # inflates. late_delivery caps at 25% because they received the goods: the harm is the
    # delay, not the price. Without this rule, "so sorry, here's your money back" is an option
    # the model can be talked into.
    cap = round(order["amount_usd"] * REFUND_CAP_FRACTION[claim_type], 2)
    if amount_usd > cap:
        return _decide(
            "deny",
            "refund_exceeds_cap",
            f"${amount_usd:.2f} exceeds the ${cap:.2f} cap for {claim_type} "
            f"(order value ${order['amount_usd']:.2f}).",
        )

    # --- Rule 7: large refunds need a human. ------------------------------------------------
    # Last, and an escalate rather than a deny: every rule above already agreed the refund is
    # legitimate, so what's left is a question of authority, not merit. Sized so the agent
    # clears the long tail unattended and a person always sees the large money.
    if amount_usd > AUTO_APPROVE_MAX_USD:
        return _decide(
            "escalate",
            "over_auto_approve_limit",
            f"${amount_usd:.2f} is over the ${AUTO_APPROVE_MAX_USD:.2f} auto-approve limit.",
        )

    return _decide("allow", "within_policy", f"${amount_usd:.2f} refund for a verified {claim_type} claim.")


def demo() -> None:
    """One runnable check per rule, against the real data/db/orders.json rather than fixtures —
    so this also fails loudly if the DB is ever regenerated into a shape the rules can't read.
    """
    orders = [o for o in (get_order(f"ORD-{i}") for i in range(1000, 1460)) if o]

    def _age(o: dict) -> int:
        return (NOW - date.fromisoformat(o["promised_date"])).days

    on_time = next(o for o in orders if o["situation"] == "on_time")
    fresh = next(o for o in orders if o["situation"] == "late" and _age(o) <= CLAIM_WINDOW_DAYS)
    stale = next(o for o in orders if o["situation"] == "late" and _age(o) > CLAIM_WINDOW_DAYS)
    owner = fresh["customer_id"]  # the real owner — every "allow"/non-identity denial below must
                                   # supply this, or rule 2 fires first and the test proves nothing
                                   # about the rule it claims to.

    # 0 — a non-positive amount is rejected before the order is even looked up
    assert check_refund(fresh["order_id"], "late_delivery", 0.0, [], owner).rule_id == "invalid_amount"
    assert check_refund(fresh["order_id"], "late_delivery", -5.0, [], owner).rule_id == "invalid_amount"

    # 1 — an order we've never heard of
    assert check_refund("ORD-999999", "late_delivery", 10.0, [], "anyone").rule_id == "unknown_order"

    # 2 — a caller with no stated connection to a real order is denied before its claim is even
    # read, whether or not that claim happens to be true. The whole reason this rule exists.
    assert check_refund(fresh["order_id"], "late_delivery", 5.0, [], "some_other_customer").rule_id == "caller_not_order_owner"
    assert check_refund(fresh["order_id"], "late_delivery", 5.0, []).rule_id == "caller_not_order_owner", (
        "no caller_id at all must deny exactly like a mismatch — there is no bypassing this rule"
    )

    # 3 — an on-time order cannot support a late_delivery claim, however insistently it's asked
    assert check_refund(on_time["order_id"], "late_delivery", 10.0, [], on_time["customer_id"]).rule_id == "claim_not_supported"

    # 4 — genuinely late, but too old: true is not the same as eligible
    assert check_refund(stale["order_id"], "late_delivery", 1.0, [], stale["customer_id"]).rule_id == "outside_claim_window"

    # 5 — the identical legitimate request, a second time
    hist = [{"tool": "issue_refund", "order_id": fresh["order_id"], "ok": True, "args": {"amount_usd": 5.0}}]
    assert check_refund(fresh["order_id"], "late_delivery", 5.0, hist, owner).rule_id == "already_refunded"

    # 6 — asking for the whole order value on a late delivery (capped at 25%)
    assert check_refund(fresh["order_id"], "late_delivery", fresh["amount_usd"], [], owner).rule_id == "refund_exceeds_cap"

    # 7 vs allow — the same verified claim on either side of the auto-approve limit.
    # This fixture must clear rules 1-6 to reach rule 7 at all, which means in-window too: the
    # first draft picked on amount alone and got denied by the claim window (158 days old), so
    # rule 7 never ran and the assert failed for the wrong reason. Uses never_arrived (cap 1.0)
    # so the full order value can straddle the limit — a 25%-capped late_delivery can't.
    big = next(
        o for o in orders
        if o["situation"] == "never_arrived"
        and o["amount_usd"] > AUTO_APPROVE_MAX_USD
        and _age(o) <= CLAIM_WINDOW_DAYS
    )
    big_owner = big["customer_id"]
    assert check_refund(big["order_id"], "never_arrived", AUTO_APPROVE_MAX_USD + 0.01, [], big_owner).action == "escalate"
    assert check_refund(big["order_id"], "never_arrived", AUTO_APPROVE_MAX_USD, [], big_owner).action == "allow"

    print(f"policy demo OK — 8 rules, clock={NOW}, {len(orders)} orders loaded")


if __name__ == "__main__":
    demo()
