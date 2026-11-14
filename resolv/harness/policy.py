"""
# resolv/harness/policy.py

**What:** every refund rule as one pure function, `check_refund`, plus `refundable` (what each claim type could get
right now) and `action_hash` (the exact refund call a human approval is bound to).
**Why:** the rules are an exact threshold table, so they live in Python, not in a prompt a model could misapply.
check_refund runs inside resolv/harness/tools.py, the only write path, so nothing can route around it. Eight rules (0-7)
run in order, first match wins, and each returns a rule_id the trail and UI can name. It reads history, which is
what lets rule 5 refuse a second refund that looks fine on its face. caller_id None is not "skip the check": it
means no identity was claimed, and rule 2 denies it like a mismatch.
**Used by:** resolv/harness/tools.py, evals/tasks.py, evals/runner.py, resolv/monitor.py.
"""
import hashlib
import json
import math
from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel

from resolv.config import AUTO_APPROVE_MAX_USD, CLAIM_WINDOW_DAYS, REFUND_CAP_FRACTION
from resolv.harness.validity import get_order, verify_claim
from resolv.schemas import CustomerClaim

# The demo clock. Olist is 2016-2018 data, so wall-clock time would put every order out of window. Pinned just
# after the dataset's last order; the data spans ~2 years, so the claim-window rule is still exercised.
NOW = date(2018, 10, 20)

Action = Literal["allow", "deny", "escalate"]


# The verdict on one proposed tool call: allow (mutate), deny (must not happen), escalate (a human decides).
class Decision(BaseModel):
    action: Action
    rule_id: str
    reason: str


# Shorthand for building a Decision.
def _decide(action: Action, rule_id: str, reason: str) -> Decision:
    return Decision(action=action, rule_id=rule_id, reason=reason)


# Canonical JSON of one concrete refund call and its sha256: exactly what a human approves, nothing else.
def action_hash(order_id: str, claim_type: str, amount_usd: float, caller_id: str | None) -> tuple[str, str]:
    action = {"tool": "issue_refund", "order_id": order_id, "claim_type": claim_type,
              "amount_cents": round(amount_usd * 100), "caller_id": caller_id}
    text = json.dumps(action, sort_keys=True, separators=(",", ":"))
    return text, hashlib.sha256(text.encode()).hexdigest()


# True when an approval is for exactly this call, still approved, unexpired, and not given by the caller.
def _approval_valid(approval: dict | None, order_id, claim_type, amount_usd, caller_id) -> bool:
    return bool(approval) \
        and approval.get("status") == "approved" \
        and approval.get("action_hash") == action_hash(order_id, claim_type, amount_usd, caller_id)[1] \
        and approval.get("expires_at", "") > datetime.now(timezone.utc).isoformat() \
        and approval.get("decided_by") not in (None, "", caller_id)


# Should this refund be issued? The single authority. Pure: same inputs, same verdict.
def check_refund(
    order_id: str, claim_type: str, amount_usd: float, history: list[dict], caller_id: str | None = None,
    approval: dict | None = None,
) -> Decision:
    # --- Rule 0: a well-formed request. ---------------------------------------------------------------
    # NaN compares False with everything, so `amount <= 0` alone let NaN through every later rule. bool is an
    # int subclass; True must not become a $1 refund. An off-enum claim_type has no cap, so it stops here too.
    if (not isinstance(amount_usd, (int, float)) or isinstance(amount_usd, bool)
            or not math.isfinite(amount_usd) or amount_usd <= 0):
        return _decide("deny", "invalid_amount", f"{amount_usd!r} is not a positive, finite refund amount.")
    if not isinstance(claim_type, str) or claim_type not in REFUND_CAP_FRACTION:
        return _decide("deny", "invalid_request",
                       f"claim_type must be one of {sorted(REFUND_CAP_FRACTION)}, got {claim_type!r}.")

    # --- Rule 1: the order must exist. ----------------------------------------------------------------
    order = get_order(order_id) if isinstance(order_id, str) else {}
    if not order:
        return _decide("deny", "unknown_order", f"Order {order_id} is not in our records.")

    # --- Rule 2: the caller must own the order. -------------------------------------------------------
    # Checked before the claim is read: a true claim from someone with no connection to the order is still denied.
    if caller_id != order.get("customer_id"):
        return _decide("deny", "caller_not_order_owner", "This order does not belong to the caller on record.")

    # --- Rule 3: the claim must be true against the record. -------------------------------------------
    # The same verify_claim that scores the extractor; saying "it was late" twice does not move a delivery date.
    finding = verify_claim(CustomerClaim(order_id=order_id, claim_type=claim_type))
    if not finding.claim_true:
        return _decide("deny", "claim_not_supported", finding.reason)

    # --- Rule 4: the claim must be inside the window. -------------------------------------------------
    age_days = (NOW - date.fromisoformat(order["promised_date"])).days
    if age_days > CLAIM_WINDOW_DAYS:
        return _decide("deny", "outside_claim_window",
                       f"Order is {age_days} days old; claims close after {CLAIM_WINDOW_DAYS} days.")

    # --- Rule 5: never refund the same order twice. ---------------------------------------------------
    # Only the past makes the second request wrong, which is why history is an input.
    prior = [h for h in history if h["tool"] == "issue_refund" and h["order_id"] == order_id and h["ok"]]
    if prior:
        paid = prior[0]["args"].get("amount_usd", 0.0)
        return _decide("deny", "already_refunded", f"Order {order_id} was already refunded (${paid:.2f}).")

    # --- Rule 6: within the cap for this claim type. --------------------------------------------------
    # A fraction of what the record says was PAID, never of the amount the customer states.
    cap = round(order["amount_usd"] * REFUND_CAP_FRACTION[claim_type], 2)
    if amount_usd > cap:
        return _decide("deny", "refund_exceeds_cap",
                       f"${amount_usd:.2f} exceeds the ${cap:.2f} cap for {claim_type} "
                       f"(order value ${order['amount_usd']:.2f}).")

    # --- Rule 7: large refunds need a human. ----------------------------------------------------------
    # Authority, not merit: rules 0-6 already passed. A valid approval bound to this exact call lets it through.
    # Once a refund on this order went to a human, a smaller one may not slip under the limit while they decide.
    waiting = any(h["tool"] == "issue_refund" and h["order_id"] == order_id
                  and h.get("rule_id") == "over_auto_approve_limit" for h in history)
    if amount_usd > AUTO_APPROVE_MAX_USD or waiting:
        if _approval_valid(approval, order_id, claim_type, amount_usd, caller_id):
            return _decide("allow", "approved_by_human",
                           f"${amount_usd:.2f} approved by {approval['decided_by']} ({approval.get('id')}).")
        if amount_usd <= AUTO_APPROVE_MAX_USD:
            return _decide("escalate", "over_auto_approve_limit",
                           f"A larger refund on {order_id} is already waiting for a human; a smaller one cannot "
                           "skip that review.")
        return _decide("escalate", "over_auto_approve_limit",
                       f"${amount_usd:.2f} is over the ${AUTO_APPROVE_MAX_USD:.2f} auto-approve limit.")

    return _decide("allow", "within_policy", f"${amount_usd:.2f} refund for a verified {claim_type} claim.")


# What each claim type could be refunded right now: {claim_type: {"max_usd": cap or None, "reason": ...}}.
# Reuses check_refund at the cap, so no rule is duplicated. Revealing this is safe: the gate re-checks every refund.
def refundable(order_id: str, caller_id: str | None, history: list[dict]) -> dict[str, dict]:
    order = get_order(order_id)
    out = {}
    for claim_type, fraction in REFUND_CAP_FRACTION.items():
        cap = round(order.get("amount_usd", 0.0) * fraction, 2)
        d = check_refund(order_id, claim_type, cap, history, caller_id)
        out[claim_type] = {"max_usd": cap if d.action in ("allow", "escalate") else None, "reason": d.reason}
    return out


# One runnable check per rule against the real data/db/orders.json.
def demo() -> None:
    orders = [o for o in (get_order(f"ORD-{i}") for i in range(1000, 1460)) if o]

    # Days between the order's promised date and the demo clock.
    def _age(o: dict) -> int:
        return (NOW - date.fromisoformat(o["promised_date"])).days

    on_time = next(o for o in orders if o["situation"] == "on_time")
    fresh = next(o for o in orders if o["situation"] == "late" and _age(o) <= CLAIM_WINDOW_DAYS)
    stale = next(o for o in orders if o["situation"] == "late" and _age(o) > CLAIM_WINDOW_DAYS)
    oid, owner = fresh["order_id"], fresh["customer_id"]

    for bad in (0.0, -5.0, float("nan"), float("inf"), "50", None, True):
        assert check_refund(oid, "late_delivery", bad, [], owner).rule_id == "invalid_amount", bad
    assert check_refund(oid, "store_credit", 5.0, [], owner).rule_id == "invalid_request"
    assert check_refund("ORD-999999", "late_delivery", 10.0, [], "anyone").rule_id == "unknown_order"
    assert check_refund(oid, "late_delivery", 5.0, [], "someone_else").rule_id == "caller_not_order_owner"
    assert check_refund(oid, "late_delivery", 5.0, []).rule_id == "caller_not_order_owner"
    assert check_refund(on_time["order_id"], "late_delivery", 10.0, [], on_time["customer_id"]).rule_id == "claim_not_supported"
    assert check_refund(stale["order_id"], "late_delivery", 1.0, [], stale["customer_id"]).rule_id == "outside_claim_window"
    hist = [{"tool": "issue_refund", "order_id": oid, "ok": True, "args": {"amount_usd": 5.0}}]
    assert check_refund(oid, "late_delivery", 5.0, hist, owner).rule_id == "already_refunded"
    assert check_refund(oid, "late_delivery", fresh["amount_usd"], [], owner).rule_id == "refund_exceeds_cap"

    # Rule 7 needs an order that clears rules 0-6 above the limit: never_arrived (cap 1.0), in window.
    big = next(o for o in orders if o["situation"] == "never_arrived"
               and o["amount_usd"] > AUTO_APPROVE_MAX_USD and _age(o) <= CLAIM_WINDOW_DAYS)
    bid, bown, over = big["order_id"], big["customer_id"], AUTO_APPROVE_MAX_USD + 0.01
    assert check_refund(bid, "never_arrived", over, [], bown).action == "escalate"
    assert check_refund(bid, "never_arrived", AUTO_APPROVE_MAX_USD, [], bown).action == "allow"

    # An approval bound to this exact call allows it; any other amount, or self-approval, escalates again.
    approval = {"id": "ESC-demo", "status": "approved", "decided_by": "approver-1", "expires_at": "9999-01-01",
                "action_hash": action_hash(bid, "never_arrived", over, bown)[1]}
    assert check_refund(bid, "never_arrived", over, [], bown, approval).rule_id == "approved_by_human"
    assert check_refund(bid, "never_arrived", over + 1, [], bown, approval).action == "escalate"
    assert check_refund(bid, "never_arrived", over, [], bown, approval | {"decided_by": bown}).action == "escalate"

    assert refundable(oid, owner, [])["late_delivery"]["max_usd"] == round(fresh["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    assert refundable(oid, owner, [])["never_arrived"]["max_usd"] is None
    print(f"policy demo OK: 8 rules, clock={NOW}, {len(orders)} orders loaded")


if __name__ == "__main__":
    demo()
