"""
# resolv/harness/tools.py

**What:** the three tools the agent can call (lookup_order, issue_refund, escalate_to_human), the refund saga
behind issue_refund, `reconcile_once` for refunds whose outcome is unknown, and `record_invalid_call`.
**Why:** this is the only code that changes anything. Every refund calls policy.check_refund first, inside one
ledger transaction, so the check cannot be skipped or raced. A refusal comes back as a plain string the model can
explain, never an exception, and every attempt is recorded, malformed ones included, with the caller on it.
Money moves as a saga: reserve a pending refund (local, undoable), call the payment provider (the point of no
return), then settle, compensate on a decline, or park as `unknown` for the reconciler.
Two operator switches sit between the policy verdict and the money: REFUNDS_FROZEN=1 (kill switch) turns every
refund policy would allow into an escalation (rule_id `refunds_frozen`), and RESOLV_SHADOW=1 records the allowed
refund as `shadow_mode` ("would have refunded $X") and moves no money. Both are read on every call; denials are
unchanged by either.
**Used by:** resolv/agents/loop.py, resolv/api/escalations.py, evals/*.
"""
import os
import sqlite3
import uuid

from resolv.config import REFUND_CAP_FRACTION
from resolv.harness import audit, ledger
from resolv.harness.policy import Decision, action_hash, check_refund, refundable
from resolv.harness.validity import get_order
from resolv.integrations import payments

MAX_RECONCILE_ATTEMPTS = 3
_PENDING_TEXT = "Refund submitted; confirmation pending. Tell the customer it is processing, not that it is done."
_DECLINED_TEXT = "Refund could not be processed by the payment provider; escalated to a person."
_SHADOW_TEXT = "Not refunded — sent to a human for review."


# True when RESOLV_SHADOW=1. Refused in test and eval runs, so a score can never silently measure a dry run.
def _shadow() -> bool:
    if os.environ.get("RESOLV_SHADOW") != "1":
        return False
    if os.environ.get("RESOLV_ENV") in ("test", "eval"):
        raise RuntimeError("RESOLV_SHADOW=1 is not allowed when RESOLV_ENV is test or eval.")
    return True


# Read an order the caller owns. Missing and foreign orders return identical text, so IDs cannot be probed.
def lookup_order(case_id: str, order_id: str, caller_id: str | None) -> str:
    order = get_order(order_id) if isinstance(order_id, str) else {}
    owned = bool(order) and order.get("customer_id") == caller_id
    if owned:
        result = (f"Order {order_id}: paid ${order['amount_usd']:.2f}, promised {order['promised_date']}, "
                  f"delivered {order['delivered_date'] or 'never'}, status {order['status']}.")
        if os.environ.get("SHOW_MAX_REFUNDABLE") == "1":  # one-variable A/B arm, off by default
            result += " " + _refundable_text(case_id, order_id, caller_id)
        rule_id = "lookup_hit"
    else:
        result = f"Order {order_id} not found on your account. Ask the customer to re-check the number."
        rule_id = "lookup_not_owner" if order else "lookup_miss"
    # Reads are recorded: whether the agent looked before it acted is the most useful fact about a run.
    decision = Decision(action="allow", rule_id=rule_id, reason=f"Read order {order_id}.")
    audit.append(case_id, "lookup_order", order_id, {}, decision, owned, result, caller_id)
    return result


# The "Refundable now: ..." sentence lookup_order adds when SHOW_MAX_REFUNDABLE=1.
def _refundable_text(case_id: str, order_id: str, caller_id: str | None) -> str:
    history = audit.read(case_id) + audit.order_history(order_id)
    parts = [f"{ct} up to ${v['max_usd']:.2f}" if v["max_usd"] else f"{ct}: not eligible ({v['reason'].rstrip('.')})"
             for ct, v in refundable(order_id, caller_id, history).items()]
    return "Refundable now: " + "; ".join(parts) + "."


# Record a malformed or unknown tool call as a deny/invalid_request attempt and return the refusal text.
def record_invalid_call(case_id: str, tool_name, raw, reason: str, caller_id: str | None = None) -> str:
    result = f"Refused: {reason}"
    order_id = raw.get("order_id") if isinstance(raw, dict) and isinstance(raw.get("order_id"), str) else "unknown"
    decision = Decision(action="deny", rule_id="invalid_request", reason=reason)
    audit.append(case_id, tool_name, order_id, {"raw": raw}, decision, False, result, caller_id)
    return result


# Why these refund arguments are malformed, or None. The amount is left to policy rule 0.
def _bad_args(order_id, claim_type) -> str | None:
    if not isinstance(order_id, str) or not order_id.strip():
        return f"order_id must be a non-empty string, got {order_id!r}."
    if not isinstance(claim_type, str) or claim_type not in REFUND_CAP_FRACTION:
        return f"claim_type must be one of {sorted(REFUND_CAP_FRACTION)}, got {claim_type!r}."
    return None


# Refund money. Check, record, reserve in one transaction; then pay, then settle or compensate.
def issue_refund(case_id: str, order_id: str, claim_type: str, amount_usd: float, caller_id: str | None,
                 approval: dict | None = None, approver_id: str | None = None,
                 idempotency_key: str | None = None) -> str:
    args = {"amount_usd": amount_usd, "claim_type": claim_type}
    problem = _bad_args(order_id, claim_type)
    if problem:
        return record_invalid_call(case_id, "issue_refund", args | {"order_id": order_id}, problem, caller_id)
    key = idempotency_key or uuid.uuid4().hex

    with ledger.tx() as conn:  # BEGIN IMMEDIATE: one writer at a time, across threads and processes
        prior = ledger.refund_by_key(key, conn)
        if prior:  # a replayed delivery of a call that already ran: same outcome, no second effect
            result = _status_text(prior)
            replay = Decision(action="allow", rule_id="idempotent_replay", reason=f"Key {key} already used.")
            ledger.append_audit(conn, case_id, "issue_refund", order_id, args, replay, False, result, caller_id, approver_id)
            return result

        if approval:  # re-read inside the lock, and only the approver who gave it may use it
            approval = ledger.escalation(approval.get("id"), conn)
            if approval and approval["decided_by"] != approver_id:
                approval = None
        history = ledger.case_trail(case_id, conn) + ledger.order_refund_records(order_id, conn)
        decision = check_refund(order_id, claim_type, amount_usd, history, caller_id, approval)
        if decision.action == "allow" and os.environ.get("REFUNDS_FROZEN") == "1":  # kill switch: a person decides
            decision = Decision(action="escalate", rule_id="refunds_frozen",
                                reason=f"Refunds are frozen; policy said {decision.rule_id}: {decision.reason}")
        if decision.action == "allow" and _shadow():  # record what would have been paid, move no money
            shadow = Decision(action="escalate", rule_id="shadow_mode",
                              reason=f"Shadow mode: would have refunded ${amount_usd:.2f} ({decision.rule_id}).")
            ledger.append_audit(conn, case_id, "issue_refund", order_id, args, shadow, False, _SHADOW_TEXT,
                                caller_id, approver_id)
            return _SHADOW_TEXT

        if decision.action != "allow":
            if decision.action == "escalate":
                result = f"Not refunded — sent to a human for approval. {decision.reason}"
            else:
                result = f"Refused: {decision.reason}"
            ledger.append_audit(conn, case_id, "issue_refund", order_id, args, decision, False, result, caller_id, approver_id)
            if decision.action == "escalate":  # reaches a human whether or not the model asks for one
                action_json, digest = action_hash(order_id, claim_type, amount_usd, caller_id)
                _escalate(conn, case_id, order_id, decision.reason, caller_id, "approval", action_json, digest, decision.rule_id)
            return result

        cents = round(amount_usd * 100)
        authorized = ledger.append_audit(conn, case_id, "authorize_refund", order_id, args, decision, True,
                                         f"Authorized ${amount_usd:.2f}; sending to the payment provider.",
                                         caller_id, approver_id)
        try:
            refund_id = ledger.insert_refund(conn, key, order_id, case_id, claim_type, cents, authorized["seq"],
                                             approval["id"] if approval else None)
        except sqlite3.IntegrityError:  # backstop under rule 5; should never fire
            blocked = Decision(action="deny", rule_id="duplicate_refund_blocked_by_db",
                               reason=f"Order {order_id} already has a live refund.")
            result = f"Refused: {blocked.reason}"
            ledger.append_audit(conn, case_id, "issue_refund", order_id, args, blocked, False, result, caller_id, approver_id)
            return result
        if approval:
            ledger.move_escalation(conn, approval["id"], "approved", "executed")

    return _pay(refund_id, key, case_id, order_id, args, cents, decision, caller_id, approver_id)


# Call the provider for a reserved refund and record what happened: settled, declined (compensated) or unknown.
def _pay(refund_id, key, case_id, order_id, args, cents, decision, caller_id, approver_id) -> str:
    try:
        receipt = payments.refund(key, order_id, cents)
    except payments.Declined as e:
        with ledger.tx() as conn:  # compensation: free the order's refund slot, hand it to a person
            ledger.set_refund_status(conn, refund_id, "failed")
            declined = Decision(action="deny", rule_id="payment_declined", reason=str(e))
            ledger.append_audit(conn, case_id, "issue_refund", order_id, args, declined, False, _DECLINED_TEXT, caller_id, approver_id)
            _escalate(conn, case_id, order_id, f"Payment declined: {e}", caller_id, "review")
        return _DECLINED_TEXT
    except Exception as e:  # timeout or 5xx: the money may or may not have moved; the reconciler finds out
        with ledger.tx() as conn:
            ledger.set_refund_status(conn, refund_id, "unknown")
            unknown = Decision(action="allow", rule_id="payment_unknown", reason=f"{type(e).__name__}: {e}")
            ledger.append_audit(conn, case_id, "issue_refund", order_id, args, unknown, False, _PENDING_TEXT, caller_id, approver_id)
        return _PENDING_TEXT

    result = f"Refunded ${cents / 100:.2f} on {order_id}."
    with ledger.tx() as conn:
        _settle(conn, refund_id, case_id, order_id, cents, receipt["psp_ref"])
        ledger.append_audit(conn, case_id, "issue_refund", order_id, args, decision, True, result, caller_id, approver_id)
    return result


# Mark a refund settled and queue the customer's confirmation in the same transaction; False if already resolved.
def _settle(conn, refund_id, case_id, order_id, cents, psp_ref) -> bool:
    if not ledger.set_refund_status(conn, refund_id, "settled", psp_ref):
        return False
    ledger.add_outbox(conn, "customer_refund_confirmation",
                      {"case_id": case_id, "order_id": order_id, "amount_usd": cents / 100, "psp_ref": psp_ref})
    return True


# The tool result for a refund row's current status (used when a key is replayed).
def _status_text(refund: dict) -> str:
    if refund["status"] == "settled":
        return f"Refunded ${refund['amount_cents'] / 100:.2f} on {refund['order_id']}."
    if refund["status"] in ("pending", "unknown"):
        return _PENDING_TEXT
    return _DECLINED_TEXT if refund["status"] == "failed" else f"Refund {refund['status']}."


# File an escalation and its trail record inside the caller's transaction; None if the same action is already pending.
def _escalate(conn, case_id, order_id, reason, caller_id, kind, action_json=None, digest=None, rule_id=None) -> str | None:
    esc_id = ledger.create_escalation(conn, case_id, order_id, kind, reason, caller_id, action_json, digest, rule_id)
    if esc_id is None:
        return None
    result = f"Escalated {order_id} to a human agent: {reason}"
    decision = Decision(action="allow", rule_id="escalation_always_permitted", reason=str(reason))
    ledger.append_audit(conn, case_id, "escalate_to_human", order_id, {"reason": reason, "escalation_id": esc_id},
                        decision, True, result, caller_id)
    return result


# Hand the case to a person. Always allowed: gating the escape hatch leaves the agent nowhere to go.
def escalate_to_human(case_id: str, order_id: str, reason: str, caller_id: str | None = None) -> str:
    with ledger.tx() as conn:
        return _escalate(conn, case_id, order_id, str(reason), caller_id, "review")


# Resolve pending/unknown refunds older than min_age_s by asking the provider; returns how many changed state.
def reconcile_once(min_age_s: float = 30) -> int:
    changed = 0
    for r in ledger.refunds(("pending", "unknown"), older_than_s=min_age_s):
        key, psp_ref, outcome = r["idempotency_key"], None, "retry"
        found = payments.get(key)
        if found:
            outcome, psp_ref = "settled", found["psp_ref"]
        elif r["attempts"] >= MAX_RECONCILE_ATTEMPTS:
            outcome = "failed"  # safe: the provider has no refund under this key
        else:
            try:
                psp_ref, outcome = payments.refund(key, r["order_id"], r["amount_cents"])["psp_ref"], "settled"
            except payments.Declined:
                outcome = "failed"
            except Exception:
                outcome = "retry"
        with ledger.tx() as conn:
            if outcome == "retry":
                ledger.bump_attempts(conn, r["refund_id"])
                continue
            if outcome == "settled":
                moved = _settle(conn, r["refund_id"], r["case_id"], r["order_id"], r["amount_cents"], psp_ref)
            else:
                moved = ledger.set_refund_status(conn, r["refund_id"], "failed")
            if not moved:
                continue
            ok = outcome == "settled"
            decision = Decision(action="allow" if ok else "deny", rule_id=f"refund_{outcome}",
                                reason=f"Reconciled refund {r['refund_id']} with the payment provider.")
            ledger.append_audit(conn, r["case_id"], "reconcile", r["order_id"],
                                {"refund_id": r["refund_id"], "amount_usd": r["amount_cents"] / 100},
                                decision, ok, f"Refund {outcome}.", "system:reconcile")
            if not ok:
                _escalate(conn, r["case_id"], r["order_id"], "Refund could not be confirmed with the payment provider.",
                          "system:reconcile", "review")
            changed += 1
    return changed


# The agent's entire surface area. If it isn't in this list, the agent cannot do it.
TOOLS = [lookup_order, issue_refund, escalate_to_human]


# Prove no bypass, recorded denials, refusals as strings, and rule 5 reading what this file wrote.
def demo() -> None:
    from datetime import date

    from resolv.config import AUTO_APPROVE_MAX_USD, CLAIM_WINDOW_DAYS
    from resolv.harness.policy import NOW

    ledger.use_temp_db()  # never touches data/resolv.db
    case = "_demo_tools"
    orders = [o for o in (get_order(f"ORD-{i}") for i in range(1000, 1460)) if o]
    fresh = next(o for o in orders if o["situation"] == "late"
                 and (NOW - date.fromisoformat(o["promised_date"])).days <= CLAIM_WINDOW_DAYS)
    oid, owner = fresh["order_id"], fresh["customer_id"]

    assert f"${fresh['amount_usd']:.2f}" in lookup_order(case, oid, owner)
    assert audit.read(case)[-1]["rule_id"] == "lookup_hit"
    assert "not found" in lookup_order(case, "ORD-999999", owner)
    assert audit.read(case)[-1]["rule_id"] == "lookup_miss"
    assert "not found" in lookup_order(case, oid, "someone_else"), "another customer's order must not leak"

    assert issue_refund(case, oid, "late_delivery", 1.0, "someone_else").startswith("Refused:")
    assert audit.read(case)[-1]["rule_id"] == "caller_not_order_owner"
    assert issue_refund(case, oid, "late_delivery", fresh["amount_usd"], owner).startswith("Refused:")
    assert audit.read(case)[-1]["rule_id"] == "refund_exceeds_cap" and audit.read(case)[-1]["ok"] is False

    cap = round(fresh["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    amount = min(cap, AUTO_APPROVE_MAX_USD)
    assert issue_refund(case, oid, "late_delivery", amount, owner).startswith("Refunded")
    assert "already refunded" in issue_refund(case, oid, "late_delivery", amount, owner).lower()
    assert escalate_to_human(case, oid, "customer asked for a manager", owner).startswith("Escalated")

    # 3 lookups + 4 refund attempts (the paid one writes authorize + settle) + 1 escalation = 9 records.
    trail = audit.read(case)
    assert len(trail) == 9, f"expected 9 recorded attempts, got {len(trail)}"
    assert all(r["caller_id"] == owner or r["caller_id"] == "someone_else" for r in trail)
    assert audit.verify_chain(case) == (True, None)
    print(f"tools demo OK: {len(TOOLS)} tools, every mutation policy-checked, {len(trail)} attempts recorded")


if __name__ == "__main__":
    demo()
