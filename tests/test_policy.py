"""
# tests/test_policy.py

**What:** every refund rule in resolv/harness/policy.py, one passing and one failing case each, plus degenerate amounts,
exact threshold boundaries, and which rule answers when a request breaks two at once.
**Why:** a rule that silently changes must fail CI. The tests read the real data/db/orders.json on purpose, so they
cannot drift from the data the rules run on. Every call passes the order's real customer_id as caller_id, except
the rule 2 tests, which exist to show what happens when it doesn't match.
**Used by:** pytest.
"""
from datetime import date

import pytest

from resolv.config import AUTO_APPROVE_MAX_USD, CLAIM_WINDOW_DAYS, REFUND_CAP_FRACTION
from resolv.harness.policy import NOW, check_refund
from resolv.harness.validity import get_order

_ORDERS = [o for o in (get_order(f"ORD-{i}") for i in range(1000, 1460)) if o]


# Days since the order's promised date, on the pinned demo clock.
def _age(o: dict) -> int:
    return (NOW - date.fromisoformat(o["promised_date"])).days


# First order matching all conditions, or skip (a smaller regenerated DB skips instead of passing vacuously).
def _pick(**cond):
    for o in _ORDERS:
        if cond.get("situation") and o["situation"] != cond["situation"]:
            continue
        if cond.get("in_window") is True and _age(o) > CLAIM_WINDOW_DAYS:
            continue
        if cond.get("in_window") is False and _age(o) <= CLAIM_WINDOW_DAYS:
            continue
        if cond.get("over_limit") and o["amount_usd"] <= AUTO_APPROVE_MAX_USD:
            continue
        return o
    pytest.skip(f"no order in data/db matches {cond}")


# The real order DB is present and non-trivial.
def test_db_loaded():
    assert len(_ORDERS) > 100, "the real order DB should be present and non-trivial"


# Rule 1: an unknown order is denied.
def test_rule1_unknown_order_denied():
    assert check_refund("ORD-999999", "late_delivery", 10.0, [], "anyone").rule_id == "unknown_order"


# Rule 2: a caller who doesn't own the order is denied, even when the claim is true.
def test_rule2_caller_mismatch_denied_even_when_claim_is_true():
    fresh = _pick(situation="late", in_window=True)
    d = check_refund(fresh["order_id"], "late_delivery", 5.0, [], "not_the_owner")
    assert d.rule_id == "caller_not_order_owner" and d.action == "deny"


# Rule 2: no caller id is denied exactly like a mismatch.
def test_rule2_no_caller_id_denied_same_as_a_mismatch():
    fresh = _pick(situation="late", in_window=True)
    assert check_refund(fresh["order_id"], "late_delivery", 5.0, []).rule_id == "caller_not_order_owner"


# Rule 3: an on-time order cannot support a late_delivery claim, however it's phrased.
def test_rule3_untrue_claim_denied():
    on_time = _pick(situation="on_time")
    d = check_refund(on_time["order_id"], "late_delivery", 10.0, [], on_time["customer_id"])
    assert d.rule_id == "claim_not_supported"


# Rule 4: an order outside the claim window is denied.
def test_rule4_out_of_window_denied():
    stale = _pick(situation="late", in_window=False)
    d = check_refund(stale["order_id"], "late_delivery", 1.0, [], stale["customer_id"])
    assert d.rule_id == "outside_claim_window" and d.action == "deny"


# Rule 5: a second refund on the same order is denied.
def test_rule5_double_refund_denied():
    fresh = _pick(situation="late", in_window=True)
    history = [{"tool": "issue_refund", "order_id": fresh["order_id"], "ok": True, "args": {"amount_usd": 5.0}}]
    d = check_refund(fresh["order_id"], "late_delivery", 5.0, history, fresh["customer_id"])
    assert d.rule_id == "already_refunded"


# Rule 6: the full order value on a late delivery (capped at 25%) is refused.
def test_rule6_over_cap_denied():
    fresh = _pick(situation="late", in_window=True)
    d = check_refund(fresh["order_id"], "late_delivery", fresh["amount_usd"], [], fresh["customer_id"])
    assert d.rule_id == "refund_exceeds_cap" and d.action == "deny"


# Rule 7: over the auto-approve limit escalates (never_arrived has cap 1.0, so the value can straddle the limit).
def test_rule7_over_limit_escalates_not_denies():
    big = _pick(situation="never_arrived", in_window=True, over_limit=True)
    d = check_refund(big["order_id"], "never_arrived", AUTO_APPROVE_MAX_USD + 0.01, [], big["customer_id"])
    assert d.action == "escalate" and d.rule_id == "over_auto_approve_limit"


# Rule 7: after an over-limit refund on an order went to a human, a smaller one under the limit escalates too.
# Found in error analysis: the agent lowered a $248.90 escalated refund to $200 and it was paid with no human.
def test_rule7_smaller_refund_cannot_skip_a_pending_approval():
    big = _pick(situation="never_arrived", in_window=True, over_limit=True)
    oid, owner = big["order_id"], big["customer_id"]
    pending = [{"tool": "issue_refund", "order_id": oid, "ok": False, "action": "escalate",
                "rule_id": "over_auto_approve_limit", "args": {"amount_usd": AUTO_APPROVE_MAX_USD + 0.01}}]
    assert check_refund(oid, "never_arrived", AUTO_APPROVE_MAX_USD, [], owner).action == "allow"
    d = check_refund(oid, "never_arrived", AUTO_APPROVE_MAX_USD, pending, owner)
    assert d.action == "escalate" and d.rule_id == "over_auto_approve_limit"


# A verified claim within cap and limit is allowed.
def test_allow_within_policy():
    fresh = _pick(situation="late", in_window=True)
    cap = round(fresh["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    amount = min(cap, AUTO_APPROVE_MAX_USD)
    d = check_refund(fresh["order_id"], "late_delivery", amount, [], fresh["customer_id"])
    assert d.action == "allow" and d.rule_id == "within_policy"


# The cap is a fraction of what the record says was paid; the customer's figure never enters.
def test_cap_uses_record_not_stated_amount():
    fresh = _pick(situation="late", in_window=True)
    cap = round(fresh["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    owner = fresh["customer_id"]
    assert check_refund(fresh["order_id"], "late_delivery", cap, [], owner).action == "allow"
    assert check_refund(fresh["order_id"], "late_delivery", cap + 0.01, [], owner).rule_id == "refund_exceeds_cap"


# --- Degenerate inputs (rule 0) --------------------------------------------------------------
# A $0 refund would burn the order's one refund (rule 5 then denies the real one); a negative one charges the customer.

# Zero and negative amounts are denied as invalid.
@pytest.mark.parametrize("amount", [0.0, -0.01, -5.0, -1000.0])
def test_non_positive_amounts_are_denied(amount):
    fresh = _pick(situation="late", in_window=True)
    decision = check_refund(fresh["order_id"], "late_delivery", amount, [], fresh["customer_id"])
    assert decision.action == "deny" and decision.rule_id == "invalid_amount"


# The boundary is at zero: a one-cent refund is still valid.
def test_smallest_positive_amount_is_allowed():
    fresh = _pick(situation="late", in_window=True)
    assert check_refund(fresh["order_id"], "late_delivery", 0.01, [], fresh["customer_id"]).action == "allow"


# --- Boundaries: the exact edge of every threshold -------------------------------------------

# Exactly at the cap is allowed; one cent over is not.
def test_exactly_at_cap_is_allowed_one_cent_over_is_not():
    fresh = _pick(situation="late", in_window=True)
    cap = round(fresh["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    owner = fresh["customer_id"]
    assert check_refund(fresh["order_id"], "late_delivery", cap, [], owner).rule_id == "within_policy"
    assert check_refund(fresh["order_id"], "late_delivery", round(cap + 0.01, 2), [], owner).rule_id == "refund_exceeds_cap"


# At the auto-approve limit is allowed; one cent over escalates (authority, not merit).
def test_exactly_at_auto_approve_limit_is_allowed_one_cent_over_escalates():
    big = _pick(situation="never_arrived", in_window=True, over_limit=True)
    owner = big["customer_id"]
    at = check_refund(big["order_id"], "never_arrived", AUTO_APPROVE_MAX_USD, [], owner)
    over = check_refund(big["order_id"], "never_arrived", AUTO_APPROVE_MAX_USD + 0.01, [], owner)
    assert at.action == "allow"
    assert over.action == "escalate" and over.rule_id == "over_auto_approve_limit"


# The oldest in-window order passes rule 4; the youngest out-of-window one does not.
def test_claim_window_boundary_separates_adjacent_orders():
    in_window = max((o for o in _ORDERS if o["situation"] == "late" and _age(o) <= CLAIM_WINDOW_DAYS),
                    key=_age, default=None)
    out_window = min((o for o in _ORDERS if o["situation"] == "late" and _age(o) > CLAIM_WINDOW_DAYS),
                     key=_age, default=None)
    if not in_window or not out_window:
        pytest.skip("need orders on both sides of the claim window")
    assert _age(in_window) <= CLAIM_WINDOW_DAYS < _age(out_window)
    d_in = check_refund(in_window["order_id"], "late_delivery", 1.0, [], in_window["customer_id"])
    d_out = check_refund(out_window["order_id"], "late_delivery", 1.0, [], out_window["customer_id"])
    assert d_in.rule_id != "outside_claim_window"
    assert d_out.rule_id == "outside_claim_window"


# --- First-match semantics: the rule ORDER is part of the contract ---------------------------
# Each request breaks two rules at once; the one that answers is the reason the customer is told.

# Rule 0 answers before rule 1.
def test_rule0_precedes_rule1_unknown_order_with_negative_amount():
    assert check_refund("ORD-999999", "late_delivery", -5.0, [], "anyone").rule_id == "invalid_amount"


# Rule 1 answers before rule 2: an unknown order is not an ownership problem.
def test_rule1_precedes_rule2_unknown_order_is_not_an_ownership_problem():
    assert check_refund("ORD-999999", "late_delivery", 10.0, [], "not_the_owner").rule_id == "unknown_order"


# Rule 2 answers before rule 6.
def test_rule2_precedes_rule6_wrong_caller_is_not_a_cap_problem():
    fresh = _pick(situation="late", in_window=True)
    d = check_refund(fresh["order_id"], "late_delivery", 999999.0, [], "not_the_owner")
    assert d.rule_id == "caller_not_order_owner"


# Rule 3 answers before rule 6.
def test_rule3_precedes_rule6_untrue_claim_is_not_a_cap_problem():
    on_time = _pick(situation="on_time")
    d = check_refund(on_time["order_id"], "late_delivery", 999999.0, [], on_time["customer_id"])
    assert d.rule_id == "claim_not_supported"


# Rule 4 answers before rule 6.
def test_rule4_precedes_rule6_stale_order_is_not_a_cap_problem():
    stale = _pick(situation="late", in_window=False)
    d = check_refund(stale["order_id"], "late_delivery", 999999.0, [], stale["customer_id"])
    assert d.rule_id == "outside_claim_window"


# Rule 5 answers before rule 6.
def test_rule5_precedes_rule6_second_refund_is_not_a_cap_problem():
    fresh = _pick(situation="late", in_window=True)
    history = [{"tool": "issue_refund", "order_id": fresh["order_id"], "ok": True,
                "args": {"amount_usd": 5.0}}]
    d = check_refund(fresh["order_id"], "late_delivery", 999999.0, history, fresh["customer_id"])
    assert d.rule_id == "already_refunded"


# Over both cap and limit is a cap denial: escalating it would present an illegitimate request as legitimate.
def test_rule6_precedes_rule7_over_cap_beats_over_limit():
    big = _pick(situation="never_arrived", in_window=True, over_limit=True)
    over_both = round(big["amount_usd"] + 100.0, 2)
    assert over_both > AUTO_APPROVE_MAX_USD
    d = check_refund(big["order_id"], "never_arrived", over_both, [], big["customer_id"])
    assert d.rule_id == "refund_exceeds_cap"


# Rule 5 counts only ok refunds: a refused attempt does not block a later valid one.
def test_denied_prior_refund_does_not_block_a_later_one():
    fresh = _pick(situation="late", in_window=True)
    history = [{"tool": "issue_refund", "order_id": fresh["order_id"], "ok": False,
                "args": {"amount_usd": 999999.0}}]
    cap = round(fresh["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    d = check_refund(fresh["order_id"], "late_delivery", min(cap, AUTO_APPROVE_MAX_USD),
                     history, fresh["customer_id"])
    assert d.action == "allow"


# Rule 5 is per order: one refund does not freeze every other order in the case.
def test_refund_on_a_different_order_is_unaffected_by_history():
    first = _pick(situation="late", in_window=True)
    others = [o for o in _ORDERS
              if o["situation"] == "late" and _age(o) <= CLAIM_WINDOW_DAYS
              and o["order_id"] != first["order_id"]]
    if not others:
        pytest.skip("need a second in-window late order")
    second = others[0]
    history = [{"tool": "issue_refund", "order_id": first["order_id"], "ok": True,
                "args": {"amount_usd": 5.0}}]
    cap = round(second["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    d = check_refund(second["order_id"], "late_delivery", min(cap, AUTO_APPROVE_MAX_USD),
                     history, second["customer_id"])
    assert d.action == "allow"
