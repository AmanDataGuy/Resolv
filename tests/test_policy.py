"""Every policy rule, one passing and one failing case each — the enforcement contract, pinned.

policy.demo() already exercises the rules against the real DB; these are the same checks as
pytest, so a rule that silently changes fails CI, not just an ad-hoc script run. They read the
real data/db/orders.json on purpose: a fixture DB could drift from the shape the rules assume,
and then the tests would pass while production breaks. Real data, real rule ids.

Every check_refund() call below passes the order's real customer_id as caller_id, except the new
rule-2 tests, which exist to prove what happens when it doesn't match. Without that, rule 2 (added
after tests/test_hallucination_collision.py proved the ownership gap real) would fire first on
every other test here and each would fail with "caller_not_order_owner" instead of the rule it's
actually named for.
"""
from datetime import date

import pytest

from config import AUTO_APPROVE_MAX_USD, CLAIM_WINDOW_DAYS, REFUND_CAP_FRACTION
from harness.policy import NOW, check_refund
from harness.validity import get_order

_ORDERS = [o for o in (get_order(f"ORD-{i}") for i in range(1000, 1460)) if o]


def _age(o: dict) -> int:
    return (NOW - date.fromisoformat(o["promised_date"])).days


def _pick(**cond):
    """First order matching all conditions, or skip if the data has none (keeps the suite honest
    if the DB is regenerated smaller rather than green on a vacuous pass)."""
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


def test_db_loaded():
    assert len(_ORDERS) > 100, "the real order DB should be present and non-trivial"


def test_rule1_unknown_order_denied():
    assert check_refund("ORD-999999", "late_delivery", 10.0, [], "anyone").rule_id == "unknown_order"


def test_rule2_caller_mismatch_denied_even_when_claim_is_true():
    fresh = _pick(situation="late", in_window=True)
    d = check_refund(fresh["order_id"], "late_delivery", 5.0, [], "not_the_owner")
    assert d.rule_id == "caller_not_order_owner" and d.action == "deny"


def test_rule2_no_caller_id_denied_same_as_a_mismatch():
    fresh = _pick(situation="late", in_window=True)
    assert check_refund(fresh["order_id"], "late_delivery", 5.0, []).rule_id == "caller_not_order_owner"


def test_rule3_untrue_claim_denied():
    # An on-time order cannot support a late_delivery claim, however it's phrased.
    on_time = _pick(situation="on_time")
    d = check_refund(on_time["order_id"], "late_delivery", 10.0, [], on_time["customer_id"])
    assert d.rule_id == "claim_not_supported"


def test_rule4_out_of_window_denied():
    stale = _pick(situation="late", in_window=False)
    d = check_refund(stale["order_id"], "late_delivery", 1.0, [], stale["customer_id"])
    assert d.rule_id == "outside_claim_window" and d.action == "deny"


def test_rule5_double_refund_denied():
    fresh = _pick(situation="late", in_window=True)
    history = [{"tool": "issue_refund", "order_id": fresh["order_id"], "ok": True, "args": {"amount_usd": 5.0}}]
    d = check_refund(fresh["order_id"], "late_delivery", 5.0, history, fresh["customer_id"])
    assert d.rule_id == "already_refunded"


def test_rule6_over_cap_denied():
    # Asking the full order value on a late delivery (capped at 25%) is refused.
    fresh = _pick(situation="late", in_window=True)
    d = check_refund(fresh["order_id"], "late_delivery", fresh["amount_usd"], [], fresh["customer_id"])
    assert d.rule_id == "refund_exceeds_cap" and d.action == "deny"


def test_rule7_over_limit_escalates_not_denies():
    # never_arrived (cap 1.0) so the full value can straddle the auto-approve limit.
    big = _pick(situation="never_arrived", in_window=True, over_limit=True)
    d = check_refund(big["order_id"], "never_arrived", AUTO_APPROVE_MAX_USD + 0.01, [], big["customer_id"])
    assert d.action == "escalate" and d.rule_id == "over_auto_approve_limit"


def test_allow_within_policy():
    fresh = _pick(situation="late", in_window=True)
    cap = round(fresh["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    amount = min(cap, AUTO_APPROVE_MAX_USD)
    d = check_refund(fresh["order_id"], "late_delivery", amount, [], fresh["customer_id"])
    assert d.action == "allow" and d.rule_id == "within_policy"


def test_cap_uses_record_not_stated_amount():
    # The cap is a fraction of what the record says was PAID — the customer's figure never enters.
    fresh = _pick(situation="late", in_window=True)
    cap = round(fresh["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    owner = fresh["customer_id"]
    assert check_refund(fresh["order_id"], "late_delivery", cap, [], owner).action == "allow"
    assert check_refund(fresh["order_id"], "late_delivery", cap + 0.01, [], owner).rule_id == "refund_exceeds_cap"


# --- Degenerate inputs (rule 0) --------------------------------------------------------------
# Found by writing these tests, not in production: the agent never proposes a non-positive
# amount, so no sweep would ever have surfaced it. A $0 refund silently burns the order's one
# allowed refund (rule 5 then denies the real one); a negative refund charges the customer.

@pytest.mark.parametrize("amount", [0.0, -0.01, -5.0, -1000.0])
def test_non_positive_amounts_are_denied(amount):
    fresh = _pick(situation="late", in_window=True)
    decision = check_refund(fresh["order_id"], "late_delivery", amount, [], fresh["customer_id"])
    assert decision.action == "deny" and decision.rule_id == "invalid_amount"


def test_smallest_positive_amount_is_allowed():
    """The boundary is at zero, not somewhere above it — a one-cent refund is still valid."""
    fresh = _pick(situation="late", in_window=True)
    assert check_refund(fresh["order_id"], "late_delivery", 0.01, [], fresh["customer_id"]).action == "allow"


# --- Boundaries: the exact edge of every threshold -------------------------------------------

def test_exactly_at_cap_is_allowed_one_cent_over_is_not():
    fresh = _pick(situation="late", in_window=True)
    cap = round(fresh["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    owner = fresh["customer_id"]
    assert check_refund(fresh["order_id"], "late_delivery", cap, [], owner).rule_id == "within_policy"
    assert check_refund(fresh["order_id"], "late_delivery", round(cap + 0.01, 2), [], owner).rule_id == "refund_exceeds_cap"


def test_exactly_at_auto_approve_limit_is_allowed_one_cent_over_escalates():
    """Rule 7 is an escalate, not a deny: every rule above already agreed it's legitimate, so
    what's left is a question of authority, not merit."""
    big = _pick(situation="never_arrived", in_window=True, over_limit=True)
    owner = big["customer_id"]
    at = check_refund(big["order_id"], "never_arrived", AUTO_APPROVE_MAX_USD, [], owner)
    over = check_refund(big["order_id"], "never_arrived", AUTO_APPROVE_MAX_USD + 0.01, [], owner)
    assert at.action == "allow"
    assert over.action == "escalate" and over.rule_id == "over_auto_approve_limit"


def test_claim_window_boundary_separates_adjacent_orders():
    """The oldest in-window order passes rule 4; the youngest out-of-window one does not."""
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
# Each of these presents a request that violates TWO rules at once and asserts which one answers.
# If the order ever changes, a denial's reason changes with it — and the reason is what the
# customer is told and what the scorecard groups by.

def test_rule0_precedes_rule1_unknown_order_with_negative_amount():
    assert check_refund("ORD-999999", "late_delivery", -5.0, [], "anyone").rule_id == "invalid_amount"


def test_rule1_precedes_rule2_unknown_order_is_not_an_ownership_problem():
    assert check_refund("ORD-999999", "late_delivery", 10.0, [], "not_the_owner").rule_id == "unknown_order"


def test_rule2_precedes_rule6_wrong_caller_is_not_a_cap_problem():
    fresh = _pick(situation="late", in_window=True)
    d = check_refund(fresh["order_id"], "late_delivery", 999999.0, [], "not_the_owner")
    assert d.rule_id == "caller_not_order_owner"


def test_rule3_precedes_rule6_untrue_claim_is_not_a_cap_problem():
    on_time = _pick(situation="on_time")
    d = check_refund(on_time["order_id"], "late_delivery", 999999.0, [], on_time["customer_id"])
    assert d.rule_id == "claim_not_supported"


def test_rule4_precedes_rule6_stale_order_is_not_a_cap_problem():
    stale = _pick(situation="late", in_window=False)
    d = check_refund(stale["order_id"], "late_delivery", 999999.0, [], stale["customer_id"])
    assert d.rule_id == "outside_claim_window"


def test_rule5_precedes_rule6_second_refund_is_not_a_cap_problem():
    fresh = _pick(situation="late", in_window=True)
    history = [{"tool": "issue_refund", "order_id": fresh["order_id"], "ok": True,
                "args": {"amount_usd": 5.0}}]
    d = check_refund(fresh["order_id"], "late_delivery", 999999.0, history, fresh["customer_id"])
    assert d.rule_id == "already_refunded"


def test_rule6_precedes_rule7_over_cap_beats_over_limit():
    """An amount over BOTH the cap and the auto-approve limit is a cap denial, not an escalation:
    escalating it would put an illegitimate request in front of a human as if it were legitimate."""
    big = _pick(situation="never_arrived", in_window=True, over_limit=True)
    over_both = round(big["amount_usd"] + 100.0, 2)
    assert over_both > AUTO_APPROVE_MAX_USD
    d = check_refund(big["order_id"], "never_arrived", over_both, [], big["customer_id"])
    assert d.rule_id == "refund_exceeds_cap"


def test_denied_prior_refund_does_not_block_a_later_one():
    """Rule 5 filters on ok=True. A refused attempt is recorded but must not count as a refund."""
    fresh = _pick(situation="late", in_window=True)
    history = [{"tool": "issue_refund", "order_id": fresh["order_id"], "ok": False,
                "args": {"amount_usd": 999999.0}}]
    cap = round(fresh["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    d = check_refund(fresh["order_id"], "late_delivery", min(cap, AUTO_APPROVE_MAX_USD),
                      history, fresh["customer_id"])
    assert d.action == "allow"


def test_refund_on_a_different_order_is_unaffected_by_history():
    """Rule 5 is scoped per order — one refund must not freeze every other order in the case."""
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
