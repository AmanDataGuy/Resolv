"""Every policy rule, one passing and one failing case each — the enforcement contract, pinned.

policy.demo() already exercises the rules against the real DB; these are the same checks as
pytest, so a rule that silently changes fails CI, not just an ad-hoc script run. They read the
real data/db/orders.json on purpose: a fixture DB could drift from the shape the rules assume,
and then the tests would pass while production breaks. Real data, real rule ids.
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
    assert check_refund("ORD-999999", "late_delivery", 10.0, []).rule_id == "unknown_order"


def test_rule2_untrue_claim_denied():
    # An on-time order cannot support a late_delivery claim, however it's phrased.
    on_time = _pick(situation="on_time")
    assert check_refund(on_time["order_id"], "late_delivery", 10.0, []).rule_id == "claim_not_supported"


def test_rule3_out_of_window_denied():
    stale = _pick(situation="late", in_window=False)
    d = check_refund(stale["order_id"], "late_delivery", 1.0, [])
    assert d.rule_id == "outside_claim_window" and d.action == "deny"


def test_rule4_double_refund_denied():
    fresh = _pick(situation="late", in_window=True)
    history = [{"tool": "issue_refund", "order_id": fresh["order_id"], "ok": True, "args": {"amount_usd": 5.0}}]
    assert check_refund(fresh["order_id"], "late_delivery", 5.0, history).rule_id == "already_refunded"


def test_rule5_over_cap_denied():
    # Asking the full order value on a late delivery (capped at 25%) is refused.
    fresh = _pick(situation="late", in_window=True)
    d = check_refund(fresh["order_id"], "late_delivery", fresh["amount_usd"], [])
    assert d.rule_id == "refund_exceeds_cap" and d.action == "deny"


def test_rule6_over_limit_escalates_not_denies():
    # never_arrived (cap 1.0) so the full value can straddle the auto-approve limit.
    big = _pick(situation="never_arrived", in_window=True, over_limit=True)
    d = check_refund(big["order_id"], "never_arrived", AUTO_APPROVE_MAX_USD + 0.01, [])
    assert d.action == "escalate" and d.rule_id == "over_auto_approve_limit"


def test_allow_within_policy():
    fresh = _pick(situation="late", in_window=True)
    cap = round(fresh["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    amount = min(cap, AUTO_APPROVE_MAX_USD)
    d = check_refund(fresh["order_id"], "late_delivery", amount, [])
    assert d.action == "allow" and d.rule_id == "within_policy"


def test_cap_uses_record_not_stated_amount():
    # The cap is a fraction of what the record says was PAID — the customer's figure never enters.
    fresh = _pick(situation="late", in_window=True)
    cap = round(fresh["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    assert check_refund(fresh["order_id"], "late_delivery", cap, []).action == "allow"
    assert check_refund(fresh["order_id"], "late_delivery", cap + 0.01, []).rule_id == "refund_exceeds_cap"

