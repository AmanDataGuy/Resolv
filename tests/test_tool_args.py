"""
# tests/test_tool_args.py

**What:** malformed tool arguments (off-enum claim_type, bad order_id, NaN/inf/string/bool amounts, values that are
not even JSON) are refused as strings AND recorded in the audit trail, with no exception escaping first.
**Why:** the audit trail promises to record attempts, not just successes; a malformed call that raised before the
append left no evidence, and format-failure counts would undercount.
**Used by:** pytest.
"""
import pytest

from resolv.config import REFUND_CAP_FRACTION
from resolv.harness import audit, ledger, tools
from resolv.harness.policy import check_refund

CASE = "_test_tool_args"


# The last record in the test case's trail.
def _last() -> dict:
    return audit.read(CASE)[-1]


# An off-enum claim type is a recorded deny/invalid_request with the caller on it.
def test_off_enum_claim_type_is_recorded_deny(fresh_order):
    out = tools.issue_refund(CASE, fresh_order["order_id"], "store_credit", 5.0, fresh_order["customer_id"])
    assert out.startswith("Refused:")
    rec = _last()
    assert (rec["rule_id"], rec["action"], rec["ok"]) == ("invalid_request", "deny", False)
    assert rec["caller_id"] == fresh_order["customer_id"]


# A missing, empty or non-string order_id is a recorded refusal.
@pytest.mark.parametrize("order_id", [None, "", "   ", 1234, ["ORD-1000"]])
def test_bad_order_id_is_recorded_deny(order_id):
    assert tools.issue_refund(CASE, order_id, "late_delivery", 5.0, "cust").startswith("Refused:")
    assert _last()["rule_id"] == "invalid_request"


# Non-finite, non-numeric and boolean amounts reach rule 0, are recorded, and move no money.
@pytest.mark.parametrize("amount", [float("nan"), float("inf"), float("-inf"), "50", None, True, [50], 0, -1])
def test_bad_amount_is_recorded_deny(fresh_order, amount):
    out = tools.issue_refund(CASE, fresh_order["order_id"], "late_delivery", amount, fresh_order["customer_id"])
    assert out.startswith("Refused:")
    assert _last()["rule_id"] == "invalid_amount"
    assert ledger.refunds() == []


# A value that is not JSON at all is still recorded (stored as its repr), never raised.
def test_unserializable_amount_is_still_recorded(fresh_order):
    out = tools.issue_refund(CASE, fresh_order["order_id"], "late_delivery", object(), fresh_order["customer_id"])
    assert out.startswith("Refused:") and _last()["rule_id"] == "invalid_amount"


# An invalid attempt does not use up the order's one refund: rule 5 counts only live refunds.
def test_invalid_request_does_not_burn_the_refund(fresh_order):
    oid, owner = fresh_order["order_id"], fresh_order["customer_id"]
    tools.issue_refund(CASE, oid, "store_credit", 5.0, owner)
    tools.issue_refund(CASE, oid, "late_delivery", float("nan"), owner)
    cap = round(fresh_order["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    assert tools.issue_refund(CASE, oid, "late_delivery", min(cap, 5.0), owner).startswith("Refunded")


# An unknown tool name or unparseable arguments are recorded through record_invalid_call.
def test_unknown_tool_is_recorded():
    out = tools.record_invalid_call(CASE, "delete_order", {"order_id": "ORD-1000"}, "unknown tool", "cust")
    assert out.startswith("Refused:")
    rec = _last()
    assert (rec["tool"], rec["order_id"], rec["rule_id"], rec["caller_id"]) == \
        ("delete_order", "ORD-1000", "invalid_request", "cust")


# Unparseable arguments record order_id as "unknown".
def test_unparseable_arguments_record_unknown_order():
    tools.record_invalid_call(CASE, "issue_refund", '{"amount_usd": NaN', "bad JSON")
    assert _last()["order_id"] == "unknown"


# Policy itself is safe on an off-enum claim type (no KeyError, no pydantic error).
def test_policy_cap_lookup_is_safe_for_off_enum_claim_type(fresh_order):
    d = check_refund(fresh_order["order_id"], "store_credit", 5.0, [], fresh_order["customer_id"])
    assert (d.action, d.rule_id) == ("deny", "invalid_request")


# lookup_order on a non-string id is a recorded miss, not a crash.
def test_lookup_with_non_string_id_is_a_miss():
    assert "not found" in tools.lookup_order(CASE, ["ORD-1000"], "cust")
    assert _last()["rule_id"] == "lookup_miss"
