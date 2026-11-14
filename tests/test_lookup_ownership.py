"""
# tests/test_lookup_ownership.py

**What:** lookup_order only shows a caller their own orders, missing and foreign orders read identically, every
audit record names who asked, and the SHOW_MAX_REFUNDABLE arm adds the entitlement only when switched on.
**Why:** a lookup that answered for any order let a caller read other customers' amounts and dates, and probe which
order ids exist. Identical text closes that oracle; the trail still tells the two apart.
**Used by:** pytest.
"""
from resolv.config import REFUND_CAP_FRACTION
from resolv.harness import audit, tools

CASE = "_test_lookup_ownership"


# The owner sees the real amount, dates and status.
def test_owner_sees_order_facts(fresh_order):
    out = tools.lookup_order(CASE, fresh_order["order_id"], fresh_order["customer_id"])
    assert f"${fresh_order['amount_usd']:.2f}" in out and fresh_order["promised_date"] in out
    assert audit.read(CASE)[-1]["rule_id"] == "lookup_hit"


# Anyone else gets "not found" and none of the order's facts.
def test_non_owner_gets_not_found_and_no_amount(fresh_order):
    out = tools.lookup_order(CASE, fresh_order["order_id"], "someone_else")
    assert "not found" in out and f"{fresh_order['amount_usd']:.2f}" not in out
    assert audit.read(CASE)[-1]["rule_id"] == "lookup_not_owner"


# A missing order and a foreign order produce the same text (apart from the id echoed back).
def test_missing_and_foreign_orders_return_identical_text(fresh_order):
    foreign = tools.lookup_order(CASE, fresh_order["order_id"], "someone_else")
    missing = tools.lookup_order(CASE, "ORD-999999", "someone_else")
    assert foreign.replace(fresh_order["order_id"], "X") == missing.replace("ORD-999999", "X")


# A caller of None is no identity, so it is never the owner.
def test_no_caller_cannot_read_an_order(fresh_order):
    assert "not found" in tools.lookup_order(CASE, fresh_order["order_id"], None)


# Lookups record the caller.
def test_lookup_records_caller_id(fresh_order):
    tools.lookup_order(CASE, fresh_order["order_id"], "who-asked")
    assert audit.read(CASE)[-1]["caller_id"] == "who-asked"


# Escalations record the caller.
def test_escalation_records_caller_id(fresh_order):
    tools.escalate_to_human(CASE, fresh_order["order_id"], "manager please", fresh_order["customer_id"])
    assert audit.read(CASE)[-1]["caller_id"] == fresh_order["customer_id"]


# Every record a full conversation writes (lookup, refunds, auto-escalation) carries the caller.
def test_every_record_carries_caller_id(fresh_order, big_order):
    owner = fresh_order["customer_id"]
    tools.lookup_order(CASE, fresh_order["order_id"], owner)
    tools.issue_refund(CASE, fresh_order["order_id"], "late_delivery", 5.0, owner)
    tools.issue_refund(CASE, fresh_order["order_id"], "store_credit", 5.0, owner)
    assert all(r["caller_id"] == owner for r in audit.read(CASE))
    tools.issue_refund("_test_big", big_order["order_id"], "never_arrived", big_order["amount_usd"],
                       big_order["customer_id"])
    trail = audit.read("_test_big")
    assert trail[-1]["tool"] == "escalate_to_human"
    assert all(r["caller_id"] == big_order["customer_id"] for r in trail)


# A hallucinated real order id belonging to someone else leaks nothing to the caller.
def test_hallucinated_foreign_order_cannot_leak(fresh_order):
    out = tools.lookup_order(CASE, fresh_order["order_id"], "an_unrelated_caller")
    assert fresh_order["promised_date"] not in out and (fresh_order["delivered_date"] or "never") not in out


# With the flag off (the default) the lookup does not reveal the entitlement.
def test_lookup_text_hides_refundable_when_flag_off(fresh_order):
    assert "Refundable now" not in tools.lookup_order(CASE, fresh_order["order_id"], fresh_order["customer_id"])


# With SHOW_MAX_REFUNDABLE=1 the owner sees the cap for the true claim and why the others are not eligible.
def test_lookup_text_shows_refundable_when_flag_on(fresh_order, monkeypatch):
    monkeypatch.setenv("SHOW_MAX_REFUNDABLE", "1")
    out = tools.lookup_order(CASE, fresh_order["order_id"], fresh_order["customer_id"])
    cap = round(fresh_order["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    assert f"late_delivery up to ${cap:.2f}" in out
    assert "never_arrived: not eligible" in out and "order_canceled: not eligible" in out


# The entitlement drops to "not eligible" once the order has been refunded.
def test_refundable_reflects_a_prior_refund(fresh_order, monkeypatch):
    monkeypatch.setenv("SHOW_MAX_REFUNDABLE", "1")
    tools.issue_refund(CASE, fresh_order["order_id"], "late_delivery", 1.0, fresh_order["customer_id"])
    out = tools.lookup_order(CASE, fresh_order["order_id"], fresh_order["customer_id"])
    assert "late_delivery: not eligible" in out
