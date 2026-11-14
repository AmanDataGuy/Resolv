"""
# tests/test_live_retry.py

**What:** what a retry of the live path (resolv/api/main.py runs a failed attempt again under a new case id) can and
cannot do to money and evidence.
**Why:** the old retry cleared the failed attempt's trail; if attempt 1 had already refunded, the evidence vanished
and the chain restarted undetectably. Now clear refuses outside test/eval, the failed attempt's trail stays, and the
ledger stops attempt 2 from paying again, even when attempt 1 crashed mid-saga.
**Used by:** pytest.
"""
import pytest

from resolv.config import REFUND_CAP_FRACTION
from resolv.harness import audit, ledger, tools
from resolv.integrations import payments


# The process dies: not an Exception, so nothing in the saga catches it.
class Crash(BaseException):
    pass


# The order's refund amount used throughout (its late_delivery cap, kept under the auto-approve limit).
def _amount(order: dict) -> float:
    return min(round(order["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2), 50.0)


# Attempt 1 refunds and then fails; attempt 2 under a new case id is refused and attempt 1's trail is intact.
def test_failed_attempt_trail_is_kept_and_second_attempt_cannot_double_pay(fresh_order):
    oid, owner, amount = fresh_order["order_id"], fresh_order["customer_id"], _amount(fresh_order)
    assert tools.issue_refund("api-x-a0", oid, "late_delivery", amount, owner).startswith("Refunded")
    second = tools.issue_refund("api-x-a1", oid, "late_delivery", amount, owner)
    assert "already refunded" in second.lower()
    assert [r for r in audit.read("api-x-a0") if r["tool"] == "issue_refund" and r["ok"]], "attempt 1's trail is kept"
    assert len(payments.all_refunds()) == 1
    assert audit.verify_chain() == (True, None)


# Attempt 1 dies after the provider paid but before settling; attempt 2 is refused and the reconciler settles once.
def test_crash_after_payment_before_settle_is_not_paid_twice(fresh_order, monkeypatch):
    oid, owner, amount = fresh_order["order_id"], fresh_order["customer_id"], _amount(fresh_order)
    real_refund = payments.refund

    # Pay for real, then die before the ledger hears back.
    def pay_then_crash(*args):
        real_refund(*args)
        raise Crash()

    monkeypatch.setattr(payments, "refund", pay_then_crash)
    with pytest.raises(Crash):
        tools.issue_refund("api-y-a0", oid, "late_delivery", amount, owner)
    monkeypatch.setattr(payments, "refund", real_refund)

    assert "already refunded" in tools.issue_refund("api-y-a1", oid, "late_delivery", amount, owner).lower()
    assert tools.reconcile_once(min_age_s=0) == 1
    assert [r["status"] for r in ledger.refunds()] == ["settled"]
    assert len(payments.all_refunds()) == 1


# clear and clear_order refuse to delete evidence unless RESOLV_ENV is test or eval.
@pytest.mark.parametrize("env", ["prod", "dev", None])
def test_clear_refuses_outside_test_env(monkeypatch, env):
    if env is None:
        monkeypatch.delenv("RESOLV_ENV", raising=False)
    else:
        monkeypatch.setenv("RESOLV_ENV", env)
    with pytest.raises(RuntimeError):
        audit.clear("any-case")
    with pytest.raises(RuntimeError):
        audit.clear_order("ORD-1000")


# The eval may still reset its own cases between repeats.
def test_clear_is_allowed_in_eval(monkeypatch, fresh_order):
    monkeypatch.setenv("RESOLV_ENV", "eval")
    tools.lookup_order("eval-case", fresh_order["order_id"], fresh_order["customer_id"])
    audit.clear("eval-case")
    audit.clear_order(fresh_order["order_id"])
    assert audit.read("eval-case") == []
