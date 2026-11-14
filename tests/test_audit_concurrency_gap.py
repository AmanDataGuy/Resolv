"""
# tests/test_audit_concurrency_gap.py

**What:** regression tests for the split-session double refund and the concurrent-write race in issue_refund.
**Why:** both once paid an order more than once (two honest cases each refunded; 7 of 8 concurrent calls refunded).
Both are now closed by the ledger: rule 5 reads every live refund on the order, and the read-check-write runs in
one BEGIN IMMEDIATE transaction with a partial UNIQUE index behind it. These assert exactly one refund lands.
**Used by:** pytest.
"""
import threading

from resolv.config import REFUND_CAP_FRACTION
from resolv.harness import ledger, tools


# A second, separate conversation about the same order sees the first refund and is denied.
def test_a_second_independent_case_on_the_same_order_is_denied(fresh_order):
    cap = round(fresh_order["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    oid, owner = fresh_order["order_id"], fresh_order["customer_id"]
    r1 = tools.issue_refund("_test_split_a", oid, "late_delivery", cap, owner)
    r2 = tools.issue_refund("_test_split_b", oid, "late_delivery", cap, owner)
    assert r1.startswith("Refunded")
    assert "already refunded" in r2.lower(), r2


# Eight threads racing the same refund: exactly one is paid.
def test_concurrent_calls_on_the_same_case_and_order_refund_exactly_once(fresh_order):
    cap = round(fresh_order["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    n, results = 8, []
    barrier = threading.Barrier(n)

    # One racing attempt, released by the barrier at the same moment as the others.
    def attempt():
        barrier.wait()
        results.append(tools.issue_refund("_test_race_case", fresh_order["order_id"], "late_delivery", cap,
                                          fresh_order["customer_id"]))

    threads = [threading.Thread(target=attempt) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(r.startswith("Refunded") for r in results) == 1, results
    assert len(ledger.refunds(("settled",))) == 1
