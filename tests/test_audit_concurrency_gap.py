"""Regression tests for the two gaps found and fixed in the 2026-08-15 audit (plan_ahead.md,
Priorities 1-2): the split-session double refund and the concurrent-write race in
harness/tools.py::issue_refund.

Both are fixed as of this commit — these tests now assert the FIXED behavior (exactly one
refund lands, in both scenarios), so a future regression in either fix fails loudly here instead
of being rediscovered the hard way.
"""
import threading
from datetime import date

import pytest

from config import CLAIM_WINDOW_DAYS, REFUND_CAP_FRACTION
from harness import audit, tools
from harness.policy import NOW
from harness.validity import get_order


@pytest.fixture(autouse=True)
def _isolated_order_index(tmp_path, monkeypatch):
    """Same isolation as tests/test_harness.py — this file also picks a real, deterministic order
    and must not see (or leave behind) order-level state from any other test module or a prior
    real run of the app."""
    monkeypatch.setattr(audit, "ORDER_DIR", tmp_path / "_by_order")


def _fresh_late_order() -> dict:
    for i in range(1000, 1460):
        o = get_order(f"ORD-{i}")
        if not o or o["situation"] != "late":
            continue
        if (NOW - date.fromisoformat(o["promised_date"])).days <= CLAIM_WINDOW_DAYS:
            return o
    pytest.skip("no fresh late order in the demo DB")


class TestSplitSessionDoubleRefund:
    def test_a_second_independent_case_on_the_same_order_is_denied(self):
        """FIXED: rule 4 now reads audit.order_history(order_id) in addition to the case trail
        (harness/tools.py::issue_refund), so a second, separate conversation about the same real
        order sees the first case's successful refund and is denied — even though nothing about
        the second request is wrong on its own. Before the fix this test's own predecessor
        (test_two_independent_cases_on_the_same_order_both_refund) demonstrated $319.72 paid on a
        $159.86-cap order across two sessions.
        """
        order = _fresh_late_order()
        cap = round(order["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
        case_a, case_b = "_test_split_a", "_test_split_b"
        audit.clear(case_a)
        audit.clear(case_b)
        audit.clear_order(order["order_id"])
        owner = order["customer_id"]
        try:
            r1 = tools.issue_refund(case_a, order["order_id"], "late_delivery", cap, owner)
            r2 = tools.issue_refund(case_b, order["order_id"], "late_delivery", cap, owner)
            assert r1.startswith("Refunded")
            assert "already refunded" in r2.lower(), r2
        finally:
            audit.clear(case_a)
            audit.clear(case_b)
            audit.clear_order(order["order_id"])


class TestConcurrentRefundRace:
    def test_concurrent_calls_on_the_same_case_and_order_refund_exactly_once(self):
        """FIXED: the read-check-append sequence in issue_refund() now runs under a process-wide
        lock (harness/tools.py::_refund_lock). Before the fix, 8 concurrent calls on the same
        case_id and order paid out 7 times ($1,119.02 vs $159.86 owed); with the lock, only the
        first to acquire it can pass rule 4 before the second sees the first's record.
        """
        order = _fresh_late_order()
        cap = round(order["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
        owner = order["customer_id"]
        case = "_test_race_case"
        audit.clear(case)
        audit.clear_order(order["order_id"])
        try:
            n = 8
            results = []
            barrier = threading.Barrier(n)

            def attempt():
                barrier.wait()
                results.append(tools.issue_refund(case, order["order_id"], "late_delivery", cap, owner))

            threads = [threading.Thread(target=attempt) for _ in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            refunded = sum(1 for r in results if r.startswith("Refunded"))
            assert refunded == 1, f"expected exactly 1 refund out of {n} concurrent calls, got {refunded}"
        finally:
            audit.clear(case)
            audit.clear_order(order["order_id"])
