"""
# tests/test_ledger.py

**What:** the SQLite ledger's guarantees: one live refund per order (enforced by the database, across processes),
cents that round-trip, idempotency keys for tool calls and HTTP requests, and the transactional outbox.
**Why:** these are constraints, not conventions; each test proves the database itself refuses the bad state.
**Used by:** pytest.
"""
import multiprocessing
import os
import sqlite3

import pytest

from resolv.config import REFUND_CAP_FRACTION
from resolv.harness import audit, ledger, tools
from resolv.harness.policy import Decision
from resolv.integrations import notify, payments

CASE = "_test_ledger"


# Write one allow record and return its seq, so a refund row has an audit record to point at.
def _audit_seq() -> int:
    return audit.append(CASE, "authorize_refund", "ORD-1000", {}, Decision(action="allow", rule_id="t", reason="t"),
                        True, "t")["seq"]


# Insert a pending refund straight into the ledger, bypassing tools and policy.
def _insert(key: str, order_id: str = "ORD-1000", cents: int = 500) -> str:
    seq = _audit_seq()
    with ledger.tx() as conn:
        return ledger.insert_refund(conn, key, order_id, CASE, "late_delivery", cents, seq)


# The partial unique index blocks a second live refund on one order even with no policy in the way.
def test_partial_unique_index_blocks_second_live_refund():
    _insert("k1")
    with pytest.raises(sqlite3.IntegrityError):
        _insert("k2")


# A failed refund frees the order's slot for a new one.
def test_failed_refund_frees_the_slot():
    refund_id = _insert("k1")
    with ledger.tx() as conn:
        assert ledger.set_refund_status(conn, refund_id, "failed")
    _insert("k2")
    assert sorted(r["status"] for r in ledger.refunds()) == ["failed", "pending"]


# A reused idempotency key is refused by the database too.
def test_reused_idempotency_key_is_refused():
    refund_id = _insert("k1")
    with ledger.tx() as conn:
        ledger.set_refund_status(conn, refund_id, "failed")
    with pytest.raises(sqlite3.IntegrityError):
        _insert("k1")


# $47.09 is stored as 4709 cents and read back as 47.09, never through float sums.
def test_cents_roundtrip():
    _insert("k1", cents=round(47.09 * 100))
    assert ledger.refunds()[0]["amount_cents"] == 4709
    assert ledger.order_refund_records("ORD-1000")[0]["args"]["amount_usd"] == 47.09


# If policy were ever skipped, the database still blocks the second refund and the attempt is recorded.
def test_db_backstop_records_duplicate_blocked(fresh_order, monkeypatch):
    allow = Decision(action="allow", rule_id="within_policy", reason="forced")
    monkeypatch.setattr(tools, "check_refund", lambda *a, **k: allow)
    oid, owner = fresh_order["order_id"], fresh_order["customer_id"]
    assert tools.issue_refund(CASE, oid, "late_delivery", 1.0, owner).startswith("Refunded")
    assert tools.issue_refund(CASE, oid, "late_delivery", 1.0, owner).startswith("Refused:")
    assert audit.read(CASE)[-1]["rule_id"] == "duplicate_refund_blocked_by_db"
    assert len(payments.all_refunds()) == 1


# Replaying a tool call with the same idempotency key returns the first outcome and pays once.
def test_tool_call_replay_is_idempotent(fresh_order):
    oid, owner = fresh_order["order_id"], fresh_order["customer_id"]
    first = tools.issue_refund(CASE, oid, "late_delivery", 2.0, owner, idempotency_key="wf-1:3")
    again = tools.issue_refund(CASE, oid, "late_delivery", 2.0, owner, idempotency_key="wf-1:3")
    assert first == again and first.startswith("Refunded")
    assert audit.read(CASE)[-1]["rule_id"] == "idempotent_replay"
    assert len(ledger.refunds()) == 1 and len(payments.all_refunds()) == 1


# Request keys: new, then in flight, then a stored replay; a different body is a conflict; a release allows a retry.
def test_request_idempotency_key_lifecycle():
    assert ledger.claim_key("idem-1", "approve", "sha-a") == ("new", None)
    assert ledger.claim_key("idem-1", "approve", "sha-a") == ("in_flight", None)
    ledger.finish_key("idem-1", {"result": "done"})
    assert ledger.claim_key("idem-1", "approve", "sha-a") == ("replay", {"result": "done"})
    assert ledger.claim_key("idem-1", "approve", "sha-b") == ("conflict", None)
    assert ledger.claim_key("idem-2", "approve", "sha-a") == ("new", None)
    ledger.finish_key("idem-2", None)
    assert ledger.claim_key("idem-2", "approve", "sha-a") == ("new", None)


# A settled refund queues the customer's confirmation in the same transaction; draining sends it once.
def test_settle_writes_outbox_and_drain_marks_sent(fresh_order):
    cap = round(fresh_order["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
    tools.issue_refund(CASE, fresh_order["order_id"], "late_delivery", min(cap, 5.0), fresh_order["customer_id"])
    assert [r["kind"] for r in ledger.unsent_outbox()] == ["customer_refund_confirmation"]
    assert notify.drain_outbox() == 1
    assert ledger.unsent_outbox() == [] and notify.drain_outbox() == 0
    assert len(list((notify.OUTBOX / "events").iterdir())) == 1


# One process's attempt at the shared refund (runs in a spawned child).
def _child_refund(args: tuple) -> str:
    db, psp, order_id, owner, amount = args
    os.environ.update(RESOLV_ENV="test", RESOLV_DB=db, RESOLV_PSP_DB=psp)
    from resolv.harness import tools as child_tools

    return child_tools.issue_refund(f"_mp_{os.getpid()}", order_id, "late_delivery", amount, owner)


# Eight processes race the same refund against one database file: exactly one wins.
def test_multiprocess_double_refund_one_wins(fresh_order, tmp_path):
    amount = min(round(fresh_order["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2), 5.0)
    args = (os.environ["RESOLV_DB"], os.environ["RESOLV_PSP_DB"], fresh_order["order_id"],
            fresh_order["customer_id"], amount)
    ledger.connect().close()  # create the schema before the race
    with multiprocessing.get_context("spawn").Pool(8) as pool:
        results = pool.map(_child_refund, [args] * 8)
    assert sum(r.startswith("Refunded") for r in results) == 1, results
    assert len(ledger.refunds(("settled",))) == 1 and len(payments.all_refunds()) == 1
