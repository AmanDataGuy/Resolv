"""
# tests/test_escalations.py

**What:** human approvals bound to the concrete tool call: escalation creation, the approval rule in policy, and the
/escalations API (list, detail, approve with Idempotency-Key, deny, approver-only access).
**Why:** a human must approve exactly the action they saw and nothing else; execution goes through the same gate,
so rules 0-6 are re-checked, an approval runs once, and an approver can never approve their own refund.
**Used by:** pytest.
"""
import threading

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from resolv.api import auth, escalations
from resolv.config import AUTO_APPROVE_MAX_USD
from resolv.harness import audit, ledger, tools
from resolv.harness.policy import action_hash, check_refund
from resolv.integrations import payments

CASE = "_test_escalations"


# A client for an app holding only the escalations router.
@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(escalations.router)
    return TestClient(app)


# Headers for an approver named alice.
@pytest.fixture
def alice() -> dict:
    return {"Authorization": f"Bearer {auth.issue_token('alice', role='approver')}"}


# The agent asks for the full value of an over-limit order: policy escalates and files an approval.
@pytest.fixture
def pending(big_order) -> dict:
    out = tools.issue_refund(CASE, big_order["order_id"], "never_arrived", big_order["amount_usd"],
                             big_order["customer_id"])
    assert "sent to a human" in out
    (esc,) = ledger.escalations("pending")
    return esc


# Approve through the API with a fresh or given key.
def _approve(client, esc, headers, key="key-1", note="ok"):
    return client.post(f"/escalations/{esc['id']}/approve", json={"note": note},
                       headers=headers | {"Idempotency-Key": key})


# The escalation carries the raw action and its hash, bound to this exact call.
def test_over_limit_refund_creates_bound_approval(pending, big_order):
    text, digest = action_hash(big_order["order_id"], "never_arrived", big_order["amount_usd"], big_order["customer_id"])
    assert pending["kind"] == "approval" and pending["action_json"] == text and pending["action_hash"] == digest
    assert pending["requested_by"] == big_order["customer_id"]


# Approving executes the refund once, records the approver, and settles with the provider.
def test_approved_action_executes_once(client, alice, pending):
    r = _approve(client, pending, alice)
    assert r.status_code == 200 and r.json()["result"].startswith("Refunded")
    assert ledger.escalation(pending["id"])["status"] == "executed"
    paid = [x for x in audit.read(CASE) if x["tool"] == "issue_refund" and x["ok"]]
    assert len(paid) == 1 and paid[0]["approver_id"] == "alice" and paid[0]["rule_id"] == "approved_by_human"
    assert _approve(client, pending, alice, key="key-2").status_code == 409
    assert len(payments.all_refunds()) == 1


# The same Idempotency-Key returns the stored response and pays nothing extra.
def test_same_key_returns_same_response(client, alice, pending):
    first = _approve(client, pending, alice).json()
    assert _approve(client, pending, alice).json() == first
    assert len(payments.all_refunds()) == 1


# Reusing a key with a different body is a 422.
def test_key_reuse_with_different_body_422(client, alice, pending):
    _approve(client, pending, alice, note="a")
    assert _approve(client, pending, alice, note="b").status_code == 422


# Approve without an Idempotency-Key is rejected.
def test_missing_idempotency_key_rejected(client, alice, pending):
    assert client.post(f"/escalations/{pending['id']}/approve", json={}, headers=alice).status_code == 422


# An approval for one amount cannot be used for another: policy escalates again.
def test_modified_amount_cannot_use_approval(pending, big_order):
    with ledger.tx() as conn:
        ledger.decide_escalation(conn, pending["id"], "approved", "alice", "")
    out = tools.issue_refund(CASE, big_order["order_id"], "never_arrived", big_order["amount_usd"] - 1,
                             big_order["customer_id"], approval={"id": pending["id"]}, approver_id="alice")
    assert "sent to a human" in out and payments.all_refunds() == []


# Only the approver who gave the approval can execute with it.
def test_approval_is_bound_to_its_approver(pending, big_order):
    with ledger.tx() as conn:
        ledger.decide_escalation(conn, pending["id"], "approved", "alice", "")
    out = tools.issue_refund(CASE, big_order["order_id"], "never_arrived", big_order["amount_usd"],
                             big_order["customer_id"], approval={"id": pending["id"]}, approver_id="mallory")
    assert "sent to a human" in out and payments.all_refunds() == []


# Approving your own refund is refused by the API and by policy.
def test_approver_equals_caller_denied(client, pending, big_order):
    owner = big_order["customer_id"]
    headers = {"Authorization": f"Bearer {auth.issue_token(owner, role='approver')}"}
    assert _approve(client, pending, headers).status_code == 403
    self_approved = pending | {"status": "approved", "decided_by": owner}
    d = check_refund(big_order["order_id"], "never_arrived", big_order["amount_usd"], [], owner, self_approved)
    assert d.action == "escalate"


# An expired escalation cannot be approved and is marked expired on read.
def test_expired_approval_rejected(client, alice, pending):
    with ledger.tx() as conn:
        conn.execute("UPDATE escalations SET expires_at=? WHERE id=?", (ledger.now(-1), pending["id"]))
    assert _approve(client, pending, alice).status_code == 409
    client.get("/escalations", headers=alice)
    assert ledger.escalation(pending["id"])["status"] == "expired"


# If the order was refunded meanwhile, rule 5 refuses even an approved action, and the escalation is marked failed.
def test_rule5_rechecked_at_execution(client, alice, pending, big_order):
    small = min(AUTO_APPROVE_MAX_USD, big_order["amount_usd"])
    assert tools.issue_refund("_other_case", big_order["order_id"], "never_arrived", small,
                              big_order["customer_id"]).startswith("Refunded")
    r = _approve(client, pending, alice)
    assert "already refunded" in r.json()["result"].lower()
    assert ledger.escalation(pending["id"])["status"] == "failed"
    assert len(payments.all_refunds()) == 1


# Two approvers clicking at once: one execution, one 409.
def test_concurrent_double_approve_one_execution(alice, pending):
    outcomes = []

    # One approver's click, called straight on the endpoint function.
    def click(key: str) -> None:
        try:
            escalations.approve(pending["id"], escalations.Decide(), idempotency_key=key, who="alice")
            outcomes.append(200)
        except HTTPException as e:
            outcomes.append(e.status_code)

    threads = [threading.Thread(target=click, args=(f"k{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == [200, 409] and len(payments.all_refunds()) == 1


# Denying records the decision with the approver and notifies the customer through the outbox.
def test_deny_records_and_notifies(client, alice, pending):
    r = client.post(f"/escalations/{pending['id']}/deny", json={"note": "not convinced"}, headers=alice)
    assert r.status_code == 200 and ledger.escalation(pending["id"])["status"] == "denied"
    last = audit.read(CASE)[-1]
    assert (last["rule_id"], last["approver_id"]) == ("denied_by_human", "alice")
    assert ledger.unsent_outbox() == [] and payments.all_refunds() == []


# The detail view shows the raw action JSON, the order facts and the full trail.
def test_detail_shows_raw_action(client, alice, pending):
    body = client.get(f"/escalations/{pending['id']}", headers=alice).json()
    assert body["action_json"] == pending["action_json"] and body["order"]["order_id"] == pending["order_id"]
    assert body["trail"] and body["policy_explanation"]


# The pending list shows the amount from the bound action.
def test_list_shows_pending_with_amount(client, alice, pending, big_order):
    (row,) = client.get("/escalations", headers=alice).json()
    assert row["id"] == pending["id"] and row["amount"] == round(big_order["amount_usd"], 2)


# A valid token that is not an approver's gets 403; a bad token gets 401.
def test_non_approver_token_403_and_bad_token_401(client, alice, pending):
    customer = {"Authorization": f"Bearer {auth.issue_token('some-customer')}"}
    assert client.get("/escalations", headers=customer).status_code == 403
    assert client.get("/escalations", headers={"Authorization": "Bearer forged.garbage"}).status_code == 401
    assert client.get("/escalations").status_code == 401
