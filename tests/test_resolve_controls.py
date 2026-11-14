"""
# tests/test_resolve_controls.py

**What:** the three operator controls on money: the `Idempotency-Key` header on /resolve, shadow mode
(RESOLV_SHADOW=1) and the kill switch (REFUNDS_FROZEN=1), driven through the real API and harness.
**Why:** each one exists to stop money moving twice or at all, so each is pinned by counting refund rows in the
ledger. The model is faked (it always asks for one small, in-policy refund); everything under it is real.
**Used by:** pytest.
"""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import resolv.agents.loop as loop
from resolv.api import auth, ratelimit
from resolv.api.main import app
from resolv.harness import audit, ledger, tools


# A client whose fake model asks for a $1 late_delivery refund on the order, then replies in plain text.
@pytest.fixture
def client(fresh_order, monkeypatch):
    args = json.dumps({"order_id": fresh_order["order_id"], "claim_type": "late_delivery", "amount_usd": 1.0})

    # Ask for the refund once; after the tool result comes back, give a neutral reply.
    async def fake_model(**kwargs):
        if kwargs["messages"][-1]["role"] == "tool":
            msg = SimpleNamespace(content="Thanks, we have handled your request.", tool_calls=[])
        else:
            call = SimpleNamespace(id="tc1", function=SimpleNamespace(name="issue_refund", arguments=args))
            msg = SimpleNamespace(content=None, tool_calls=[call])
        msg.model_dump = lambda: {"role": "assistant", "content": msg.content}
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    # An extractor that reads nothing.
    async def no_claim(message):
        return {"order_id": None, "claim_type": None}

    monkeypatch.setattr(loop, "acomplete", fake_model)
    monkeypatch.setattr(loop, "extract", no_claim)
    ratelimit._hits.clear()
    return TestClient(app)


# The /resolve body for the order's owner.
def _body(order: dict, message: str = "my order was late, please refund") -> dict:
    return {"message": message, "token": auth.issue_token(order["customer_id"])}


# The same Idempotency-Key twice runs one case and pays one refund; the second call returns the stored response.
def test_resolve_idempotency_key_runs_once(client, fresh_order):
    headers = {"Idempotency-Key": "retry-1"}
    first = client.post("/resolve", json=_body(fresh_order), headers=headers)
    second = client.post("/resolve", json=_body(fresh_order), headers=headers)
    assert first.status_code == second.status_code == 200
    assert second.json() == first.json()
    assert len(ledger.refunds()) == 1
    assert first.json()["ticket"]["outcome"] == "resolved"


# Reusing a key with a different body is refused (422), and runs nothing.
def test_resolve_idempotency_key_conflict(client, fresh_order):
    headers = {"Idempotency-Key": "retry-2"}
    assert client.post("/resolve", json=_body(fresh_order), headers=headers).status_code == 200
    other = client.post("/resolve", json=_body(fresh_order, "something else"), headers=headers)
    assert other.status_code == 422
    assert len(ledger.refunds()) == 1


# Without the header every request is a new case (the one-live-refund rule still blocks a second payment).
def test_resolve_without_key_runs_each_time(client, fresh_order):
    a = client.post("/resolve", json=_body(fresh_order)).json()
    b = client.post("/resolve", json=_body(fresh_order)).json()
    assert a["case_id"] != b["case_id"]
    assert len(ledger.refunds()) == 1


# Shadow mode: the agent and policy run, the decision is recorded as "would have refunded", and no money moves.
def test_shadow_moves_no_money(client, fresh_order, monkeypatch):
    monkeypatch.setenv("RESOLV_ENV", "dev")  # shadow is refused in test/eval runs
    monkeypatch.setenv("RESOLV_DURABLE", "0")  # the fake model reply cannot be journaled
    monkeypatch.setenv("RESOLV_SHADOW", "1")
    out = client.post("/resolve", json=_body(fresh_order)).json()
    assert ledger.refunds() == []
    [rec] = [r for r in audit.read(out["case_id"]) if r["tool"] == "issue_refund"]
    assert (rec["rule_id"], rec["action"], rec["ok"]) == ("shadow_mode", "escalate", False)
    assert "would have refunded $1.00 (within_policy)" in rec["reason"]
    assert out["ticket"]["outcome"] == "escalated"


# Shadow mode is refused when RESOLV_ENV is test or eval, so a score can never measure a dry run.
@pytest.mark.parametrize("env", ["test", "eval"])
def test_shadow_refused_in_test_and_eval(fresh_order, monkeypatch, env):
    monkeypatch.setenv("RESOLV_ENV", env)
    monkeypatch.setenv("RESOLV_SHADOW", "1")
    with pytest.raises(RuntimeError, match="RESOLV_SHADOW"):
        tools.issue_refund("_shadow", fresh_order["order_id"], "late_delivery", 1.0, fresh_order["customer_id"])
    assert ledger.refunds() == []


# Kill switch: a refund policy would allow is escalated for a person instead, and nothing is paid.
def test_frozen_escalates_everything(client, fresh_order, monkeypatch):
    monkeypatch.setenv("REFUNDS_FROZEN", "1")
    out = client.post("/resolve", json=_body(fresh_order)).json()
    assert ledger.refunds() == []
    [rec] = [r for r in audit.read(out["case_id"]) if r["tool"] == "issue_refund"]
    assert (rec["rule_id"], rec["action"], rec["ok"]) == ("refunds_frozen", "escalate", False)
    assert out["ticket"]["outcome"] == "escalated"
    [esc] = ledger.escalations("pending")
    assert esc["policy_rule_id"] == "refunds_frozen"


# Kill switch leaves denials alone: a refund over the cap is still denied for its real reason.
def test_frozen_keeps_denials(fresh_order, monkeypatch):
    monkeypatch.setenv("REFUNDS_FROZEN", "1")
    result = tools.issue_refund("_frozen", fresh_order["order_id"], "late_delivery", fresh_order["amount_usd"],
                                fresh_order["customer_id"])
    assert result.startswith("Refused:")
    assert audit.read("_frozen")[-1]["rule_id"] == "refund_exceeds_cap"
