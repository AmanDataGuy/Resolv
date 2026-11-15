"""
# tests/test_stream_metrics.py

**What:** POST /resolve/stream and GET /metrics, driven through the real API and harness with a scripted model.
**Why:** the stream must show each step, end with exactly the /resolve body, and never carry PII or the prompt; the
metrics must count what the harness decided (allow, deny, escalate) and the money that moved. No network, no keys.
**Used by:** pytest.
"""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

import resolv.agents.loop as loop
from resolv.api import auth, ratelimit
from resolv.api.main import app
from resolv.harness import pii

PII = ["jane.doe@example.com", "415-555-0134", "4111 1111 1111 1111"]


# A client whose scripted model proposes `call` (tool name, args) once, then replies with `reply`.
@pytest.fixture
def scripted(monkeypatch):
    def make(tool: str, args: dict, reply: str = "Thanks, we have handled your request."):
        # One tool call, then a plain-text reply once the tool result is back.
        async def fake_model(**kwargs):
            if kwargs["messages"][-1]["role"] == "tool":
                msg = SimpleNamespace(content=reply, tool_calls=[])
            else:
                call = SimpleNamespace(id="tc1", function=SimpleNamespace(name=tool, arguments=json.dumps(args)))
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
    return make


# The /resolve body for the order's owner.
def _body(order: dict, message: str = "my order was late, please refund") -> dict:
    return {"message": message, "token": auth.issue_token(order["customer_id"])}


# The SSE data payloads of a /resolve/stream response, in order.
def _events(resp) -> list[dict]:
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
    return [json.loads(line[len("data: "):]) for line in resp.text.splitlines() if line.startswith("data: ")]


# {(metric sample name, labels tuple): value} from the /metrics text.
def _scrape(client) -> dict:
    text = client.get("/metrics").text
    return {(s.name, tuple(sorted(s.labels.items()))): s.value
            for fam in text_string_to_metric_families(text) for s in fam.samples}


# Sum every sample of `name` whose labels include `match`.
def _sum(scrape: dict, name: str, **match) -> float:
    return sum(v for (n, labels), v in scrape.items() if n == name and set(match.items()) <= set(labels))


# (a) The stream shows intake, the proposal, the decision, the saga step and the reply, then the /resolve body.
def test_stream_steps_end_with_the_resolve_body(scripted, fresh_order):
    args = {"order_id": fresh_order["order_id"], "claim_type": "late_delivery", "amount_usd": 1.0}
    client = scripted("issue_refund", args)
    headers = {"Idempotency-Key": "stream-1"}
    events = _events(client.post("/resolve/stream", json=_body(fresh_order), headers=headers))

    assert [e["type"] for e in events] == ["intake", "proposal", "decision", "saga", "reply", "result"]
    proposal, decision, saga = events[1:4]
    assert (proposal["tool"], proposal["args"]) == ("issue_refund", args)
    assert (decision["tool"], decision["decision"], decision["rule_id"]) == ("authorize_refund", "allow",
                                                                            "within_policy")
    assert (saga["tool"], saga["ok"]) == ("issue_refund", True)

    # The same key on /resolve replays the stored response: the stream's result is exactly the /resolve body.
    result = events[-1]["response"]
    assert client.post("/resolve", json=_body(fresh_order), headers=headers).json() == result
    assert result["ticket"]["outcome"] == "resolved" and events[-2]["reply"] == result["reply"]
    # And a replay on the stream sends only that stored body.
    assert _events(client.post("/resolve/stream", json=_body(fresh_order), headers=headers)) == \
        [{"type": "result", "response": result}]


# The stream route has the same gate as /resolve: a bad token is a 401 before anything runs.
def test_stream_rejects_bad_token(scripted, fresh_order):
    client = scripted("lookup_order", {"order_id": fresh_order["order_id"]})
    assert client.post("/resolve/stream", json={"message": "hi", "token": "forged.x"}).status_code == 401


# (b) After one allowed, one denied and one escalated refund, /metrics shows each decision once and $1 paid.
def test_metrics_count_decisions_and_money(scripted, fresh_order, big_order, monkeypatch):
    before = _scrape(scripted("lookup_order", {"order_id": fresh_order["order_id"]}))
    runs = [
        (fresh_order, {"order_id": fresh_order["order_id"], "claim_type": "late_delivery",
                       "amount_usd": fresh_order["amount_usd"]}),  # over the 25% late cap: denied
        (fresh_order, {"order_id": fresh_order["order_id"], "claim_type": "late_delivery", "amount_usd": 1.0}),
        (big_order, {"order_id": big_order["order_id"], "claim_type": "never_arrived",
                     "amount_usd": big_order["amount_usd"]}),  # over the auto-approve limit: a person decides
    ]
    for order, args in runs:
        assert scripted("issue_refund", args).post("/resolve", json=_body(order)).status_code == 200
    monkeypatch.setenv("REFUNDS_FROZEN", "1")
    after = _scrape(scripted("lookup_order", {"order_id": fresh_order["order_id"]}))

    def delta(name, **match):
        return _sum(after, name, **match) - _sum(before, name, **match)

    assert delta("resolv_refund_decisions_total", decision="allow", rule="within_policy") == 1
    assert delta("resolv_refund_decisions_total", decision="deny", rule="refund_exceeds_cap") == 1
    assert delta("resolv_refund_decisions_total", decision="escalate") == 1
    assert delta("resolv_refund_decisions_total") == 3
    assert delta("resolv_refund_usd_total") == pytest.approx(1.0)
    assert delta("resolv_resolve_seconds_count") == 3
    assert _sum(after, "resolv_refunds_frozen") == 1.0


# (c) Customer PII and the system prompt never appear in a step event, even when the model echoes the PII.
def test_stream_events_carry_no_pii(scripted, fresh_order):
    reason = f"customer {PII[0]} ({PII[1]}) wants a manager"
    client = scripted("escalate_to_human", {"order_id": fresh_order["order_id"], "reason": reason},
                      reply=f"I have passed this on; we will email {PII[0]}.")
    message = f"I'm {PII[0]}, call {PII[1]}, card {PII[2]}. Order {fresh_order['order_id']} was late."
    resp = client.post("/resolve/stream", json=_body(fresh_order, message))
    steps = [e for e in _events(resp) if e["type"] != "result"]

    assert {"proposal", "decision", "reply"} <= {e["type"] for e in steps}
    text = json.dumps(steps)
    assert not any(p in text for p in PII) and pii.find(text) == []
    assert "[EMAIL]" in text  # masked, not dropped
    assert loop.SYSTEM[:60] not in resp.text
