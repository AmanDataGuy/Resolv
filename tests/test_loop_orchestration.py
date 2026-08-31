"""agents/loop.py's own orchestration logic — the one thing 240+ prior tests never touched.

Everything else in this suite tests the deterministic harness the loop calls INTO. Nothing tested
the loop's own control flow: what happens when the model never stops calling tools (MAX_STEPS),
and what happens when it calls one with unparseable arguments. Both are mocked here — no LLM key
needed, no network — by faking complete()'s return shape and extract()'s result directly, so the
orchestration logic itself is what's under test, not any model's behaviour.

Found as a confirmed gap in the 2026-08-20 audit (plan_ahead.md, "not started" list).
"""
import json
from types import SimpleNamespace

import pytest

import agents.loop as loop
from harness import audit


class _FakeToolCall:
    def __init__(self, name: str, arguments: str, call_id: str = "tc1"):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class _FakeMessage:
    def __init__(self, content: str | None = None, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self) -> dict:
        return {"role": "assistant", "content": self.content}


def _fake_response(message: _FakeMessage):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


async def _fake_extract(message: str) -> dict:
    return {"order_id": None, "claim_type": None}


@pytest.fixture
def case():
    case_id = "_test_loop_orchestration"
    audit.clear(case_id)
    yield case_id
    audit.clear(case_id)


@pytest.mark.asyncio
async def test_max_steps_exhaustion_escalates_instead_of_looping_forever(case, monkeypatch):
    """The model calling a harmless tool forever (never producing a final reply) must not hang
    the loop or crash it — agents/loop.py:219-221 escalates with a reason and replies to the
    customer instead of returning silence. Confirms it actually fires at exactly MAX_STEPS, not
    one step early or late, and that the escalation is recorded on the real audit trail.
    """
    always_lookup = _fake_response(
        _FakeMessage(tool_calls=[_FakeToolCall("lookup_order", json.dumps({"order_id": "ORD-1000"}))])
    )
    monkeypatch.setattr(loop, "extract", _fake_extract)
    monkeypatch.setattr(loop, "complete", lambda **kw: always_lookup)

    events = [e async for e in loop.run_case_events(case, "where is my order")]

    done = events[-1]
    assert done["type"] == "done"
    assert done["result"]["steps"] == loop.MAX_STEPS
    assert "colleague" in done["result"]["reply"].lower()

    # tool_call events: one per step (MAX_STEPS of them, all lookup_order).
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_calls) == loop.MAX_STEPS
    assert all(tc["name"] == "lookup_order" for tc in tool_calls)

    # The escalation actually landed on the real trail, with the exhaustion reason — not just a
    # reply string the customer sees with nothing behind it.
    trail = audit.read(case)
    escalations = [r for r in trail if r["tool"] == "escalate_to_human"]
    assert len(escalations) == 1
    assert f"{loop.MAX_STEPS} steps" in escalations[0]["args"]["reason"]


@pytest.mark.asyncio
async def test_malformed_tool_arguments_recover_instead_of_crashing(case, monkeypatch):
    """agents/loop.py:199-206: a tool call with unparseable JSON arguments must not crash the
    request — it's the model's mistake to recover from mid-conversation, same reasoning as a
    policy denial being a result rather than an exception. Confirms the loop yields a tool_call
    event (with empty args, since nothing parsed), keeps running, and reaches a normal finish.
    """
    broken_call = _fake_response(
        _FakeMessage(tool_calls=[_FakeToolCall("issue_refund", "{not valid json")])
    )
    final_reply = _fake_response(_FakeMessage(content="Sorry, let me try that again."))
    responses = iter([broken_call, final_reply])
    monkeypatch.setattr(loop, "extract", _fake_extract)
    monkeypatch.setattr(loop, "complete", lambda **kw: next(responses))

    events = [e async for e in loop.run_case_events(case, "refund me")]

    tool_call_events = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_call_events) == 1
    assert tool_call_events[0]["name"] == "issue_refund"
    assert tool_call_events[0]["args"] == {}, "unparseable arguments must not crash json.loads' caller"

    # The malformed call must NOT reach the audit trail as a real attempt — no order_id was ever
    # successfully parsed out of it, so there is nothing valid to record.
    assert audit.read(case) == []

    # The loop recovered and reached a normal, non-escalated finish on the very next step.
    done = events[-1]
    assert done["type"] == "done"
    assert done["result"]["steps"] == 2
    assert done["result"]["reply"] == "Sorry, let me try that again."
