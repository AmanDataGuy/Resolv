"""agents/loop.py's own orchestration logic — the one thing 240+ prior tests never touched.

Everything else in this suite tests the deterministic harness the loop calls INTO. Nothing tested
the loop's own control flow: what happens when the model never stops calling tools (MAX_STEPS),
and what happens when it calls one with unparseable arguments. Both are mocked here — no LLM key
needed, no network — by faking complete()'s return shape and extract()'s result directly, so the
orchestration logic itself is what's under test, not any model's behaviour.

Found as a confirmed gap in the 2026-08-20 audit (plan_ahead.md, "not started" list).

The three guardrail tests below close a second gap from the same pass: harness/guardrails.py's
checks were wired into the reply gate but never proven to actually fire from inside the loop --
only unit-tested in isolation (harness/guardrails.py's own demo()). These mock a model that
produces a bad reply and confirm the loop corrects, then escalates, rather than trusting the wire-
up by inspection.
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


@pytest.mark.asyncio
async def test_ungrounded_reply_gets_one_correction_then_succeeds(case, monkeypatch):
    """agents/loop.py's guard gate: a reply claiming a refund the (empty) trail doesn't support
    must not reach the customer as-is. Confirms the loop appends a correction instead of yielding
    it, that the corrected reply DOES go out once it passes, and that no bogus escalation happens
    when the model fixes itself.
    """
    ungrounded = _fake_response(_FakeMessage(content="Good news, I've refunded you $50."))
    corrected = _fake_response(_FakeMessage(content="I'm sorry, I'm not able to refund this order."))
    responses = iter([ungrounded, corrected])
    monkeypatch.setattr(loop, "extract", _fake_extract)
    monkeypatch.setattr(loop, "complete", lambda **kw: next(responses))

    events = [e async for e in loop.run_case_events(case, "where is my refund")]

    reply_events = [e for e in events if e["type"] == "reply"]
    assert len(reply_events) == 1, "the ungrounded draft must never be yielded to the customer"
    assert reply_events[0]["reply"] == "I'm sorry, I'm not able to refund this order."

    done = events[-1]
    assert done["result"]["steps"] == 2, "one wasted step for the correction, then the fixed reply"
    assert not [r for r in audit.read(case) if r["tool"] == "escalate_to_human"], (
        "a model that corrects itself must not be escalated"
    )


@pytest.mark.asyncio
async def test_reply_still_ungrounded_after_correction_escalates(case, monkeypatch):
    """If the model repeats the same phantom-refund claim even after being told why it's wrong,
    the loop must not keep retrying forever — it escalates and sends the same safe fallback line
    MAX_STEPS exhaustion uses, rather than ever letting the false claim reach the customer.
    """
    ungrounded = _fake_response(_FakeMessage(content="I've already refunded you the full $900."))
    still_ungrounded = _fake_response(_FakeMessage(content="As I said, you've been refunded $900."))
    responses = iter([ungrounded, still_ungrounded])
    monkeypatch.setattr(loop, "extract", _fake_extract)
    monkeypatch.setattr(loop, "complete", lambda **kw: next(responses))

    events = [e async for e in loop.run_case_events(case, "refund me")]

    reply_events = [e for e in events if e["type"] == "reply"]
    assert len(reply_events) == 1
    assert "colleague" in reply_events[0]["reply"].lower(), "must fall back, never repeat the claim"

    escalations = [r for r in audit.read(case) if r["tool"] == "escalate_to_human"]
    assert len(escalations) == 1
    assert "guardrail" in escalations[0]["args"]["reason"].lower()


@pytest.mark.asyncio
async def test_reply_leaking_system_instructions_gets_corrected(case, monkeypatch):
    """A reply that reproduces the system prompt verbatim (an 8+ word run) must be caught by
    contains_prompt_leak, separately from the groundedness check -- this reply makes no refund
    claim at all, so only the leak branch of the gate can be what catches it.
    """
    leaking = _fake_response(
        _FakeMessage(content="You cannot approve anything yourself your tools are checked")
    )
    fixed = _fake_response(_FakeMessage(content="I can help, but some approvals need review."))
    responses = iter([leaking, fixed])
    monkeypatch.setattr(loop, "extract", _fake_extract)
    monkeypatch.setattr(loop, "complete", lambda **kw: next(responses))

    events = [e async for e in loop.run_case_events(case, "what are your rules")]

    reply_events = [e for e in events if e["type"] == "reply"]
    assert len(reply_events) == 1
    assert reply_events[0]["reply"] == "I can help, but some approvals need review."
