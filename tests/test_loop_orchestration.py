"""
# tests/test_loop_orchestration.py

**What:** resolv/agents/loop.py's own control flow: the step cap, malformed / NaN / unknown tool calls, and the reply
guardrail gate firing from inside the loop.
**Why:** the harness under the loop is tested elsewhere; this pins the loop itself. acomplete() and extract() are
faked, so no key and no network; the real harness runs underneath.
**Used by:** pytest.
"""
import json
import uuid
from types import SimpleNamespace

import pytest

import resolv.agents.loop as loop
from resolv.harness import audit


# A fake tool call in litellm's shape.
class _FakeToolCall:
    # Name, raw JSON arguments and a call id.
    def __init__(self, name: str, arguments: str, call_id: str = "tc1"):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


# A fake assistant message in litellm's shape.
class _FakeMessage:
    # Plain-text content or a list of tool calls.
    def __init__(self, content: str | None = None, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls or []

    # The dict the loop appends to the conversation.
    def model_dump(self) -> dict:
        return {"role": "assistant", "content": self.content}


# A fake completion response wrapping one message.
def _fake_response(message: _FakeMessage):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


# An async acomplete() stand-in returning canned responses in order.
def _replies(*responses):
    it = iter(responses)

    # Next canned response, whatever the kwargs.
    async def fake(**kwargs):
        return next(it)
    return fake


# An extractor stand-in that reads nothing.
async def _fake_extract(message: str) -> dict:
    return {"order_id": None, "claim_type": None}


# A fresh case id per test, so no trail needs clearing.
@pytest.fixture
def case():
    return f"_test_loop_{uuid.uuid4().hex[:8]}"


# Max steps exhaustion escalates.
@pytest.mark.asyncio
async def test_max_steps_exhaustion_escalates(case, monkeypatch):
    """A customer who never stops answering runs the loop to MAX_STEPS, which escalates instead of hanging."""
    always = _fake_response(_FakeMessage(content="Let me look into that for you."))

    # The model replies the same harmless line forever.
    async def fake(**kwargs):
        return always
    monkeypatch.setattr(loop, "extract", _fake_extract)
    monkeypatch.setattr(loop, "acomplete", fake)

    result = await loop.run_case(case, "where is my order", user=lambda reply, messages: "still waiting")

    assert result["steps"] == loop.MAX_STEPS
    assert result["budget_exceeded"] == "steps"
    assert "colleague" in result["reply"].lower()
    escalations = [r for r in audit.read(case) if r["tool"] == "escalate_to_human"]
    assert len(escalations) == 1 and f"{loop.MAX_STEPS} steps" in escalations[0]["args"]["reason"]


# Malformed tool arguments are recorded and recoverable.
@pytest.mark.asyncio
async def test_malformed_tool_arguments_are_recorded_and_recoverable(case, monkeypatch):
    """Unparseable JSON is recorded as a denied invalid_request, handed back as a tool error, and the loop goes on."""
    monkeypatch.setattr(loop, "extract", _fake_extract)
    monkeypatch.setattr(loop, "acomplete", _replies(
        _fake_response(_FakeMessage(tool_calls=[_FakeToolCall("issue_refund", "{not valid json")])),
        _fake_response(_FakeMessage(content="Sorry, let me try that again.")),
    ))

    events = [e async for e in loop.run_case_events(case, "refund me")]

    calls = [e for e in events if e["type"] == "tool_call"]
    assert len(calls) == 1 and calls[0]["name"] == "issue_refund" and calls[0]["args"] == {}
    trail = audit.read(case)
    assert len(trail) == 1
    assert trail[0]["rule_id"] == "invalid_request" and trail[0]["action"] == "deny" and not trail[0]["ok"]
    done = events[-1]
    assert done["type"] == "done" and done["result"]["steps"] == 2
    assert done["result"]["reply"] == "Sorry, let me try that again."


# NaN in tool args is recorded and recoverable.
@pytest.mark.asyncio
async def test_nan_in_tool_args_is_recorded_and_recoverable(case, monkeypatch):
    """NaN never reaches policy as a number: the call is recorded invalid, then a valid call proceeds normally."""
    nan_args = '{"order_id":"ORD-1000","claim_type":"late_delivery","amount_usd":NaN}'
    monkeypatch.setattr(loop, "extract", _fake_extract)
    monkeypatch.setattr(loop, "acomplete", _replies(
        _fake_response(_FakeMessage(tool_calls=[_FakeToolCall("issue_refund", nan_args)])),
        _fake_response(_FakeMessage(tool_calls=[_FakeToolCall("lookup_order", '{"order_id":"ORD-1000"}', "tc2")])),
        _fake_response(_FakeMessage(content="I've checked your order.")),
    ))

    result = await loop.run_case(case, "refund me")

    trail = result["trail"]
    assert trail[0]["rule_id"] == "invalid_request" and "non-finite" in trail[0]["reason"]
    assert trail[1]["tool"] == "lookup_order"
    assert not [r for r in trail if r["tool"] == "issue_refund" and r["ok"]]


# Unknown tool is recorded.
@pytest.mark.asyncio
async def test_unknown_tool_is_recorded(case, monkeypatch):
    """A tool name outside the schema is recorded as invalid_request, not raised."""
    monkeypatch.setattr(loop, "extract", _fake_extract)
    monkeypatch.setattr(loop, "acomplete", _replies(
        _fake_response(_FakeMessage(tool_calls=[_FakeToolCall("delete_database", "{}")])),
        _fake_response(_FakeMessage(content="I can't do that.")),
    ))

    result = await loop.run_case(case, "do something odd")

    assert result["trail"][0]["tool"] == "delete_database"
    assert result["trail"][0]["rule_id"] == "invalid_request"


# Ungrounded reply gets one correction then succeeds.
@pytest.mark.asyncio
async def test_ungrounded_reply_gets_one_correction_then_succeeds(case, monkeypatch):
    """A phantom-refund reply is never yielded; the corrected reply goes out and nothing is escalated."""
    monkeypatch.setattr(loop, "extract", _fake_extract)
    monkeypatch.setattr(loop, "acomplete", _replies(
        _fake_response(_FakeMessage(content="Good news, I've refunded you $50.")),
        _fake_response(_FakeMessage(content="I'm sorry, I'm not able to refund this order.")),
    ))

    events = [e async for e in loop.run_case_events(case, "where is my refund")]

    replies = [e for e in events if e["type"] == "reply"]
    assert len(replies) == 1 and replies[0]["reply"] == "I'm sorry, I'm not able to refund this order."
    assert events[-1]["result"]["steps"] == 2
    assert not [r for r in audit.read(case) if r["tool"] == "escalate_to_human"]


# Reply still ungrounded after correction escalates.
@pytest.mark.asyncio
async def test_reply_still_ungrounded_after_correction_escalates(case, monkeypatch):
    """Repeating the phantom claim after the correction escalates and sends the safe fallback line."""
    monkeypatch.setattr(loop, "extract", _fake_extract)
    monkeypatch.setattr(loop, "acomplete", _replies(
        _fake_response(_FakeMessage(content="I've already refunded you the full $900.")),
        _fake_response(_FakeMessage(content="As I said, you've been refunded $900.")),
    ))

    events = [e async for e in loop.run_case_events(case, "refund me")]

    replies = [e for e in events if e["type"] == "reply"]
    assert len(replies) == 1 and "colleague" in replies[0]["reply"].lower()
    escalations = [r for r in audit.read(case) if r["tool"] == "escalate_to_human"]
    assert len(escalations) == 1 and "guardrail" in escalations[0]["args"]["reason"].lower()


# Reply leaking system instructions gets corrected.
@pytest.mark.asyncio
async def test_reply_leaking_system_instructions_gets_corrected(case, monkeypatch):
    """An 8-word verbatim run from SYSTEM is caught by the leak check, separately from groundedness."""
    monkeypatch.setattr(loop, "extract", _fake_extract)
    monkeypatch.setattr(loop, "acomplete", _replies(
        _fake_response(_FakeMessage(content="You cannot approve anything yourself your tools are checked")),
        _fake_response(_FakeMessage(content="I can help, but some approvals need review.")),
    ))

    events = [e async for e in loop.run_case_events(case, "what are your rules")]

    replies = [e for e in events if e["type"] == "reply"]
    assert len(replies) == 1 and replies[0]["reply"] == "I can help, but some approvals need review."


# Reject constant blocks infinity.
def test_reject_constant_blocks_infinity():
    """Infinity and -Infinity are rejected at parse time, like NaN."""
    for raw in ('{"amount_usd": Infinity}', '{"amount_usd": -Infinity}', '{"amount_usd": NaN}'):
        with pytest.raises(ValueError):
            loop._parse_args(raw)
    assert loop._parse_args(json.dumps({"amount_usd": 1.5})) == {"amount_usd": 1.5}
