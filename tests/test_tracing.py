"""
# tests/test_tracing.py

**What:** the span tree one resolution produces (invoke_agent > chat / execute_tool), its attributes, content
masking, Langfuse scores via a fake client, and the zero-network no-op without keys.
**Why:** traces are only useful if the tree and attributes match docs/observability.md. An in-memory
exporter is installed for this test module only; no Langfuse, no network.
**Used by:** pytest.
"""
import json
import socket
import uuid
from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import resolv.agents.loop as loop
from resolv import observability
from resolv.harness.validity import get_order

EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
trace.set_tracer_provider(_provider)

OWNER = get_order("ORD-1000")["customer_id"]


# A fake litellm response: a tool call or a plain reply, with usage.
def _resp(content=None, tool=None, args=None, call_id="tc1"):
    calls = [SimpleNamespace(id=call_id, function=SimpleNamespace(name=tool, arguments=json.dumps(args)))] if tool else []
    msg = SimpleNamespace(content=content, tool_calls=calls, model_dump=lambda: {"role": "assistant", "content": content})
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=15, prompt_tokens_details=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage,
                           model="openai/gpt-oss-120b")


# Patch litellm.acompletion to return the given responses in order, and stub the extractor.
def _script(monkeypatch, *responses):
    import litellm

    it = iter(responses)

    # Next canned response.
    async def fake(**kwargs):
        return next(it)

    # An extractor stand-in.
    async def no_claim(message):
        return {"order_id": "ORD-1000", "claim_type": "late_delivery"}
    monkeypatch.setattr(litellm, "acompletion", fake)
    monkeypatch.setattr(loop, "extract", no_claim)


# Spans by name prefix.
def _named(spans, prefix):
    return [s for s in spans if s.name.startswith(prefix)]


# Empty the in-memory span exporter before and after each test.
@pytest.fixture(autouse=True)
def _clear():
    EXPORTER.clear()
    yield
    EXPORTER.clear()


# Span tree for one refund case.
@pytest.mark.asyncio
async def test_span_tree_for_one_refund_case(monkeypatch):
    case_id = f"_trace_{uuid.uuid4().hex[:8]}"
    _script(monkeypatch,
            _resp(tool="lookup_order", args={"order_id": "ORD-1000"}),
            _resp(tool="issue_refund", args={"order_id": "ORD-1000", "claim_type": "late_delivery",
                                             "amount_usd": 159.86}, call_id="tc2"),
            _resp(content="I've refunded you $159.86."))

    result = await loop.run_case(case_id, "my order ORD-1000 was late", caller_id=OWNER)

    spans = EXPORTER.get_finished_spans()
    root = _named(spans, "invoke_agent")[0]
    assert root.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert root.attributes["gen_ai.conversation.id"] == case_id
    assert root.attributes["langfuse.session.id"] == case_id
    assert root.attributes["resolv.outcome"] == "resolved"
    assert result["trace_id"] == format(root.context.trace_id, "032x")

    chats = _named(spans, "chat ")
    tools = _named(spans, "execute_tool ")
    assert len(chats) == 3 and len(tools) == 2
    for child in chats + tools:
        assert child.parent.span_id == root.context.span_id
    assert chats[0].attributes["gen_ai.response.model"] == "openai/gpt-oss-120b"
    assert chats[0].attributes["gen_ai.usage.input_tokens"] == 120
    assert chats[0].attributes["resolv.role"] == "agent"
    refund = [t for t in tools if t.attributes["gen_ai.tool.name"] == "issue_refund"][0]
    assert refund.attributes["gen_ai.operation.name"] == "execute_tool"
    assert refund.attributes["resolv.policy.rule_id"] == "within_policy"


# Denied refund span carries verdict.
@pytest.mark.asyncio
async def test_denied_refund_span_carries_verdict(monkeypatch):
    _script(monkeypatch,
            _resp(tool="issue_refund", args={"order_id": "ORD-1000", "claim_type": "late_delivery", "amount_usd": 900}),
            _resp(content="I can't refund $900."))
    await loop.run_case(f"_trace_{uuid.uuid4().hex[:8]}", "refund 900", caller_id=OWNER)
    tool = _named(EXPORTER.get_finished_spans(), "execute_tool")[0]
    assert tool.attributes["resolv.policy.action"] == "deny"
    assert tool.attributes["resolv.policy.rule_id"] == "refund_exceeds_cap"


# Content not exported by default.
@pytest.mark.asyncio
async def test_content_not_exported_by_default(monkeypatch):
    monkeypatch.delenv("RESOLV_TRACE_CONTENT", raising=False)
    _script(monkeypatch, _resp(content="Hello."))
    await loop.run_case(f"_trace_{uuid.uuid4().hex[:8]}", "mail me at a@b.co", caller_id=OWNER)
    for span in EXPORTER.get_finished_spans():
        assert "gen_ai.input.messages" not in (span.attributes or {})


# Content is masked when enabled.
@pytest.mark.asyncio
async def test_content_is_masked_when_enabled(monkeypatch):
    monkeypatch.setenv("RESOLV_TRACE_CONTENT", "1")
    _script(monkeypatch, _resp(content="Hello."))
    await loop.run_case(f"_trace_{uuid.uuid4().hex[:8]}", "mail me at jane@example.com", caller_id=OWNER)
    content = _named(EXPORTER.get_finished_spans(), "chat ")[0].attributes["gen_ai.input.messages"]
    assert "jane@example.com" not in content and "[EMAIL]" in content


# No tracing provider means no network calls.
@pytest.mark.asyncio
async def test_no_provider_no_network(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    observability._client.cache_clear()

    # Any socket connect is a test failure.
    def no_connect(*args, **kwargs):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(socket.socket, "connect", no_connect)
    _script(monkeypatch, _resp(content="Hello."))
    try:
        assert observability.init() is False and observability.enabled() is False
        result = await loop.run_case(f"_trace_{uuid.uuid4().hex[:8]}", "hi", caller_id=OWNER)
        observability.log_request("x", "hi", result, {"unauthorized": False})
    finally:
        observability._client.cache_clear()


# Scores sent to trace with fake client.
def test_scores_sent_to_trace_with_fake_client(monkeypatch):
    scores = []
    fake = SimpleNamespace(create_score=lambda **kw: scores.append(kw))
    monkeypatch.setattr(observability, "_client", lambda: fake)
    metrics = {"unauthorized": False, "reply_grounded": True, "looked_up_first": True}
    observability.log_request("c", "m", {"trace_id": "ab" * 16, "steps": 3}, metrics)
    assert {s["name"] for s in scores} == {"unauthorized", "reply_grounded", "looked_up_first", "steps"}
    assert all(s["trace_id"] == "ab" * 16 for s in scores)


# No scores without client.
def test_no_scores_without_client(monkeypatch):
    monkeypatch.setattr(observability, "_client", lambda: None)
    observability.log_request("c", "m", {"trace_id": "ab" * 16}, {"unauthorized": True})  # must not raise
