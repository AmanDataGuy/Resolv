"""
# tests/test_budgets.py

**What:** the loop's hard limits (tool calls, input tokens, USD), loop detection, the provider circuit breaker,
the byte-stable prompt prefix, the cache-hit ratio and the context-local token meter.
**Why:** each limit must escalate with a named reason instead of spending without bound. LLM calls are faked.
**Used by:** pytest.
"""
import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest

import resolv.agents.loop as loop
import resolv.agents.llm as ru
from resolv.harness import audit


# A fake response: one message, optional usage with cached tokens.
def _resp(content=None, tool=None, args=None, prompt=100, completion=10, cached=None, call_id="tc"):
    calls = [SimpleNamespace(id=call_id, function=SimpleNamespace(name=tool, arguments=json.dumps(args)))] if tool else []
    msg = SimpleNamespace(content=content, tool_calls=calls, model_dump=lambda: {"role": "assistant", "content": content})
    details = SimpleNamespace(cached_tokens=cached) if cached is not None else None
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, prompt_tokens_details=details)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage, model="served-x")


# An async acomplete() stand-in built from a function of the call index.
def _fake(make):
    count = {"n": 0}

    # Return make(i) for the i-th call.
    async def fake(**kwargs):
        count["n"] += 1
        return make(count["n"] - 1)
    return fake


# An extractor stand-in that reads nothing.
async def _no_claim(message):
    return {"order_id": None, "claim_type": None}


# A fresh case id per test.
@pytest.fixture
def case(monkeypatch):
    monkeypatch.setattr(loop, "extract", _no_claim)
    return f"_test_budget_{uuid.uuid4().hex[:8]}"


# Reasons of the escalation records on a case's trail.
def _escalations(case_id):
    return [r["args"]["reason"] for r in audit.read(case_id) if r["tool"] == "escalate_to_human"]


# Tool call cap escalates.
@pytest.mark.asyncio
async def test_tool_call_cap_escalates(case, monkeypatch):
    monkeypatch.setattr(loop, "acomplete", _fake(lambda i: _resp(tool="lookup_order", args={"order_id": f"ORD-{1000 + i}"})))
    result = await loop.run_case(case, "hi")
    assert result["budget_exceeded"] == "tool_calls"
    assert len([r for r in result["trail"] if r["tool"] == "lookup_order"]) == loop.MAX_TOOL_CALLS
    assert _escalations(case) == ["budget_exceeded:tool_calls"]


# Token budget escalates.
@pytest.mark.asyncio
async def test_token_budget_escalates(case, monkeypatch):
    big = loop.MAX_INPUT_TOKENS_PER_CASE + 1
    monkeypatch.setattr(loop, "acomplete", _fake(lambda i: _resp(tool="lookup_order", args={"order_id": "ORD-1000"},
                                                                  prompt=big)))
    result = await loop.run_case(case, "hi")
    assert result["budget_exceeded"] == "tokens"
    assert _escalations(case) == ["budget_exceeded:tokens"]


# USD budget escalates.
@pytest.mark.asyncio
async def test_usd_budget_escalates(case, monkeypatch):
    monkeypatch.setattr(loop.config, "price_usd", lambda model, p, c: 1.0)
    monkeypatch.setattr(loop, "acomplete", _fake(lambda i: _resp(tool="lookup_order", args={"order_id": f"ORD-{1000 + i}"})))
    result = await loop.run_case(case, "hi")
    assert result["budget_exceeded"] == "usd"


# Repeated identical call nudged then escalated.
@pytest.mark.asyncio
async def test_repeated_identical_call_nudged_then_escalated(case, monkeypatch):
    seen = []

    # Always the same call; remember what the model was shown.
    async def fake(**kwargs):
        seen.append([m for m in kwargs["messages"] if isinstance(m, dict) and m.get("role") == "tool"])
        return _resp(tool="lookup_order", args={"order_id": "ORD-1000"})
    monkeypatch.setattr(loop, "acomplete", fake)

    result = await loop.run_case(case, "hi")

    assert result["budget_exceeded"] == "loop"
    assert len([r for r in result["trail"] if r["tool"] == "lookup_order"]) == 1  # the repeat never ran
    assert seen[-1][-1]["content"].startswith("[System guardrail] You already did that")
    assert _escalations(case) == ["loop_detected"]


# Cache hit ratio and none when unreported.
@pytest.mark.asyncio
async def test_cache_hit_ratio_and_none_when_unreported(case, monkeypatch):
    monkeypatch.setattr(loop, "acomplete", _fake(lambda i: _resp(content="Hello there.", prompt=200, cached=50)))
    assert (await loop.run_case(case, "hi"))["cache_hit_ratio"] == 0.25

    monkeypatch.setattr(loop, "acomplete", _fake(lambda i: _resp(content="Hello there.")))
    result = await loop.run_case(case + "b", "hi")
    assert result["cache_hit_ratio"] is None and result["cached_input_tokens"] is None


# Prefix is byte stable.
@pytest.mark.asyncio
async def test_prefix_is_byte_stable(monkeypatch):
    captured = []

    # Record the system message and tools of each call.
    async def fake(**kwargs):
        captured.append(json.dumps([kwargs["messages"][0], kwargs["tools"]], sort_keys=True))
        return _resp(content="Hello there.")
    monkeypatch.setattr(loop, "extract", _no_claim)
    monkeypatch.setattr(loop, "acomplete", fake)
    await loop.run_case(f"_prefix_a_{uuid.uuid4().hex[:6]}", "my order ORD-1000 is late", caller_id="x")
    await loop.run_case(f"_prefix_b_{uuid.uuid4().hex[:6]}", "never arrived ORD-1001", caller_id="y")
    assert captured[0] == captured[1]


# Breaker opens and half opens.
def test_breaker_opens_and_half_opens(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(ru, "_now", lambda: clock["t"])
    monkeypatch.setattr(ru, "_open_until", 0.0)
    monkeypatch.setattr(ru, "_tripped", False)
    ru._failures.clear()

    for _ in range(ru.BREAKER_FAILURES):
        ru._breaker_check()
        ru._breaker_record(False)
    with pytest.raises(ru.CircuitOpen):
        ru._breaker_check()

    clock["t"] += ru.BREAKER_COOLDOWN_S + 1
    ru._breaker_check()  # half-open: one trial allowed
    ru._breaker_record(False)  # the trial fails: open again at once
    with pytest.raises(ru.CircuitOpen):
        ru._breaker_check()

    clock["t"] += ru.BREAKER_COOLDOWN_S + 1
    ru._breaker_record(True)  # a trial succeeds: closed
    ru._breaker_check()
    ru._failures.clear()


# Daily quota raises typed error.
def test_daily_quota_raises_typed_error():
    error = Exception("Rate limit reached ... on tokens per day (TPD): Limit 100000. Please try again in 7m.")
    with pytest.raises(ru.QuotaExhausted):
        ru._on_error(error, 0, "groq/openai/gpt-oss-120b")


# Selfhosted fails fast without retry.
def test_selfhosted_fails_fast_without_retry():
    with pytest.raises(RuntimeError, match="SELFHOSTED_BASE_URL"):
        ru._on_error(Exception("Connection refused"), 0, "hosted_vllm/x")
    with pytest.raises(Exception, match="429"):
        ru._on_error(Exception("429 busy"), 0, "hosted_vllm/x")
    ru._failures.clear()


# Meter isolated across concurrent cases.
@pytest.mark.asyncio
async def test_meter_isolated_across_concurrent_cases(monkeypatch):
    import litellm

    # Token counts depend on the request so each case is distinguishable.
    async def fake(**kwargs):
        await asyncio.sleep(0.01)
        n = kwargs["metadata"]["n"]
        return _resp(content="ok", prompt=n, completion=n)
    monkeypatch.setattr(litellm, "acompletion", fake)

    # One "request": a fresh meter, a few calls, then read the meter.
    async def one(n):
        ru.new_meter()
        for _ in range(3):
            await ru.acomplete(model="groq/x", messages=[], metadata={"n": n})
        return ru.tokens_split(), ru.served_model()

    (a, served_a), (b, _) = await asyncio.gather(one(10), one(1000))
    assert a == (30, 30) and b == (3000, 3000) and served_a == "served-x"
