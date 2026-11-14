"""
# tests/test_async_health.py

**What:** /resolve never blocks the event loop: /health answers while a slow resolution is in flight; retries use
asyncio.sleep; a wall-clock timeout escalates instead of failing; ADK sessions are deleted.
**Why:** one slow provider call must not freeze every other request on the server. LLM calls are faked.
**Used by:** pytest.
"""
import asyncio
import time
import uuid
from types import SimpleNamespace

import httpx
import pytest

import resolv.agents.loop as loop
import resolv.agents.llm as ru
from resolv.api import auth, ratelimit
from resolv.api.main import app


# A fake plain-text completion response.
def _reply(text="Thanks, I'll look into it."):
    msg = SimpleNamespace(content=text, tool_calls=[], model_dump=lambda: {"role": "assistant", "content": text})
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None, model="m")


# An extractor stand-in that reads nothing.
async def _no_claim(message):
    return {"order_id": None, "claim_type": None}


# Health responds while resolve in flight.
@pytest.mark.asyncio
async def test_health_responds_while_resolve_in_flight(monkeypatch):
    import litellm

    # A provider call that takes two seconds without blocking the loop.
    async def slow(**kwargs):
        await asyncio.sleep(2)
        return _reply()
    monkeypatch.setattr(litellm, "acompletion", slow)
    monkeypatch.setattr(loop, "extract", _no_claim)
    ratelimit._hits.clear()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        pending = asyncio.create_task(
            client.post("/resolve", json={"message": "where is my order", "token": auth.issue_token("c1")}))
        await asyncio.sleep(0.1)
        started = time.perf_counter()
        health = await client.get("/health")
        assert health.status_code == 200
        assert time.perf_counter() - started < 0.5
        assert (await pending).status_code == 200


# acomplete() retries with asyncio sleep.
@pytest.mark.asyncio
async def test_acomplete_retries_with_asyncio_sleep(monkeypatch):
    import litellm

    sleeps = []
    calls = {"n": 0}

    # Rate-limited once, then fine.
    async def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception("429 rate limit. Please try again in 1.5s")
        return _reply()

    # Record the wait instead of sleeping.
    async def fake_sleep(seconds):
        sleeps.append(seconds)
    monkeypatch.setattr(litellm, "acompletion", flaky)
    monkeypatch.setattr(ru.asyncio, "sleep", fake_sleep)

    await ru.acomplete(model="groq/x", messages=[])
    assert sleeps == [1.5] and calls["n"] == 2


# Resolve times out to escalation.
@pytest.mark.asyncio
async def test_resolve_times_out_to_escalation(monkeypatch):
    import resolv.api.main as main

    # A run that never finishes in time.
    async def stuck(*args, **kwargs):
        await asyncio.sleep(5)
    monkeypatch.setattr(main, "run_case", stuck)
    monkeypatch.setattr(main, "RESOLVE_TIMEOUT_S", 0.2)

    result = await main._attempt(f"_timeout_{uuid.uuid4().hex[:6]}", "hi", "c1")

    assert result["budget_exceeded"] == "wall_clock"
    assert result["reply"] == loop.FALLBACK_REPLY
    assert result["trail"][-1]["tool"] == "escalate_to_human"


# ADK sessions are deleted after each run.
@pytest.mark.asyncio
async def test_adk_sessions_deleted(monkeypatch):
    created, deleted = [], []

    # A session service that only counts.
    class FakeSessions:
        # Count a create.
        async def create_session(self, **kw):
            created.append(kw["session_id"])

        # Count a delete.
        async def delete_session(self, **kw):
            deleted.append(kw["session_id"])

        # Return a session with the extractor's output in state.
        async def get_session(self, **kw):
            return SimpleNamespace(state={"customer_claim": {"order_id": "ORD-1000"}})

    # A runner that yields one final event.
    class FakeRunner:
        # Accept the real constructor's kwargs.
        def __init__(self, **kw):
            pass

        # One final event with text.
        async def run_async(self, **kw):
            part = SimpleNamespace(text="{}")
            yield SimpleNamespace(is_final_response=lambda: True, content=SimpleNamespace(parts=[part]),
                                  usage_metadata=None, model_version=None)

    monkeypatch.setattr(ru, "_session_service", FakeSessions())
    monkeypatch.setattr(ru, "Runner", FakeRunner)
    agent = SimpleNamespace(model="groq/x")

    out = await ru.run_agent_once(agent, "hi", "customer_claim")

    assert out == {"order_id": "ORD-1000"}
    assert created and created == deleted
