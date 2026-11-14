"""
# tests/test_crash_resume.py

**What:** durable execution, in-process and keyless: a resolution "crashes" (an exception) after some steps, then
the same case_id is run again and must finish from its step journal.
**Why:** the two promises of the journal are that completed steps are not redone (no second model call, no second
lookup) and that a refund which committed just before the crash is still paid exactly once, through its
idempotency key. scripts/crash_resume_demo.py shows the same with a real process kill.
**Used by:** pytest.
"""
import json

import litellm
import pytest

from resolv.agents import loop
from resolv.config import AUTO_APPROVE_MAX_USD, REFUND_CAP_FRACTION
from resolv.harness import audit, ledger

CASE = "_test_crash_resume"


# Journal steps in this test even though RESOLV_ENV=test turns durability off by default.
@pytest.fixture(autouse=True)
def _durable(monkeypatch):
    monkeypatch.setenv("RESOLV_DURABLE", "1")


# (order_id, owner, amount owed) for a refund-owed late order, and a fixed extractor read for it.
@pytest.fixture
def owed(fresh_order, monkeypatch):
    amount = min(round(fresh_order["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2), AUTO_APPROVE_MAX_USD)

    # the extractor's read, without a model
    async def fake_extract(message):
        return {"order_id": fresh_order["order_id"], "claim_type": "late_delivery"}
    monkeypatch.setattr(loop, "extract", fake_extract)
    return fresh_order["order_id"], fresh_order["customer_id"], amount


# A scripted model (lookup -> refund -> reply, chosen by tool results seen) that raises on call number crash_on.
def model(order_id: str, amount: float, calls: list, crash_on: int | None = None):
    # one fake model call
    async def fake(**kwargs):
        calls.append(1)
        if len(calls) == crash_on:
            raise RuntimeError("simulated crash")
        seen = sum(1 for m in kwargs["messages"] if m.get("role") == "tool")
        plan = [("lookup_order", {"order_id": order_id}),
                ("issue_refund", {"order_id": order_id, "claim_type": "late_delivery", "amount_usd": amount})]
        if seen < len(plan):
            name, args = plan[seen]
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"tc{seen}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
        else:
            message = {"role": "assistant", "content": f"I've issued a refund of ${amount:.2f} for the late delivery."}
        return litellm.ModelResponse(model="fake", choices=[{"index": 0, "finish_reason": "stop", "message": message}])
    return fake


# Ok issue_refund records in the case trail.
def _paid() -> list[dict]:
    return [r for r in audit.read(CASE) if r["tool"] == "issue_refund" and r["ok"]]


# A crash on the second model call resumes without redoing the extract, the first model call or the lookup.
@pytest.mark.asyncio
async def test_resume_replays_journaled_steps(owed, monkeypatch):
    order_id, owner, amount = owed
    first: list = []
    monkeypatch.setattr(loop, "acomplete", model(order_id, amount, first, crash_on=2))
    with pytest.raises(RuntimeError, match="simulated crash"):
        await loop.run_case(CASE, f"My order {order_id} was late", caller_id=owner)
    assert ledger.steps(CASE)[0]["order_id"] == order_id, "the extractor read is the first journaled step"

    second: list = []
    monkeypatch.setattr(loop, "acomplete", model(order_id, amount, second))
    result = await loop.run_case(CASE, f"My order {order_id} was late", caller_id=owner)

    assert len(second) == 2, "only the refund call and the reply are new; the first call came from the journal"
    assert sum(1 for r in result["trail"] if r["tool"] == "lookup_order") == 1, "the lookup must not run again"
    assert len(_paid()) == 1 and audit.verify_chain() == (True, None)


# A crash after the refund committed but before it was journaled re-runs the tool with the same key: paid once.
@pytest.mark.asyncio
async def test_refund_committed_before_journal_is_paid_once(owed, monkeypatch):
    order_id, owner, amount = owed
    real_add = ledger.add_step

    # journal every step except the refund's result, then die: the worst-placed crash
    def crash_after_refund(case_id, step, kind, payload):
        if kind == "tool" and str(payload).startswith("Refunded"):
            raise RuntimeError("simulated crash")
        real_add(case_id, step, kind, payload)

    monkeypatch.setattr(loop, "acomplete", model(order_id, amount, []))
    monkeypatch.setattr(ledger, "add_step", crash_after_refund)
    with pytest.raises(RuntimeError, match="simulated crash"):
        await loop.run_case(CASE, f"My order {order_id} was late", caller_id=owner)
    assert len(_paid()) == 1, "the refund itself committed before the crash"

    monkeypatch.setattr(ledger, "add_step", real_add)
    result = await loop.run_case(CASE, f"My order {order_id} was late", caller_id=owner)

    assert len(_paid()) == 1, "the re-run refund must replay its key, not pay again"
    assert "idempotent_replay" in [r["rule_id"] for r in result["trail"]]
    assert len([r for r in ledger.refunds(("settled",)) if r["order_id"] == order_id]) == 1
    assert audit.verify_chain() == (True, None)


# With RESOLV_DURABLE unset, test runs journal nothing (eval and tests stay throwaway).
@pytest.mark.asyncio
async def test_journal_off_by_default_in_test(owed, monkeypatch):
    order_id, owner, amount = owed
    monkeypatch.delenv("RESOLV_DURABLE")
    monkeypatch.setattr(loop, "acomplete", model(order_id, amount, []))
    await loop.run_case(CASE, f"My order {order_id} was late", caller_id=owner)
    assert ledger.steps(CASE) == []
