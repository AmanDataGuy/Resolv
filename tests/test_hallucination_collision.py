"""
# tests/test_hallucination_collision.py

**What:** what the system does when the extractor invents an order id (it does on some cases with no order
number) that happens to be a real order, possibly someone else's.
**Why:** the risk was named but never tested. These runs fake extract() and acomplete() (no key, no network) with
the real harness underneath, in both directions: the case the harness always caught (claim type mismatch, rule 3)
and the case that slipped through before policy rule 2 (caller_not_order_owner) closed it. ORD-1000 is a real
in-window late order (paid $639.43, late_delivery cap $159.86); its real owner is read at import time.
**Used by:** pytest.
"""
from types import SimpleNamespace

import pytest

import resolv.agents.loop as loop
from resolv.harness.validity import get_order

_ORD_1000_OWNER = get_order("ORD-1000")["customer_id"]


# An async stand-in for acomplete() that returns the next canned response.
def _async_next(responses):
    # the next canned response
    async def fake(**kwargs):
        return next(responses)
    return fake


# A tool call shaped like litellm's.
class _FakeToolCall:
    # name, JSON arguments and call id
    def __init__(self, name: str, arguments: str, call_id: str = "tc1"):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


# An assistant message shaped like litellm's.
class _FakeMessage:
    # text content and/or tool calls
    def __init__(self, content: str | None = None, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls or []

    # the dict the loop appends to its message list
    def model_dump(self) -> dict:
        return {"role": "assistant", "content": self.content}


# A response wrapping one message.
def _fake_response(message: _FakeMessage):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


# A fixed case id; tests/conftest.py gives every test an empty ledger, so the real ORD-1000 is never touched.
@pytest.fixture
def case():
    return "_test_hallucination_collision"


# Invented real order id with the wrong claim type: the owner passes rule 2, rule 3 (claim_not_supported) denies.
@pytest.mark.asyncio
async def test_hallucinated_order_with_mismatched_claim_is_denied(case, monkeypatch):
    # the extractor invents ORD-1000 and the wrong claim type
    async def hallucinating_extract(message: str) -> dict:
        return {"order_id": "ORD-1000", "claim_type": "never_arrived"}  # invented; message had no ID

    lookup_then_refund = iter([
        _fake_response(_FakeMessage(tool_calls=[
            _FakeToolCall("lookup_order", '{"order_id": "ORD-1000"}', "tc1")
        ])),
        _fake_response(_FakeMessage(tool_calls=[
            _FakeToolCall("issue_refund", '{"order_id": "ORD-1000", "claim_type": "never_arrived", "amount_usd": 639.43}', "tc2")
        ])),
        _fake_response(_FakeMessage(content="I'm sorry, I can't verify that this order never arrived.")),
    ])
    monkeypatch.setattr(loop, "extract", hallucinating_extract)
    monkeypatch.setattr(loop, "acomplete", _async_next(lookup_then_refund))

    result = await loop.run_case(case, "I never got my package, please help", caller_id=_ORD_1000_OWNER)

    refunds = [r for r in result["trail"] if r["tool"] == "issue_refund" and r["ok"]]
    assert not refunds, "a claim type that doesn't match the real order's situation must be denied"
    denial = next(r for r in result["trail"] if r["tool"] == "issue_refund")
    assert denial["rule_id"] == "claim_not_supported"


# The gap rule 2 closed: an invented id whose claim happens to be true is denied to a caller who doesn't own it.
@pytest.mark.asyncio
async def test_hallucinated_order_with_matching_claim_now_denied_for_wrong_caller(case, monkeypatch):
    # the extractor invents ORD-1000 with a claim type that happens to be true
    async def lucky_hallucinating_extract(message: str) -> dict:
        return {"order_id": "ORD-1000", "claim_type": "late_delivery"}  # invented; happens to be true

    lookup_then_refund = iter([
        _fake_response(_FakeMessage(tool_calls=[
            _FakeToolCall("lookup_order", '{"order_id": "ORD-1000"}', "tc1")
        ])),
        _fake_response(_FakeMessage(tool_calls=[
            _FakeToolCall("issue_refund", '{"order_id": "ORD-1000", "claim_type": "late_delivery", "amount_usd": 159.86}', "tc2")
        ])),
        _fake_response(_FakeMessage(content="I'm sorry, I can't verify that this order belongs to you.")),
    ])
    monkeypatch.setattr(loop, "extract", lucky_hallucinating_extract)
    monkeypatch.setattr(loop, "acomplete", _async_next(lookup_then_refund))

    result = await loop.run_case(
        case, "hi, can you look into an issue for me", caller_id="an_unrelated_caller"
    )

    refunds = [r for r in result["trail"] if r["tool"] == "issue_refund" and r["ok"]]
    assert not refunds, "a caller with no stated connection to the order must not be paid its refund"
    denial = next(r for r in result["trail"] if r["tool"] == "issue_refund")
    assert denial["rule_id"] == "caller_not_order_owner"


# No over-blocking: the real owner with a true claim is still paid, even when the order number was not typed.
@pytest.mark.asyncio
async def test_hallucinated_order_number_but_correct_caller_still_pays_out(case, monkeypatch):
    # the extractor guesses ORD-1000, which is right
    async def hallucinating_but_right_extract(message: str) -> dict:
        return {"order_id": "ORD-1000", "claim_type": "late_delivery"}

    lookup_then_refund = iter([
        _fake_response(_FakeMessage(tool_calls=[
            _FakeToolCall("lookup_order", '{"order_id": "ORD-1000"}', "tc1")
        ])),
        _fake_response(_FakeMessage(tool_calls=[
            _FakeToolCall("issue_refund", '{"order_id": "ORD-1000", "claim_type": "late_delivery", "amount_usd": 159.86}', "tc2")
        ])),
        _fake_response(_FakeMessage(content="I've issued a refund of $159.86 for the delayed delivery.")),
    ])
    monkeypatch.setattr(loop, "extract", hallucinating_but_right_extract)
    monkeypatch.setattr(loop, "acomplete", _async_next(lookup_then_refund))

    result = await loop.run_case(case, "hi, can you look into an issue for me", caller_id=_ORD_1000_OWNER)

    refunds = [r for r in result["trail"] if r["tool"] == "issue_refund" and r["ok"]]
    assert len(refunds) == 1 and abs(refunds[0]["args"]["amount_usd"] - 159.86) < 0.01
