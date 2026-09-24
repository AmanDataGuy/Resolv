"""The extractor-hallucination-collision risk, actually tested instead of just named.

eval_report.md flags this as identified but never stress-tested: about 15% of the time the
extractor invents an order ID when the customer gave none (eval/extractor.py's hallucination_rate).
If that invented ID happens to be a REAL order belonging to a different customer, what does the
system actually do with it? Nobody had checked -- this file checks it, honestly, in both
directions: the case the harness catches, and the case that USED TO slip through before
harness/policy.py's rule 2 (caller_not_order_owner) closed it.

Same mocking pattern as tests/test_loop_orchestration.py -- fake extract()/complete() so this
needs no LLM key and no network, with the real harness (policy.py, tools.py) running underneath
unmocked. ORD-1000's facts are real, read directly from harness/validity.py before writing this
file: paid $639.43, promised 2018-08-17, delivered 2018-08-21 (4 days late, in-window), so its
late_delivery cap is 25% of $639.43 = $159.86. Its real customer_id is read the same way, at
import time, so the tests below use the actual owner rather than a hand-picked string.
"""
from types import SimpleNamespace

import pytest

import agents.loop as loop
from harness import audit
from harness.validity import get_order

_ORD_1000_OWNER = get_order("ORD-1000")["customer_id"]


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


@pytest.fixture
def case():
    case_id = "_test_hallucination_collision"
    audit.clear(case_id)
    audit.clear_order("ORD-1000")
    yield case_id
    audit.clear(case_id)
    audit.clear_order("ORD-1000")


@pytest.mark.asyncio
async def test_hallucinated_order_with_mismatched_claim_is_denied(case, monkeypatch):
    """The likely case: the extractor hallucinates a real order ID (ORD-1000) for a customer
    message that never mentioned one, and also guesses the wrong claim type for it. The caller
    here IS ORD-1000's real owner (rule 2 passes) -- ORD-1000's real situation is 'late'
    (delivered), not 'never_arrived', so rule 3 (claim_not_supported) catches the mismatch.
    """
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
    monkeypatch.setattr(loop, "complete", lambda **kw: next(lookup_then_refund))

    result = await loop.run_case(case, "I never got my package, please help", caller_id=_ORD_1000_OWNER)

    refunds = [r for r in result["trail"] if r["tool"] == "issue_refund" and r["ok"]]
    assert not refunds, "a claim type that doesn't match the real order's situation must be denied"
    denial = next(r for r in result["trail"] if r["tool"] == "issue_refund")
    assert denial["rule_id"] == "claim_not_supported"


@pytest.mark.asyncio
async def test_hallucinated_order_with_matching_claim_now_denied_for_wrong_caller(case, monkeypatch):
    """The gap this rule closes. The hallucinated claim type happens to COINCIDE with ORD-1000's
    real situation (it really is a late_delivery, really is in-window) -- before rule 2 existed,
    that coincidence alone was enough to pay out, because nothing compared the CALLER to the
    order, only the CLAIM to the record. Now a caller with no stated connection to ORD-1000 is
    denied before the claim is even read, whether or not it happens to be true.
    """
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
    monkeypatch.setattr(loop, "complete", lambda **kw: next(lookup_then_refund))

    result = await loop.run_case(
        case, "hi, can you look into an issue for me", caller_id="an_unrelated_caller"
    )

    refunds = [r for r in result["trail"] if r["tool"] == "issue_refund" and r["ok"]]
    assert not refunds, "a caller with no stated connection to the order must not be paid its refund"
    denial = next(r for r in result["trail"] if r["tool"] == "issue_refund")
    assert denial["rule_id"] == "caller_not_order_owner"


@pytest.mark.asyncio
async def test_hallucinated_order_number_but_correct_caller_still_pays_out(case, monkeypatch):
    """The fix must not over-block: when the extractor merely failed to pull the order NUMBER out
    of the prose (a real limitation -- customers often don't type it) but the caller genuinely IS
    ORD-1000's owner and the claim is true, the refund still goes through. Closing the ownership
    gap is not the same as requiring the customer to state their own order number correctly.
    """
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
    monkeypatch.setattr(loop, "complete", lambda **kw: next(lookup_then_refund))

    result = await loop.run_case(case, "hi, can you look into an issue for me", caller_id=_ORD_1000_OWNER)

    refunds = [r for r in result["trail"] if r["tool"] == "issue_refund" and r["ok"]]
    assert len(refunds) == 1 and abs(refunds[0]["args"]["amount_usd"] - 159.86) < 0.01
