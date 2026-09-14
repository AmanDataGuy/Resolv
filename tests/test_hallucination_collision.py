"""The extractor-hallucination-collision risk, actually tested instead of just named.

eval_report.md flags this as identified but never stress-tested: about 15% of the time the
extractor invents an order ID when the customer gave none (eval/extractor.py's hallucination_rate).
If that invented ID happens to be a REAL order belonging to a different customer, what does the
system actually do with it? Nobody had checked -- this file checks it, honestly, in both
directions: the case the harness catches, and the residual case it does not.

Same mocking pattern as tests/test_loop_orchestration.py -- fake extract()/complete() so this
needs no LLM key and no network, with the real harness (policy.py, tools.py) running underneath
unmocked. ORD-1000's facts are real, read directly from harness/validity.py before writing this
file: paid $639.43, promised 2018-08-17, delivered 2018-08-21 (4 days late, in-window), so its
late_delivery cap is 25% of $639.43 = $159.86.
"""
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
    message that never mentioned one, and also guesses the wrong claim type for it. ORD-1000's
    real situation is 'late' (delivered), not 'never_arrived' -- rule 2 (claim_not_supported)
    catches the mismatch and denies it, regardless of whose order it actually is.
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

    result = await loop.run_case(case, "I never got my package, please help")

    refunds = [r for r in result["trail"] if r["tool"] == "issue_refund" and r["ok"]]
    assert not refunds, "a claim type that doesn't match the real order's situation must be denied"
    denial = next(r for r in result["trail"] if r["tool"] == "issue_refund")
    assert denial["rule_id"] == "claim_not_supported"


@pytest.mark.asyncio
async def test_hallucinated_order_with_matching_claim_still_pays_out(case, monkeypatch):
    """The genuine residual risk, demonstrated rather than asserted away. This time the
    hallucinated claim type happens to COINCIDE with ORD-1000's real situation (it really is a
    late_delivery, really is in-window). The customer's actual message named no order and has no
    real connection to ORD-1000 at all -- but nothing in the system checks that the CALLER owns
    the order, only that the CLAIM about the order is true. The refund goes through.

    This is not a bug in check_refund() -- every rule it has fires correctly. It's the gap named
    in eval_report.md's open items: the order record already carries a real customer_id
    (confirmed by reading harness/validity.py's data directly), but nothing in agents/loop.py,
    api/main.py, or harness/tools.py ever compares a caller's identity against it. Fixing this
    needs an identity check added to the request path, not a change to the policy rules
    themselves -- this test exists to make that gap concrete instead of theoretical.
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
        _fake_response(_FakeMessage(content="I've issued a refund of $159.86 for the delayed delivery.")),
    ])
    monkeypatch.setattr(loop, "extract", lucky_hallucinating_extract)
    monkeypatch.setattr(loop, "complete", lambda **kw: next(lookup_then_refund))

    result = await loop.run_case(case, "hi, can you look into an issue for me")

    refunds = [r for r in result["trail"] if r["tool"] == "issue_refund" and r["ok"]]
    assert len(refunds) == 1 and abs(refunds[0]["args"]["amount_usd"] - 159.86) < 0.01, (
        "documents the real gap: a caller with no stated connection to ORD-1000 receives its "
        "refund, because policy checks the claim's truth, not the caller's ownership of it"
    )
