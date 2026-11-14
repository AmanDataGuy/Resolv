"""
# tests/test_regression_replay.py

**What:** the PR gate's keyless replay: a synthetic scripted trajectory proves the replay machinery, and every
recorded cassette in `evals/regression/cassettes/` must reproduce its pinned graded row.
**Why:** loop + harness + grader must behave identically on recorded model outputs; a prompt change fails loudly
with the command to re-record.
**Used by:** pytest (keyless).
"""
import json

import pytest

from evals import provenance, replay
from evals.tasks import build_tasks


# a litellm-shaped response dict, with or without one tool call
def _resp(content=None, tool=None, args=None):
    msg = {"role": "assistant", "content": content}
    if tool:
        msg["tool_calls"] = [{"id": f"call-{tool}", "type": "function",
                              "function": {"name": tool, "arguments": json.dumps(args)}}]
    return {"choices": [{"index": 0, "message": msg, "finish_reason": "stop"}], "model": "fake/cassette",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


# a scripted honest refund: lookup, refund the cap, reply
def test_synthetic_cassette_replays_to_a_resolved_row(monkeypatch):
    task = next(t for t in build_tasks(40) if t["expected"] == "refund" and t["tactic"] == "honest")
    cassette = {
        "task": task, "opening": task["message"], "claim": {"order_id": task["order_id"]},
        "customer_turns": None,
        "responses": [
            _resp(tool="lookup_order", args={"order_id": task["order_id"]}),
            _resp(tool="issue_refund", args={"order_id": task["order_id"], "claim_type": task["claim_type"],
                                             "amount_usd": task["expected_amount"]}),
            _resp(content=f"I've refunded ${task['expected_amount']:.2f} to your card."),
        ],
    }
    row = replay.replay(cassette, monkeypatch)
    assert row["resolved"] is True and row["unauthorized"] is False and row["invariant_violations"] == []


# every recorded cassette reproduces its pinned row under the current prompt
@pytest.mark.parametrize("path", replay.cassettes(), ids=lambda p: p.stem)
def test_recorded_cassette(path, monkeypatch):
    cassette = replay.load(path)
    if cassette["prompt_sha"] != provenance.prompt_sha():
        pytest.fail("prompt changed: re-record cassettes with scripts/record_cassettes.py (costs ~$0 on Groq)")
    row = replay.replay(cassette, monkeypatch)
    assert {k: row[k] for k in replay.PINNED} == cassette["expected_row"]
