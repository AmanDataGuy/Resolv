"""
# tests/test_provenance.py

**What:** the run manifest, per-row usage fields, error classification, atomic writes, the openings fixture,
the model-aware cache keys, and the independent invariant grader.
**Why:** a row that misstates what produced it, or a half-written results file, silently corrupts every number
built on it; the invariant grader must catch policy bugs the replay cannot.
**Used by:** pytest (keyless).
"""
import json
import subprocess
import sys

import pytest

from resolv import config
from resolv import invariants
from evals import provenance, simulator

ROLE_KEYS = ("agent_model", "extractor_model", "simulator_model", "judge_model")


# the manifest names every role model, the hashes, seed and start time
def test_manifest_keys(monkeypatch):
    for name in ("AGENT_MODEL", "EXTRACTOR_MODEL", "SIMULATOR_MODEL", "JUDGE_MODEL"):
        monkeypatch.setattr(config, name, "groq/test-model", raising=False)
    m = provenance.manifest("unit", temperature=0.7)
    for key in ("run_id", "git_sha", "git_dirty", "provider", "prompt_sha", "dataset_sha", "openings_sha",
                "temperature", "seed", "started_at", *ROLE_KEYS):
        assert key in m
    assert m["provider"] == "groq" and m["run_id"].endswith("-groq-unit")


# prompt_sha changes when the system prompt changes
def test_prompt_sha_tracks_system_prompt(monkeypatch):
    from resolv.agents import loop
    before = provenance.prompt_sha()
    monkeypatch.setattr(loop, "SYSTEM", loop.SYSTEM + " changed")
    assert provenance.prompt_sha() != before


# usage fields carry the served model and provider-aware usd
def test_usage_fields(monkeypatch):
    monkeypatch.setattr(config, "price_usd", lambda m, p, c: 0.25, raising=False)
    monkeypatch.setattr(config, "is_priced", lambda m: True, raising=False)
    row = provenance.usage_fields("groq/openai/gpt-oss-120b", 1000, 200, "openai/gpt-oss-120b")
    assert row["served_model"] == "openai/gpt-oss-120b" and row["usd"] == 0.25 and row["tokens"] == 1200


# selfhosted models map to the selfhosted provider
def test_provider_of():
    assert provenance.provider_of("hosted_vllm/qwen") == "selfhosted"
    assert provenance.provider_of("gemini/gemini-3.6-flash") == "gemini"


# exceptions are classified so infra failures can be separated from agent failures
@pytest.mark.parametrize("exc, kind", [(Exception("429 Too Many Requests"), "rate_limit"),
                                       (Exception("RESOURCE_EXHAUSTED: quota exceeded"), "quota"),
                                       (TimeoutError("slow"), "timeout"),
                                       (KeyError("x"), "agent_exception")])
def test_error_kind(exc, kind):
    assert provenance.error_kind(exc) == kind


# a write that fails mid-way leaves the original file intact
def test_rewrite_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "rows.jsonl"
    provenance.write_jsonl(path, [{"a": 1}, {"a": 2}])
    original = path.read_text()
    calls = {"n": 0}
    real_dumps = json.dumps

    # raise on the second row
    def flaky(obj, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("disk full")
        return real_dumps(obj, *a, **k)

    monkeypatch.setattr(provenance.json, "dumps", flaky)
    with pytest.raises(RuntimeError):
        provenance.write_jsonl(path, [{"b": 1}, {"b": 2}])
    assert path.read_text() == original


# the committed openings fixture covers every task and records its generator
def test_openings_fixture_covers_all_tasks():
    from evals.tasks import build_tasks
    fixture = json.loads(simulator.FIXTURE.read_text(encoding="utf-8"))
    for t in build_tasks(40):
        assert t["task_id"] in fixture
        if t["tactic"] != "honest":
            assert fixture[t["task_id"]]["generated_by"]


# openings come from the fixture, never the cache, and a missing task fails loudly
def test_opening_reads_fixture_not_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(simulator, "CACHE", tmp_path / "empty")
    simulator._openings.cache_clear()
    monkeypatch.setattr(simulator, "_openings", lambda: {"t-pressure": {"text": "frozen"}})
    assert simulator.opening({"task_id": "t-pressure", "tactic": "pressure"}) == "frozen"
    with pytest.raises(KeyError):
        simulator.opening({"task_id": "missing", "tactic": "pressure"})


# the simulator cache key changes with model and temperature
def test_simulator_cache_key_changes_with_model():
    msgs = [{"role": "user", "content": "hi"}]
    assert simulator.cache_key("a", 0.0, msgs) != simulator.cache_key("b", 0.0, msgs)
    assert simulator.cache_key("a", 0.0, msgs) != simulator.cache_key("a", 0.7, msgs)


# the judge cache key changes with the judge model
def test_judge_cache_key_changes_with_judge_model():
    pytest.importorskip("deepeval")
    from evals import quality
    assert quality.cache_key("a", "reply", "ctx") != quality.cache_key("b", "reply", "ctx")


# an ok refund record in the audit shape
def _ok(order: dict, amount: float, claim: str = "late_delivery", caller="owner", **extra) -> dict:
    return {"tool": "issue_refund", "order_id": order["order_id"], "ok": True,
            "caller_id": order["customer_id"] if caller == "owner" else caller,
            "args": {"amount_usd": amount, "claim_type": claim, **extra}}


# an in-window late order under the auto-approve limit
@pytest.fixture
def late_order():
    from datetime import date
    for o in invariants._orders().values():
        cap = round(o["amount_usd"] * 0.25, 2)
        if o["situation"] == "late" and (invariants.NOW - date.fromisoformat(o["promised_date"])).days <= 90 \
                and cap <= 200:
            return o
    pytest.skip("no suitable order")


# a NaN refund the old policy allowed is flagged
def test_invariants_catch_nan_refund(late_order):
    assert invariants.check([_ok(late_order, float("nan"))])


# a correct refund is clean; over-cap, second refund, wrong caller and untrue claim are each flagged
def test_invariants_cases(late_order):
    cap = round(late_order["amount_usd"] * 0.25, 2)
    assert invariants.check([_ok(late_order, cap)]) == []
    assert invariants.check([_ok(late_order, cap + 1)])
    assert invariants.check([_ok(late_order, 1.0), _ok(late_order, 1.0)])
    assert invariants.check([_ok(late_order, 1.0, caller="someone-else")])
    assert invariants.check([_ok(late_order, 1.0, claim="order_canceled")])
    assert invariants.check([{**_ok(late_order, 1.0), "order_id": "ORD-999999"}])


# importing the invariant grader never imports the policy module
def test_invariants_independent_of_policy_module():
    code = "import sys, resolv.invariants; assert 'resolv.harness.policy' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code], cwd=str(provenance.ROOT)).returncode == 0


# the restated constants still match config and policy
def test_fraction_table_matches_config():
    from resolv.harness import policy
    assert invariants.FRACTION == config.REFUND_CAP_FRACTION
    assert invariants.AUTO_APPROVE_MAX_USD == config.AUTO_APPROVE_MAX_USD
    assert invariants.WINDOW_DAYS == config.CLAIM_WINDOW_DAYS
    assert invariants.NOW == policy.NOW
