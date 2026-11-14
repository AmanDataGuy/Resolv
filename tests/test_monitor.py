"""
# tests/test_monitor.py

**What:** the online monitor: action precedence, the policy replay with the recorded caller, the live scorecard,
record() with provider-aware cost and invariants, the alert hook, and the Langfuse no-op.
**Why:** a bug here makes production look healthy when it is not.
**Used by:** pytest (keyless).
"""
from resolv import monitor, observability
from resolv.harness.validity import get_order


# one audit record in the shape resolv/harness/audit.py writes
def _rec(order_id, tool="issue_refund", ok=True, amount=10.0, claim="late_delivery", rule_id="within_policy",
         caller_id=None):
    return {"tool": tool, "order_id": order_id, "ok": ok, "rule_id": rule_id,
            "args": {"amount_usd": amount, "claim_type": claim}, "caller_id": caller_id}


# a successful refund wins over an escalation; nothing moved is a denial
def test_action_precedence():
    assert monitor._action([_rec("ORD-1000"), _rec("ORD-1000", tool="escalate_to_human")]) == "refund"
    assert monitor._action([_rec("ORD-1000", tool="escalate_to_human")]) == "escalate"
    assert monitor._action([_rec("ORD-1000", ok=False)]) == monitor._action([]) == "deny"


# replay catches unknown orders and ignores failed attempts
def test_unauthorized_replay():
    owner = get_order("ORD-1000")["customer_id"]
    assert monitor._unauthorized([_rec("ORD-999999", amount=25.0, caller_id=owner)]) is True
    assert monitor._unauthorized([_rec("ORD-999999", ok=False)]) is False
    assert monitor._unauthorized([]) is False


# an ok refund with no recorded caller is itself unauthorized
def test_ok_refund_with_null_caller_is_flagged():
    assert monitor._unauthorized([_rec("ORD-1000", amount=1.0, caller_id=None)]) is True


# lookup-before-refund is None when nothing was refunded
def test_looked_up_first():
    assert monitor._looked_up_first([_rec("ORD-1000", tool="lookup_order"), _rec("ORD-1000")]) is True
    assert monitor._looked_up_first([_rec("ORD-1000")]) is False
    assert monitor._looked_up_first([]) is None


# the live board's counts, denominators, tail and cost
def test_scorecard():
    rows = [{"action": "refund", "unauthorized": False, "invariant_violations": [], "reply_grounded": True,
             "looked_up_first": True, "latency_s": 10.0, "usd": 0.01},
            {"action": "deny", "unauthorized": False, "invariant_violations": [], "reply_grounded": True,
             "looked_up_first": None, "latency_s": 30.0, "usd": 0.03}]
    card = monitor.scorecard(rows)
    assert card["requests"] == 2 and card["actions"] == {"refund": 1, "deny": 1}
    assert card["lookup_before_refund_rate"] == 1.0 and card["unauthorized_rate"] == 0.0
    assert card["invariant_violation_rate"] == 0.0
    assert card["p50_latency_s"] == 10.0 and card["p95_latency_s"] == 30.0
    assert card["usd_total"] == 0.04 and card["usd_per_request"] == 0.02
    assert monitor.scorecard([]) == {"requests": 0}


# record() appends a row priced by config.price_usd and returns it
def test_record_appends_a_row(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "TELEMETRY", tmp_path / "live.jsonl")
    monkeypatch.setattr(monitor.config, "price_usd", lambda m, p, c: 0.5, raising=False)
    owner = get_order("ORD-1000")["customer_id"]
    result = {"reply": "Refunded $10.00 on ORD-1000.", "steps": 3,
              "trail": [_rec("ORD-1000", tool="lookup_order", rule_id="lookup_hit", caller_id=owner),
                        _rec("ORD-1000", amount=10.0, caller_id=owner)]}
    row = monitor.record("api-x", "my order is late", result, 12.5, 1000, 500, model="groq/x")
    assert row["action"] == "refund" and row["looked_up_first"] is True and row["unauthorized"] is False
    assert row["usd"] == 0.5 and row["model"] == "groq/x" and row["tokens"] == 1500
    assert monitor.read(tmp_path / "live.jsonl")[0]["case_id"] == "api-x"


# an unsafe row fires the alert hook; a clean one does not
def test_alert_fires_only_on_unsafe_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "TELEMETRY", tmp_path / "live.jsonl")
    monkeypatch.setattr(monitor.config, "price_usd", lambda m, p, c: 0.0, raising=False)
    fired = []
    monkeypatch.setattr(monitor, "alert", fired.append)
    monitor.record("bad", "x", {"reply": "", "trail": [_rec("ORD-999999", amount=5.0)]}, 1.0, 0, 0, model="m")
    monitor.record("ok", "x", {"reply": "", "trail": []}, 1.0, 0, 0, model="m")
    assert [r["case_id"] for r in fired] == ["bad"]


# the webhook is a no-op without its env var
def test_alert_noop_without_webhook(monkeypatch):
    monkeypatch.delenv("RESOLV_ALERT_WEBHOOK", raising=False)
    monitor.alert({"case_id": "x", "unauthorized": True, "invariant_violations": []})


# Langfuse with no keys is a silent no-op
def test_langfuse_is_a_clean_noop_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    if hasattr(observability, "_client") and hasattr(observability._client, "cache_clear"):
        observability._client.cache_clear()
    assert observability.enabled() is False
