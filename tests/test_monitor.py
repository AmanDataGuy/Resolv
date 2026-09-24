"""The online-eval monitor — aggregation, the safety replay, and the Langfuse no-op. No network.

The live board reports what production is doing. A bug in `scorecard` or `_unauthorized` means the
system can look healthy while it isn't — the same stakes as the offline grader, so the same gating.
The Langfuse layer is tested only for the one property that matters here: with no keys it must be a
silent no-op, never a crash on the request path.
"""
from eval import monitor, observability
from harness.validity import get_order


def _rec(order_id, tool="issue_refund", ok=True, amount=10.0,
         claim="late_delivery", rule_id="within_policy", caller_id=None):
    return {"tool": tool, "order_id": order_id, "ok": ok, "rule_id": rule_id,
            "args": {"amount_usd": amount, "claim_type": claim}, "caller_id": caller_id}


class TestAction:
    def test_a_successful_refund_wins(self):
        trail = [_rec("ORD-1000"), _rec("ORD-1000", tool="escalate_to_human")]
        assert monitor._action(trail) == "refund"

    def test_escalate_when_no_refund_moved(self):
        assert monitor._action([_rec("ORD-1000", tool="escalate_to_human")]) == "escalate"

    def test_deny_when_nothing_moved(self):
        assert monitor._action([_rec("ORD-1000", ok=False)]) == "deny"
        assert monitor._action([]) == "deny"


class TestUnauthorizedReplay:
    """The one safety signal that survives having no answer key — it asks policy, not a label."""

    def test_a_refund_on_an_unknown_order_is_caught(self):
        assert monitor._unauthorized([_rec("ORD-999999", amount=25.0)]) is True

    def test_a_failed_attempt_is_never_judged(self):
        assert monitor._unauthorized([_rec("ORD-999999", ok=False, amount=25.0)]) is False

    def test_an_empty_trail_is_clean(self):
        assert monitor._unauthorized([]) is False


class TestScorecard:
    @staticmethod
    def _rows():
        return [
            {"action": "refund", "unauthorized": False, "reply_grounded": True,
             "looked_up_first": True, "latency_s": 10.0, "usd": 0.01},
            {"action": "deny", "unauthorized": False, "reply_grounded": True,
             "looked_up_first": None, "latency_s": 30.0, "usd": 0.03},
        ]

    def test_counts_and_action_breakdown(self):
        card = monitor.scorecard(self._rows())
        assert card["requests"] == 2 and card["actions"] == {"refund": 1, "deny": 1}

    def test_lookup_rate_excludes_the_rows_that_never_refunded(self):
        """The deny row has looked_up_first=None (no refund attempted) and must stay out of the
        denominator — otherwise a healthy agent looks like it skipped a lookup."""
        assert monitor.scorecard(self._rows())["lookup_before_refund_rate"] == 1.0

    def test_unauthorized_is_the_headline_and_is_zero(self):
        assert monitor.scorecard(self._rows())["unauthorized_rate"] == 0.0

    def test_latency_reports_the_tail(self):
        card = monitor.scorecard(self._rows())
        assert card["p50_latency_s"] == 10.0 and card["p95_latency_s"] == 30.0

    def test_cost_totals_and_per_request(self):
        card = monitor.scorecard(self._rows())
        assert card["usd_total"] == 0.04 and card["usd_per_request"] == 0.02

    def test_escalation_rate(self):
        rows = self._rows() + [{"action": "escalate", "unauthorized": False, "reply_grounded": None,
                                "looked_up_first": None, "latency_s": 5.0, "usd": 0.0}]
        assert monitor.scorecard(rows)["escalation_rate"] == round(1 / 3, 4)

    def test_empty_window_does_not_crash(self):
        assert monitor.scorecard([]) == {"requests": 0}


def test_record_appends_a_row_and_returns_it(tmp_path, monkeypatch):
    """record() writes one telemetry row per live request and hands the same dict back for Langfuse.

    TELEMETRY is redirected to a tmp file so the test never touches the real log.
    """
    monkeypatch.setattr(monitor, "TELEMETRY", tmp_path / "live.jsonl")
    owner = get_order("ORD-1000")["customer_id"]
    result = {
        "reply": "Refunded $10.00 on ORD-1000.",
        "steps": 3,
        "trail": [_rec("ORD-1000", tool="lookup_order", rule_id="lookup_hit"),
                  _rec("ORD-1000", amount=10.0, caller_id=owner)],
    }
    row = monitor.record("api-x", "my order is late", result, 12.5, 1000, 500)

    assert row["action"] == "refund" and row["looked_up_first"] is True
    assert row["unauthorized"] is False and row["reply_grounded"] is True
    assert row["latency_s"] == 12.5 and row["tokens"] == 1500 and row["usd"] > 0

    back = monitor.read(tmp_path / "live.jsonl")
    assert len(back) == 1 and back[0]["case_id"] == "api-x"


def test_langfuse_is_a_clean_noop_without_keys(monkeypatch):
    """The safety property of the whole observability layer: no keys -> disabled -> never raises."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    observability._client.cache_clear()
    try:
        assert observability.enabled() is False
        # Must not raise even with a live-shaped payload — the endpoint depends on this.
        observability.log_request("x", "msg", {"reply": "y", "trail": []},
                                  {"action": "deny", "unauthorized": False, "reply_grounded": True})
    finally:
        observability._client.cache_clear()
