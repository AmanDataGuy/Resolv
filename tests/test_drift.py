"""Phase 3 drift detector — PSI math and the alarm logic. No network, no baseline file.

The drift gate can silence a real regression (miss an alarm) or cry wolf (fire on noise); both erode
trust in the board, so the thresholds and the direction of each alarm are pinned here.
"""
from eval import drift


def _baseline(refund=180, escalate=15, deny=5):
    """A synthetic pinned baseline: an action mix (as rows) + a card of safety rates."""
    rows = ([{"paid": 5.0, "escalated": False}] * refund
            + [{"paid": 0.0, "escalated": True}] * escalate
            + [{"paid": 0.0, "escalated": False}] * deny)
    return {"rows": rows,
            "card": {"lookup_before_refund_rate": 1.0, "reply_grounded_rate": 0.99}}


def _rows(refund=36, escalate=3, deny=1, unauthorized=False, grounded=True, looked=True):
    out = []
    for action, count in (("refund", refund), ("escalate", escalate), ("deny", deny)):
        for _ in range(count):
            out.append({"action": action, "unauthorized": unauthorized,
                        "reply_grounded": grounded if action != "deny" else None,
                        "looked_up_first": looked if action == "refund" else None,
                        "latency_s": 12.0, "usd": 0.01})
    return out


class TestPSI:
    def test_identical_distributions_are_zero(self):
        d = {"refund": 0.9, "escalate": 0.075, "deny": 0.025}
        assert drift.psi(d, d) == 0.0

    def test_a_big_shift_clears_the_significant_band(self):
        base = {"refund": 0.9, "escalate": 0.075, "deny": 0.025}
        live = {"refund": 0.3, "escalate": 0.6, "deny": 0.1}   # refunds collapsed into escalations
        assert drift.psi(live, base) > drift.PSI_SIGNIFICANT

    def test_a_new_category_does_not_divide_by_zero(self):
        assert drift.psi({"refund": 1.0}, {"refund": 0.5, "escalate": 0.5}) > 0


class TestReport:
    def test_too_few_requests_reports_not_alarms(self):
        out = drift.report(_rows(refund=5, escalate=0, deny=0), _baseline())
        assert out["verdict"] == "insufficient_data" and "drift" not in out

    def test_matching_traffic_is_stable(self):
        out = drift.report(_rows(), _baseline())
        assert out["verdict"] == "stable" and out["drift"] is False

    def test_any_unauthorized_is_drift(self):
        out = drift.report(_rows(unauthorized=True), _baseline())
        assert out["drift"] is True
        assert any("unauthorized" in a for a in out["alarms"])

    def test_a_grounded_collapse_is_drift(self):
        out = drift.report(_rows(grounded=False), _baseline())
        assert out["drift"] is True
        assert any("reply_grounded_rate" in a for a in out["alarms"])

    def test_action_mix_flip_trips_psi(self):
        # Mostly-escalate live traffic against a mostly-refund baseline.
        out = drift.report(_rows(refund=3, escalate=35, deny=2), _baseline())
        assert out["drift"] is True and out["action_psi"] > drift.PSI_SIGNIFICANT
