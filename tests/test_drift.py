"""
# tests/test_drift.py

**What:** the drift detector: PSI math, alarm directions, the live-baseline requirement, and that baseline rows
are bucketed with the monitor's precedence.
**Why:** a drift board that misses a real shift or cries wolf stops being read.
**Used by:** pytest (keyless).
"""
from resolv import monitor
from evals import drift


# a live-traffic baseline as freeze_baseline writes it
def _baseline(refund=180, escalate=15, deny=5):
    rows = ([{"action": "refund"}] * refund + [{"action": "escalate"}] * escalate + [{"action": "deny"}] * deny)
    return {"rows": rows, "card": {"lookup_before_refund_rate": 1.0, "reply_grounded_rate": 0.99}}


# live telemetry rows with the given mix and flags
def _rows(refund=36, escalate=3, deny=1, unauthorized=False, grounded=True, looked=True):
    out = []
    for action, count in (("refund", refund), ("escalate", escalate), ("deny", deny)):
        for _ in range(count):
            out.append({"action": action, "unauthorized": unauthorized, "invariant_violations": [],
                        "reply_grounded": grounded if action != "deny" else None,
                        "looked_up_first": looked if action == "refund" else None, "latency_s": 12.0, "usd": 0.01})
    return out


# identical distributions score zero
def test_psi_identical_is_zero():
    d = {"refund": 0.9, "escalate": 0.075, "deny": 0.025}
    assert drift.psi(d, d) == 0.0


# a large shift clears the significant band
def test_psi_big_shift():
    assert drift.psi({"refund": 0.3, "escalate": 0.6, "deny": 0.1},
                     {"refund": 0.9, "escalate": 0.075, "deny": 0.025}) > drift.PSI_SIGNIFICANT


# a category missing on one side does not divide by zero
def test_psi_new_category():
    assert drift.psi({"refund": 1.0}, {"refund": 0.5, "escalate": 0.5}) > 0


# too few requests or no live baseline report insufficient data, never an alarm
def test_insufficient_data():
    assert drift.report(_rows(refund=5, escalate=0, deny=0), _baseline())["verdict"] == "insufficient_data"
    assert drift.report(_rows(), None)["verdict"] == "insufficient_data"


# matching traffic is stable
def test_matching_traffic_is_stable():
    out = drift.report(_rows(), _baseline())
    assert out["verdict"] == "stable" and out["drift"] is False


# any unauthorized action is drift
def test_any_unauthorized_is_drift():
    out = drift.report(_rows(unauthorized=True), _baseline())
    assert out["drift"] is True and any("unauthorized" in a for a in out["alarms"])


# a groundedness collapse is drift
def test_grounded_collapse_is_drift():
    assert drift.report(_rows(grounded=False), _baseline())["drift"] is True


# an action-mix flip trips PSI
def test_action_mix_flip():
    out = drift.report(_rows(refund=3, escalate=35, deny=2), _baseline())
    assert out["drift"] is True and out["action_psi"] > drift.PSI_SIGNIFICANT


# a row that both refunded and escalated is a refund, exactly as monitor._action decides
def test_drift_uses_monitor_precedence():
    row = {"paid": 5.0, "escalated": True}
    trail = [{"tool": "issue_refund", "ok": True}, {"tool": "escalate_to_human", "ok": True}]
    assert drift._row_action(row) == monitor._action(trail) == "refund"


# freezing keeps only the first week of traffic
def test_freeze_baseline_keeps_first_week():
    rows = [{"ts": "2026-11-01T00:00:00", "action": "refund"}, {"ts": "2026-11-07T23:00:00", "action": "deny"},
            {"ts": "2026-11-09T00:00:00", "action": "escalate"}]
    assert [r["action"] for r in drift.freeze_baseline(rows)["rows"]] == ["refund", "deny"]
