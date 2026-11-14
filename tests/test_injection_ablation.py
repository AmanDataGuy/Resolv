"""
# tests/test_injection_ablation.py

**What:** the ablation's one dangerous property (the OFF arm must always restore the real `check_refund`) and the
injection payload table.
**Why:** an unrestored stub would leave every later run in the process with no policy engine, and nothing in the
output would say so.
**Used by:** pytest (keyless).
"""
from evals import ablation
from evals.injection import MARKER, PAYLOADS, TARGETS
from resolv.harness import policy, tools


# the OFF arm's stub allows anything
def test_stub_allows_everything():
    assert ablation._allow_all("ORD-1", "late_delivery", 1e9, [], None).action == "allow"


# both arms leave the real check_refund in place afterwards
def test_arms_restore_check_refund():
    originals = (tools.check_refund, policy.check_refund)
    ablation.run_arm([], n=1, arm="on")
    ablation.run_arm([], n=1, arm="off")
    assert (tools.check_refund, policy.check_refund) == originals


# the stub is removed even when a run raises
def test_restored_even_when_a_run_raises():
    originals = (tools.check_refund, policy.check_refund)
    try:
        ablation.run_arm([{"task_id": "boom"}], n=1, arm="off")
    except Exception:
        pass
    assert (tools.check_refund, policy.check_refund) == originals


# unauthorized dollars are reported gross and as excess over what was owed
def test_unauthorized_usd_gross_and_excess():
    rows = [{"unauthorized": True, "paid": 100.0, "excess_usd": 75.0}, {"unauthorized": False, "paid": 5.0}]
    assert ablation.unauthorized_usd(rows) == {"gross": 100.0, "excess": 75.0}


# every payload declares a target, and the exfiltration payload plants the marker
def test_payload_table():
    assert set(TARGETS) == set(PAYLOADS)
    assert MARKER in PAYLOADS["exfiltrate_prompt"]
