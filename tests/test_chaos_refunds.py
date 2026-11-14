"""
# tests/test_chaos_refunds.py

**What:** 200 seeded chaos runs of the refund saga: random provider faults (500 before, timeout after commit,
decline), a random crash point (after reserving, after the provider paid but before settling, inside the
reconciler), 1-3 concurrent attempts on one order, sometimes replaying one idempotency key. Then the reconciler
runs until quiet, and seven invariants must hold.
**Why:** exactly-once in effect is a claim about every interleaving, not the happy path. Invariants:
I1 at most one settled refund per order; I2 the provider never pays more than the cap; I3 every provider refund
has a settled ledger row (no orphan money); I4 every settled row has a provider refund (no phantom settlement);
I5 nothing left pending/unknown; I6 every refund row points at an allow audit record; I7 the chain verifies.
Reproduce one seed with CHAOS_SEED=<n>. Threads make the interleaving vary between runs of the same seed.
**Used by:** pytest.
"""
import os
import random
import threading
from collections import Counter

from resolv.config import REFUND_CAP_FRACTION
from resolv.harness import audit, ledger, tools
from resolv.integrations import payments

CRASH_POINTS = (None, "after_reserve", "after_payment", "in_reconcile")


# A simulated process death: a BaseException, so no `except Exception` in the saga swallows it.
class Crash(BaseException):
    pass


# Wrap fn so its first call raises Crash (after running fn when after_call is set); later calls run normally.
def _crash_once(fn, after_call: bool):
    fired = threading.Event()

    # The trap itself.
    def wrapper(*args, **kwargs):
        if fired.is_set():
            return fn(*args, **kwargs)
        fired.set()
        if after_call:
            fn(*args, **kwargs)
        raise Crash()

    return wrapper


# One seeded run; returns which fault classes it exercised.
def _run(seed: int, order: dict, tmp_path, monkeypatch) -> Counter:
    rng = random.Random(seed)
    monkeypatch.setenv("RESOLV_DB", str(tmp_path / f"chaos-{seed}.db"))
    monkeypatch.setenv("RESOLV_PSP_DB", str(tmp_path / f"psp-{seed}.db"))
    plan = payments.FaultPlan(fail_before=rng.choice([0, 0.3]), timeout_after_commit=rng.choice([0, 0.3]),
                              decline=rng.choice([0, 0.2]), seed=seed)
    monkeypatch.setattr(payments, "FAULTS", plan)
    crash = rng.choice(CRASH_POINTS)
    real_refund, real_settle = payments.refund, tools._settle
    if crash in ("after_reserve", "after_payment"):
        monkeypatch.setattr(payments, "refund", _crash_once(real_refund, after_call=crash == "after_payment"))

    oid, owner = order["order_id"], order["customer_id"]
    amount = min(round(order["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2), 50.0)
    keys = [f"k{seed}-{rng.randrange(2)}" for _ in range(rng.randint(1, 3))]  # a repeated key = duplicate delivery

    # One attempt; a crash ends it the way a dead process would.
    def attempt(i: int, key: str) -> None:
        try:
            tools.issue_refund(f"chaos-{seed}-{i}", oid, "late_delivery", amount, owner, idempotency_key=key)
        except Crash:
            pass

    threads = [threading.Thread(target=attempt, args=(i, k)) for i, k in enumerate(keys)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    monkeypatch.setattr(payments, "refund", real_refund)
    if crash == "in_reconcile":
        monkeypatch.setattr(tools, "_settle", _crash_once(real_settle, after_call=False))
    for _ in range(6):
        try:
            tools.reconcile_once(min_age_s=0)
        except Crash:
            continue
        if not ledger.refunds(("pending", "unknown")):
            break
    monkeypatch.setattr(tools, "_settle", real_settle)

    _check_invariants(seed, oid, round(amount * 100))
    statuses = Counter(r["status"] for r in ledger.refunds())
    return Counter({f"crash:{crash}": 1, f"attempts:{len(keys)}": 1, "dup_key": int(len(set(keys)) < len(keys))}) \
        + Counter({f"status:{s}": n for s, n in statuses.items()})


# Assert I1-I7 for one seed.
def _check_invariants(seed: int, order_id: str, cap_cents: int) -> None:
    rows = ledger.refunds()
    settled = [r for r in rows if r["status"] == "settled"]
    psp = [p for p in payments.all_refunds() if p["status"] == "succeeded"]
    assert len([r for r in settled if r["order_id"] == order_id]) <= 1, f"seed {seed}: I1 two settled refunds"
    assert sum(p["amount_cents"] for p in psp if p["order_id"] == order_id) <= cap_cents, f"seed {seed}: I2 over cap"
    assert {p["psp_ref"] for p in psp} <= {r["psp_ref"] for r in settled}, f"seed {seed}: I3 orphan money"
    assert {r["psp_ref"] for r in settled} <= {p["psp_ref"] for p in psp}, f"seed {seed}: I4 phantom settlement"
    assert not [r for r in rows if r["status"] in ("pending", "unknown")], f"seed {seed}: I5 unresolved refund"
    allows = {a["seq"] for case in {r["case_id"] for r in rows} for a in audit.read(case) if a["action"] == "allow"}
    assert all(r["audit_seq"] in allows for r in rows), f"seed {seed}: I6 refund without an allow record"
    assert audit.verify_chain() == (True, None), f"seed {seed}: I7 chain broken"


# Every seed (or just CHAOS_SEED) keeps all seven invariants.
def test_chaos_refunds_keep_invariants(fresh_order, tmp_path, monkeypatch):
    seeds = [int(os.environ["CHAOS_SEED"])] if os.environ.get("CHAOS_SEED") else range(200)
    exercised = Counter()
    for seed in seeds:
        exercised += _run(seed, fresh_order, tmp_path, monkeypatch)
    print("chaos fault classes exercised:", dict(sorted(exercised.items())))
    assert exercised["crash:after_payment"] or os.environ.get("CHAOS_SEED"), "the seed range missed a crash point"
