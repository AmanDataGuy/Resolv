"""
# resolv/integrations/payments.py

**What:** FakePSP, a simulated payment provider with its own SQLite file (RESOLV_PSP_DB, default data/psp.db) and
seeded fault injection (`FAULTS`).
**Why:** a refund needs a real external side effect for the saga in resolv/harness/tools.py to mean anything, and a separate
file makes it a separate system with its own state. Like Stripe, the same idempotency key always returns the same
refund, so a retry never pays twice. Faults cover every saga branch: a 500 before any effect, a timeout after the
effect (the "unknown outcome" case), a decline, and latency. Payments are simulated in every environment;
production runs with no faults.
**Used by:** resolv/harness/tools.py (issue_refund, reconcile_once), tests/test_chaos_refunds.py.
"""
import os
import random
import sqlite3
import time
import uuid
from dataclasses import dataclass, field


# The provider refused (a 4xx): nothing happened and a retry will not help.
class Declined(Exception):
    pass


# The provider failed before doing anything (a 5xx): nothing happened and a retry may succeed.
class ProviderError(Exception):
    pass


# Per-call fault probabilities, drawn from one seeded RNG so a failing run can be replayed.
@dataclass
class FaultPlan:
    fail_before: float = 0.0
    timeout_after_commit: float = 0.0
    decline: float = 0.0
    latency_s: float = 0.0
    seed: int = 0
    rng: random.Random = field(init=False, repr=False)

    # Build the RNG from the seed once the fields are set.
    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)


FAULTS = FaultPlan()  # tests swap this; production keeps the zero-fault default


# Open the provider's own database file.
def _db() -> sqlite3.Connection:
    path = os.environ.get("RESOLV_PSP_DB", "data/psp.db")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS psp_refunds(idempotency_key TEXT PRIMARY KEY, psp_ref TEXT UNIQUE NOT NULL,"
                 " order_id TEXT NOT NULL, amount_cents INTEGER NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL)")
    return conn


# Run one statement on a short-lived connection and return the rows as dicts.
def _q(sql: str, params=()) -> list[dict]:
    conn = _db()
    try:
        return [dict(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


# The refund made under this idempotency key, or None if the provider never made one.
def get(idempotency_key: str) -> dict | None:
    rows = _q("SELECT * FROM psp_refunds WHERE idempotency_key=?", (idempotency_key,))
    return rows[0] if rows else None


# Every refund the provider holds (tests check ledger invariants against this).
def all_refunds() -> list[dict]:
    return _q("SELECT * FROM psp_refunds ORDER BY created_at")


# Refund money once per key: {"psp_ref", "status": "succeeded", ...}; may raise ProviderError, Declined, TimeoutError.
def refund(idempotency_key: str, order_id: str, amount_cents: int) -> dict:
    plan = FAULTS
    if plan.latency_s:
        time.sleep(plan.latency_s)
    done = get(idempotency_key)
    if done:
        return done
    roll = plan.rng.random()
    if roll < plan.fail_before:
        raise ProviderError("HTTP 500 from the payment provider")
    if roll < plan.fail_before + plan.decline:
        raise Declined("the payment provider declined the refund")
    # INSERT OR IGNORE: two concurrent calls with one key still make one refund.
    _q("INSERT OR IGNORE INTO psp_refunds VALUES (?,?,?,?,'succeeded',?)",
       (idempotency_key, f"re_{uuid.uuid4().hex[:16]}", order_id, amount_cents, time.time()))
    if plan.rng.random() < plan.timeout_after_commit:
        raise TimeoutError("the payment provider timed out after accepting the refund")
    return get(idempotency_key)


# Reverse a refund by its provider reference; raises KeyError if there is no such refund.
def reverse(psp_ref: str) -> dict:
    _q("UPDATE psp_refunds SET status='reversed' WHERE psp_ref=?", (psp_ref,))
    rows = _q("SELECT * FROM psp_refunds WHERE psp_ref=?", (psp_ref,))
    if not rows:
        raise KeyError(psp_ref)
    return rows[0]
