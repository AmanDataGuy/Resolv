"""
# tests/conftest.py

**What:** one autouse fixture that gives every test its own empty ledger, payment provider, outbox and telemetry
file under tmp_path, sets RESOLV_ENV=test, and turns Langfuse tracing off.
**Why:** tests must never write into the developer's real data/ (the audit trail is evidence), and no test may see
another test's refunds. The ledger and FakePSP read RESOLV_DB / RESOLV_PSP_DB on every call, so env vars are enough;
module-level paths are monkeypatched.
**Used by:** every test under tests/.
"""
import sys

import pytest


# Isolate all on-disk state per test and mark the environment as test.
@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    from resolv.integrations import notify, payments

    monkeypatch.setenv("RESOLV_ENV", "test")
    monkeypatch.setenv("RESOLV_DB", str(tmp_path / "resolv.db"))
    monkeypatch.setenv("RESOLV_PSP_DB", str(tmp_path / "psp.db"))
    monkeypatch.delenv("SHOW_MAX_REFUNDABLE", raising=False)
    monkeypatch.delenv("AUDIT_HMAC_KEY", raising=False)
    # No tracing in tests, even when the developer's .env has Langfuse keys: the suite stays offline.
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    observability = sys.modules.get("resolv.observability")
    if observability is not None:
        observability._client.cache_clear()
    monkeypatch.setattr(payments, "FAULTS", payments.FaultPlan())
    monkeypatch.setattr(notify, "OUTBOX", tmp_path / "outbox")
    monitor = sys.modules.get("resolv.monitor")  # only patch it if some test imported it; never import it here
    if monitor is not None:
        monkeypatch.setattr(monitor, "TELEMETRY", tmp_path / "telemetry.jsonl")


# First real order matching a situation, in the claim window, optionally worth more than the auto-approve limit.
def _pick(situation: str, over_limit: bool = False) -> dict:
    from datetime import date

    from resolv.config import AUTO_APPROVE_MAX_USD, CLAIM_WINDOW_DAYS
    from resolv.harness.policy import NOW
    from resolv.harness.validity import get_order

    for i in range(1000, 1460):
        o = get_order(f"ORD-{i}")
        if (o and o["situation"] == situation
                and (NOW - date.fromisoformat(o["promised_date"])).days <= CLAIM_WINDOW_DAYS
                and (not over_limit or o["amount_usd"] > AUTO_APPROVE_MAX_USD)):
            return o
    pytest.skip(f"no in-window {situation} order (over_limit={over_limit}) in data/db")


# A real in-window late order: a refund up to 25% of its value clears rules 0-6.
@pytest.fixture
def fresh_order() -> dict:
    return _pick("late")


# A real in-window never_arrived order above the auto-approve limit: its full value reaches rule 7.
@pytest.fixture
def big_order() -> dict:
    return _pick("never_arrived", over_limit=True)
