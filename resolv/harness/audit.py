"""
# resolv/harness/audit.py

**What:** the audit trail API (append, read, order_history, verify_chain, clear, clear_order), a thin facade over
resolv/harness/ledger.py.
**Why:** policy rule 5, the eval grader, the API and the demo all read the trail through these calls, so their
signatures and the record shape (`tool, order_id, args, ok, rule_id, ...`) stay fixed while storage is the SQLite
ledger with a keyed hash chain. It records ATTEMPTS, not successes: a denied call is the most useful line in it.
`clear` and `clear_order` delete evidence, so they refuse to run unless RESOLV_ENV is test or eval.
**Used by:** resolv/harness/tools.py, resolv/agents/loop.py, resolv/api/escalations.py, app.py, evals/*.
"""
import os

from resolv.harness import ledger

GENESIS_HASH = ledger.GENESIS


# Record one attempted tool call in its own transaction and return the stored record.
def append(case_id: str, tool: str, order_id: str, args: dict, decision, ok: bool, result: str,
           caller_id: str | None = None, approver_id: str | None = None) -> dict:
    with ledger.tx() as conn:
        return ledger.append_audit(conn, case_id, tool, order_id, args, decision, ok, result, caller_id, approver_id)


# Every record for a case, oldest first; [] when the case has none.
def read(case_id: str) -> list[dict]:
    return ledger.case_trail(case_id)


# Every live refund (pending, settled or unknown) on this order across all cases, shaped like an issue_refund record.
def order_history(order_id: str) -> list[dict]:
    return ledger.order_refund_records(order_id)


# Verify the one global keyed chain: (True, None), or (False, seq of the first bad record). case_id is kept for callers.
def verify_chain(case_id: str | None = None, head: dict | None = None) -> tuple[bool, int | None]:
    return ledger.verify(head)


# Refuse unless RESOLV_ENV (read now, not at import) is test or eval.
def _test_or_eval_only(name: str) -> None:
    if os.environ.get("RESOLV_ENV") not in ("test", "eval"):
        raise RuntimeError(f"{name} deletes audit evidence and only runs with RESOLV_ENV=test or eval")


# Delete a case's trail (test setup and eval repeats only).
def clear(case_id: str) -> None:
    _test_or_eval_only("audit.clear")
    ledger.clear_case(case_id)


# Delete an order's refunds so the next eval repeat sees it unrefunded (test and eval only).
def clear_order(order_id: str) -> None:
    _test_or_eval_only("audit.clear_order")
    ledger.clear_order(order_id)


# Round-trip records, check the shape rule 5 reads, and catch an edit through the chain.
def demo() -> None:
    from resolv.harness.policy import Decision

    ledger.use_temp_db()
    case = "_demo_audit"
    d = Decision(action="allow", rule_id="within_policy", reason="ok")
    rec = append(case, "issue_refund", "ORD-9999999", {"amount_usd": 12.5}, d, True, "refunded $12.50", "cust-1")
    assert read(case) == [rec] and rec["caller_id"] == "cust-1"
    assert [h for h in read(case) if h["tool"] == "issue_refund" and h["ok"]][0]["args"]["amount_usd"] == 12.5
    append(case, "issue_refund", "ORD-9999999", {"amount_usd": 1.0}, d, False, "denied", "cust-1")
    assert verify_chain(case) == (True, None)

    conn = ledger.connect()
    conn.execute("UPDATE audit SET ok=0 WHERE seq=1")  # flip a refund into a denial after the fact
    conn.close()
    assert verify_chain(case) == (False, 1), "an edited record must fail its own mac"
    print("audit demo OK: append-only, rule 5 shape intact, keyed chain verified and tamper-detected")


if __name__ == "__main__":
    demo()
