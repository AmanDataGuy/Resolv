"""The audit trail — append-only JSONL, one record per attempted tool call.

THIS IS NOT LOGGING. Logging is for humans debugging later; it can be sampled, buffered, or
dropped, and nothing breaks. Three things here depend on this file being complete and exact:

  1. harness/policy.py rule 4 reads it. "Has this order already been refunded?" is answered
     from these records and nowhere else. A dropped record is a double refund.
  2. The eval scores from it. Whether a run actually did the right thing is judged on what the
     trail says happened — not on what the agent claimed in prose.
  3. The demo renders it. The right-hand panel IS this file, streamed.

So it records ATTEMPTS, not successes. A denied call is the most interesting record in here:
"the model tried to refund $900 and rule refund_exceeds_cap stopped it" is the entire value
proposition, and a trail that kept only successes would discard exactly the evidence worth
having.

WHY JSONL AND NOT A DATABASE. Append-only is the property that matters, and a file opened in
"a" mode has it natively — no transaction, no ORM, no migration. One line per call, readable
with `tail -f`, diffable, trivially replayable. Postgres would buy durability and concurrent
writers; neither is a demo's problem, and swapping this for a table later is a change inside
append() that no caller sees. (ponytail: file-backed; move to a real table when more than one
process writes, not before.)

TAMPER-EVIDENCE. Append-only at the filesystem level is a convention, not a guarantee — nothing
stops someone with disk access from opening a case's file and editing a line to make a denied
refund look allowed. Each record now carries `record_id` (a fresh UUID) and `prev_hash` (the
SHA-256 of the ENTIRE previous record, including that record's own prev_hash). That makes the
records a hash chain: editing any record changes what every later record's prev_hash should have
been, and verify_chain() below catches exactly that. This answers EU AI Act Article 12
(record-keeping); harness/policy.py's deterministic gating already answers Article 14 (oversight).

WHAT THIS DOES NOT COVER, STATED PLAINLY. Two honest limits, not glossed over:
  1. The chain is per CASE FILE, not one ledger across the whole system. Deleting an entire
     case's file leaves no trace in any OTHER file — this proves a record wasn't altered once
     written, not that no case was ever deleted outright. A tamper-proof deletion guarantee needs
     a separate, append-only index of "which case_ids exist," which isn't built here.
  2. Canonicalization is `json.dumps(record, sort_keys=True, separators=(",", ":"))`, not full
     RFC 8785 (JSON Canonicalization Scheme) — sort_keys removes key-order ambiguity, which is
     the only ambiguity that matters for a single Python process writing and verifying its own
     hashes. RFC 8785 also pins down float/number serialization for cross-LANGUAGE verification;
     nothing here reads these files in anything but Python, so that guarantee isn't needed yet.
     (ponytail: sort_keys canonicalization; upgrade to RFC 8785 only if a non-Python verifier
     ever needs to independently recompute these hashes.)
"""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDIT_DIR = Path(__file__).parent.parent / "data" / "audit"

# The prev_hash of the first record in any chain — there is no real "previous record" to hash,
# so a fixed, obviously-synthetic value (not a real SHA-256 output of any data) marks "this is
# where the chain starts," the same way a Merkle tree's implementation pins a genesis value.
GENESIS_HASH = "0" * 64


def _canonical(record: dict) -> str:
    """See the module docstring's tamper-evidence section for exactly what this does and doesn't
    guarantee. sort_keys is the whole trick: it's what makes hashing the same record twice, or
    reading it back from disk in whatever order json gave it, produce the identical digest."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _hash(record: dict) -> str:
    return hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()

# Order-scoped index, separate from the per-case trail. A case's trail answers "what happened in
# THIS conversation"; this answers "has this order EVER been refunded, in any conversation." Rule
# 4 in harness/policy.py originally only saw the case trail, which meant two separate, honest
# contacts about the same order both got refunded in full — found and reproduced in the
# 2026-08-15 audit (see plan_ahead.md, Priority 1). Written automatically by append() below,
# never by a second, separate call a caller could forget to make.
ORDER_DIR = Path(__file__).parent.parent / "data" / "audit" / "_by_order"


def _path(case_id: str) -> Path:
    return AUDIT_DIR / f"{case_id}.jsonl"


def _order_path(order_id: str) -> Path:
    return ORDER_DIR / f"{order_id}.jsonl"


def _last_record(case_id: str) -> dict | None:
    """The last record currently in this case's file, or None if it has no trail yet. This is
    the ONE record append() needs to extend the chain — everything before it is already baked
    into that record's own prev_hash, transitively."""
    existing = read(case_id)
    return existing[-1] if existing else None


def append(
    case_id: str, tool: str, order_id: str, args: dict, decision: Any, ok: bool, result: str,
    caller_id: str | None = None,
) -> dict:
    """Record one attempted tool call and return the record.

    Returned as well as written so callers needn't re-read the file to see what they just
    wrote — harness/tools.py appends this to the in-memory history it hands the next policy
    check.

    The record shape IS the contract policy.check_refund() reads (`tool`, `order_id`, `ok`,
    `args`), so renaming a key here silently breaks rule 5. That coupling is why both files
    live in harness/, and demo() asserts it.

    caller_id is recorded so eval/monitor.py's live replay of check_refund() (which has no task
    answer key to fall back on, only the trail) can honestly re-check rule 2 against who actually
    called — the trail is "the truth" per this module's own docstring, so it has to carry that
    fact rather than assume it.

    record_id / prev_hash make this case's file a hash chain -- see the module docstring's
    tamper-evidence section. prev_hash is computed from the case file's last record BEFORE this
    one is appended, which is why _last_record() runs first.
    """
    prior = _last_record(case_id)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "tool": tool,
        "order_id": order_id,
        "args": args,
        "caller_id": caller_id,
        # Flattened, not nested. `rule_id` is what you group by when asking "what is this agent
        # actually being stopped by?" — burying it a level down makes the most useful query in
        # the file the most awkward one to write.
        "action": decision.action,
        "rule_id": decision.rule_id,
        "reason": decision.reason,
        "ok": ok,
        "result": result,
        "record_id": uuid.uuid4().hex,
        "prev_hash": _hash(prior) if prior else GENESIS_HASH,
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(_path(case_id), "a") as f:
        f.write(json.dumps(record) + "\n")

    # Mirror successful refunds into the order-scoped index. Only successful issue_refund calls —
    # a denial or a lookup says nothing about whether this order has been paid, and duplicating
    # those would just be noise in the one file rule 4 actually needs to be complete.
    if tool == "issue_refund" and ok:
        ORDER_DIR.mkdir(parents=True, exist_ok=True)
        with open(_order_path(order_id), "a") as f:
            f.write(json.dumps(record) + "\n")

    return record


def read(case_id: str) -> list[dict]:
    """Every record for a case, oldest first. [] if the case has no trail yet."""
    p = _path(case_id)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def order_history(order_id: str) -> list[dict]:
    """Every successful refund ever recorded against this order, across every case. [] if none.

    This is what closes the split-session gap: rule 4 in harness/policy.py needs to know "has
    this order been refunded" as a fact about the ORDER, not a fact about one conversation.
    """
    p = _order_path(order_id)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def verify_chain(case_id: str) -> tuple[bool, int | None]:
    """Recompute every record's prev_hash from the record before it and compare against what's
    stored. Returns (True, None) if the chain is intact, or (False, i) naming the first index
    where it isn't -- meaning that record, or the one before it, was altered after being written,
    or a record was deleted from the middle. i == 0 failing means the first record's own
    prev_hash was changed (it should always equal GENESIS_HASH).

    This is the verifier promised by the module docstring's tamper-evidence section, and it's the
    demo: a hash chain nobody ever checks is a hash chain in name only. See that same docstring
    for what this does NOT prove -- it cannot detect a case file deleted outright, only edits to
    records that still exist.
    """
    records = read(case_id)
    for i, r in enumerate(records):
        expected = GENESIS_HASH if i == 0 else _hash(records[i - 1])
        if r.get("prev_hash") != expected:
            return False, i
    return True, None


def clear(case_id: str) -> None:
    """Delete a case's trail. Test setup and eval runs only — each of the n=5 repeats of a task
    must start from an empty history, or rule 4 would deny run 2 because run 1 refunded.

    Never call this from the live path. An audit trail that can be erased in production isn't
    an audit trail.
    """
    _path(case_id).unlink(missing_ok=True)


def clear_order(order_id: str) -> None:
    """Delete an order's cross-case refund index. Test setup and eval runs only, same rule as
    clear(): each of the n=5 repeats of a task touches the SAME real order_id across DIFFERENT
    case_ids, so the order-level index must also be reset per repeat, or repeat 2 would see
    repeat 1's refund and fail for a reason that has nothing to do with the agent being graded.

    Never call this from the live path — for the same reason clear() must not be.
    """
    _order_path(order_id).unlink(missing_ok=True)


def demo() -> None:
    """Round-trip a record and assert the exact shape policy.py rule 4 depends on."""
    from harness.policy import Decision

    case = "_demo_audit"
    order = "ORD-9999999"  # a demo-only id, well outside the real 1000-1459 range — never a
                            # real order, so this demo can never collide with harness/tools.py's
                            # own demo() (or anything else) touching an actual DB order.
    clear(case)
    clear_order(order)
    assert read(case) == [] and order_history(order) == []

    d = Decision(action="allow", rule_id="within_policy", reason="ok")
    rec = append(case, "issue_refund", order, {"amount_usd": 12.5}, d, True, "refunded $12.50")

    back = read(case)
    assert len(back) == 1 and back[0] == rec, "what we wrote is not what we read back"

    # The exact access pattern policy.check_refund() rule 4 uses. If a key is renamed, this
    # fails here — loudly, now — instead of as a silent double refund inside the eval.
    prior = [h for h in back if h["tool"] == "issue_refund" and h["order_id"] == order and h["ok"]]
    assert prior and prior[0]["args"]["amount_usd"] == 12.5

    # The order-scoped index picked up the same successful refund automatically (2026-08-15 fix)
    # — this is what lets rule 4 see it from a SECOND, different case_id.
    assert order_history(order) == [rec]

    # Append-only: a second call adds, never replaces.
    append(case, "issue_refund", order, {"amount_usd": 1.0}, d, False, "denied")
    assert len(read(case)) == 2
    # A denied attempt must NOT be mirrored into the order index — only successful refunds are.
    assert len(order_history(order)) == 1

    # The chain: genesis on the first record, then each prev_hash matches the record before it.
    trail = read(case)
    assert trail[0]["prev_hash"] == GENESIS_HASH
    assert trail[1]["prev_hash"] == _hash(trail[0])
    assert verify_chain(case) == (True, None)

    # Tamper with the first record in place -- the exact attack the chain exists to catch -- and
    # confirm verify_chain() names the SECOND record (index 1) as the first one that no longer
    # lines up, since record 0's own prev_hash (genesis) is untouched.
    p = _path(case)
    lines = p.read_text().splitlines()
    tampered = json.loads(lines[0])
    tampered["ok"] = False  # flip a real refund into a denial after the fact
    lines[0] = json.dumps(tampered)
    p.write_text("\n".join(lines) + "\n")
    ok, break_at = verify_chain(case)
    assert ok is False and break_at == 1, "tampering with record 0 must be caught at record 1"

    clear(case)
    clear_order(order)
    print("audit demo OK — append-only, order index intact, hash chain verified and tamper-detected")


if __name__ == "__main__":
    demo()
