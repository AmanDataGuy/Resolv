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
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDIT_DIR = Path(__file__).parent.parent / "data" / "audit"


def _path(case_id: str) -> Path:
    return AUDIT_DIR / f"{case_id}.jsonl"


def append(case_id: str, tool: str, order_id: str, args: dict, decision: Any, ok: bool, result: str) -> dict:
    """Record one attempted tool call and return the record.

    Returned as well as written so callers needn't re-read the file to see what they just
    wrote — harness/tools.py appends this to the in-memory history it hands the next policy
    check.

    The record shape IS the contract policy.check_refund() reads (`tool`, `order_id`, `ok`,
    `args`), so renaming a key here silently breaks rule 4. That coupling is why both files
    live in harness/, and demo() asserts it.
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "tool": tool,
        "order_id": order_id,
        "args": args,
        # Flattened, not nested. `rule_id` is what you group by when asking "what is this agent
        # actually being stopped by?" — burying it a level down makes the most useful query in
        # the file the most awkward one to write.
        "action": decision.action,
        "rule_id": decision.rule_id,
        "reason": decision.reason,
        "ok": ok,
        "result": result,
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(_path(case_id), "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def read(case_id: str) -> list[dict]:
    """Every record for a case, oldest first. [] if the case has no trail yet."""
    p = _path(case_id)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def clear(case_id: str) -> None:
    """Delete a case's trail. Test setup and eval runs only — each of the n=5 repeats of a task
    must start from an empty history, or rule 4 would deny run 2 because run 1 refunded.

    Never call this from the live path. An audit trail that can be erased in production isn't
    an audit trail.
    """
    _path(case_id).unlink(missing_ok=True)


def demo() -> None:
    """Round-trip a record and assert the exact shape policy.py rule 4 depends on."""
    from harness.policy import Decision

    case = "_demo_audit"
    clear(case)
    assert read(case) == []

    d = Decision(action="allow", rule_id="within_policy", reason="ok")
    rec = append(case, "issue_refund", "ORD-1000", {"amount_usd": 12.5}, d, True, "refunded $12.50")

    back = read(case)
    assert len(back) == 1 and back[0] == rec, "what we wrote is not what we read back"

    # The exact access pattern policy.check_refund() rule 4 uses. If a key is renamed, this
    # fails here — loudly, now — instead of as a silent double refund inside the eval.
    prior = [h for h in back if h["tool"] == "issue_refund" and h["order_id"] == "ORD-1000" and h["ok"]]
    assert prior and prior[0]["args"]["amount_usd"] == 12.5

    # Append-only: a second call adds, never replaces.
    append(case, "issue_refund", "ORD-1000", {"amount_usd": 1.0}, d, False, "denied")
    assert len(read(case)) == 2

    clear(case)
    print("audit demo OK — append-only, and rule 4's read path is intact")


if __name__ == "__main__":
    demo()
