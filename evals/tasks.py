"""
# evals/tasks.py

**What:** the 40-task benchmark: a real complaint, a tactic the customer will use, and the answer key, built
deterministically from the case file and the policy engine.
**Why:** the answer key is derived from `resolv.harness.policy.check_refund` at build time, so it can never drift from
the rules it tests. Tasks are sampled to a 50/50 owed/not-owed split and asserted, so "deny everything" or
"refund everything" cannot beat a coin flip. The case file is read only when `build_tasks` is called.
`expected` is one of: refund (owed exactly `expected_amount`), escalate (owed, but over the auto-approve
limit), deny (nothing owed).
**Used by:** evals/runner.py, ablation.py, injection.py, cost.py, latency.py, reliability.py, gate.py,
scripts/freeze_openings.py, tests.
"""
import json
import random
from pathlib import Path

from resolv.config import REFUND_CAP_FRACTION
from resolv.harness.policy import check_refund
from resolv.harness.validity import get_order

CASES = Path(__file__).parent.parent / "data" / "datasets" / "complaint_cases.json"

# assigned round-robin over the sorted tasks, so every tactic lands on owed and not-owed cases alike
TACTIC_ORDER = ["honest", "inflate_amount", "wrong_order_id", "change_story", "pressure"]
TARGET_REFUND_SHARE = 0.5
TOLERANCE = 0.1


# what should happen and the amount owed, asked of the policy engine at the cap with an empty history
def expected_outcome(order_id: str, claim_type: str) -> tuple[str, float]:
    order = get_order(order_id)
    if not order:
        return ("deny", 0.0)
    cap = round(order["amount_usd"] * REFUND_CAP_FRACTION[claim_type], 2)
    # the caller is the order's own customer: every task is the real account holder, whatever they lie about
    decision = check_refund(order_id, claim_type, cap, [], order["customer_id"])
    if decision.action == "deny":
        return ("deny", 0.0)
    if decision.action == "escalate":
        return ("escalate", cap)
    return ("refund", cap)


# n tasks balanced between owed and not owed, deterministic for a seed; cases without an order id are dropped
def build_tasks(n: int = 40, seed: int = 0) -> list[dict]:
    if not CASES.exists():
        raise FileNotFoundError(f"{CASES} is missing; the benchmark needs the committed case file.")
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    rng = random.Random(seed)

    pool = []
    for c in sorted(cases, key=lambda c: c["case_id"]):
        oid = c["ground_truth"]["order_id"]
        order = get_order(oid) if oid else None
        if not order:
            continue
        claim = c["ground_truth"]["claim_type"]
        outcome, amount = expected_outcome(oid, claim)
        pool.append({
            "case_id": c["case_id"],
            "order_id": oid,
            "claim_type": claim,
            "amount_usd": order["amount_usd"],
            "customer_id": order["customer_id"],
            "message": c["message"],
            "difficulty": c["difficulty"],
            # a plausible order number that is not in the DB, stable per case, for the wrong_order_id tactic
            "fake_order_id": f"ORD-{9000 + int(c['case_id'].split('-')[1])}",
            "expected": outcome,
            "expected_amount": amount,
        })

    owed = [t for t in pool if t["expected"] != "deny"]
    denied = [t for t in pool if t["expected"] == "deny"]
    want_owed = round(n * TARGET_REFUND_SHARE)
    if len(owed) < want_owed or len(denied) < n - want_owed:
        raise ValueError(
            f"Pool cannot fill n={n} at the target split: {len(owed)} owed, {len(denied)} denied. "
            "Regenerate cases with scripts/gen_complaint_cases.py (see MIN_PROMISED)."
        )

    tasks = rng.sample(owed, want_owed) + rng.sample(denied, n - want_owed)
    tasks.sort(key=lambda t: t["case_id"])
    for i, t in enumerate(tasks):
        t["tactic"] = TACTIC_ORDER[i % len(TACTIC_ORDER)]
        t["task_id"] = f"{t['case_id']}-{t['tactic']}"

    share = sum(1 for t in tasks if t["expected"] != "deny") / len(tasks)
    assert abs(share - TARGET_REFUND_SHARE) <= TOLERANCE, (
        f"refund share {share:.2f} is off target; 'deny everything' would score {1 - share:.0%}"
    )
    return tasks
