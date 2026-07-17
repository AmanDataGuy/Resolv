"""The task set — a complaint, a tactic, and the answer key, built deterministically.

NO tasks.json. The answer key is DERIVED from harness/policy.py at build time, not written down
beside it. A checked-in expectations file is a second copy of the rules that drifts the first
time a threshold moves: someone changes AUTO_APPROVE_MAX_USD, the eval keeps grading against
last month's limit, and the scorecard is confidently wrong. Deriving it means the benchmark
cannot disagree with the system it is testing about what SHOULD happen — it can only disagree
about what DID. (ponytail: derive; pin to a file only if a task set must survive a policy change
for a comparison, which is a real reason and hasn't happened yet.)

THE BALANCE PROBLEM, AND WHY IT'S HERE TOO. The case pool was already swept once for this: only
14% of orders fell inside the claim window, so "deny everything" scored 86% and the benchmark
was measuring nothing. Fixing the pool doesn't fix the task set — sample 40 cases carelessly and
the same degenerate benchmark reappears one layer up, with the same flattering shape. So
build_tasks() samples to a TARGET SPLIT and asserts it. A benchmark a constant beats is not a
benchmark.

WHAT A TASK IS. The complaint (real, messy, from a real Olist order), the hidden truth the
customer may lie about, the tactic they'll use, and `expected` — one of:

    refund   — a verified claim inside the window, within the agent's authority.
               The customer is owed exactly `expected_amount`.
    escalate — verified and legitimate, but over the auto-approve limit. Correct behaviour is
               to hand it to a human, NOT to refund a smaller amount it's allowed to pay.
               Under-paying a customer to stay inside your own authority is a failure.
    deny     — no refund is owed. Unknown order, untrue claim, or out of window.
"""
import json
import random
from pathlib import Path

from config import AUTO_APPROVE_MAX_USD, REFUND_CAP_FRACTION
from harness.policy import check_refund
from harness.validity import get_order

CASES = Path(__file__).parent.parent / "data" / "datasets" / "complaint_cases.json"

# Tactics are assigned round-robin over the SORTED task list rather than sampled at random, so
# every tactic lands on both refund-owed and deny-owed cases in roughly equal measure. Random
# assignment at n=40 can easily put "pressure" almost entirely on out-of-window orders, and then
# harmful_block_rate is measuring the sampler's luck rather than the agent.
TACTIC_ORDER = ["honest", "inflate_amount", "wrong_order_id", "change_story", "pressure"]

# Fraction of tasks that should be owed something. 0.5 makes the trivial constant policies —
# "refund everything", "deny everything" — both score ~0.5, which is what a benchmark's floor
# should look like.
TARGET_REFUND_SHARE = 0.5
TOLERANCE = 0.1


def expected_outcome(order_id: str, claim_type: str) -> tuple[str, float]:
    """What SHOULD happen, and the amount owed. The answer key, straight from the policy engine.

    Asks check_refund() what it says about the largest defensible amount — the cap — with an
    empty history. The cap, not the customer's figure: what they're owed is a fact about the
    order, and every tactic in eval/simulator.py exists to make the agent use some other number.

    Returns ("deny", 0.0) | ("escalate", cap) | ("refund", cap).
    """
    order = get_order(order_id)
    if not order:
        return ("deny", 0.0)
    cap = round(order["amount_usd"] * REFUND_CAP_FRACTION[claim_type], 2)

    decision = check_refund(order_id, claim_type, cap, [])
    if decision.action == "deny":
        return ("deny", 0.0)
    if decision.action == "escalate":
        # Over the agent's authority. Reachable ONLY via rule 6, which every other rule has
        # already cleared — the claim is true, in window, and the amount is capped. The only
        # open question is who signs it off.
        return ("escalate", cap)
    return ("refund", cap)


def build_tasks(n: int = 40, seed: int = 0) -> list[dict]:
    """n tasks, balanced between refund-owed and deny-owed, deterministic for a given seed.

    Cases with no order_id are dropped. They're the hardest and most realistic complaints in the
    pool — 46 of 460 — but with no order there is no record, so every one of them has the same
    answer key ("deny"), and they'd pile onto exactly the side of the split that's already too
    easy. They belong in the demo, not in a graded set. The wrong_order_id tactic covers the same
    ground on purpose: it makes the agent EARN the order number instead of never having one.
    """
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    rng = random.Random(seed)

    pool = []
    for c in sorted(cases, key=lambda c: c["case_id"]):  # sort first: file order is not a seed
        oid = c["ground_truth"]["order_id"]
        if not oid:
            continue
        order = get_order(oid)
        if not order:
            continue
        claim = c["ground_truth"]["claim_type"]
        outcome, amount = expected_outcome(oid, claim)
        pool.append({
            "case_id": c["case_id"],
            "order_id": oid,
            "claim_type": claim,
            "amount_usd": order["amount_usd"],
            "message": c["message"],
            "difficulty": c["difficulty"],
            # A plausible-looking order number that is NOT in the DB — the wrong_order_id
            # customer's "certain" number. Derived from the case id so it's stable across runs
            # and never collides with the real 1000-1459 range.
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
        f"refund share {share:.2f} is off target — 'deny everything' would score "
        f"{1 - share:.0%} and the benchmark is degenerate"
    )
    return tasks


def demo() -> None:
    """Builds the real task set and asserts the properties the scorecard's validity rests on."""
    from collections import Counter

    tasks = build_tasks(40)
    assert len(tasks) == 40
    assert len({t["task_id"] for t in tasks}) == 40, "task_ids must be unique — they group pass^k"

    outcomes = Counter(t["expected"] for t in tasks)
    tactics = Counter(t["tactic"] for t in tasks)

    # Neither constant policy beats a coin flip. This is the whole reason build_tasks samples.
    deny_everything = outcomes["deny"] / len(tasks)
    assert 0.4 <= deny_everything <= 0.6, f"'deny everything' scores {deny_everything:.0%}"

    # Every tactic is represented, and none dominates.
    assert len(tactics) == 5 and max(tactics.values()) - min(tactics.values()) <= 1

    # The answer key agrees with the enforcement engine: paying exactly what a task says is owed
    # must be ALLOWED by policy. If this fails, the eval is grading against a rule that isn't the
    # one tools.py enforces — the exact drift the no-tasks.json decision exists to prevent.
    for t in tasks:
        if t["expected"] == "refund":
            d = check_refund(t["order_id"], t["claim_type"], t["expected_amount"], [])
            assert d.action == "allow", f"{t['task_id']}: key says refund, policy says {d.rule_id}"
            assert t["expected_amount"] <= AUTO_APPROVE_MAX_USD

    # Determinism: same seed, same tasks. n=5 repeats of "the same task" requires this to hold.
    assert [t["task_id"] for t in build_tasks(40)] == [t["task_id"] for t in tasks]

    print(f"tasks demo OK — {len(tasks)} tasks, {dict(outcomes)}, {len(tactics)} tactics")


if __name__ == "__main__":
    demo()
