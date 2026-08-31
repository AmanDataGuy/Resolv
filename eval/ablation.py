"""Harness ON vs harness OFF — the control arm for the project's central claim.

    python -m eval.ablation                      # 20 tasks x 1 repeat x 2 arms, ~$0.70
    python -m eval.ablation --tasks 40 --n 5     # the full paired sweep, ~$3.50

Every other number in this repo is measured with the policy engine switched on, and they all say
the same thing: zero unauthorized refunds. That is a fact about the system, but on its own it is
not evidence about the DESIGN, because it never establishes that the harness is what produced it.
Maybe the model is simply good and the harness is decoration. This file is the only thing that
can tell those apart: same tasks, same adversarial customer, same model, same temperature — one
arm with policy in front of the money and one without.

If the OFF arm also scores zero, the honest conclusion is that the harness bought nothing on this
task set and the project should say so. If it doesn't, the gap between the arms is the measured
value of the design rather than an argument for it. Either way it is a result, which is why the
run is worth its cost.

WHY THE POLICY-FREE TOOLS LIVE HERE AND NOT BEHIND A FLAG IN harness/tools.py. A `skip_policy=True`
parameter would be a second write path — the exact thing the README claims does not exist, added
for the express purpose of measuring how important it is that it does not exist. Whatever the
ablation then measured, "tools.py is the only path to the money" would no longer be literally
true, and that sentence is the product. So the bypass is built by rebinding agents.loop._bind for
the duration of the OFF arm and restoring it after. Production code is untouched; the seam is in
the experiment, where a seam belongs.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

os.environ.setdefault("LITELLM_LOG", "ERROR")

from tqdm import tqdm

from agents import loop
from config import LLM_PROVIDER, MODEL
from eval import simulator
from eval.metrics import scorecard
from eval.runner import _grade
from eval.tasks import build_tasks
from harness import audit
from harness.policy import Decision
from harness.validity import get_order

OUT_DIR = Path(__file__).parent.parent / "data" / "eval"

# The synthetic verdict the unchecked tools record. rule_id says "no_harness" rather than
# "within_policy" so an ablation trail can never be mistaken for a real one if the files are ever
# read together — and so the grader's replay, which re-runs the REAL check_refund over these rows,
# still reports honestly on what policy would have said.
_BYPASSED = Decision(action="allow", rule_id="no_harness", reason="Policy engine disabled (ablation).")


def _unchecked_refund(case_id: str, order_id: str, claim_type: str, amount_usd: float) -> str:
    """Pay whatever is asked. No lookup, no cap, no window, no double-refund check.

    This is what "let the LLM decide" actually means: the model's own judgment is the only thing
    between the customer's story and the money. Still writes the audit record, because the grader
    replays trails — an arm that logged nothing would be unfalsifiable rather than safe.
    """
    audit.append(case_id, "issue_refund", order_id,
                 {"amount_usd": amount_usd, "claim_type": claim_type},
                 _BYPASSED, True, f"Refunded ${amount_usd:.2f} on {order_id}.")
    return f"Refunded ${amount_usd:.2f} on {order_id}."


def _unchecked_lookup(case_id: str, order_id: str) -> str:
    """Unchanged in substance — lookup mutates nothing, so it has no policy check to remove.

    Reimplemented here only so the OFF arm imports nothing from harness/tools.py at all. An
    ablation that borrows half its subject is hard to argue about later.
    """
    order = get_order(order_id)
    result = (
        f"Order {order_id}: ${order['amount_usd']:.2f}, promised {order['promised_date']}, "
        f"delivered {order.get('delivered_date') or 'not delivered'}, status {order.get('status')}."
        if order else f"Order {order_id} not found."
    )
    audit.append(case_id, "lookup_order", order_id, {}, _BYPASSED, bool(order), result)
    return result


def _unchecked_escalate(case_id: str, order_id: str, reason: str) -> str:
    audit.append(case_id, "escalate_to_human", order_id, {"reason": reason}, _BYPASSED, True, "Escalated.")
    return "Escalated to a human agent."


def _bind_without_harness(case_id: str) -> dict:
    """The drop-in replacement for agents.loop._bind — same three names, no policy engine."""
    return {
        "lookup_order": lambda order_id: _unchecked_lookup(case_id, order_id),
        "issue_refund": lambda order_id, claim_type, amount_usd: _unchecked_refund(
            case_id, order_id, claim_type, amount_usd
        ),
        "escalate_to_human": lambda order_id, reason: _unchecked_escalate(case_id, order_id, reason),
    }


def _run(task: dict, repeat: int, arm: str) -> dict:
    """One attempt at one task in one arm. Mirrors eval/runner.py::_one_run deliberately.

    The two arms must differ in exactly one thing. Same case_id scheme, same simulator, same
    temperature, same grading — anything else that varied would be a confound, and the whole
    point of a control arm is that there isn't one.
    """
    case_id = f"abl-{arm}-{task['task_id']}-r{repeat}"
    audit.clear(case_id)
    audit.clear_order(task["order_id"])

    def user(agent_said: str, history: list[dict]) -> str | None:
        turns = [{"role": m["role"], "content": m.get("content") or ""}
                 for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
        return simulator.reply(task, agent_said, turns)

    try:
        result = asyncio.run(loop.run_case(case_id, simulator.opening(task), temperature=0.7, user=user))
        row = _grade(task, result["trail"], result["steps"], result["reply"])
    except Exception as e:
        row = _grade(task, audit.read(case_id), 0)
        row["resolved"] = False
        row["error"] = f"{type(e).__name__}: {e}"
    row["arm"] = arm
    row["repeat"] = repeat
    return row


def run_arm(tasks: list[dict], n: int, arm: str) -> list[dict]:
    """Every task x n repeats in one arm. Restores the real tools even if a run raises."""
    original = loop._bind
    if arm == "off":
        loop._bind = _bind_without_harness
    try:
        jobs = [(t, i) for t in tasks for i in range(n)]
        return [_run(t, i, arm) for t, i in tqdm(jobs, unit="run", desc=f"harness {arm:<3}")]
    finally:
        loop._bind = original  # production tools back, whatever happened above


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure what the policy harness is actually worth.")
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--n", type=int, default=1, help="repeats per task (raise for pass^k in both arms)")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--arm", choices=["both", "on", "off"], default="both")
    ap.add_argument("--out", default="ablation.jsonl")
    args = ap.parse_args()

    tasks = build_tasks(args.tasks)
    arms = ["on", "off"] if args.arm == "both" else [args.arm]
    print(f"provider={LLM_PROVIDER}  model={MODEL}")
    print(f"{len(tasks)} tasks x {args.n} repeats x {len(arms)} arm(s) = {len(tasks) * args.n * len(arms)} runs")

    rows = [r for arm in arms for r in run_arm(tasks, args.n, arm)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / args.out
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    print(f"\nablation over {len(rows)} runs -> {out.name}\n")
    cards = {arm: scorecard([r for r in rows if r["arm"] == arm], k=args.k) for arm in arms}
    keys = ["runs", f"pass^{args.k}", "unauthorized_rate", "harmful_block_rate", "over_block_rate",
            "resolve_rate"]
    width = max(len(key) for key in keys)
    print(f"  {'metric':<{width}}  " + "  ".join(f"{a:>10}" for a in arms))
    for key in keys:
        print(f"  {key:<{width}}  " + "  ".join(f"{cards[a].get(key)!s:>10}" for a in arms))

    if len(arms) == 2:
        # The headline of the whole experiment: money that moved without authorisation, per arm.
        off = sum(r["paid"] for r in rows if r["arm"] == "off" and r.get("unauthorized"))
        on = sum(r["paid"] for r in rows if r["arm"] == "on" and r.get("unauthorized"))
        print(f"\n  unauthorized dollars, harness OFF: ${off:,.2f}")
        print(f"  unauthorized dollars, harness ON:  ${on:,.2f}")
        if cards["off"]["unauthorized_rate"] == 0:
            print("\n  note: the OFF arm also leaked nothing on this task set. The harness is not")
            print("        shown to be load-bearing here — say that, rather than the reverse.")


if __name__ == "__main__":
    main()
