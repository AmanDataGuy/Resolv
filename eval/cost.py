"""
Operations eval: COST.

    python -m eval.cost

Cost here is DERIVED, not measured directly: cost = tokens x price. Tokens are
near-deterministic (same task, same tools available -> similar counts each run), so
unlike latency this is a stable, OFFLINE-answerable number: "can we afford N
resolutions/day?" is answerable before anything ships.

Shape borrowed from the reference eval_cost.py (see the moreeval/ study repo): fixed
question set, few repeats (cost barely moves run to run), a $/request budget line, and
a projection to daily/monthly spend at expected volume. Adapted here to call the REAL
agent loop on REAL complaint tasks (eval/tasks.py) instead of a RAG chain, and to price
with the pair-rate constants eval/runner.py already defines rather than a second table.
"""
import asyncio

from agents.loop import run_case
from agents.runner_utils import tokens_split
from config import LLM_PROVIDER, MODEL
from eval.runner import USD_PER_MTOK_IN, USD_PER_MTOK_OUT
from eval.tasks import build_tasks
from harness import audit

# --- CONFIG ----------------------------------------------------------------------------
TASKS = build_tasks(5)   # 5 real complaints. Single-shot below -- we're pricing one
                          # resolution, not a haggling session, so the simulator sits out.
REPEATS = 3               # cost is stable -> needs far fewer repeats than latency does

REQUESTS_PER_DAY = 2000               # set to your expected traffic
COST_BUDGET_PER_REQUEST_USD = 0.02    # the offline pass/fail line -- tune to your economics


# --- MEASURE -----------------------------------------------------------------------------
def measure_one(task: dict, repeat: int) -> dict:
    """One resolution, one token/cost row. Fresh case_id + order state each time, same
    reason eval/runner.py does it: rule 4 would deny a second refund on a reused order.
    """
    case_id = f"cost-{task['task_id']}-r{repeat}"
    audit.clear(case_id)
    audit.clear_order(task["order_id"])

    before = tokens_split()
    asyncio.run(run_case(case_id, task["message"], temperature=0.7))
    prompt, completion = (a - b for a, b in zip(tokens_split(), before))

    cost = prompt / 1e6 * USD_PER_MTOK_IN + completion / 1e6 * USD_PER_MTOK_OUT
    return {"prompt": prompt, "completion": completion, "cost": cost}


def benchmark() -> list[dict]:
    return [measure_one(task, r) for task in TASKS for r in range(REPEATS)]


# --- REPORT ------------------------------------------------------------------------------
def report(rows: list[dict]) -> None:
    n = len(rows)
    avg_cost = sum(r["cost"] for r in rows) / n
    avg_prompt = sum(r["prompt"] for r in rows) / n
    avg_completion = sum(r["completion"] for r in rows) / n
    min_cost, max_cost = min(r["cost"] for r in rows), max(r["cost"] for r in rows)

    print(f"provider={LLM_PROVIDER}  model={MODEL}")
    print("=" * 60)
    print("COST")
    print("=" * 60)
    print(f"samples             : {n}")
    print(f"avg prompt tokens   : {avg_prompt:.0f}")
    print(f"avg completion tok  : {avg_completion:.0f}")
    print(f"avg cost / request  : ${avg_cost:.6f}")
    print(f"   min / max        : ${min_cost:.6f} / ${max_cost:.6f}  "
          f"(tight range = stable, unlike latency)")
    print("-" * 60)

    daily = avg_cost * REQUESTS_PER_DAY
    print(f"projection @ {REQUESTS_PER_DAY}/day : ${daily:.2f}/day  ${daily * 30:.2f}/month")
    print("-" * 60)

    verdict = "PASS" if avg_cost <= COST_BUDGET_PER_REQUEST_USD else "FAIL"
    print(f"BUDGET: cost/request <= ${COST_BUDGET_PER_REQUEST_USD}  ->  "
          f"${avg_cost:.6f}   [{verdict}]")


def main() -> None:
    report(benchmark())


if __name__ == "__main__":
    main()
