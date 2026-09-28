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

# eval/runner.py's USD_PER_MTOK_IN/OUT are Gemini's rates specifically -- fine as the shared
# scorecard basis everywhere else (a stable comparison unit), but wrong to print as THIS report's
# dollar figure on any other provider, which is exactly the gap eval_report.md flagged. Real
# published rates, not guesses: Groq's is openai/gpt-oss-120b's paid/dev-tier list price
# (console.groq.com/docs/model/openai/gpt-oss-120b, checked when this was written).
_GROQ_PAID_USD_PER_MTOK = (0.15, 0.60)


# --- MEASURE -----------------------------------------------------------------------------
def measure_one(task: dict, repeat: int) -> dict:
    """One resolution, one token/cost row. Fresh case_id + order state each time, same
    reason eval/runner.py does it: rule 4 would deny a second refund on a reused order.
    """
    case_id = f"cost-{task['task_id']}-r{repeat}"
    audit.clear(case_id)
    audit.clear_order(task["order_id"])

    before = tokens_split()
    asyncio.run(run_case(case_id, task["message"], temperature=0.7, caller_id=task["customer_id"]))
    prompt, completion = (a - b for a, b in zip(tokens_split(), before))

    cost = prompt / 1e6 * USD_PER_MTOK_IN + completion / 1e6 * USD_PER_MTOK_OUT
    return {"prompt": prompt, "completion": completion, "cost": cost}


def benchmark() -> list[dict]:
    return [measure_one(task, r) for task in TASKS for r in range(REPEATS)]


# --- REPORT ------------------------------------------------------------------------------
def report(rows: list[dict]) -> None:
    n = len(rows)
    avg_prompt = sum(r["prompt"] for r in rows) / n
    avg_completion = sum(r["completion"] for r in rows) / n

    print(f"provider={LLM_PROVIDER}  model={MODEL}")
    print("=" * 60)
    print("COST")
    print("=" * 60)
    print(f"samples             : {n}")
    print(f"avg prompt tokens   : {avg_prompt:.0f}")
    print(f"avg completion tok  : {avg_completion:.0f}")

    # Token counts are trustworthy on any provider (measured directly). The DOLLAR figure is
    # provider-specific and was wrong to compute with Gemini's rate on anything else -- that
    # was the exact gap eval_report.md flagged. Show the real number for whoever is actually
    # configured, not a borrowed one.
    if LLM_PROVIDER == "groq":
        paid_in, paid_out = _GROQ_PAID_USD_PER_MTOK
        paid_costs = [r["prompt"] / 1e6 * paid_in + r["completion"] / 1e6 * paid_out for r in rows]
        avg_paid = sum(paid_costs) / n
        print("avg cost / request  : $0.000000  (Groq free tier -- $0/token, bounded by the "
              "100k-tokens/day/org quota instead of a bill)")
        print(f"   if on Groq's paid/dev tier instead: ${avg_paid:.6f}/request "
              f"(${paid_in}/${paid_out} per Mtok in/out)")
        avg_cost, budget_note = avg_paid, " (paid-tier estimate; free tier is $0)"
    elif LLM_PROVIDER == "openrouter":
        avg_gemini_equiv = sum(
            r["prompt"] / 1e6 * USD_PER_MTOK_IN + r["completion"] / 1e6 * USD_PER_MTOK_OUT
            for r in rows
        ) / n
        print("avg cost / request  : not stated -- OpenRouter's real rate depends entirely on "
              "which underlying model LLM_MODEL routes to (free vs paid vary by 10-100x)")
        print(f"   at Gemini-equivalent rates, for comparison only: ${avg_gemini_equiv:.6f}/request")
        avg_cost, budget_note = avg_gemini_equiv, " (comparison estimate, not OpenRouter's real rate)"
    else:
        avg_cost = sum(r["cost"] for r in rows) / n
        min_cost, max_cost = min(r["cost"] for r in rows), max(r["cost"] for r in rows)
        print(f"avg cost / request  : ${avg_cost:.6f}")
        print(f"   min / max        : ${min_cost:.6f} / ${max_cost:.6f}  "
              f"(tight range = stable, unlike latency)")
        budget_note = ""
    print("-" * 60)

    daily = avg_cost * REQUESTS_PER_DAY
    print(f"projection @ {REQUESTS_PER_DAY}/day : ${daily:.2f}/day  ${daily * 30:.2f}/month{budget_note}")
    print("-" * 60)

    verdict = "PASS" if avg_cost <= COST_BUDGET_PER_REQUEST_USD else "FAIL"
    print(f"BUDGET: cost/request <= ${COST_BUDGET_PER_REQUEST_USD}  ->  "
          f"${avg_cost:.6f}   [{verdict}]{budget_note}")


def main() -> None:
    report(benchmark())


if __name__ == "__main__":
    main()
