"""
Operations eval: LATENCY.

    python -m eval.latency

Deterministic measurement, no golden set and no LLM judge: run the pipeline N times,
collect a distribution, report percentiles against an SLO -- not against a ground truth.

Two numbers, because they answer different questions (same split the reference
eval_latency.py makes between end-to-end and time-to-first-token):

  end-to-end    -- how long until the case is fully resolved (all tool calls done, reply sent)
  first-action  -- how long until the agent's FIRST tool call or reply lands, i.e. how long
                   the customer stares at nothing before anything visibly happens

run_case_events() (agents/loop.py) already yields both endpoints as a stream, one
implementation shared with the Streamlit demo -- so this clocks the real event stream
rather than timing a black-box call.
"""
import asyncio
import time

from agents.loop import run_case_events
from config import LLM_PROVIDER, MODEL
from eval.metrics import _percentile
from eval.tasks import build_tasks
from harness import audit

TASKS = build_tasks(5)
REPEATS = 5
WARMUP_RUNS = 1                # discarded -- cold start (first provider call) skews the tail

SLO_P95_S = 15.0                # full resolution, tail budget
SLO_FIRST_ACTION_P95_S = 6.0    # perceived: time to the first visible thing happening


async def _timed_run(case_id: str, message: str) -> dict:
    t0 = time.perf_counter()
    first_action_s = None
    async for event in run_case_events(case_id, message, temperature=0.7):
        if event["type"] in ("tool_call", "reply") and first_action_s is None:
            first_action_s = time.perf_counter() - t0
    return {"total": time.perf_counter() - t0, "first_action": first_action_s}


def measure_one(task: dict, repeat: int) -> dict:
    case_id = f"lat-{task['task_id']}-r{repeat}"
    audit.clear(case_id)
    audit.clear_order(task["order_id"])
    return asyncio.run(_timed_run(case_id, task["message"]))


def benchmark() -> list[dict]:
    print(f"warming up ({WARMUP_RUNS} run(s), discarded)...")
    for i in range(WARMUP_RUNS):
        measure_one(TASKS[i % len(TASKS)], repeat=999)

    return [measure_one(task, r) for task in TASKS for r in range(REPEATS)]


def report(rows: list[dict]) -> None:
    total = [r["total"] for r in rows]
    first = [r["first_action"] for r in rows if r["first_action"] is not None]

    print(f"provider={LLM_PROVIDER}  model={MODEL}")
    print("=" * 60)
    print("LATENCY (seconds)")
    print("=" * 60)
    print(f"samples            : {len(rows)}")
    print(f"end-to-end   p50={_percentile(total, 50)}  p95={_percentile(total, 95)}")
    print(f"first-action p50={_percentile(first, 50)}  p95={_percentile(first, 95)}")
    print("-" * 60)

    p95_total = _percentile(total, 95)
    p95_first = _percentile(first, 95)
    print(f"SLO end-to-end   <= {SLO_P95_S}s   ->  p95={p95_total}s   "
          f"[{'PASS' if p95_total <= SLO_P95_S else 'FAIL'}]")
    print(f"SLO first-action <= {SLO_FIRST_ACTION_P95_S}s   ->  p95={p95_first}s   "
          f"[{'PASS' if p95_first <= SLO_FIRST_ACTION_P95_S else 'FAIL'}]")


def main() -> None:
    report(benchmark())


if __name__ == "__main__":
    main()
