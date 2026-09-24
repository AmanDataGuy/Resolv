"""
Operations eval: RELIABILITY.

    python -m eval.reliability

Does the agent finish, or does it crash? agents/runner_utils.py::complete() already
survives transient rate-limit errors on its own (key rotation, then wait-and-retry --
see its docstring; this project's rate limits are the documented binding constraint,
not an edge case). This measures what's left AFTER that safety net: does run_case()
finish cleanly, or does something else -- a malformed tool call the model can't
recover from, MAX_STEPS exhaustion, a non-rate-limit provider error -- still take the
whole run down?

No retry loop here on purpose: retrying would hide exactly the failures this file
exists to count. eval/runner.py's sweep DOES retry (resumable JSONL), because a sweep's
job is to finish; this file's job is to report the truth about single attempts.
"""
import asyncio

from agents.loop import run_case
from config import LLM_PROVIDER, MODEL
from eval.tasks import build_tasks
from harness import audit

TASKS = build_tasks(5)
REPEATS = 5

SLO_MIN_SUCCESS_RATE = 0.95


def try_one(task: dict, repeat: int) -> dict:
    case_id = f"rel-{task['task_id']}-r{repeat}"
    audit.clear(case_id)
    audit.clear_order(task["order_id"])
    try:
        asyncio.run(run_case(case_id, task["message"], temperature=0.7, caller_id=task["customer_id"]))
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def benchmark() -> list[dict]:
    return [try_one(task, r) for task in TASKS for r in range(REPEATS)]


def report(rows: list[dict]) -> None:
    n = len(rows)
    ok = sum(r["ok"] for r in rows)
    success_rate = ok / n if n else 0.0

    print(f"provider={LLM_PROVIDER}  model={MODEL}")
    print("=" * 60)
    print("RELIABILITY")
    print("=" * 60)
    print(f"total requests : {n}")
    print(f"successful     : {ok}")
    print(f"failed         : {n - ok}")
    print(f"success rate   : {success_rate:.2%}")

    failures = [r for r in rows if not r["ok"]]
    if failures:
        print("-" * 60)
        for r in failures:
            print(f"  FAILED: {r['error']}")

    print("-" * 60)
    verdict = "PASS" if success_rate >= SLO_MIN_SUCCESS_RATE else "FAIL"
    print(f"SLO: success_rate >= {SLO_MIN_SUCCESS_RATE:.0%}  ->  {success_rate:.2%}   [{verdict}]")


def main() -> None:
    report(benchmark())


if __name__ == "__main__":
    main()
