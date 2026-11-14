"""
# evals/latency.py

**What:** operations eval for latency: runs real single-turn resolutions, records end-to-end and first-action
time per request, and reports p50/p90/p95/p99 against SLOs with a task-clustered CI of p95. Also holds the timed
single-run helper and the row/card writer the cost and reliability evals share.
**Why:** a distribution against an SLO, not a ground truth. First-action time (case start to the first tool call
or reply event) is what the customer feels; end-to-end is what the queue feels. Rows are saved, so every
published latency resolves to a file. Warm-up runs are recorded with `warmup: true` and left out of the stats.
**Used by:** `python -m evals.latency`, evals/cost.py, evals/reliability.py.
"""
import os

os.environ.setdefault("RESOLV_ENV", "eval")
os.environ.setdefault("LITELLM_LOG", "ERROR")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import time  # noqa: E402

from resolv import config  # noqa: E402
from evals import provenance  # noqa: E402
from evals.metrics import quantile_ci  # noqa: E402
from resolv.monitor import percentile  # noqa: E402

SLO_P95_S = 15.0
SLO_FIRST_ACTION_P95_S = 6.0


# one single-turn resolution, timed from the event stream, with tokens, cost and error kind
def timed_run(task: dict, repeat: int, prefix: str, warmup: bool = False) -> dict:
    from resolv.agents.loop import run_case_events
    from resolv.agents.llm import new_meter, served_model, tokens_split
    from resolv.harness import audit

    case_id = f"{prefix}-{task['task_id']}-r{repeat}"
    audit.clear(case_id)
    audit.clear_order(task["order_id"])

    # drain the event stream, noting the first visible action and the final step count
    async def drive():
        new_meter()
        t0, first, steps, err = time.perf_counter(), None, 0, None
        try:
            async for event in run_case_events(case_id, task["message"], temperature=0.7,
                                               caller_id=task["customer_id"]):
                if event["type"] in ("tool_call", "reply") and first is None:
                    first = time.perf_counter() - t0
                if event["type"] == "done":
                    steps = event["result"].get("steps", 0)
        except Exception as e:
            err = e
        return time.perf_counter() - t0, first, steps, err, *tokens_split(), served_model()

    e2e, first, steps, err, prompt, completion, served = asyncio.run(drive())
    return {
        "task_id": task["task_id"], "repeat": repeat, "warmup": warmup,
        "first_action_s": round(first, 3) if first is not None else None, "e2e_s": round(e2e, 3), "steps": steps,
        "crashed": err is not None, "error_kind": provenance.error_kind(err) if err else None,
        **({"error": f"{type(err).__name__}: {err}"} if err else {}),
        **provenance.usage_fields(config.AGENT_MODEL, prompt, completion, served),
    }


# write ops rows and their card under evals/results/ops_<script>_<run_id>.*
def save(script: str, rows: list[dict], card: dict, run: dict) -> None:
    out = provenance.RESULTS / f"ops_{script}_{run['run_id']}.jsonl"
    provenance.write_jsonl(out, [{**r, **run} for r in rows])
    provenance.write_json(out.with_name(out.stem + "_card.json"), {"manifest": run, "card": card})
    print(f"rows -> {out}")


# p50/p90/p95/p99 of a field over rows
def percentiles(rows: list[dict], field: str) -> dict:
    values = [r[field] for r in rows if r.get(field) is not None]
    return {f"p{p}": percentile(values, p) for p in (50, 90, 95, 99)}


# the latency card with SLO verdicts and a clustered CI of p95
def card(rows: list[dict]) -> dict:
    measured = [r for r in rows if not r["warmup"]]
    e2e, first = percentiles(measured, "e2e_s"), percentiles(measured, "first_action_s")
    return {
        "samples": len(measured),
        "e2e_s": e2e, "e2e_p95_ci95_clustered": quantile_ci(measured, "e2e_s", 95),
        "first_action_s": first, "first_action_p95_ci95_clustered": quantile_ci(measured, "first_action_s", 95),
        "slo_e2e_p95_pass": e2e["p95"] is not None and e2e["p95"] <= SLO_P95_S,
        "slo_first_action_p95_pass": first["p95"] is not None and first["p95"] <= SLO_FIRST_ACTION_P95_S,
        "note": "small n makes p95 noisy; read the clustered CI",
    }


# CLI: warm up, measure, save, print
def main() -> None:
    from evals.tasks import build_tasks

    ap = argparse.ArgumentParser(description="Latency distribution against SLOs.")
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    tasks = build_tasks(40)[: args.tasks]
    run = provenance.manifest("latency", temperature=0.7)
    rows = [timed_run(tasks[i % len(tasks)], 900 + i, "lat", warmup=True) for i in range(args.warmup)]
    rows += [timed_run(t, r, "lat") for t in tasks for r in range(args.n)]
    c = card(rows)
    save("latency", rows, c, run)
    for key, value in c.items():
        print(f"  {key:<34} {value}")


if __name__ == "__main__":
    main()
