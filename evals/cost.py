"""
# evals/cost.py

**What:** operations eval for cost: real single-turn resolutions priced per request with the provider-aware
`config.price_usd`, projected to daily and monthly spend, checked against a per-request budget.
**Why:** tokens are near-deterministic per task, so cost is answerable offline before anything ships. Pricing comes
from one table in config, so a Groq run is never priced at Gemini rates. Rows are saved, so every published cost
resolves to a file.
**Used by:** `python -m evals.cost`.
"""
import os

os.environ.setdefault("RESOLV_ENV", "eval")
os.environ.setdefault("LITELLM_LOG", "ERROR")

import argparse  # noqa: E402

from resolv import config  # noqa: E402
from evals import provenance  # noqa: E402
from evals.latency import save, timed_run  # noqa: E402

REQUESTS_PER_DAY = 2000
COST_BUDGET_PER_REQUEST_USD = 0.02


# the cost card: mean tokens, mean and range of usd per request, projection, budget verdict
def card(rows: list[dict]) -> dict:
    n = len(rows)
    usd = [r["usd"] for r in rows if r.get("usd") is not None]
    avg = sum(usd) / len(usd) if usd else None
    return {
        "samples": n,
        "model": config.AGENT_MODEL,
        "avg_input_tokens": round(sum(r["input_tokens"] for r in rows) / n) if n else None,
        "avg_output_tokens": round(sum(r["output_tokens"] for r in rows) / n) if n else None,
        "avg_usd_per_request": round(avg, 6) if avg is not None else None,
        "min_max_usd": (min(usd), max(usd)) if usd else None,
        "usd_per_day": round(avg * REQUESTS_PER_DAY, 2) if avg is not None else None,
        "usd_per_month": round(avg * REQUESTS_PER_DAY * 30, 2) if avg is not None else None,
        "budget_pass": avg is not None and avg <= COST_BUDGET_PER_REQUEST_USD,
        "note": "list price; a free tier bills $0 but is capped by its quota",
    }


# CLI: measure, save, print
def main() -> None:
    from evals.tasks import build_tasks

    ap = argparse.ArgumentParser(description="Cost per resolution and spend projection.")
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    run = provenance.manifest("cost", temperature=0.7)
    rows = [timed_run(t, r, "cost") for t in build_tasks(40)[: args.tasks] for r in range(args.n)]
    c = card(rows)
    save("cost", rows, c, run)
    for key, value in c.items():
        print(f"  {key:<22} {value}")


if __name__ == "__main__":
    main()
