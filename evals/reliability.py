"""
# evals/reliability.py

**What:** operations eval for reliability: does a single resolution finish, or does something still take it down
after the provider-level retries (a malformed tool call, a non-rate-limit provider error)? Crashes are counted by
kind and checked against a success-rate SLO.
**Why:** no retry loop here on purpose; retrying would hide exactly the failures this exists to count. Rows are
saved, so the published success rate resolves to a file.
**Used by:** `python -m evals.reliability`.
"""
import os

os.environ.setdefault("RESOLV_ENV", "eval")
os.environ.setdefault("LITELLM_LOG", "ERROR")

import argparse  # noqa: E402

from evals import provenance  # noqa: E402
from evals.latency import save, timed_run  # noqa: E402

SLO_MIN_SUCCESS_RATE = 0.95


# the reliability card: success rate, crashes by kind, SLO verdict
def card(rows: list[dict]) -> dict:
    n = len(rows)
    ok = sum(1 for r in rows if not r["crashed"])
    kinds: dict[str, int] = {}
    for r in rows:
        if r["crashed"]:
            kinds[r["error_kind"]] = kinds.get(r["error_kind"], 0) + 1
    rate = ok / n if n else 0.0
    return {"requests": n, "successful": ok, "success_rate": round(rate, 4), "crashes_by_kind": kinds,
            "slo_pass": rate >= SLO_MIN_SUCCESS_RATE}


# CLI: measure, save, print
def main() -> None:
    from evals.tasks import build_tasks

    ap = argparse.ArgumentParser(description="Single-attempt success rate.")
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    run = provenance.manifest("reliability", temperature=0.7)
    rows = [timed_run(t, r, "rel") for t in build_tasks(40)[: args.tasks] for r in range(args.n)]
    c = card(rows)
    save("reliability", rows, c, run)
    for key, value in c.items():
        print(f"  {key:<16} {value}")
    for r in rows:
        if r["crashed"]:
            print(f"  FAILED [{r['error_kind']}]: {r.get('error')}")


if __name__ == "__main__":
    main()
