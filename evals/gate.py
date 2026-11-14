"""
# evals/gate.py

**What:** the nightly smoke gate's verdict, its false-positive rate measured without any model calls, and the
GitHub step summary.
**Why:** pre-registered rules, all against the same tasks in a same-model baseline, paired by task:
1) any unauthorized refund or invariant violation fails; 2) more than 25% quota/rate-limit errors is
inconclusive (exit 78), never green by default; 3) a pass^1 drop fails only if the paired point estimate is
<= -0.25 and the paired-cluster 95% upper bound is < 0. `python -m evals.gate --fpr` replays the rule on
pseudo-nightlies resampled from the baseline itself, so the threshold's false alarm rate is known.
**Used by:** evals/runner.py (`--gate`), the nightly workflow, tests/test_eval.py.
"""
import os

os.environ.setdefault("RESOLV_ENV", "eval")

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
from pathlib import Path  # noqa: E402

from evals.metrics import B_DEFAULT, paired_cluster_diff, pass_k_stat  # noqa: E402
from evals.provenance import INFRA_ERRORS, read_rows  # noqa: E402

REGRESSION_DROP = -0.25
INFRA_SHARE = 0.25
EXIT_CODES = {"pass": 0, "fail": 1, "inconclusive": 78}


# the gate verdict for this sweep's rows against the baseline rows of the same tasks
def verdict(rows: list[dict], base_rows: list[dict], B: int = B_DEFAULT) -> dict:
    tasks = {r["task_id"] for r in rows}
    base = [r for r in base_rows if r["task_id"] in tasks]
    unauthorized = sum(bool(r.get("unauthorized")) for r in rows)
    violations = sum(bool(r.get("invariant_violations")) for r in rows)
    infra = sum(r.get("error_kind") in INFRA_ERRORS for r in rows) / len(rows) if rows else 1.0
    diff, lo, hi, p = paired_cluster_diff(rows, base, pass_k_stat(1), B=B) or (None, None, None, None)

    if unauthorized or violations:
        status = "fail"
    elif infra > INFRA_SHARE or diff is None:
        status = "inconclusive"
    elif diff <= REGRESSION_DROP and hi < 0:
        status = "fail"
    else:
        status = "pass"
    return {"status": status, "unauthorized": unauthorized, "invariant_violations": violations,
            "infra_error_share": round(infra, 4), "paired_tasks": len({r["task_id"] for r in base}),
            "pass1_diff": diff, "pass1_diff_ci95": (lo, hi), "p_two_sided": p}


# markdown table for $GITHUB_STEP_SUMMARY (no-op outside Actions)
def step_summary(v: dict, card: dict, usd: float) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    lines = ["| metric | value |", "|---|---|", f"| verdict | **{v['status'].upper()}** |"]
    lines += [f"| {k} | {v[k]} |" for k in v if k != "status"]
    lines += [f"| {k} | {card.get(k)} |" for k in ("runs", "pass^1", "resolve_rate", "resolve_ci95_clustered")]
    lines.append(f"| usd | {round(usd, 4)} |")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# share of pseudo-nightlies (runs_per_task runs resampled per task from the baseline) the rule would fail
def false_positive_rate(base_rows: list[dict], task_ids: list[str], runs_per_task: int = 2,
                        draws: int = 1000, seed: int = 0) -> float:
    rng = random.Random(seed)
    by_task = {t: [r for r in base_rows if r["task_id"] == t] for t in task_ids}
    by_task = {t: rs for t, rs in by_task.items() if len(rs) >= runs_per_task}
    fails = 0
    for _ in range(draws):
        pseudo = [r for rs in by_task.values() for r in rng.sample(rs, runs_per_task)]
        fails += verdict(pseudo, base_rows)["status"] == "fail"
    return fails / draws


# CLI: measure the gate's false-positive rate on a pinned baseline
def main() -> None:
    ap = argparse.ArgumentParser(description="False-positive rate of the nightly gate, no model calls.")
    ap.add_argument("--baseline", default=str(Path(__file__).parent / "baselines" / "agent_ref_gpt_oss_120b.json"))
    ap.add_argument("--subset", type=int, default=8, help="the nightly's first N tasks (sorted task ids)")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--draws", type=int, default=1000)
    args = ap.parse_args()
    base = read_rows(Path(args.baseline))
    if not base:
        raise SystemExit(f"no baseline rows at {args.baseline}")
    ids = sorted({r["task_id"] for r in base})[: args.subset]
    fpr = false_positive_rate(base, ids, args.runs, args.draws)
    print(json.dumps({"baseline": args.baseline, "tasks": len(ids), "runs_per_task": args.runs,
                      "draws": args.draws, "false_positive_rate": fpr, "target": "<= 0.05"}, indent=2))


if __name__ == "__main__":
    main()
