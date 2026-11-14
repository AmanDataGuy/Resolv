"""
# evals/ablation.py

**What:** harness ON vs harness OFF on the same tasks, simulator, model and temperature. The OFF arm keeps every
tool and every tool result string and only swaps `check_refund` for an always-allow stub, so it is a one-variable
ablation. Reports unauthorized dollars both gross and as excess over what was owed. A second pairing
(`--arm extractor`) runs the harness-on arm with and without the extractor stage (`EXTRACTOR_BACKEND=none`: no
extract() call, no intake hint in the opening), to test whether the extractor earns its serial LLM call.
**Why:** "zero unauthorized refunds" with the harness on does not show the harness caused it; the OFF arm is the
control. The stub is installed for the OFF arm only and restored in a `finally`, so production code has no bypass
flag and a crash can never leave the policy engine disabled. The grader's replay uses its own import of the real
`check_refund`, so OFF-arm refunds are still judged honestly.
**Used by:** `python -m evals.ablation`; tests/test_injection_ablation.py.
"""
import os

os.environ.setdefault("RESOLV_ENV", "eval")
os.environ.setdefault("LITELLM_LOG", "ERROR")

import argparse  # noqa: E402

from tqdm import tqdm  # noqa: E402

from resolv import config  # noqa: E402
from evals import provenance  # noqa: E402
from evals.metrics import scorecard  # noqa: E402
from evals.runner import _one_run  # noqa: E402
from evals.tasks import build_tasks  # noqa: E402
from resolv.harness import policy, tools  # noqa: E402
from resolv.harness.policy import Decision  # noqa: E402

_BYPASSED = Decision(action="allow", rule_id="no_harness", reason="Policy engine disabled (ablation).")


# the OFF arm's policy: allow everything, whatever is asked
def _allow_all(*args, **kwargs) -> Decision:
    return _BYPASSED


# every task x n repeats in one arm; the real check_refund is restored even if a run raises
def run_arm(tasks: list[dict], n: int, arm: str, run: dict | None = None) -> list[dict]:
    originals = (tools.check_refund, policy.check_refund)
    backend = config.EXTRACTOR_BACKEND
    if arm == "off":
        tools.check_refund = policy.check_refund = _allow_all
    if arm == "no_extractor":  # skip extract() and send no intake hint; the harness stays on
        config.EXTRACTOR_BACKEND = "none"
    try:
        rows = []
        for t, i in tqdm([(t, i) for t in tasks for i in range(n)], unit="run", desc=f"harness {arm:<3}"):
            row = _one_run(t, i, run or {}, case_prefix=f"abl-{arm}")
            row["arm"] = arm
            rows.append(row)
        return rows
    finally:
        tools.check_refund, policy.check_refund = originals
        config.EXTRACTOR_BACKEND = backend


# unauthorized dollars in one arm: gross paid, and only the part above what was owed
def unauthorized_usd(rows: list[dict]) -> dict:
    bad = [r for r in rows if r.get("unauthorized")]
    return {"gross": round(sum(r["paid"] for r in bad), 2), "excess": round(sum(r.get("excess_usd") or 0 for r in bad), 2)}


# CLI: run the arms, write rows and card, print the comparison
def main() -> None:
    ap = argparse.ArgumentParser(description="Measure what the policy harness is actually worth.")
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--arm", choices=["both", "on", "off", "no_extractor", "extractor"], default="both",
                    help="both = harness on vs off; extractor = harness on with vs without the extractor")
    ap.add_argument("--out", default="ablation.jsonl")
    args = ap.parse_args()

    tasks = build_tasks(args.tasks)
    arms = {"both": ["on", "off"], "extractor": ["on", "no_extractor"]}.get(args.arm, [args.arm])
    run = provenance.manifest("ablation", temperature=0.7)
    print(f"run_id={run['run_id']}  agent={config.AGENT_MODEL}  "
          f"{len(tasks)} tasks x {args.n} x {len(arms)} arm(s) = {len(tasks) * args.n * len(arms)} runs")
    rows = [r for arm in arms for r in run_arm(tasks, args.n, arm, run)]

    out = provenance.RESULTS / args.out
    provenance.write_jsonl(out, rows)
    cards = {arm: {**scorecard([r for r in rows if r["arm"] == arm], k=args.k),
                   "unauthorized_usd": unauthorized_usd([r for r in rows if r["arm"] == arm])} for arm in arms}
    provenance.write_json(out.with_name(out.stem + "_card.json"), {"manifest": run, "cards": cards})

    keys = ["runs", f"pass^{args.k}", "unauthorized_rate", "invariant_violation_rate", "harmful_block_rate",
            "over_block_rate", "resolve_rate", "unauthorized_usd"]
    print(f"\nablation over {len(rows)} runs -> {out}\n")
    for key in keys:
        print(f"  {key:<26}" + "  ".join(f"{cards[a].get(key)!s:>24}" for a in arms))
    if "off" in cards and cards["off"]["unauthorized_rate"] == 0:
        print("\n  note: the OFF arm leaked nothing on this task set; the harness is not shown to be load-bearing here.")


if __name__ == "__main__":
    main()
