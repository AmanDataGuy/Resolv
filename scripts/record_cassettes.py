"""
# scripts/record_cassettes.py

**What:** records PR-gate cassettes: runs chosen benchmark tasks once against the real provider and saves every
model response, the extractor's claim, the customer's turns and the graded row to
`evals/regression/cassettes/<task_id>.json`. Costs real (small) model calls; run it by hand only.
**Why:** cassettes let CI replay real trajectories with no key. Re-record whenever the agent prompt changes (the
replay test fails loudly on a prompt hash mismatch).
With `--redteam A6-nan A7 ...` it records red-team variants instead (inside their carrier-note overlay, with their
scripted turns) and writes `evals/regression/redteam_subset.jsonl` with each one's expected rule ids and harm flag.
**Used by:** run by hand: `python scripts/record_cassettes.py --tasks case-0026-honest case-0031-pressure ...`.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RESOLV_ENV", "eval")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resolv.agents import loop, llm  # noqa: E402
from evals import provenance, redteam, simulator  # noqa: E402
from evals.replay import CASSETTES, PINNED  # noqa: E402
from evals.runner import _one_run  # noqa: E402
from evals.tasks import build_tasks  # noqa: E402


# run one task (or red-team case) with recording wrappers around the model, extractor and customer
def record(task: dict, redteam_case: bool = False) -> dict:
    responses, turns, claim = [], [], {}
    real_acomplete, real_extract, real_reply = llm.acomplete, loop.extract, simulator.reply

    # record each agent model response
    async def rec_acomplete(**kwargs):
        resp = await real_acomplete(**kwargs)
        responses.append(resp.model_dump())
        return resp

    # record the extractor's read
    async def rec_extract(message):
        claim.update(await real_extract(message) or {})
        return claim

    # record each customer turn
    def rec_reply(t, agent_said, so_far):
        back = real_reply(t, agent_said, so_far)
        turns.append(back)
        return back

    loop.acomplete = llm.acomplete = rec_acomplete
    loop.extract, simulator.reply = rec_extract, rec_reply
    try:
        # a red-team case runs inside its carrier-note overlay with its scripted turns as the customer
        row = redteam.run_one(task, 0, {}) if redteam_case else _one_run(task, 0, {}, case_prefix="rec")
    finally:
        loop.acomplete = llm.acomplete = real_acomplete
        loop.extract, simulator.reply = real_extract, real_reply
    if redteam_case:
        opening, customer = task["turns"][0], task["turns"][1:]  # the replay pops them in order, like scripted_user
    else:
        opening, customer = simulator.opening(task), [t for t in turns if t is not None]
    return {"prompt_sha": provenance.prompt_sha(), "task": task, "opening": opening, "claim": claim,
            "customer_turns": customer, "responses": responses, "expected_row": {k: row[k] for k in PINNED},
            "harm": {h: row[h] for h in redteam.HARMS} if redteam_case else None}


# write one cassette file and return its path
def _save(name: str, cassette: dict) -> Path:
    path = CASSETTES / f"{name}.json"
    path.write_text(json.dumps(cassette, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: {cassette['expected_row']} -> {path}")
    return path


# CLI: record the named benchmark tasks and/or red-team variants (docs/eval/REDTEAM.md section 4)
def main() -> None:
    ap = argparse.ArgumentParser(description="Record replay cassettes for the PR gate (real model calls).")
    ap.add_argument("--tasks", nargs="*", default=[], help="task ids from build_tasks(40)")
    ap.add_argument("--redteam", nargs="*", default=[], help="variant ids from evals/redteam_cases.json")
    ap.add_argument("--orders", type=int, default=1, help="red team: record each variant on this many orders")
    args = ap.parse_args()
    if not args.tasks and not args.redteam:
        ap.error("name at least one --tasks or --redteam id")
    CASSETTES.mkdir(parents=True, exist_ok=True)
    by_id = {t["task_id"]: t for t in build_tasks(40)} if args.tasks else {}
    for tid in args.tasks:
        _save(tid, record(by_id[tid]))

    subset = []
    for case in [c for c in redteam.build_cases(args.orders) if c["variant"] in args.redteam]:
        cassette = record(case, redteam_case=True)
        path = _save(case["task_id"], cassette)
        subset.append({"cassette": path.name, "family": case["family"], "variant": case["variant"],
                       "expected_rule_ids": cassette["expected_row"]["rules_hit"],
                       "harm": int(any(cassette["harm"].values()))})
    if subset:  # the gate wants successful defenses only: harmful recordings are listed so they can be dropped
        out = CASSETTES.parent / "redteam_subset.jsonl"
        provenance.write_jsonl(out, subset)
        print(f"{len(subset)} red-team cassettes -> {out}; harmful: {[s['variant'] for s in subset if s['harm']]}")


if __name__ == "__main__":
    main()
